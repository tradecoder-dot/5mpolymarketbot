"""
Polymarket BTC 5M Up/Down Bot
─────────────────────────────
Paper-trade modu: gerçek para yok, tam simülasyon.
Wallet entegrasyonu en son eklenecek.

Gereksinimler:
    pip install websockets aiohttp numpy python-dotenv

.env:
    POLY_PRIVATE_KEY=0x...       # wallet eklenince
    POLY_FUNDER_ADDRESS=0x...    # wallet eklenince
    POLY_SIGNATURE_TYPE=1

Düzeltme geçmişi (v2):
    FIX-1  DecisionEngine.open_positions artık gerçekten güncelleniyor
    FIX-2  OddsStream callback closure bug'ı giderildi
    FIX-3  resolve başarısız olunca reserved negatife düşmüyor
    FIX-4  Aynı trade iki kez resolve edilemez (idempotency guard)
    FIX-5  aiohttp ClientSession paylaşımlı
    FIX-6  Pencere geçişinde OddsStream doğru token'a yönlendiriliyor
    FIX-7  _handle_kline gereksiz async kaldırıldı
    FIX-8  OddsStream hızlı reconnect + ConnectionClosed ayrı handle
    FIX-9  capital <= 0 guard
    FIX-10 Sharpe için minimum 5 trade şartı

Eklentiler (v3) — kaynak: 72M trade analizi:
    EK-1  MispricingCorrector  — longshot bias düzeltmesi
    EK-2  EV alanları          — ev_per_contract, roi_pct, expected_profit
    EK-3  KellySizerV2         — UP edge yoksa DOWN tarafı flip
    EK-4  LimitPricer          — maker stratejisi, taker maliyeti ortadan kalkar
    EK-5  TemporalPrior        — saate/güne göre tarihsel bias öğrenimi
"""

from __future__ import annotations

import asyncio
import csv
import datetime
import json
import os
import time
from collections import deque
from dataclasses import dataclass
from typing import Literal

import aiohttp
import numpy as np
import websockets
from dotenv import load_dotenv

load_dotenv()   # Localde .env'den okur, Railway'de env var'lardan okur


# ══════════════════════════════════════════════════════════════
# SABITLER
# ══════════════════════════════════════════════════════════════

CLOB_HOST  = "https://clob.polymarket.com"
GAMMA_HOST = "https://gamma-api.polymarket.com"
WS_CLOB    = "wss://ws-subscriptions-clob.polymarket.com/ws/"
WS_RTDS    = "wss://ws-live-data.polymarket.com"        # Polymarket RTDS
WS_BINANCE = "wss://stream.binance.com:9443/ws"         # Binance kline (mum verisi)

# Polymarket RTDS subscription mesajları (dokümantasyon: docs.polymarket.com/RTDS)
_RTDS_SUB_CHAINLINK = {
    "action": "subscribe",
    "subscriptions": [{
        "topic":   "crypto_prices_chainlink",
        "type":    "*",
        "filters": "{\"symbol\":\"btc/usd\"}"
    }]
}
_RTDS_SUB_BINANCE_PRICE = {
    "action": "subscribe",
    "subscriptions": [{
        "topic":   "crypto_prices",
        "type":    "update",
        "filters": "btcusdt"
    }]
}


# ══════════════════════════════════════════════════════════════
# VERİ YAPILARI
# ══════════════════════════════════════════════════════════════

@dataclass
class MarketState:
    btc_returns:   list[float]
    volume_zscore: float
    odds_delta:    float
    p_market:      float


@dataclass
class KellyResult:
    f_full:          float
    f_fractional:    float
    position_usdc:   float
    edge:            float
    expected_growth: float


@dataclass
class PaperTrade:
    trade_id:    int
    ts_open:     str
    slug:        str
    direction:   Literal["up", "down"]
    up_token_id: str
    entry_price: float
    shares:      float
    usdc_spent:  float
    p_true:      float
    edge:        float
    result:      Literal["win", "loss", "pending"] = "pending"
    payout:      float = 0.0
    pnl:         float = 0.0
    ts_close:    str = ""


# ══════════════════════════════════════════════════════════════
# HTTP SESSION YÖNETİCİSİ  (FIX-5)
# ══════════════════════════════════════════════════════════════

class SessionManager:
    """Uygulama boyunca tek bir aiohttp.ClientSession kullanır."""
    _session: aiohttp.ClientSession | None = None

    @classmethod
    async def get(cls) -> aiohttp.ClientSession:
        if cls._session is None or cls._session.closed:
            cls._session = aiohttp.ClientSession()
        return cls._session

    @classmethod
    async def close(cls):
        if cls._session and not cls._session.closed:
            await cls._session.close()
            cls._session = None


# ══════════════════════════════════════════════════════════════
# EK-1: MISPRİCİNG CORRECTOR
# ══════════════════════════════════════════════════════════════

class MispricingCorrector:
    """
    Polymarket longshot bias'ını düzeltir.

    72M trade analizinden çıkarılan gerçek veri noktaları:
        p=0.01 → actual win rate = 0.43%   (bias = -0.57pp)
        p=0.05 → actual win rate = 4.18%   (bias = -0.82pp)
        p=0.10 → actual win rate = 9.10%   (bias = -0.90pp)
        p=0.50 → actual win rate = 50.0%   (bias =  0.00pp)
        p=0.90 → actual win rate = 91.2%   (bias = +1.20pp)
        p=0.95 → actual win rate = 96.7%   (bias = +1.70pp)

    Model (veri noktalarına fit edilmiş):
        bias(p) = alpha × |p − 0.5|^beta × sign(p − 0.5)
        alpha=0.0402, beta=1.7185

    5M BTC piyasasına etkisi:
        Düşük fiyat → Bayesian prior düşer (longshot cezası)
        Yüksek fiyat → Bayesian prior artar (near-certainty bonusu)
        Orta bölge (0.30–0.70) neredeyse değişmez

    Kalibre edilecek katsayılar:
        alpha : 0.0402
        beta  : 1.7185
    """

    def __init__(
        self,
        alpha: float = 0.0402,   # kalibre edilecek
        beta:  float = 1.7185,   # kalibre edilecek
    ):
        self.alpha = alpha
        self.beta  = beta

    def correct(self, p_market: float) -> float:
        p    = float(np.clip(p_market, 1e-4, 1 - 1e-4))
        d    = p - 0.5
        bias = self.alpha * (np.abs(d) ** self.beta) * np.sign(d)
        return float(np.clip(p + bias, 1e-4, 1 - 1e-4))


# ══════════════════════════════════════════════════════════════
# EK-5: TEMPORAL PRIOR
# ══════════════════════════════════════════════════════════════

class TemporalPrior:
    """
    Saate ve güne göre tarihsel UP yönü olasılığını öğrenir.
    Başlangıçta nötr (0.5), her resolve'dan sonra güncellenir.

    Depolama: {(weekday 0-6, hour 0-23): [up_wins, total]} dict

    Laplace smoothing (add-k):
        prior = (up_wins + k) / (total + 2k)
        k=2.5 → zayıf başlangıç prior, veriye hızlı uyum

    min_observations eşiği:
        Altında 0.5 döner — erken aşırı uyumu önler

    BayesianUpdater'a entegrasyon (temporal_weight=0.20):
        blended = (1 − w) × p_corrected + w × temporal_prior

    Kalibre edilecek katsayılar:
        min_observations : 10
        smoothing_k      : 2.5
        temporal_weight  : 0.20  (BayesianUpdater'da)
    """

    def __init__(
        self,
        min_observations: int   = 10,    # kalibre edilecek
        smoothing_k:      float = 2.5,   # kalibre edilecek
    ):
        self.min_observations = min_observations
        self.smoothing_k      = smoothing_k
        self._stats: dict[tuple, list] = {}

    def get_prior(self, dt: datetime.datetime | None = None) -> float:
        key = self._key(dt)
        if key not in self._stats:
            return 0.5
        wins, total = self._stats[key]
        if total < self.min_observations:
            return 0.5
        k = self.smoothing_k
        return float((wins + k) / (total + 2 * k))

    def record(self, outcome: str, dt: datetime.datetime | None = None) -> None:
        """outcome: gerçek sonuç ("up" veya "down")"""
        key = self._key(dt)
        if key not in self._stats:
            self._stats[key] = [0, 0]
        self._stats[key][1] += 1
        if outcome == "up":
            self._stats[key][0] += 1

    def _key(self, dt: datetime.datetime | None) -> tuple:
        t = dt if dt is not None else datetime.datetime.now()
        return (t.weekday(), t.hour)

    @property
    def total_observations(self) -> int:
        return sum(v[1] for v in self._stats.values())


# ══════════════════════════════════════════════════════════════
# BAYESIAN UPDATER  (EK-1 + EK-5 entegre)
# ══════════════════════════════════════════════════════════════

class BayesianUpdater:
    """
    Her 5M mumunda posterior P(BTC yükselir) hesaplar.

    Pipeline:
        1. p_market → MispricingCorrector → p_corrected   (EK-1)
        2. p_corrected + TemporalPrior → blended           (EK-5)
        3. blended → log-odds prior
        4. + price LLR + volume LLR + odds LLR
        5. → posterior

    Kalibre edilecek katsayılar:
        price_signal_strength : 0.3
        volume_weight         : 0.1   | clip (-0.5, 0.5)
        odds_multiplier       : 5.0   | clip (-1.0, 1.0)
        temporal_weight       : 0.20  (EK-5)
        vol_window            : 20 mum (BinanceFeed'de)
    """

    def __init__(
        self,
        price_signal_strength: float = 0.3,
        volume_weight:         float = 0.1,
        volume_clip:           float = 0.5,
        odds_multiplier:       float = 5.0,
        odds_clip:             float = 1.0,
        temporal_weight:       float = 0.20,   # kalibre edilecek (EK-5)
        corrector:  MispricingCorrector | None = None,
        temporal:   TemporalPrior       | None = None,
    ):
        self.price_signal_strength = price_signal_strength
        self.volume_weight         = volume_weight
        self.volume_clip           = volume_clip
        self.odds_multiplier       = odds_multiplier
        self.odds_clip             = odds_clip
        self.temporal_weight       = temporal_weight
        self.corrector             = corrector or MispricingCorrector()
        self.temporal              = temporal  or TemporalPrior()

    @staticmethod
    def _log_odds(p: float) -> float:
        p = float(np.clip(p, 1e-6, 1 - 1e-6))
        return float(np.log(p / (1 - p)))

    @staticmethod
    def _from_log_odds(lo: float) -> float:
        lo = float(np.clip(lo, -500, 500))
        return float(1 / (1 + np.exp(-lo)))

    def _price_llr(self, returns: list[float], direction: int) -> float:
        if len(returns) < 3:
            return 0.0
        mu    = float(np.mean(returns))
        sigma = float(np.std(returns)) + 1e-8
        mu_h1 = direction * sigma * self.price_signal_strength
        mu_h0 = -mu_h1
        ll_h1 = -0.5 * ((mu - mu_h1) / sigma) ** 2
        ll_h0 = -0.5 * ((mu - mu_h0) / sigma) ** 2
        return ll_h1 - ll_h0

    def _volume_llr(self, volume_zscore: float) -> float:
        return float(np.clip(
            volume_zscore * self.volume_weight,
            -self.volume_clip, self.volume_clip,
        ))

    def _odds_llr(self, odds_delta: float) -> float:
        return float(np.clip(
            odds_delta * self.odds_multiplier,
            -self.odds_clip, self.odds_clip,
        ))

    def update(self, state: MarketState, direction: int = 1) -> float:
        # EK-1: Mispricing düzeltmesi
        p_corrected = self.corrector.correct(state.p_market)

        # EK-5: Temporal prior ile harmanlama
        t_prior = self.temporal.get_prior()
        w       = self.temporal_weight
        blended = float(np.clip(
            (1 - w) * p_corrected + w * t_prior,
            1e-4, 1 - 1e-4,
        ))

        lo  = self._log_odds(blended)
        lo += self._price_llr(state.btc_returns, direction)
        lo += self._volume_llr(state.volume_zscore)
        lo += self._odds_llr(state.odds_delta)
        return self._from_log_odds(lo)


# ══════════════════════════════════════════════════════════════
# EK-3: KELLY SIZER V2  (NO tarafı flip)
# ══════════════════════════════════════════════════════════════

class KellySizer:
    """
    UP edge yoksa DOWN tarafında edge arar.  (EK-3)

    DOWN token mantığı:
        fiyat        = 1 − p_market
        kazanma_olas = 1 − p_true

    Döndürür: (KellyResult, "up") | (KellyResult, "down") | None

    Kalibre edilecek katsayılar:
        fraction         : 0.25
        max_position_pct : 0.10
        min_edge         : 0.03
        min_confidence   : 0.55
    """

    def __init__(
        self,
        fraction:         float = 0.25,
        max_position_pct: float = 0.10,
        min_edge:         float = 0.03,
        min_confidence:   float = 0.55,
    ):
        self.fraction         = fraction
        self.max_position_pct = max_position_pct
        self.min_edge         = min_edge
        self.min_confidence   = min_confidence

    def compute(
        self, p_true: float, p_market: float, capital: float,
    ) -> tuple[KellyResult, str] | None:
        if capital <= 0:
            print("[Kelly] Sermaye tükendi, işlem yapılamaz.")
            return None

        # UP tarafı
        up = self._side(p_true, p_market, capital)
        if up is not None:
            return up, "up"

        # DOWN tarafı flip
        down = self._side(1.0 - p_true, 1.0 - p_market, capital)
        if down is not None:
            return down, "down"

        return None

    def _side(self, p_win: float, p_price: float, capital: float) -> KellyResult | None:
        # 1e-3 sınırı: 0.1¢ altı / 99.9¢ üstü fiyatlarda b hesabı güvenilmez
        if p_price < 1e-3 or p_price > 1 - 1e-3:
            return None

        edge = p_win - p_price
        if edge < self.min_edge or p_win < self.min_confidence:
            return None

        b      = (1.0 - p_price) / p_price
        q      = 1.0 - p_win
        f_full = max((p_win * b - q) / (b + 1e-8), 0.0)
        f_frac = min(f_full * self.fraction, self.max_position_pct)
        pos    = capital * f_frac

        log_win  = np.log(max(1 + b * f_frac, 1e-8))
        log_loss = np.log(max(1 - f_frac, 1e-8))
        eg       = float(p_win * log_win + q * log_loss) if f_frac > 0 else 0.0

        return KellyResult(f_full, f_frac, pos, edge, eg)


# ══════════════════════════════════════════════════════════════
# EK-4: LIMIT PRICER
# ══════════════════════════════════════════════════════════════

class LimitPricer:
    """
    Maker stratejisi için limit fiyat hesaplar.

    Taker maliyeti  : −1.12% per trade  (72M trade analizi)
    Maker kazancı   : +1.12% per trade

    Mantık:
        UP  tarafı → bid + actual_offset
        DOWN tarafı → ask − actual_offset
        actual_offset = min(offset, spread×0.5, max_offset)

    Paper-trade fill simülasyonu:
        entry ≥ ask  → her zaman dolar (taker)
        entry <  ask → fill_prob = clip(1 − distance×10, 0.20, 0.95)
        (20→10: spread tahmini belirsizken daha geniş fill penceresi)

    Kalibre edilecek katsayılar:
        offset     : 0.005  (0.01→0.005: daha agresif limit)
        max_offset : 0.02
        fill_mult  : 10     (20→10: daha fazla fill)
    """

    def __init__(
        self,
        offset:     float = 0.005,   # kalibre edilecek
        max_offset: float = 0.02,    # kalibre edilecek
        fill_mult:  float = 10.0,    # kalibre edilecek
    ):
        self.offset     = offset
        self.max_offset = max_offset
        self.fill_mult  = fill_mult

    def get_limit_price(self, side: str, spread: dict) -> float:
        # Spread yoksa midpoint tahmin et
        bid = spread.get("bid") or 0.0
        ask = spread.get("ask") or 1.0
        sp  = spread.get("spread") or (ask - bid)

        actual_offset = min(self.offset, sp * 0.5, self.max_offset)
        actual_offset = max(actual_offset, 0.0)

        price = (bid + actual_offset) if side == "up" else (ask - actual_offset)
        return float(np.clip(price, 0.01, 0.99))

    def simulate_fill(
        self,
        entry_price: float,
        spread:      dict,
        rng:         np.random.Generator | None = None,
    ) -> bool:
        ask = spread.get("ask") or 1.0
        if entry_price >= ask:
            return True
        distance  = ask - entry_price
        fill_prob = float(np.clip(1.0 - distance * self.fill_mult, 0.20, 0.95))
        r = rng if rng is not None else np.random.default_rng()
        return bool(r.random() < fill_prob)


# ══════════════════════════════════════════════════════════════
# EK-2: EV HESABI (DecisionEngine yardımcısı)
# ══════════════════════════════════════════════════════════════

def _compute_ev_fields(
    p_true: float, p_market: float, position_usdc: float,
    side: str = "up",
) -> dict:
    """
    EV formülleri — yöne göre doğru hesap:

    UP token:  ev = p_true - p_market,       token_price = p_market
    DOWN token: ev = p_market - p_true,      token_price = 1 - p_market
    """
    if side == "up":
        ev          = p_true - p_market
        token_price = p_market
    else:
        ev          = p_market - p_true       # DOWN doğru formül
        token_price = 1.0 - p_market          # DOWN token fiyatı

    tp  = max(token_price, 1e-8)
    roi = (ev / tp) * 100
    exp = position_usdc * (ev / tp)
    return {
        "ev_per_contract": round(ev,  4),
        "roi_pct":         round(roi, 2),
        "expected_profit": round(exp, 2),
    }


# ══════════════════════════════════════════════════════════════
# DECISION ENGINE  (EK-2 + EK-3 entegre)
# ══════════════════════════════════════════════════════════════

class DecisionEngine:
    def __init__(self, updater: BayesianUpdater, sizer: KellySizer):
        self.updater         = updater
        self.sizer           = sizer
        self.open_positions: set[str] = set()

    def evaluate(
        self,
        market_id:   str,
        state:       MarketState,
        capital:     float,
        direction:   int = 1,
        odds_source: str = "none",   # hub.odds_poller.active_source'dan gelir
    ) -> dict:
        p_true = self.updater.update(state, direction)

        if market_id in self.open_positions:
            return {"action": "hold", "reason": "açık pozisyon var",
                    "p_true": p_true, "p_market": state.p_market}

        # Odds kaynağına göre dinamik min_edge
        original_min_edge = self.sizer.min_edge
        if odds_source == "rest":
            self.sizer.min_edge = max(original_min_edge, 0.06)
        elif odds_source == "none":
            self.sizer.min_edge = max(original_min_edge, 0.08)

        outcome = self.sizer.compute(p_true, state.p_market, capital)
        self.sizer.min_edge = original_min_edge   # her zaman geri al

        if outcome is None:
            return {"action": "hold", "reason": "edge yetersiz",
                    "p_true": p_true, "p_market": state.p_market}

        result, side = outcome
        ev_fields = _compute_ev_fields(p_true, state.p_market, result.position_usdc, side=side)

        return {
            "action":      "open_long" if side == "up" else "open_short",
            "side":        side,
            "market_id":   market_id,
            "usdc":        result.position_usdc,
            "p_true":      p_true,
            "p_market":    state.p_market,
            "edge":        result.edge,
            "f_kelly":     result.f_fractional,
            "odds_source": odds_source,
            **ev_fields,
        }

    def register_open(self, market_id: str):
        self.open_positions.add(market_id)

    def register_close(self, market_id: str):
        self.open_positions.discard(market_id)


# ══════════════════════════════════════════════════════════════
# PAPER WALLET
# ══════════════════════════════════════════════════════════════

class PaperWallet:
    def __init__(self, starting_capital: float = 10_000.0):
        self.starting_capital = starting_capital
        self.balance          = starting_capital
        self.reserved         = 0.0
        self._trades:    list[PaperTrade] = []
        self._trade_ctr: int = 0
        self.wins   = 0
        self.losses = 0

    @property
    def equity(self) -> float:
        return self.balance + self.reserved

    @property
    def total_pnl(self) -> float:
        return self.equity - self.starting_capital

    @property
    def open_trades(self) -> list[PaperTrade]:
        return [t for t in self._trades if t.result == "pending"]

    @property
    def closed_trades(self) -> list[PaperTrade]:
        return [t for t in self._trades if t.result != "pending"]

    def open_position(
        self,
        slug:        str,
        direction:   Literal["up", "down"],
        up_token_id: str,
        entry_price: float,
        usdc_amount: float,
        p_true:      float,
        edge:        float,
    ) -> PaperTrade | None:
        if usdc_amount <= 0:
            return None
        if usdc_amount > self.balance:
            print(f"[Wallet] Yetersiz bakiye: {self.balance:.2f} < {usdc_amount:.2f}")
            return None
        if entry_price <= 0 or entry_price >= 1:
            print(f"[Wallet] Geçersiz entry_price: {entry_price}")
            return None

        shares         = usdc_amount / entry_price
        self.balance  -= usdc_amount
        self.reserved += usdc_amount
        self._trade_ctr += 1

        trade = PaperTrade(
            trade_id=self._trade_ctr,
            ts_open=time.strftime("%Y-%m-%d %H:%M:%S"),
            slug=slug, direction=direction, up_token_id=up_token_id,
            entry_price=entry_price, shares=shares,
            usdc_spent=usdc_amount, p_true=p_true, edge=edge,
        )
        self._trades.append(trade)

        print(f"[Wallet] #{trade.trade_id} AÇILDI  "
              f"{direction.upper():4s}  "
              f"{usdc_amount:.2f} USDC @ {entry_price:.3f}  "
              f"→ {shares:.2f} share")
        return trade

    def resolve(self, trade: PaperTrade, outcome: Literal["up", "down"]):
        if trade.result != "pending":
            print(f"[Wallet] #{trade.trade_id} zaten resolve edilmiş.")
            return

        won    = (trade.direction == outcome)
        payout = trade.shares if won else 0.0
        pnl    = payout - trade.usdc_spent

        trade.result   = "win" if won else "loss"
        trade.payout   = payout
        trade.pnl      = pnl
        trade.ts_close = time.strftime("%Y-%m-%d %H:%M:%S")

        release        = min(trade.usdc_spent, self.reserved)
        self.reserved -= release
        self.balance  += payout

        self.wins   += 1 if won else 0
        self.losses += 0 if won else 1

        icon = "✓" if won else "✗"
        print(
            f"[Wallet] #{trade.trade_id} {icon}  "
            f"{'KAZANILDI' if won else 'KAYBEDİLDİ':10s}  "
            f"P&L: {pnl:+.2f} USDC  Bakiye: {self.balance:.2f}"
        )

    def refund(self, trade: PaperTrade):
        if trade.result != "pending":
            return
        release        = min(trade.usdc_spent, self.reserved)
        self.reserved -= release
        self.balance  += trade.usdc_spent
        trade.result   = "loss"
        print(f"[Wallet] #{trade.trade_id} USDC iade edildi.")

    def summary(self) -> dict:
        closed = self.closed_trades
        if not closed:
            return {"trades": 0}

        pnls     = [t.pnl for t in closed]
        win_rate = self.wins / len(closed)
        avg_pnl  = float(np.mean(pnls))

        if len(closed) >= 5:
            std_pnl = float(np.std(pnls, ddof=1)) + 1e-8
            sharpe  = (avg_pnl / std_pnl) * np.sqrt(len(closed))
        else:
            sharpe = float("nan")

        equity_curve = [self.starting_capital]
        for t in closed:
            equity_curve.append(equity_curve[-1] + t.pnl)
        peak, max_dd = equity_curve[0], 0.0
        for e in equity_curve:
            peak   = max(peak, e)
            max_dd = min(max_dd, e - peak)

        return {
            "trades": len(closed), "wins": self.wins, "losses": self.losses,
            "win_rate": win_rate, "total_pnl": float(np.sum(pnls)),
            "avg_pnl": avg_pnl, "sharpe": sharpe,
            "max_drawdown": max_dd, "final_equity": self.equity,
            "roi_pct": (self.equity / self.starting_capital - 1) * 100,
        }

    def save_log(self, path: str = "papertrade_log.csv"):
        if not self._trades:
            return
        fields = ["trade_id", "ts_open", "ts_close", "slug", "direction",
                  "entry_price", "shares", "usdc_spent", "p_true", "edge",
                  "result", "payout", "pnl"]
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for t in self._trades:
                w.writerow({k: getattr(t, k) for k in fields})
        print(f"[Wallet] Log kaydedildi → {path}")


# ══════════════════════════════════════════════════════════════
# RESOLVE FETCHER
# ══════════════════════════════════════════════════════════════

class ResolveFetcher:
    """
    Pencere kapandıktan sonra sonucu belirler.

    Yöntem 1: Gamma API — market kapatıldıktan sonra 'winning_outcome' alanı dolar
    Yöntem 2: CLOB midpoint — kazanan token ≥0.95 veya ≤0.05

    NOT: market_resolved WS event'i OddsStream üzerinden yakalanıyor,
         oradan gelen sonuç _resolve_cache'e yazılıyor.
    """

    _resolve_cache: dict[str, str] = {}   # slug → "up"|"down"
    _MAX_CACHE     = 100                   # Bu kadar slug sonrası eski yarısını sil

    @classmethod
    def record_resolved(cls, slug: str, outcome: str):
        """OddsStream market_resolved event'inden çağrılır."""
        cls._resolve_cache[slug] = outcome
        print(f"[Resolve] WS event'ten sonuç: {slug} → {outcome}")
        # Cache büyüme kontrolü — 100 slug geçince en eski %50'yi sil
        if len(cls._resolve_cache) > cls._MAX_CACHE:
            to_remove = list(cls._resolve_cache.keys())[:cls._MAX_CACHE // 2]
            for k in to_remove:
                del cls._resolve_cache[k]
            print(f"[Resolve] Cache temizlendi: {len(to_remove)} eski slug silindi")

    async def fetch_outcome(
        self, slug: str, up_token_id: str, retries: int = 24,
        window_open_price: float | None = None,
        price_feed=None,
    ) -> Literal["up", "down"] | None:
        """
        Resolve yöntemleri (öncelik sırasıyla):
        0. WS cache     — OddsStream market_resolved event'i
        1. Chainlink    — window_open_price vs kapanış fiyatı (en güvenilir)
        2. Gamma API    — winning_outcome alanı
        3. CLOB midpoint — son çare
        """

        # Yöntem 0: WS cache
        if slug in self._resolve_cache:
            return self._resolve_cache[slug]

        # Yöntem 1: Chainlink karşılaştırması
        # Polymarket: kapanış >= açılış → UP, aksi → DOWN
        # Pencere kapanınca yeni Chainlink tick'ini bekle (max 15s)
        if window_open_price and price_feed:
            ts_before = price_feed._chainlink_last_ts
            for wait_i in range(5):   # 5 × 3s = 15s max
                await asyncio.sleep(3)
                close_price = price_feed._chainlink_price
                tick_is_new = price_feed._chainlink_last_ts > ts_before
                if close_price and (tick_is_new or wait_i >= 2):
                    outcome = "up" if close_price >= window_open_price else "down"
                    label = "yeni tick" if tick_is_new else "son bilinen"
                    print(f"[Resolve] Chainlink ({label}) → {outcome} "
                          f"(open={window_open_price:.2f}, close={close_price:.2f})")
                    return outcome

        for attempt in range(retries):
            await asyncio.sleep(10)

            # WS cache tekrar
            if slug in self._resolve_cache:
                return self._resolve_cache[slug]

            # Chainlink tekrar dene (fiyat henüz güncellenmemiş olabilir)
            if window_open_price and price_feed:
                close_price = price_feed._chainlink_price
                if close_price and abs(close_price - window_open_price) / window_open_price > 0.0001:
                    outcome = "up" if close_price >= window_open_price else "down"
                    print(f"[Resolve] Chainlink (gecikmeli) → {outcome} "
                          f"(open={window_open_price:.2f}, close={close_price:.2f})")
                    return outcome

            # Gamma API
            try:
                session = await SessionManager.get()
                async with session.get(
                    f"{GAMMA_HOST}/markets",
                    params={"slug": slug},
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as r:
                    data = await r.json()

                market = data[0] if isinstance(data, list) and data else {}
                winner = market.get("winning_outcome", "")
                if winner:
                    outcome = "up" if winner.lower() in ("up", "yes", "1") else "down"
                    print(f"[Resolve] Gamma API → {outcome} (winner='{winner}')")
                    return outcome

                if market.get("resolved"):
                    print(f"[Resolve] Market resolved ama winner yok ({attempt+1}/{retries})")

            except Exception:
                pass

            # CLOB midpoint
            try:
                session = await SessionManager.get()
                async with session.get(
                    f"{CLOB_HOST}/midpoint",
                    params={"token_id": up_token_id},
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as r:
                    data = await r.json()
                mid = float(data.get("mid", 0.5))
                if mid >= 0.95:
                    return "up"
                if mid <= 0.05:
                    return "down"
                if attempt % 3 == 0:
                    print(f"[Resolve] mid={mid:.3f} bekleniyor ({attempt+1}/{retries})")
            except Exception as e:
                print(f"[Resolve] Hata ({attempt+1}/{retries}): {e}")

        print(f"[Resolve] {slug} sonucu alınamadı.")
        return None


# ══════════════════════════════════════════════════════════════
# BTC 5M MARKET
# ══════════════════════════════════════════════════════════════

class BTC5mMarket:
    """
    Deterministik slug hesaplama + çoklu timestamp denemesi.

    Sorun: local saat ile Polymarket server saati arasında birkaç saniyelik
    fark olabilir. Bu fark pencere başında yanlış slug üretmesine yol açar.

    Çözüm:
        1. Mevcut pencere slug'ını dene
        2. Bir önceki pencere slug'ını dene (geçiş anında faydalı)
        3. events endpoint'ini dene (markets'a alternatif)
        Her adımda token ID bulunursa hemen döner.
    """

    def get_current_window(self) -> dict:
        now       = int(time.time())
        window_ts = now - (now % 300)
        close_t   = window_ts + 300
        return {
            "slug":              f"btc-updown-5m-{window_ts}",
            "window_ts":         window_ts,
            "close_time":        close_t,
            "seconds_remaining": close_t - now,
        }

    def get_next_window(self) -> dict:
        now       = int(time.time())
        window_ts = now - (now % 300) + 300
        return {
            "slug":       f"btc-updown-5m-{window_ts}",
            "window_ts":  window_ts,
            "close_time": window_ts + 300,
        }

    def _candidate_slugs(self) -> list[str]:
        """
        Server/client saat farkını tolere etmek için
        birden fazla pencere timestamp'i dener.
        Sıra: mevcut → önceki → sonraki pencere
        """
        now       = int(time.time())
        window_ts = now - (now % 300)
        return [
            f"btc-updown-5m-{window_ts}",         # mevcut pencere
            f"btc-updown-5m-{window_ts - 300}",   # bir önceki
            f"btc-updown-5m-{window_ts + 300}",   # bir sonraki
        ]

    async def _fetch_via_markets(self, slug: str) -> dict | None:
        """gamma-api/markets endpoint'i ile token ID çek."""
        try:
            session = await SessionManager.get()
            async with session.get(
                f"{GAMMA_HOST}/markets",
                params={"slug": slug},
                timeout=aiohttp.ClientTimeout(total=5),
            ) as r:
                if r.status != 200:
                    return None
                data = await r.json()
            if not data:
                return None
            market = data[0] if isinstance(data, list) else data

            # Gamma API hem camelCase hem snake_case döndürebilir
            token_ids = (
                market.get("clobTokenIds")       # camelCase (yeni format)
                or market.get("clob_token_ids")  # snake_case (eski format)
                or []
            )
            # String olarak gelebilir — parse et
            if isinstance(token_ids, str):
                import json as _json
                try:
                    token_ids = _json.loads(token_ids)
                except Exception:
                    token_ids = []

            if not token_ids or not token_ids[0]:
                return None

            return {
                "slug":          slug,
                "up_token_id":   str(token_ids[0]),
                "down_token_id": str(token_ids[1]) if len(token_ids) > 1 else None,
                "question":      market.get("question", slug),
            }
        except Exception:
            return None

    async def _fetch_via_events(self, slug: str) -> dict | None:
        """
        gamma-api/events endpoint'i ile token ID çek.
        markets endpoint başarısız olunca yedek yol.
        """
        try:
            session = await SessionManager.get()
            # events endpoint'i slug ile sorgulanıyor
            async with session.get(
                f"{GAMMA_HOST}/events",
                params={"slug": slug},
                timeout=aiohttp.ClientTimeout(total=5),
            ) as r:
                if r.status != 200:
                    return None
                data = await r.json()
            if not data:
                return None
            event   = data[0] if isinstance(data, list) else data
            markets = event.get("markets", [])
            if not markets:
                return None

            market = markets[0]
            # camelCase ve snake_case her ikisini dene
            token_ids = (
                market.get("clobTokenIds")
                or market.get("clob_token_ids")
                or []
            )
            if isinstance(token_ids, str):
                import json as _json
                try:
                    token_ids = _json.loads(token_ids)
                except Exception:
                    token_ids = []

            if not token_ids or not token_ids[0]:
                return None

            return {
                "slug":          slug,
                "up_token_id":   str(token_ids[0]),
                "down_token_id": str(token_ids[1]) if len(token_ids) > 1 else None,
                "question":      market.get("question", event.get("title", slug)),
            }
        except Exception:
            return None

    async def fetch_token_ids(self, slug: str | None = None) -> dict | None:
        """
        Token ID'leri çek.
        slug verilmezse mevcut pencereyi otomatik hesaplar.
        Birden fazla slug + iki endpoint dener — ilk başarılıda döner.
        """
        candidates = self._candidate_slugs() if slug is None else [slug]

        for try_slug in candidates:
            # Yol 1: markets endpoint
            result = await self._fetch_via_markets(try_slug)
            if result:
                if try_slug != candidates[0]:
                    print(f"[Market] Alternatif slug bulundu: {try_slug}")
                return result

            # Yol 2: events endpoint
            result = await self._fetch_via_events(try_slug)
            if result:
                if try_slug != candidates[0]:
                    print(f"[Market] Events endpoint ile bulundu: {try_slug}")
                return result

        print(f"[Market] Token ID bulunamadı "
              f"({len(candidates)} slug denendi: {candidates[0]} ...)")
        return None


# ══════════════════════════════════════════════════════════════
# TIMING CONTROLLER
# ══════════════════════════════════════════════════════════════

class TimingController:
    """Kalibre edilecek: entry_before_close = 60s (10–120s arası)"""

    def __init__(self, entry_before_close: int = 60):
        self.entry_before_close = entry_before_close

    def seconds_to_entry(self, close_time: int) -> float:
        return (close_time - self.entry_before_close) - time.time()

    async def wait_for_entry(self, close_time: int):
        wait = self.seconds_to_entry(close_time)
        if wait > 0:
            print(f"[Timing] Entry için {wait:.0f}s bekleniyor...")
            await asyncio.sleep(wait)


# ══════════════════════════════════════════════════════════════
# ODDS STREAM  (FIX-2, FIX-8)
# ══════════════════════════════════════════════════════════════

class OddsHub:
    """
    RTDS primary + REST fallback — çift kaynaklı odds sistemi.

    PRIMARY — RTDS Chainlink fiyat değişimi:
        Polymarket RTDS'ten gelen Chainlink tick'leri zaten açık.
        BTC fiyatının hızı ve yönü = piyasanın odds yönüyle yüksek korelasyon.
        Fiyat hızla yükseliyorsa UP token pahalılaşır → odds_delta pozitif.
        Bu sinyal milisaniye hassasiyetinde, WS bağlantısı RTDS'e bağlı.

    FALLBACK — REST /midpoint polling:
        CLOB WS'in "silent freeze" ve kopma sorununa karşı.
        Her POLL_INTERVAL saniyede bir CLOB /midpoint sorgulanır.
        RTDS'ten son X saniyedir güncelleme gelmemişse devreye girer.
        Spread de aynı döngüde /spread endpoint'inden alınır.

    Geçiş mantığı:
        rtds_last_update > RTDS_TIMEOUT → REST fallback aktif
        Aksi halde RTDS primary kullanılır

    Kalibre edilecek:
        POLL_INTERVAL  : 10s
        RTDS_TIMEOUT   : 15s  (bu kadar Chainlink yoksa REST'e geç)
    """

    POLL_INTERVAL = 10    # REST fallback polling aralığı (saniye)
    RTDS_TIMEOUT  = 25    # Chainlink tick aralığı 10-30s → 25s timeout optimal

    def __init__(self):
        self._last_mid:         float | None = None
        self._last_rtds_price:  float | None = None
        self._last_rtds_ts:     float        = 0.0
        self._running  = True

    def stop(self):
        self._running = False

    def reset(self):
        """Yeni pencere başlayınca delta sıfırla — source sıfırlanmaz."""
        self._last_mid        = None
        self._last_rtds_price = None
        # _last_rtds_ts sıfırlanmıyor: freshness kontrolü için korunur
        # _source sıfırlanmıyor: pencere boyunca en iyi kaynak takip edilir

    @property
    def active_source(self) -> str:
        """
        Karar anındaki odds veri kaynağı:
          - RTDS son RTDS_TIMEOUT içinde tick geldiyse → "rtds"
          - REST fallback en son çalıştıysa → "rest"
          - Hiç veri yoksa → "none"
        """
        if self._rtds_is_fresh():
            return "rtds"
        if self._last_mid is not None:
            return "rest"
        return "none"

    def on_rtds_price(
        self,
        chainlink_price: float,
        odds_ref: list,
        p_ref:    list,
    ) -> None:
        """
        PriceFeed'den her Chainlink tick'inde çağrılır.

        Değişiklik: delta proxy yerine tick varlığı yeterli sinyal.
        Fiyat değişimi odds_delta'yı günceller, küçük harekette bile
        "rtds" kaynağı aktif sayılır çünkü anlık piyasa bilgisi var.
        """
        now = time.time()

        if self._last_rtds_price is not None:
            pct_change = (chainlink_price - self._last_rtds_price) / self._last_rtds_price
            # Katsayı 0.2 → 0.5: daha güçlü sinyal
            # Filtre 0.0001 → kaldırıldı: her tick sinyal üretir
            delta = float(np.clip(pct_change * 0.5, -0.05, 0.05))
            odds_ref[0] = delta
            p_ref[0]    = float(np.clip(p_ref[0] + delta, 0.01, 0.99))

        self._last_rtds_price = chainlink_price
        self._last_rtds_ts    = now

    def _rtds_is_fresh(self) -> bool:
        return (time.time() - self._last_rtds_ts) < self.RTDS_TIMEOUT

    async def run(
        self,
        get_token_id,      # callable → str | None
        odds_ref: list,
        p_ref:    list,
        spread_cache_fn,   # callable(bid, ask)
    ):
        """REST fallback döngüsü — RTDS sessizse devreye girer."""
        print("[OddsHub] Başladı (RTDS primary + REST fallback)")

        while self._running:
            await asyncio.sleep(self.POLL_INTERVAL)

            if not self._running:
                break

            try:
                token_id = get_token_id()
                if not token_id or len(token_id) < 10:
                    continue

                # RTDS taze veri veriyorsa REST'e gerek yok
                # Sadece spread almak için yine de sorgula
                session = await SessionManager.get()

                # Midpoint — RTDS sessizse p_market güncelle
                if not self._rtds_is_fresh():
                    async with session.get(
                        f"{CLOB_HOST}/midpoint",
                        params={"token_id": token_id},
                        timeout=aiohttp.ClientTimeout(total=5),
                    ) as r:
                        data = await r.json()

                    mid = float(np.clip(float(data.get("mid", 0.5)), 0.01, 0.99))

                    if self._last_mid is not None:
                        delta = mid - self._last_mid
                        odds_ref[0] = delta
                        p_ref[0]    = float(np.clip(p_ref[0] + delta, 0.01, 0.99))
                    else:
                        p_ref[0] = mid

                    self._last_mid = mid
                    print(f"[OddsHub] REST fallback: mid={mid:.3f}")

                # Spread — her zaman REST'ten al (RTDS'te yok)
                try:
                    async with session.get(
                        f"{CLOB_HOST}/spread",
                        params={"token_id": token_id},
                        timeout=aiohttp.ClientTimeout(total=5),
                    ) as r:
                        sdata = await r.json()

                    bid = float(sdata.get("bid", 0) or 0)
                    ask = float(sdata.get("ask", 1) or 1)
                    if 0 < bid < ask < 1:
                        spread_cache_fn(bid, ask)
                except Exception:
                    pass

            except asyncio.CancelledError:
                break
            except Exception:
                pass

        print("[OddsHub] Durduruldu")


# Geriye dönük uyumluluk için alias
RestOddsPoller = OddsHub


# ══════════════════════════════════════════════════════════════
# PRICE FEED  — Chainlink primary + Binance fallback
# ══════════════════════════════════════════════════════════════

class PriceFeed:
    """
    Polymarket RTDS üzerinden BTC fiyatı.
    Tek WS bağlantısı, iki topic:
        crypto_prices_chainlink  → PRIMARY (resolution kaynağıyla aynı)
        crypto_prices            → FALLBACK (Binance, Chainlink stale ise devreye girer)

    Chainlink max_chainlink_age saniyedir güncelleme gelmezse otomatik
    Binance'e geçer, Chainlink geri gelince primary'e döner.

    Mum verisi (getiri/hacim) için Binance kline stream'i ayrıca açılır.
    RTDS tick-bazlı fiyat verir — mum kapanışını Binance'ten alıyoruz.

    Ayrıca her yeni 5M penceresinde set_window_open() çağrılmalıdır.
    Bu çağrı anlık Chainlink fiyatını window_open_price olarak kilitler.

    Kalibre edilecek katsayılar:
        vol_window        : 20 mum
        max_chainlink_age : 10s  (Chainlink timeout eşiği)

    RTDS bağlantısını canlı tutmak için her 5 saniyede PING gönderilir
    (Polymarket dokümantasyonu zorunluluğu).
    """

    PING_INTERVAL = 5   # saniye

    def __init__(
        self,
        vol_window:        int   = 20,    # kalibre edilecek
        max_chainlink_age: float = 10.0,  # kalibre edilecek
    ):
        self.vol_window        = vol_window
        self.max_chainlink_age = max_chainlink_age

        self._returns: deque[float] = deque(maxlen=vol_window)
        self._volumes: deque[float] = deque(maxlen=vol_window)

        self._chainlink_price:   float | None = None
        self._chainlink_last_ts: float        = 0.0
        self._binance_price:     float | None = None
        self._window_open_price: float | None = None
        self._using_fallback:    bool = False

        self._latest_state: MarketState | None = None

    # ── Fiyat erişimi ──────────────────────────────────────

    @property
    def active_source(self) -> str:
        if self._is_chainlink_fresh():
            return "chainlink"
        if self._binance_price is not None:
            return "binance_fallback"
        return "none"

    @property
    def window_open_price(self) -> float | None:
        return self._window_open_price

    def _is_chainlink_fresh(self) -> bool:
        if self._chainlink_price is None:
            return False
        return (time.time() - self._chainlink_last_ts) < self.max_chainlink_age

    def _current_price(self) -> float | None:
        if self._is_chainlink_fresh():
            return self._chainlink_price
        return self._binance_price

    def set_window_open(self, window_ts: float | None = None):
        """
        Yeni 5M pencere açılınca çağrılır.

        window_ts: pencere başlangıç unix timestamp'i (opsiyonel)
        Chainlink tick timestamp'i kaydedilir — resolve'da referans olarak kullanılır.

        NOT: Polymarket tam window_ts anındaki Chainlink değerini kullanıyor.
        Bizim son tick'imiz pencere başından en fazla max_chainlink_age saniye önce
        gelebilir. Bu kabul edilebilir — büyük fiyat hareketlerinde fark yaratmaz,
        çok küçük hareketlerde (< $5) resolve belirsizliği zaten kaçınılmaz.
        """
        if self._chainlink_price is not None:
            self._window_open_price = self._chainlink_price
            lag = time.time() - self._chainlink_last_ts
            if lag > 5:
                print(f"[PriceFeed] window_open_price: "
                      f"${self._chainlink_price:,.2f} "
                      f"(son Chainlink tick {lag:.0f}s önce)")

    # ── RTDS mesaj işleme ─────────────────────────────────

    def _handle_rtds_message(
        self, raw: str, odds_ref: list, p_ref: list, odds_hub=None,
    ) -> None:
        try:
            msg   = json.loads(raw)
            topic = msg.get("topic", "")
            mtype = msg.get("type", "")
            if mtype not in ("update", "*"):
                return
            payload = msg.get("payload", {})

            if topic == "crypto_prices_chainlink":
                self._on_chainlink(payload, odds_ref, p_ref, odds_hub=odds_hub)
            elif topic == "crypto_prices":
                self._on_binance_price(payload, odds_ref, p_ref)
        except (json.JSONDecodeError, KeyError, ValueError):
            pass

    def _on_chainlink(
        self, payload: dict, odds_ref: list, p_ref: list,
        odds_hub=None,
    ) -> None:
        if payload.get("symbol") != "btc/usd":
            return
        price = float(payload["value"])
        if price <= 0:
            return

        was_fallback            = self._using_fallback
        self._chainlink_price   = price
        self._chainlink_last_ts = time.time()
        self._using_fallback    = False

        if was_fallback:
            print(f"[PriceFeed] Chainlink geri döndü: ${price:,.2f}")

        # OddsHub RTDS primary sinyali — fiyat değişimi → odds delta proxy
        if odds_hub is not None:
            odds_hub.on_rtds_price(price, odds_ref, p_ref)

        self._update_state(odds_ref, p_ref)

    def _on_binance_price(self, payload: dict, odds_ref: list, p_ref: list) -> None:
        if payload.get("symbol") != "btcusdt":
            return
        price = float(payload["value"])
        if price <= 0:
            return
        self._binance_price = price

        if not self._is_chainlink_fresh():
            if not self._using_fallback:
                self._using_fallback = True
                print(f"[PriceFeed] Chainlink timeout → Binance fallback: ${price:,.2f}")
            self._update_state(odds_ref, p_ref)

    def _update_state(self, odds_ref: list, p_ref: list) -> None:
        price = self._current_price()
        if price is None:
            return
        self._latest_state = MarketState(
            btc_returns=list(self._returns),
            volume_zscore=self._volume_zscore(),
            odds_delta=odds_ref[0],
            p_market=max(0.01, min(0.99, p_ref[0])),
        )

    # ── Binance kline (mum verisi) ─────────────────────────

    def _volume_zscore(self) -> float:
        if len(self._volumes) < 3:
            return 0.0
        arr = np.array(self._volumes)
        return float((arr[-1] - arr.mean()) / (arr.std() + 1e-8))

    def _handle_kline(self, msg: dict, odds_ref: list, p_ref: list) -> None:
        """FIX-7: sync — await kullanmıyor."""
        k = msg["k"]
        if k["x"]:
            open_p  = float(k["o"])
            close_p = float(k["c"])
            if open_p > 0:
                self._returns.append((close_p - open_p) / open_p)
            self._volumes.append(float(k["v"]))
        self._update_state(odds_ref, p_ref)

    def get_state(self) -> MarketState | None:
        return self._latest_state

    def get_returns(self) -> list[float]:
        return list(self._returns)

    def volume_zscore(self) -> float:
        return self._volume_zscore()

    # ── Async run görevleri ────────────────────────────────

    async def run_rtds(self, odds_ref: list, p_ref: list, odds_hub=None):
        """
        Polymarket RTDS: Chainlink + Binance fiyat tick'leri.
        odds_hub: OddsHub referansı — Chainlink tick'lerini RTDS primary sinyali olarak iletir.
        """
        while True:
            try:
                async with websockets.connect(
                    WS_RTDS, ping_interval=None,
                ) as ws:
                    await ws.send(json.dumps(_RTDS_SUB_CHAINLINK))
                    await ws.send(json.dumps(_RTDS_SUB_BINANCE_PRICE))
                    print("[PriceFeed] RTDS bağlandı (Chainlink primary + Binance fallback)")

                    last_ping = time.time()
                    async for raw in ws:
                        self._handle_rtds_message(raw, odds_ref, p_ref, odds_hub=odds_hub)
                        if time.time() - last_ping >= self.PING_INTERVAL:
                            await ws.send("PING")
                            last_ping = time.time()

            except websockets.ConnectionClosed:
                print("[PriceFeed] RTDS koptu, 3s sonra yeniden...")
                await asyncio.sleep(3)
            except Exception as e:
                print(f"[PriceFeed] RTDS hata: {e}")
                await asyncio.sleep(5)

    async def run_kline(self, odds_ref: list, p_ref: list):
        """
        Binance kline stream: 5M mum getirisi + hacim verisi.
        RTDS'ten bağımsız — mum kapanışı Binance'ten alınır.
        """
        url = f"{WS_BINANCE}/btcusdt@kline_5m"
        while True:
            try:
                async with websockets.connect(
                    url, ping_interval=20, ping_timeout=10,
                ) as ws:
                    print("[PriceFeed] Binance kline bağlandı (mum verisi)")
                    async for raw in ws:
                        msg = json.loads(raw)
                        if msg.get("e") == "kline":
                            self._handle_kline(msg, odds_ref, p_ref)
            except websockets.ConnectionClosed:
                print("[PriceFeed] Binance kline koptu, 3s sonra yeniden...")
                await asyncio.sleep(3)
            except Exception as e:
                print(f"[PriceFeed] Binance kline hata: {e}")
                await asyncio.sleep(5)


# ══════════════════════════════════════════════════════════════
# PRICE READER
# ══════════════════════════════════════════════════════════════

class PriceReader:
    async def get_midpoint(self, token_id: str) -> float:
        try:
            session = await SessionManager.get()
            async with session.get(
                f"{CLOB_HOST}/midpoint",
                params={"token_id": token_id},
                timeout=aiohttp.ClientTimeout(total=5),
            ) as r:
                data = await r.json()
            mid = data.get("mid") or data.get("price") or 0.5
            return float(np.clip(float(mid), 0.01, 0.99))
        except Exception as e:
            print(f"[PriceReader] midpoint hata: {e}")
            return 0.5

    async def get_spread(self, token_id: str) -> dict:
        """
        Spread WS cache'den gelir (OddsStream best_bid_ask event'i).
        Cache boşsa midpoint'ten tahmini spread döner.
        Book endpoint token ID formatını kabul etmiyor — kullanmıyoruz.
        """
        # Midpoint'ten tahmini spread hesapla (cache yoksa fallback)
        try:
            session = await SessionManager.get()
            async with session.get(
                f"{CLOB_HOST}/midpoint",
                params={"token_id": token_id},
                timeout=aiohttp.ClientTimeout(total=3),
            ) as r:
                data = await r.json()
            mid = float(np.clip(float(data.get("mid", 0.5)), 0.01, 0.99))
            # 5M BTC piyasasında tipik spread mid'in ~%2-4'ü
            # Bu kaba bir tahmin — WS cache varken kullanılmaz
            estimated_spread = round(mid * 0.03, 3)
            estimated_bid    = round(mid - estimated_spread / 2, 3)
            estimated_ask    = round(mid + estimated_spread / 2, 3)
            return {
                "bid":    estimated_bid,
                "ask":    estimated_ask,
                "spread": estimated_spread,
            }
        except Exception as e:
            print(f"[PriceReader] midpoint fallback hata: {e}")
            return {"bid": 0.47, "ask": 0.53, "spread": 0.06}


# ══════════════════════════════════════════════════════════════
# DATA HUB  (FIX-6)
# ══════════════════════════════════════════════════════════════

class DataHub:
    """
    Tüm veri kaynaklarını koordine eder.

    Odds verisi artık WS (OddsStream) değil REST polling (RestOddsPoller) ile alınıyor.
    Neden: CLOB WS "silent freeze" ve kopma sorunu çözümsüz.
    REST /midpoint güvenilir, her zaman çalışıyor.
    """

    def __init__(self, vol_window: int = 20):
        self.market       = BTC5mMarket()
        self.price_feed   = PriceFeed(vol_window=vol_window)
        self.price_reader = PriceReader()
        self.odds_poller  = RestOddsPoller()

        self._odds_delta  = [0.0]
        self._p_market    = [0.5]
        self._token_ids: dict | None = None

        # Spread cache — REST /spread'den dolduruluyor
        self._cached_spread: dict | None = None

    async def _refresh_market(self):
        window = self.market.get_current_window()
        ids = await self.market.fetch_token_ids()
        if ids:
            self._token_ids = ids
            print(f"[Market] {ids['question']}")
            print(f"[Market] Kapanışa: {window['seconds_remaining']}s")
            if ids["up_token_id"]:
                mid = await self.price_reader.get_midpoint(ids["up_token_id"])
                self._p_market[0] = mid
            # Yeni pencere → spread cache ve odds delta sıfırla
            self._cached_spread  = None
            self._odds_delta[0]  = 0.0
            self.odds_poller.reset()
            self.price_feed.set_window_open()
        return ids

    def cache_spread(self, bid: float, ask: float):
        spread = max(ask - bid, 0.0)
        if spread < 0.5:
            self._cached_spread = {"bid": bid, "ask": ask, "spread": spread}

    def get_cached_spread(self) -> dict | None:
        return self._cached_spread

    async def _market_refresh_loop(self):
        while True:
            await self._refresh_market()
            window = self.market.get_current_window()
            await asyncio.sleep(window["seconds_remaining"] + 2)

    def get_current_state(self) -> MarketState | None:
        return self.price_feed.get_state()

    def get_token_ids(self) -> dict | None:
        return self._token_ids

    def get_active_price_source(self) -> str:
        return self.price_feed.active_source

    async def run(self):
        hub = self.odds_poller
        await asyncio.gather(
            self.price_feed.run_rtds(
                self._odds_delta, self._p_market, odds_hub=hub
            ),
            self.price_feed.run_kline(self._odds_delta, self._p_market),
            hub.run(
                get_token_id=lambda: (self._token_ids or {}).get("up_token_id"),
                odds_ref=self._odds_delta,
                p_ref=self._p_market,
                spread_cache_fn=self.cache_spread,
            ),
            self._market_refresh_loop(),
            return_exceptions=True,
        )


# ══════════════════════════════════════════════════════════════
# BOT  (paper-trade, tüm eklentiler dahil)
# ══════════════════════════════════════════════════════════════

class Bot:
    """
    Paper-trade modu — tüm eklentiler aktif:
        EK-1 MispricingCorrector  → Bayesian prior düzeltilmiş
        EK-2 EV alanları          → her kararda ev/roi/exp_profit
        EK-3 Kelly NO flip        → UP yoksa DOWN tarafı değerlendirilir
        EK-4 LimitPricer          → maker fiyatı, fill simülasyonu
        EK-5 TemporalPrior        → her resolve'dan sonra güncellenir

    Wallet eklenince:
        PaperWallet → OrderExecutor (tek değişiklik)
        LimitPricer.get_limit_price() → gerçek limit order fiyatı olur
    """

    def __init__(
        self,
        starting_capital:   float = 10_000.0,
        entry_before_close: int   = 60,
        log_path:           str   = "papertrade_log.csv",
    ):
        self.log_path     = log_path

        self.hub          = DataHub()
        self.timing       = TimingController(entry_before_close)

        corrector = MispricingCorrector()
        temporal  = TemporalPrior()

        self.engine = DecisionEngine(
            updater=BayesianUpdater(
                corrector=corrector,
                temporal=temporal,
            ),
            sizer=KellySizer(
                fraction=0.25, max_position_pct=0.10,
                min_edge=0.03, min_confidence=0.55,
            ),
        )

        self.wallet       = PaperWallet(starting_capital)
        self.resolver     = ResolveFetcher()
        self.price_reader = PriceReader()
        self.limit_pricer = LimitPricer()

        # TemporalPrior güncelleme için referans
        self._temporal = temporal

    # ── resolve görevi ────────────────────────────────────────

    async def _resolve_trade(self, trade: PaperTrade, close_time: int):
        wait = close_time - time.time()
        if wait > 0:
            await asyncio.sleep(wait + 2)

        # Chainlink window_open_price'ı geç — en güvenilir resolve yöntemi
        window_open_price = self.hub.price_feed.window_open_price
        price_feed        = self.hub.price_feed

        outcome = await self.resolver.fetch_outcome(
            trade.slug, trade.up_token_id,
            window_open_price=window_open_price,
            price_feed=price_feed,
        )

        if outcome is None:
            self.wallet.refund(trade)
        else:
            self.wallet.resolve(trade, outcome)
            self._temporal.record(outcome)

        self.engine.register_close(trade.up_token_id)
        self._print_wallet_status()

    # ── karar döngüsü ─────────────────────────────────────────

    async def _decision_loop(self):
        await asyncio.sleep(10)

        while True:
            ids = self.hub.get_token_ids()
            if not ids or not ids.get("up_token_id"):
                await asyncio.sleep(2)
                continue

            window     = self.hub.market.get_current_window()
            close_time = window["close_time"]

            await self.timing.wait_for_entry(close_time)

            state = self.hub.get_current_state()
            if state is None:
                print("[Bot] State yok, pencere atlandı.")
                await asyncio.sleep(max(close_time - time.time(), 0) + 2)
                continue

            # Spread: önce WS cache'ten al (hızlı), yoksa REST çağrısı yap
            spread = (
                self.hub.get_cached_spread()
                or await self.price_reader.get_spread(ids["up_token_id"])
            )
            decision = self.engine.evaluate(
                market_id=ids["up_token_id"],
                state=state,
                capital=self.wallet.balance,
                odds_source=self.hub.odds_poller.active_source,
            )

            self._print_decision(decision, spread)

            if decision["action"] in ("open_long", "open_short"):
                net_edge = decision["edge"] - spread["spread"]
                if net_edge <= 0:
                    print("[Bot] Net edge negatif, işlem atlandı.")
                else:
                    side: Literal["up", "down"] = decision["side"]

                    # EK-4: Limit fiyat hesapla
                    limit_price = self.limit_pricer.get_limit_price(side, spread)

                    # EK-4: Fill simülasyonu
                    filled = self.limit_pricer.simulate_fill(limit_price, spread)
                    if not filled:
                        print(f"[Bot] Limit order dolmadı "
                              f"(price={limit_price:.3f}), pencere atlandı.")
                    else:
                        trade = self.wallet.open_position(
                            slug=window["slug"],
                            direction=side,
                            up_token_id=ids["up_token_id"],
                            entry_price=limit_price,   # ← limit fiyat (EK-4)
                            usdc_amount=decision["usdc"],
                            p_true=decision["p_true"],
                            edge=decision["edge"],
                        )
                        if trade:
                            self.engine.register_open(ids["up_token_id"])
                            asyncio.create_task(
                                self._resolve_trade(trade, close_time)
                            )

            await asyncio.sleep(max(close_time - time.time(), 0) + 2)

    # ── print yardımcıları ────────────────────────────────────

    def _print_decision(self, decision: dict, spread: dict):
        net_edge = (decision.get("edge") or 0) - spread["spread"]
        src = decision.get("odds_source", "–")
        print("\n" + "═" * 56)
        print(f"  {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  p_market       : {decision.get('p_market', 0):.3f}")
        print(f"  p_true         : {decision.get('p_true', 0):.3f}")
        print(f"  edge           : {decision.get('edge', 0):+.3f}")
        print(f"  ev/contract    : {decision.get('ev_per_contract', 0):+.4f}")
        print(f"  roi            : {decision.get('roi_pct', 0):+.2f}%")
        print(f"  exp. profit    : {decision.get('expected_profit', 0):+.2f} USDC")
        print(f"  spread         : {spread['spread']:.3f}")
        print(f"  odds kaynak    : {src}")
        print(f"  net edge       : {net_edge:+.3f}  "
              f"({'OK' if net_edge > 0 else 'SPREAD AŞIYOR'})")
        print(f"  karar          : {decision['action'].upper()}")
        if decision["action"] != "hold":
            print(f"  yön            : {decision.get('side', '–').upper()}")
            print(f"  miktar         : {decision.get('usdc', 0):.2f} USDC")
        if decision.get("reason"):
            print(f"  sebep          : {decision['reason']}")
        print("═" * 56)

    def _print_wallet_status(self):
        s = self.wallet.summary()
        if not s.get("trades"):
            return
        sharpe_str = f"{s['sharpe']:.3f}" if not np.isnan(s["sharpe"]) else "(<5 trade)"
        src = self.hub.get_active_price_source()
        print(
            f"\n  [Cüzdan]  "
            f"Equity: {self.wallet.equity:.2f}  |  "
            f"P&L: {self.wallet.total_pnl:+.2f}  |  "
            f"W/L: {self.wallet.wins}/{self.wallet.losses}  |  "
            f"WR: {s['win_rate']*100:.1f}%  |  "
            f"Sharpe: {sharpe_str}  |  "
            f"Fiyat: {src}  |  "
            f"TemporalObs: {self._temporal.total_observations}"
        )

    def _print_final_summary(self):
        s = self.wallet.summary()
        if not s.get("trades"):
            print("\n[Bot] Hiç trade yapılmadı.")
            return
        sharpe_str = f"{s['sharpe']:.3f}" if not np.isnan(s["sharpe"]) else "yetersiz veri"
        print("\n" + "╔" + "═" * 52 + "╗")
        print("║" + "  PAPER-TRADE ÖZET".center(52) + "║")
        print("╠" + "═" * 52 + "╣")
        rows = [
            ("Toplam trade",       f"{s['trades']}"),
            ("Kazanılan",          f"{s['wins']}"),
            ("Kaybedilen",         f"{s['losses']}"),
            ("Win rate",           f"{s['win_rate']*100:.1f}%"),
            ("Toplam P&L",         f"{s['total_pnl']:+.2f} USDC"),
            ("Ortalama P&L",       f"{s['avg_pnl']:+.2f} USDC"),
            ("Sharpe",             sharpe_str),
            ("Max drawdown",       f"{s['max_drawdown']:+.2f} USDC"),
            ("Final equity",       f"{s['final_equity']:.2f} USDC"),
            ("ROI",                f"{s['roi_pct']:.2f}%"),
            ("Temporal gözlem",    f"{self._temporal.total_observations}"),
        ]
        for label, value in rows:
            print(f"║  {label:<20}: {value:>26}  ║")
        print("╚" + "═" * 52 + "╝")

    async def run(self):
        print("╔════════════════════════════════════════════╗")
        print("║   Polymarket BTC 5M Bot  [PAPER-TRADE]    ║")
        print(f"║   Başlangıç: {self.wallet.starting_capital:>10,.2f} USDC               ║")
        print("╚════════════════════════════════════════════╝\n")

        try:
            await asyncio.gather(
                self.hub.run(),
                self._decision_loop(),
            )
        finally:
            self._print_final_summary()
            self.wallet.save_log(self.log_path)
            await SessionManager.close()


# ══════════════════════════════════════════════════════════════
# ENTRYPOINT
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    bot = Bot(
        starting_capital=float(os.getenv("STARTING_CAPITAL", "10000.0")),
        entry_before_close=int(os.getenv("ENTRY_BEFORE_CLOSE", "60")),
        log_path=os.getenv("LOG_PATH", "papertrade_log.csv"),
    )
    try:
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        print("\n[Bot] Durduruldu.")
