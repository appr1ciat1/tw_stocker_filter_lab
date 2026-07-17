"""Asymmetric entry confirmation variants for v8.5 and SURGE PRO.

The variants intentionally change only entry timing.  The original alpha,
ATR exits and (for SURGE PRO) regime sizing remain untouched, which makes the
incremental value of the new filters measurable rather than confounded.
"""

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd

from strategies.base import EngineStrategy, ExecConfig, MarketData
from strategies.momentum_v85 import MomentumV85
from strategies.optimized_v85 import SURGE_PRO_PARAMS, _build_engine
from strategies.registry import register
from strategy.event_backtest import EventDrivenBacktester


@dataclass
class ConfirmationLayers:
    entry_gate: pd.DataFrame
    entry_scale: pd.DataFrame
    diagnostics: Dict[str, object]


def _rank_centered(frame: pd.DataFrame, universe_mask=None) -> pd.DataFrame:
    work = frame
    if universe_mask is not None:
        work = work.where(universe_mask.reindex_like(work).fillna(False))
    return (work.rank(axis=1, pct=True) - 0.5) * 2.0


def _consecutive_true(frame: pd.DataFrame) -> pd.DataFrame:
    """Count consecutive True rows independently for every column."""
    values = frame.fillna(False).to_numpy(dtype=bool)
    counts = np.zeros(values.shape, dtype=np.int16)
    if len(values):
        counts[0] = values[0]
    for i in range(1, len(values)):
        counts[i] = np.where(values[i], counts[i - 1] + 1, 0)
    return pd.DataFrame(counts, index=frame.index, columns=frame.columns)


def _atr14(data: MarketData) -> pd.DataFrame:
    prior_close = data.close.shift(1)
    tr = np.maximum(
        data.high - data.low,
        np.maximum((data.high - prior_close).abs(), (data.low - prior_close).abs()),
    )
    return tr.rolling(14, min_periods=10).mean()


def _market_at_open(data: MarketData):
    index = data.close.index
    if data.market_close is None or len(data.market_close) == 0:
        neutral = pd.Series(False, index=index)
        return neutral, neutral, pd.Series(0.0, index=index)
    market = data.market_close.copy()
    market.index = pd.DatetimeIndex(market.index).tz_localize(None).normalize()
    market = market.reindex(index, method="ffill")
    # Row T must only know Taiwan data through T-1.
    known = market.shift(1)
    ma20 = market.rolling(20, min_periods=15).mean().shift(1)
    ma60 = market.rolling(60, min_periods=40).mean().shift(1)
    ma20_slope = ma20 > ma20.shift(3)
    bull = (known > ma60) & (known > ma20)
    recovery = (known <= ma60) & (known > ma20) & ma20_slope
    strength = ((known / ma60 - 1) / 0.08).clip(-1, 1).fillna(0.0)
    return bull.fillna(False), recovery.fillna(False), strength


def build_confirmation_layers(data: MarketData, profile: str = "v85") -> ConfirmationLayers:
    """Build open-time gates without using any same-day Taiwan information."""
    close = data.close.astype(float)
    index, columns = close.index, close.columns
    known_close = close.shift(1)
    previous_close = close.shift(2)
    atr = _atr14(data).shift(1)
    high20 = close.rolling(20, min_periods=15).max().shift(1)
    high60 = close.rolling(60, min_periods=40).max().shift(1)
    low10 = data.low.rolling(10, min_periods=7).min().shift(1)
    ma20 = close.rolling(20, min_periods=15).mean().shift(1)

    pullback = (1.0 - known_close / high20).clip(lower=0)
    rebound = (known_close > previous_close) & (known_close > low10 * 1.01)
    extension_atr = (known_close - ma20) / atr.replace(0, np.nan)

    # Prospective structure-based reward/risk.  A breakout target receives a
    # modest 1.5 ATR extension; the stop reference is the recent swing low.
    target = high60 + 1.5 * atr
    stop = low10 - 0.5 * atr
    upside = (target - known_close).clip(lower=0.5 * atr)
    downside = (known_close - stop).clip(lower=1.0 * atr)
    reward_risk = (upside / downside.replace(0, np.nan)).clip(0, 8)

    bull_market, recovering_market, market_strength = _market_at_open(data)
    global_score = pd.Series(0.0, index=index)
    leader_score = pd.DataFrame(0.0, index=index, columns=columns)
    if data.global_context is not None:
        overnight = getattr(data.global_context, "overnight", None)
        if overnight is not None and "global_risk_score" in overnight:
            global_score = pd.to_numeric(
                overnight["global_risk_score"].reindex(index), errors="coerce"
            ).fillna(0.0)
        leaders = getattr(data.global_context, "leader_score", None)
        if leaders is not None:
            leader_score = leaders.reindex(index=index, columns=columns).fillna(0.0)

    # Quantity-based chip layer.  Every component is lagged one Taiwan session
    # before it can affect the opening trade.
    inst = pd.DataFrame(np.nan, index=index, columns=columns)
    if data.inst_flow_df is not None:
        inst = data.inst_flow_df.reindex_like(close)
    inst_factor = _rank_centered(inst, data.universe_mask).shift(1).fillna(0.0)

    def balance_factor(frame, lookback=20):
        if frame is None:
            return pd.DataFrame(0.0, index=index, columns=columns)
        change = frame.reindex_like(close).pct_change(lookback, fill_method=None)
        # Rising leverage/borrowed short supply is a crowding warning.
        return (-_rank_centered(change, data.universe_mask)).shift(1).fillna(0.0)

    margin_factor = balance_factor(data.margin_balance_df)
    margin_short_factor = balance_factor(data.margin_short_df)
    sbl_factor = balance_factor(data.short_sale_df)
    chip_score = (
        0.45 * inst_factor + 0.25 * margin_factor
        + 0.10 * margin_short_factor + 0.20 * sbl_factor
    ).clip(-1, 1)

    global_matrix = pd.DataFrame(
        np.repeat(global_score.to_numpy()[:, None], len(columns), axis=1),
        index=index, columns=columns,
    )
    bull_matrix = pd.DataFrame(
        np.repeat(bull_market.to_numpy()[:, None], len(columns), axis=1),
        index=index, columns=columns,
    )
    recovery_matrix = pd.DataFrame(
        np.repeat(recovering_market.to_numpy()[:, None], len(columns), axis=1),
        index=index, columns=columns,
    )

    strong_bull = bull_matrix & (global_matrix >= 0.00)
    normal_bull = bull_matrix & ~strong_bull
    # Severe overnight/leader/chip disagreement is a veto; ordinary noise is
    # handled by consecutive confirmation rather than a brittle one-day cut.
    support = (
        (global_matrix > -0.85)
        & (leader_score > -0.80)
        & (chip_score > -0.55)
    )
    support_days = _consecutive_true(support)
    required = pd.DataFrame(3, index=index, columns=columns, dtype=np.int16)
    required = required.mask(normal_bull, 2).mask(strong_bull, 1)

    # Do not force a momentum breakout to look like mean reversion.  In a
    # confirmed bull market, extension control is enough; pullback/RR decides
    # sizing.  The hard relative-low requirement is reserved for recovery.
    setup_strong = extension_atr <= 3.00
    setup_normal = (
        (extension_atr <= 2.50)
        & ((pullback <= 0.04) | rebound)
        & (reward_risk >= 0.75)
    )
    setup_recovery = (pullback >= 0.03) & (pullback <= 0.16) & rebound & (reward_risk >= 1.60)
    if profile == "surge":
        # SURGE PRO already has breadth/VIX/regime sizing.  Repeating a local
        # pullback gate destroys its strong-cluster alpha, so the first-task
        # overlay adds persistence/extreme-veto only and keeps recovery strict.
        setup = bull_matrix | (recovery_matrix & setup_recovery)
    else:
        setup = (
            (strong_bull & setup_strong)
            | (normal_bull & setup_normal)
            | (recovery_matrix & setup_recovery)
        )
    gate = setup & (support_days >= required)
    if data.universe_mask is not None:
        # Universe membership is also Taiwan T-1 information at the T open.
        gate &= data.universe_mask.reindex_like(gate).shift(1).fillna(False)

    scale = pd.DataFrame(1.0, index=index, columns=columns)
    if profile != "surge":
        scale = scale.mask(strong_bull & gate, 1.05)
    scale = scale.mask(
        normal_bull & (pullback >= 0.03) & (pullback <= 0.09) & rebound, 1.08,
    )
    scale = scale.mask(recovery_matrix & setup_recovery, 1.12)
    scale = scale * np.where(chip_score > 0.35, 1.04, 1.0)
    scale = scale * np.where(global_matrix < -0.15, 0.85, 1.0)
    scale = scale.clip(0.75, 1.15).where(gate, 0.0)

    return ConfirmationLayers(
        entry_gate=gate.fillna(False),
        entry_scale=scale.fillna(0.0),
        diagnostics={
            "global_score": global_score,
            "leader_score": leader_score,
            "chip_score": chip_score,
            "pullback": pullback,
            "reward_risk": reward_risk,
            "support_days": support_days,
            "required_days": required,
            "market_strength": market_strength,
        },
    )


REQUIRES_CONFIRMATION = frozenset({
    "global_context", "inst_flow", "chip_indicators",
})


@register("momentum_v85_confirmed")
class MomentumV85Confirmed(EngineStrategy):
    """v8.5 with the first-task entry filters; legacy exits stay unchanged."""

    description = "v8.5 CONFIRMED｜隔夜SPX/SOX/TSM ADR+全球龍頭+籌碼+非對稱延後進場"
    requires = REQUIRES_CONFIRMATION

    def run_engine(self, data: MarketData, exec_cfg: ExecConfig):
        bundle = MomentumV85().prepare(data)
        layers = build_confirmation_layers(data, profile="v85")
        bt = EventDrivenBacktester(
            tp_sl_mode="atr", tp_atr_mult=4.0, sl_atr_mult=3.0,
            max_hold_days=20, initial_capital=exec_cfg.initial_capital,
            position_size=0.10, regime_filter=True, gap_filter_atr=1.5,
            hybrid_tiered=False, buy_cost=exec_cfg.buy_cost,
            sell_cost=exec_cfg.sell_cost, slippage=exec_cfg.slippage,
        )
        result = bt.run(
            bundle.total_score, data.close, data.open, data.high, data.low,
            bundle.ma_long, top_k=exec_cfg.top_k, threshold=exec_cfg.threshold,
            market_close=data.market_close, vol_df=data.volume,
            universe_mask=data.universe_mask,
            entry_gate_df=layers.entry_gate, entry_scale_df=layers.entry_scale,
        )
        self.last_positions = getattr(bt, "last_positions", {})
        self.last_cash = getattr(bt, "last_cash", exec_cfg.initial_capital)
        self.last_confirmation_layers = layers
        return result


@register("mom_surge_pro_confirmed")
class MomSurgeProConfirmed(EngineStrategy):
    """SURGE PRO with first-task confirmation; regime tiers stay unchanged."""

    description = "SURGE PRO CONFIRMED｜原分段加碼+隔夜/龍頭/籌碼/相對低點確認"
    requires = REQUIRES_CONFIRMATION

    def run_engine(self, data: MarketData, exec_cfg: ExecConfig):
        bundle = MomentumV85().prepare(data)
        layers = build_confirmation_layers(data, profile="surge")
        bt = _build_engine({**SURGE_PRO_PARAMS, **self.params}, exec_cfg)
        result = bt.run(
            bundle.total_score, data.close, data.open, data.high, data.low,
            bundle.ma_long, top_k=exec_cfg.top_k, threshold=exec_cfg.threshold,
            market_close=data.market_close, vol_df=data.volume,
            universe_mask=data.universe_mask,
            entry_gate_df=layers.entry_gate, entry_scale_df=layers.entry_scale,
        )
        self.last_positions = getattr(bt, "last_positions", {})
        self.last_cash = getattr(bt, "last_cash", exec_cfg.initial_capital)
        self.last_confirmation_layers = layers
        return result


__all__ = [
    "ConfirmationLayers", "build_confirmation_layers",
    "MomentumV85Confirmed", "MomSurgeProConfirmed",
]
