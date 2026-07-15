"""Capital-rotation warning layer and forward 20% drawdown diagnostics.

This module observes SURGE PRO without changing its positions.  A warning
requires the same source sector to weaken toward the same destination sector
for consecutive sessions.  Forward drawdowns are calibration outcomes, not a
claim that the warning predicts a crash.
"""

from dataclasses import dataclass, replace
from typing import Optional

import numpy as np
import pandas as pd

from strategies.base import EngineStrategy, ExecConfig, MarketData
from strategies.momentum_v85 import MomentumV85
from strategies.optimized_v85 import SURGE_PRO_PARAMS, _build_engine
from strategies.registry import register
from strategy.sector_flow import SECTOR_MAP, classify_sector


@dataclass
class RotationModel:
    sector_score: pd.DataFrame
    sector_return: pd.DataFrame
    sector_return_5d: pd.DataFrame
    sector_return_20d: pd.DataFrame
    turnover_acceleration: pd.DataFrame
    breadth_above_20d: pd.DataFrame
    institutional_flow: pd.DataFrame
    score_change_5d: pd.DataFrame
    average_correlation: pd.DataFrame
    destination: pd.DataFrame
    candidate_outflow: pd.DataFrame
    confirmed_outflow: pd.DataFrame
    confirmation_count: pd.DataFrame
    confirm_days: int


def _cs_rank(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.rank(axis=1, pct=True).fillna(0.5)


def _sector_columns(columns):
    groups = {}
    for ticker in columns:
        groups.setdefault(classify_sector(str(ticker)), []).append(ticker)
    return {sector: cols for sector, cols in groups.items() if len(cols) >= 2}


def _same_pair_confirmation(candidate, destination, confirm_days):
    counts = pd.DataFrame(0, index=candidate.index, columns=candidate.columns, dtype=np.int16)
    for col in candidate.columns:
        running = 0
        prior_dest = None
        out = []
        for ok, dest in zip(candidate[col].fillna(False), destination[col]):
            if bool(ok) and pd.notna(dest) and dest == prior_dest:
                running += 1
            elif bool(ok) and pd.notna(dest):
                running = 1
            else:
                running = 0
            prior_dest = dest if bool(ok) else None
            out.append(running)
        counts[col] = out
    return counts, counts >= int(confirm_days)


def compute_sector_rotation(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    inst_flow: Optional[pd.DataFrame] = None,
    universe_mask: Optional[pd.DataFrame] = None,
    confirm_days: int = 3,
) -> RotationModel:
    """Compute price/turnover/breadth/chip flow scores with no future data."""
    close = close.astype(float)
    volume = volume.reindex_like(close).astype(float)
    groups = _sector_columns(close.columns)
    sectors = sorted(groups)
    ret1 = close.pct_change(fill_method=None)
    ret5 = close.pct_change(5, fill_method=None)
    ret20 = close.pct_change(20, fill_method=None)
    turnover = close * volume
    turn5 = turnover.rolling(5, min_periods=3).mean()
    turn20 = turnover.rolling(20, min_periods=12).mean()
    turn_accel = turn5 / turn20.replace(0, np.nan) - 1
    above20 = close > close.rolling(20, min_periods=15).mean()
    mask = close.notna() & volume.gt(0)
    if universe_mask is not None:
        mask &= universe_mask.reindex_like(mask).fillna(False)

    sector_ret1 = pd.DataFrame(index=close.index, columns=sectors, dtype=float)
    sector_ret5 = sector_ret1.copy()
    sector_ret20 = sector_ret1.copy()
    sector_turn = sector_ret1.copy()
    sector_breadth = sector_ret1.copy()
    sector_inst = sector_ret1.copy()
    for sector, cols in groups.items():
        valid = mask[cols]
        sector_ret1[sector] = ret1[cols].where(valid).median(axis=1)
        sector_ret5[sector] = ret5[cols].where(valid).median(axis=1)
        sector_ret20[sector] = ret20[cols].where(valid).median(axis=1)
        sector_turn[sector] = turn_accel[cols].where(valid).median(axis=1)
        sector_breadth[sector] = above20[cols].where(valid).mean(axis=1)
        if inst_flow is not None:
            aligned_inst = inst_flow.reindex(index=close.index, columns=cols)
            sector_inst[sector] = aligned_inst.where(valid).median(axis=1)

    score = (
        0.30 * _cs_rank(sector_ret5)
        + 0.22 * _cs_rank(sector_ret20)
        + 0.23 * _cs_rank(sector_turn)
        + 0.15 * _cs_rank(sector_breadth)
        + 0.10 * _cs_rank(sector_inst)
    ).clip(0, 1)
    destination_name = score.idxmax(axis=1)
    destination_score = score.max(axis=1)

    avg_corr = pd.DataFrame(index=close.index, columns=sectors, dtype=float)
    for sector in sectors:
        pair_corr = [
            sector_ret1[sector].rolling(20, min_periods=12).corr(sector_ret1[other])
            for other in sectors if other != sector
        ]
        avg_corr[sector] = (
            pd.concat(pair_corr, axis=1).replace([np.inf, -np.inf], np.nan).mean(axis=1).clip(-1, 1)
            if pair_corr else np.nan
        )

    candidate = pd.DataFrame(False, index=close.index, columns=sectors)
    destinations = pd.DataFrame(index=close.index, columns=sectors, dtype=object)
    score_change5 = score - score.shift(5)
    for sector in sectors:
        dest = destination_name.where(destination_name != sector)
        destinations[sector] = dest
        spread = destination_score - score[sector]
        correlation_rollover = (avg_corr[sector] >= 0.65) & (score_change5[sector] <= -0.10)
        candidate[sector] = (
            (score[sector] <= 0.40)
            & (score_change5[sector] <= -0.06)
            & (destination_score >= 0.72)
            & (spread >= 0.32)
            & (dest.notna())
            & ((sector_turn[sector] < 0) | correlation_rollover)
        ).fillna(False)

    counts, confirmed = _same_pair_confirmation(candidate, destinations, confirm_days)
    return RotationModel(
        sector_score=score,
        sector_return=sector_ret1,
        sector_return_5d=sector_ret5,
        sector_return_20d=sector_ret20,
        turnover_acceleration=sector_turn,
        breadth_above_20d=sector_breadth,
        institutional_flow=sector_inst,
        score_change_5d=score_change5,
        average_correlation=avg_corr,
        destination=destinations,
        candidate_outflow=candidate,
        confirmed_outflow=confirmed,
        confirmation_count=counts,
        confirm_days=int(confirm_days),
    )


def active_rotation_alerts(model: RotationModel, as_of=None) -> pd.DataFrame:
    """Return active warnings known after ``as_of`` close; never emit orders."""
    if model.confirmed_outflow.empty:
        return pd.DataFrame()
    if as_of is None:
        dt = model.confirmed_outflow.index[-1]
    else:
        eligible = model.confirmed_outflow.index[
            model.confirmed_outflow.index <= pd.Timestamp(as_of)
        ]
        if not len(eligible):
            return pd.DataFrame()
        dt = eligible[-1]
    rows = []
    for source in model.confirmed_outflow.columns:
        if not bool(model.confirmed_outflow.at[dt, source]):
            continue
        destination = model.destination.at[dt, source]
        if pd.isna(destination):
            continue
        rows.append({
            "signal_date": dt,
            "available_next_session": True,
            "source_sector": source,
            "destination_sector": destination,
            "confirmation_days": int(model.confirmation_count.at[dt, source]),
            "source_score": float(model.sector_score.at[dt, source]),
            "destination_score": float(model.sector_score.at[dt, destination]),
            "score_spread": float(
                model.sector_score.at[dt, destination]
                - model.sector_score.at[dt, source]
            ),
            "source_score_change_5d": float(model.score_change_5d.at[dt, source]),
            "destination_score_change_5d": float(
                model.score_change_5d.at[dt, destination]
            ),
            "source_return_5d": float(model.sector_return_5d.at[dt, source]),
            "destination_return_5d": float(
                model.sector_return_5d.at[dt, destination]
            ),
            "source_turnover_acceleration": float(
                model.turnover_acceleration.at[dt, source]
            ),
            "destination_turnover_acceleration": float(
                model.turnover_acceleration.at[dt, destination]
            ),
            "source_breadth_above_20d": float(model.breadth_above_20d.at[dt, source]),
            "destination_breadth_above_20d": float(
                model.breadth_above_20d.at[dt, destination]
            ),
            "source_institutional_flow": float(model.institutional_flow.at[dt, source]),
            "destination_institutional_flow": float(
                model.institutional_flow.at[dt, destination]
            ),
            "average_correlation_20d": float(model.average_correlation.at[dt, source]),
            "warning_only": True,
            "automatic_trade_action": False,
        })
    return pd.DataFrame(rows)


def extract_rotation_events(model: RotationModel, cooldown_days: int = 20) -> pd.DataFrame:
    """Return one row per newly confirmed source→destination rotation event."""
    rows = []
    for sector in model.confirmed_outflow.columns:
        confirmed = model.confirmed_outflow[sector].fillna(False)
        trigger = confirmed & ~confirmed.shift(1, fill_value=False)
        last_i = -10_000
        for dt in trigger[trigger].index:
            i = model.confirmed_outflow.index.get_loc(dt)
            if i - last_i < cooldown_days:
                continue
            dest = model.destination.at[dt, sector]
            if pd.isna(dest):
                continue
            rows.append({
                "signal_date": dt,
                "source_sector": sector,
                "destination_sector": dest,
                "confirmation_days": int(model.confirmation_count.at[dt, sector]),
                "source_score": float(model.sector_score.at[dt, sector]),
                "destination_score": float(model.sector_score.at[dt, dest]),
                "score_spread": float(
                    model.sector_score.at[dt, dest] - model.sector_score.at[dt, sector]
                ),
                "source_score_change_5d": float(model.score_change_5d.at[dt, sector]),
                "destination_score_change_5d": float(model.score_change_5d.at[dt, dest]),
                "source_return_5d": float(model.sector_return_5d.at[dt, sector]),
                "destination_return_5d": float(model.sector_return_5d.at[dt, dest]),
                "source_return_20d": float(model.sector_return_20d.at[dt, sector]),
                "destination_return_20d": float(model.sector_return_20d.at[dt, dest]),
                "source_turnover_acceleration": float(
                    model.turnover_acceleration.at[dt, sector]
                ),
                "destination_turnover_acceleration": float(
                    model.turnover_acceleration.at[dt, dest]
                ),
                "source_breadth_above_20d": float(model.breadth_above_20d.at[dt, sector]),
                "destination_breadth_above_20d": float(
                    model.breadth_above_20d.at[dt, dest]
                ),
                "source_institutional_flow": float(model.institutional_flow.at[dt, sector]),
                "destination_institutional_flow": float(
                    model.institutional_flow.at[dt, dest]
                ),
                "average_correlation_20d": float(model.average_correlation.at[dt, sector]),
            })
            last_i = i
    if not rows:
        return pd.DataFrame(columns=[
            "signal_date", "source_sector", "destination_sector", "confirmation_days",
            "source_score", "destination_score", "score_spread", "average_correlation_20d",
        ])
    return pd.DataFrame(rows).sort_values("signal_date").reset_index(drop=True)


def reconfirm_rotation(model: RotationModel, confirm_days: int) -> RotationModel:
    """Reuse expensive sector features while testing a different persistence window."""
    counts, confirmed = _same_pair_confirmation(
        model.candidate_outflow, model.destination, int(confirm_days),
    )
    return replace(
        model, confirmed_outflow=confirmed, confirmation_count=counts,
        confirm_days=int(confirm_days),
    )


def analyze_forward_drawdowns(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    events: pd.DataFrame,
    drawdown: float = 0.20,
    pre_peak_days: int = 20,
    forward_days: int = 120,
    max_stocks_per_sector: int = 30,
):
    """Measure each liquid source-sector stock's first post-signal 20% drawdown."""
    records = []
    groups = _sector_columns(close.columns)
    turnover60 = (close * volume.reindex_like(close)).rolling(60, min_periods=30).mean()
    for event_id, event in events.iterrows():
        dt = pd.Timestamp(event["signal_date"])
        if dt not in close.index:
            continue
        signal_i = close.index.get_loc(dt)
        sector = event["source_sector"]
        cols = groups.get(sector, [])
        liquid = turnover60.loc[dt, cols].dropna().nlargest(max_stocks_per_sector).index
        for ticker in liquid:
            start_i = max(0, signal_i - pre_peak_days + 1)
            peak = close[ticker].iloc[start_i:signal_i + 1].max()
            if pd.isna(peak) or peak <= 0:
                continue
            future = close[ticker].iloc[signal_i + 1:signal_i + 1 + forward_days]
            hit = future[future <= peak * (1.0 - drawdown)]
            hit_date = hit.index[0] if len(hit) else pd.NaT
            lead = close.index.get_loc(hit_date) - signal_i if pd.notna(hit_date) else np.nan
            records.append({
                "event_id": int(event_id), "signal_date": dt,
                "source_sector": sector,
                "destination_sector": event["destination_sector"],
                "ticker": str(ticker), "reference_peak": float(peak),
                "drawdown_date": hit_date, "lead_trading_days": lead,
                "hit_20pct": bool(len(hit)),
            })
    detail = pd.DataFrame(records)
    if detail.empty:
        return detail, pd.DataFrame()
    summaries = []
    for event_id, group in detail.groupby("event_id"):
        hit = group.loc[group["hit_20pct"], "lead_trading_days"].dropna()
        summaries.append({
            "event_id": int(event_id),
            "signal_date": group["signal_date"].iloc[0],
            "source_sector": group["source_sector"].iloc[0],
            "destination_sector": group["destination_sector"].iloc[0],
            "stocks_tested": int(len(group)),
            "stocks_hit_20pct": int(len(hit)),
            "hit_rate": float(len(hit) / len(group)),
            # Earliest observed outcome helps set monitoring cadence.  It is
            # deliberately not labelled a safe exit or an automatic sell date.
            "earliest_observed_drawdown_days": float(hit.min()) if len(hit) else np.nan,
            "median_lead_days": float(hit.median()) if len(hit) else np.nan,
            "p80_lead_days": float(hit.quantile(0.80)) if len(hit) else np.nan,
            "max_lead_days": float(hit.max()) if len(hit) else np.nan,
        })
    return detail, pd.DataFrame(summaries)


@register("mom_surge_pro_rotation_alert")
class MomSurgeProRotationAlert(EngineStrategy):
    description = "SURGE PRO ROTATION ALERT｜連續3日同向資金輪動警示；不自動減碼、不改變母策略交易"
    requires = frozenset({"inst_flow"})

    def run_engine(self, data: MarketData, exec_cfg: ExecConfig):
        confirm_days = int(self.params.get("rotation_confirm_days", 3))
        model = compute_sector_rotation(
            data.close, data.volume, inst_flow=data.inst_flow_df,
            universe_mask=data.universe_mask, confirm_days=confirm_days,
        )
        bundle = MomentumV85().prepare(data)
        bt = _build_engine(SURGE_PRO_PARAMS, exec_cfg)
        trades, equity = bt.run(
            bundle.total_score, data.close, data.open, data.high, data.low,
            bundle.ma_long, top_k=exec_cfg.top_k, threshold=exec_cfg.threshold,
            market_close=data.market_close, vol_df=data.volume,
            universe_mask=data.universe_mask,
        )
        # Reporting attributes only: no gate, sizing multiplier or forced exit
        # is passed to the execution engine.
        self.last_rotation_model = model
        self.last_rotation_alerts = active_rotation_alerts(model)
        self.last_positions = getattr(bt, "last_positions", {})
        self.last_cash = getattr(bt, "last_cash", exec_cfg.initial_capital)
        return trades, equity


__all__ = [
    "RotationModel", "compute_sector_rotation", "active_rotation_alerts",
    "extract_rotation_events", "reconfirm_rotation", "analyze_forward_drawdowns",
    "MomSurgeProRotationAlert",
]
