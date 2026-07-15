#!/usr/bin/env python3
"""Ten-year capital-rotation warning calibration and outcome study.

Uses the existing full-market caches.  It writes every confirmed rotation
event, every tested stock and the number of trading days from confirmation to
the first 20% drawdown from its recent peak.  Outcomes calibrate warning
urgency; they are not trading orders or a claim of predictive probability.
"""

import argparse
import json
import os
import pickle

import numpy as np
import pandas as pd

from strategies.rotation_exit import (
    analyze_forward_drawdowns,
    compute_sector_rotation,
    extract_rotation_events,
    reconfirm_rotation,
)
from strategy.sector_flow import SECTOR_MAP


ARTIFACTS = "artifacts"
PRICE_CACHE = os.path.join(ARTIFACTS, "wbr_fullmkt_prices.pkl")
INST_CACHE = os.path.join(ARTIFACTS, "wbr_fullmkt_inst.pkl")


def _json_value(value):
    if pd.isna(value):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    return value


def _load(start, end, use_inst=True):
    if not os.path.exists(PRICE_CACHE):
        raise FileNotFoundError(f"缺少 {PRICE_CACHE}；請先執行 wbr_fullmkt_prices.py")
    with open(PRICE_CACHE, "rb") as f:
        prices = pickle.load(f)
    close = prices["close"].copy()
    volume = prices["vol"].copy()
    # Warm-up is retained, while event output is clipped to the requested ten years.
    warmup = pd.Timestamp(start) - pd.Timedelta(days=150)
    keep_dates = (close.index >= warmup) & (close.index <= pd.Timestamp(end))
    close, volume = close.loc[keep_dates], volume.loc[keep_dates]
    good = (close.notna().sum() >= 200) & (volume.gt(0).sum() >= 120)
    close, volume = close.loc[:, good], volume.loc[:, good]

    inst = None
    if use_inst and os.path.exists(INST_CACHE):
        with open(INST_CACHE, "rb") as f:
            raw = pickle.load(f)
        inst = raw["total"].reindex(index=close.index, columns=close.columns)
        # Normalise net lots by 5-day volume so small and large stocks are comparable.
        inst = inst.rolling(5, min_periods=3).sum() / (
            volume.rolling(5, min_periods=3).sum() / 1000.0
        ).replace(0, np.nan)
    return close, volume, inst


def _summary(detail, event_summary, events, confirm_days):
    if len(detail) and {"hit_20pct", "lead_trading_days"}.issubset(detail.columns):
        leads = detail.loc[detail["hit_20pct"], "lead_trading_days"].dropna()
        stock_hit_rate = float(detail["hit_20pct"].mean())
    else:
        leads = pd.Series(dtype=float)
        stock_hit_rate = np.nan
    event_hits = event_summary["stocks_hit_20pct"].gt(0) if len(event_summary) else pd.Series(dtype=bool)
    earliest = (
        event_summary["earliest_observed_drawdown_days"].dropna()
        if len(event_summary) else pd.Series(dtype=float)
    )
    return {
        "confirmation_days": int(confirm_days),
        "events": int(len(events)),
        "events_with_20pct_drawdown": int(event_hits.sum()),
        "event_hit_rate": float(event_hits.mean()) if len(event_hits) else np.nan,
        "stocks_tested": int(len(detail)),
        "stock_hit_rate": stock_hit_rate,
        "median_lead_days": float(leads.median()) if len(leads) else np.nan,
        "p20_lead_days": float(leads.quantile(0.20)) if len(leads) else np.nan,
        "p80_lead_days": float(leads.quantile(0.80)) if len(leads) else np.nan,
        # The 10th percentile of each event's earliest outcome is a monitoring
        # urgency statistic, not a recommended exit deadline.
        "alert_urgency_p10_days": (
            float(earliest.quantile(0.10)) if len(earliest) else np.nan
        ),
    }


def _matched_control_events(model, events, start, end, controls_per_event=2, seed=85):
    """Select same-year/source-sector non-signal dates as a base-rate control."""
    rng = np.random.default_rng(seed)
    rows = []
    all_index = model.confirmed_outflow.index
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    for _, event in events.iterrows():
        source = event["source_sector"]
        signal = pd.Timestamp(event["signal_date"])
        eligible = all_index[
            (all_index >= start) & (all_index <= end)
            & (all_index.year == signal.year)
            & ~model.confirmed_outflow[source].reindex(all_index).fillna(False).to_numpy()
        ]
        # Exclude the local event neighbourhood; otherwise controls can still
        # capture the same drawdown episode.
        eligible = eligible[np.abs((eligible - signal).days) >= 30]
        if not len(eligible):
            continue
        take = min(controls_per_event, len(eligible))
        for dt in rng.choice(eligible.to_numpy(), size=take, replace=False):
            rows.append({
                "signal_date": pd.Timestamp(dt), "source_sector": source,
                "destination_sector": "matched_control",
            })
    return pd.DataFrame(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description="十年資金輪動 → 20% 回撤領先天數")
    parser.add_argument("--start", default="2016-07-01")
    parser.add_argument("--end", default="2026-07-07")
    parser.add_argument("--confirm-days", type=int, default=3)
    parser.add_argument("--forward-days", type=int, default=120)
    parser.add_argument("--no-inst", action="store_true", help="不使用三大法人流量")
    args = parser.parse_args(argv)

    close, volume, inst = _load(args.start, args.end, use_inst=not args.no_inst)
    print(f"全市場十年資料：{close.shape[0]} 日 × {close.shape[1]} 檔")
    avg_turnover = (close * volume).rolling(60, min_periods=40).mean()
    # Top half by turnover keeps a broad market view while suppressing stale microcaps.
    liquid_mask = avg_turnover.rank(axis=1, pct=True) >= 0.50
    base_model = compute_sector_rotation(
        close, volume, inst_flow=inst, universe_mask=liquid_mask,
        confirm_days=args.confirm_days,
    )

    sensitivity_rows = []
    chosen = None
    for n in (2, 3, 4, 5):
        model = reconfirm_rotation(base_model, n)
        events = extract_rotation_events(model)
        events = events[
            (events["signal_date"] >= pd.Timestamp(args.start))
            & (events["signal_date"] <= pd.Timestamp(args.end))
        ].reset_index(drop=True)
        detail, event_summary = analyze_forward_drawdowns(
            close, volume, events, forward_days=args.forward_days,
        )
        row = _summary(detail, event_summary, events, n)
        sensitivity_rows.append(row)
        print(
            f"確認 {n} 日：{row['events']} 次，事件命中率 "
            f"{row['event_hit_rate'] * 100 if pd.notna(row['event_hit_rate']) else 0:.1f}% ，"
            f"20%回撤中位領先 {row['median_lead_days']:.1f} 交易日"
        )
        if n == args.confirm_days:
            chosen = (model, events, detail, event_summary, row)

    if chosen is None:
        model = reconfirm_rotation(base_model, args.confirm_days)
        events = extract_rotation_events(model)
        events = events[
            (events["signal_date"] >= pd.Timestamp(args.start))
            & (events["signal_date"] <= pd.Timestamp(args.end))
        ].reset_index(drop=True)
        detail, event_summary = analyze_forward_drawdowns(
            close, volume, events, forward_days=args.forward_days,
        )
        chosen = (model, events, detail, event_summary,
                  _summary(detail, event_summary, events, args.confirm_days))

    _, events, detail, event_summary, headline = chosen
    controls = _matched_control_events(base_model, events, args.start, args.end)
    control_detail, control_event_summary = analyze_forward_drawdowns(
        close, volume, controls, forward_days=args.forward_days,
    )
    control = _summary(
        control_detail, control_event_summary, controls, args.confirm_days,
    )
    headline["matched_control_stock_hit_rate"] = control["stock_hit_rate"]
    headline["stock_hit_rate_lift"] = (
        headline["stock_hit_rate"] - control["stock_hit_rate"]
        if pd.notna(headline["stock_hit_rate"]) and pd.notna(control["stock_hit_rate"])
        else np.nan
    )
    headline["matched_control_median_lead_days"] = control["median_lead_days"]
    label = {key: value.get("label", key) for key, value in SECTOR_MAP.items()}
    for frame in (events, detail, event_summary):
        if len(frame):
            if "source_sector" in frame:
                frame["source_sector_label"] = frame["source_sector"].map(label)
            if "destination_sector" in frame:
                frame["destination_sector_label"] = frame["destination_sector"].map(label)

    os.makedirs(ARTIFACTS, exist_ok=True)
    paths = {
        "events": os.path.join(ARTIFACTS, "capital_rotation_events_10y.csv"),
        "detail": os.path.join(ARTIFACTS, "capital_rotation_drawdowns_10y.csv"),
        "event_summary": os.path.join(ARTIFACTS, "capital_rotation_event_summary_10y.csv"),
        "sensitivity": os.path.join(ARTIFACTS, "capital_rotation_sensitivity_10y.csv"),
        "json": os.path.join(ARTIFACTS, "capital_rotation_summary_10y.json"),
        "control": os.path.join(ARTIFACTS, "capital_rotation_matched_control_10y.csv"),
    }
    events.to_csv(paths["events"], index=False, encoding="utf-8-sig")
    detail.to_csv(paths["detail"], index=False, encoding="utf-8-sig")
    event_summary.to_csv(paths["event_summary"], index=False, encoding="utf-8-sig")
    pd.DataFrame(sensitivity_rows).to_csv(paths["sensitivity"], index=False, encoding="utf-8-sig")
    control_detail.to_csv(paths["control"], index=False, encoding="utf-8-sig")
    payload = {
        "period": {"start": args.start, "end": args.end},
        "chosen_confirmation_days": args.confirm_days,
        "headline": {k: _json_value(v) for k, v in headline.items()},
        "sensitivity": [
            {k: _json_value(v) for k, v in row.items()} for row in sensitivity_rows
        ],
        "matched_control": {k: _json_value(v) for k, v in control.items()},
        "method": {
            "drawdown": 0.20, "pre_signal_peak_days": 20,
            "forward_trading_days": args.forward_days,
            "max_liquid_stocks_per_sector_event": 30,
            "purpose": "warning_calibration_only",
            "automatic_trade_action": False,
            "predictive_probability_claim": False,
            "event_features": [
                "score_spread", "score_change_5d", "sector_return_5d",
                "sector_return_20d", "turnover_acceleration",
                "breadth_above_20d", "institutional_flow",
                "average_correlation_20d",
            ],
        },
    }
    with open(paths["json"], "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\n已輸出：{paths}")
    if pd.notna(headline["alert_urgency_p10_days"]):
        print(
            "歷史警報急迫度 P10：較早的 20% 回撤約在確認後 "
            f"{headline['alert_urgency_p10_days']:.0f} 個交易日出現；"
            "此數字只決定監控頻率，不是自動賣出期限。"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
