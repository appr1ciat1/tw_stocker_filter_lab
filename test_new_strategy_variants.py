import numpy as np
import pandas as pd

from strategies.registry import get_strategy, list_strategies
from strategies.rotation_exit import _same_pair_confirmation
from strategy.event_backtest import EventDrivenBacktester
from twstk.data.global_context import align_completed_us_session


def test_completed_us_session_uses_previous_exchange_date():
    us = pd.DataFrame(
        {"close": [100.0, 101.0, 102.0]},
        index=pd.to_datetime(["2026-07-10", "2026-07-13", "2026-07-14"]),
    )
    tw = pd.to_datetime(["2026-07-13", "2026-07-14", "2026-07-15"])
    aligned = align_completed_us_session(us, tw)
    assert aligned["close"].tolist() == [100.0, 101.0, 102.0]


def test_optional_entry_gate_controls_first_entry():
    dates = pd.bdate_range("2025-01-01", periods=85)
    ticker = "2330"
    close = pd.DataFrame(100.0, index=dates, columns=[ticker])
    open_ = close.copy()
    high = pd.DataFrame(101.0, index=dates, columns=[ticker])
    low = pd.DataFrame(99.0, index=dates, columns=[ticker])
    volume = pd.DataFrame(1_000_000.0, index=dates, columns=[ticker])
    score = pd.DataFrame(3.0, index=dates, columns=[ticker])
    ma = pd.DataFrame(50.0, index=dates, columns=[ticker])
    gate = pd.DataFrame(False, index=dates, columns=[ticker])
    gate.loc[dates[65], ticker] = True
    bt = EventDrivenBacktester(
        tp_sl_mode="fixed", tp_pct=0.50, sl_pct=0.50, max_hold_days=1,
        regime_filter=False, gap_filter_atr=0, initial_capital=300_000,
        position_size=0.10, integer_shares=True,
    )
    trades, _ = bt.run(
        score, close, open_, high, low, ma, top_k=1, threshold=2.0,
        vol_df=volume, entry_gate_df=gate,
    )
    assert len(trades) == 1
    assert trades.iloc[0]["Entry_Date"] == dates[65].strftime("%Y-%m-%d")
    assert trades.iloc[0]["Exit_Date"] == dates[66].strftime("%Y-%m-%d")


def test_rotation_confirmation_requires_same_destination():
    dates = pd.bdate_range("2026-01-01", periods=6)
    candidate = pd.DataFrame({"semiconductor": [True] * 6}, index=dates)
    destination = pd.DataFrame(
        {"semiconductor": ["shipping", "shipping", "finance", "finance", "finance", "finance"]},
        index=dates,
    )
    counts, confirmed = _same_pair_confirmation(candidate, destination, confirm_days=3)
    assert counts["semiconductor"].tolist() == [1, 2, 1, 2, 3, 4]
    assert confirmed["semiconductor"].tolist() == [False, False, False, False, True, True]


def test_five_isolated_versions_are_registered_and_capital_is_capped():
    names = list_strategies()
    expected = {
        "momentum_v85_confirmed", "momentum_v85_300k",
        "mom_surge_pro_confirmed", "mom_surge_pro_300k",
        "mom_surge_pro_rotation_alert",
    }
    assert expected.issubset(names)
    assert get_strategy("momentum_v85_300k").capital_cap == 300_000
    assert get_strategy("mom_surge_pro_300k").capital_cap == 300_000
    alert = get_strategy("mom_surge_pro_rotation_alert")
    assert "不自動減碼" in alert.description
