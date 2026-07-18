"""
純邏輯驗證：資料契約 (twstk.data.contract) + 容量門 (validation.capacity)。
用合成資料，不碰 yfinance / CI。可直接 `python test_data_contract_capacity.py`。
"""
import os
import tempfile

import numpy as np
import pandas as pd

from twstk.data.contract import (
    validate_panel, freeze_snapshot, load_snapshot, build_manifest,
    most_recent_session, FIELDS,
)
from validation.capacity import capacity_gate, participation_table

AS_OF = pd.Timestamp("2026-07-15")  # 週三
CAL = None  # 強制用近似工作日路徑，讓測試不依賴 exchange_calendars 是否安裝

PASS = []
FAIL = []


def check(name, cond):
    (PASS if cond else FAIL).append(name)
    print(f"  {'✅' if cond else '❌'} {name}")


def make_panel(tickers, n=60, last_date=AS_OF, seed=0, include_0050=True):
    """造一份乾淨的 OHLCV 面板，index 為工作日、結束於 last_date。"""
    rng = np.random.default_rng(seed)
    cols = (["0050"] if include_0050 else []) + list(tickers)
    idx = pd.bdate_range(end=last_date, periods=n)
    base = rng.uniform(50, 500, size=len(cols))
    steps = rng.normal(0, 0.01, size=(n, len(cols)))
    close = pd.DataFrame(base * np.exp(np.cumsum(steps, axis=0)), index=idx, columns=cols)
    open_ = close.shift(1).fillna(close)
    high = np.maximum(close, open_)
    low = np.minimum(close, open_)
    vol = pd.DataFrame(rng.uniform(2e3, 2e4, size=(n, len(cols))), index=idx, columns=cols)
    return {"Close": close, "Open": open_, "High": high, "Low": low, "Volume": vol}


print("── 資料契約 ─────────────────────────────")

# 1) 乾淨面板 → 通過
p = make_panel(["2330", "2317", "2454"])
r = validate_panel(p, AS_OF, scheduled=True, calendar=CAL, min_completeness=0.9)
check("乾淨面板通過", r.ok)

# 2) 過期面板（最後 bar 是上一個工作日）→ fail freshness
stale = make_panel(["2330", "2317"], last_date=AS_OF - pd.Timedelta(days=1))
r = validate_panel(stale, AS_OF, scheduled=True, calendar=CAL)
check("過期面板被擋(freshness)", (not r.ok) and any("預期 session" in x for x in r.reasons))

# 2b) dispatch 模式對同一過期面板較寬鬆（不因 freshness 擋）
r2 = validate_panel(stale, AS_OF, scheduled=False, calendar=CAL)
check("dispatch 模式放寬 freshness", r2.ok)

# 3) 最後 bar 完整度不足 → fail
p3 = make_panel(["2330", "2317", "2454", "2412"])
p3["Close"].iloc[-1, 1:] = np.nan  # 只剩 1/5 有值
r = validate_panel(p3, AS_OF, scheduled=True, calendar=CAL, min_completeness=0.9)
check("完整度不足被擋", (not r.ok) and any("完整度" in x for x in r.reasons))

# 4) 缺 0050 → fail key ticker
p4 = make_panel(["2330", "2317"], include_0050=False)
r = validate_panel(p4, AS_OF, scheduled=True, calendar=CAL)
check("缺關鍵標的0050被擋", (not r.ok) and any("0050" in x for x in r.reasons))

# 5) 跳變 → fail jump
p5 = make_panel(["2330", "2317"])
p5["Close"].iloc[-1, p5["Close"].columns.get_loc("2330")] *= 1.8  # +80%
r = validate_panel(p5, AS_OF, scheduled=True, calendar=CAL)
check("異常跳變被擋", (not r.ok) and any("跳變" in x or "報酬" in x for x in r.reasons))

# 6) 最後 bar 無量 → fail volume
p6 = make_panel(["2330", "2317", "2454"])
p6["Volume"].iloc[-1, :] = 0
r = validate_panel(p6, AS_OF, scheduled=True, calendar=CAL, require_volume=True)
check("最後bar無量被擋", (not r.ok) and any("有量" in x for x in r.reasons))

print("── 快照 round-trip ─────────────────────")
p = make_panel(["2330", "2317", "2454"], seed=7)
man = build_manifest(p, AS_OF, provider="yfinance", auto_adjust=True,
                     contract=validate_panel(p, AS_OF, scheduled=True, calendar=CAL))
with tempfile.TemporaryDirectory() as d:
    snap_dir = os.path.join(d, "snap")
    freeze_snapshot(p, snap_dir, man)
    close2, open2, high2, low2, vol2 = load_snapshot(snap_dir)
    check("round-trip Close 完全一致", close2.equals(p["Close"].sort_index()))
    check("round-trip 五欄位齊全", all(x is not None for x in (close2, open2, high2, low2, vol2)))
    # subset：只取 2330 + 一段日期
    c_sub, *_ = load_snapshot(snap_dir, tickers=["2330"], start=p["Close"].index[10])
    check("snapshot subset 正確", list(c_sub.columns) == ["2330"] and len(c_sub) == len(p["Close"]) - 10)
    check("manifest 帶 hash/provider/auto_adjust",
          bool(man["panel_sha256"]) and man["provider"] == "yfinance" and man["auto_adjust"] is True)

# 同資料兩次 freeze → hash 相同（可重現性）
man_b = build_manifest(p, AS_OF, provider="yfinance", auto_adjust=True)
check("相同面板 hash 穩定", man["panel_sha256"] == man_b["panel_sha256"])

print("── 容量門 ──────────────────────────────")
# 造 ADV：一檔大量、一檔極小量
idx = pd.bdate_range(end=AS_OF, periods=30)
close = pd.DataFrame({"BIG": 100.0, "TINY": 100.0}, index=idx)
vol = pd.DataFrame({"BIG": 5_000_000.0, "TINY": 1_000.0}, index=idx)  # ADV: BIG=5億, TINY=10萬
# 等權 2 檔、資金 1,000,000 → 每檔 50 萬
r = capacity_gate(["BIG", "TINY"], capital=1_000_000, close_df=close, vol_df=vol,
                  max_participation=0.10, max_days_to_exit=1.0)
check("容量門擋下小量股(TINY)", (not r.ok) and any("TINY" in v for v in r.violations))
check("容量門不誤殺大量股(BIG)", all("BIG" not in v for v in r.violations))

# 全大量 → 通過
r = capacity_gate({"BIG": 0.5}, capital=1_000_000, close_df=close, vol_df=vol,
                  max_participation=0.10)
check("大量小部位通過", r.ok)

# ADV=0 → inf participation → 擋
close0 = pd.DataFrame({"DEAD": 100.0}, index=idx)
vol0 = pd.DataFrame({"DEAD": 0.0}, index=idx)
r = capacity_gate(["DEAD"], capital=100_000, close_df=close0, vol_df=vol0)
check("零成交量標的被擋", not r.ok)

# orders JSON 形狀（list[dict]，無 weight → 等權）
orders = [{"ticker": "BIG"}, {"ticker": "TINY"}]
r = capacity_gate(orders, capital=1_000_000, close_df=close, vol_df=vol, max_participation=0.10)
check("接受 orders list[dict] 形狀", (not r.ok) and any("TINY" in v for v in r.violations))

print("\n========================================")
print(f"通過 {len(PASS)} / 失敗 {len(FAIL)}")
if FAIL:
    print("失敗項：", FAIL)
    raise SystemExit(1)
print("全部通過 ✅")
