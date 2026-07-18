"""
preflight.py — 每日資料前置關卡（fail-closed）+ 四策略共享快照

每天在四個策略跑之前，只做一次：
  1. 抓一次完整面板（EXTENDED_TICKERS + 0050）。
  2. 用資料契約驗證：最後一根 bar 的新鮮度 / 完整度 / 關鍵標的 / 跳變。
  3. 通過 → 凍結成 snapshot（panel.pkl + manifest.json），設 TWSTK_SNAPSHOT，
     讓四策略共讀同一份 → 輸入完全一致、且當日 run 可重現。
  4. 不通過 → 非零 exit code（fail-closed）：workflow 後續步驟被跳過，
     不產生、不發布任何新報表，保留前一份。

用法：
  python preflight.py --start-date 2019-01-01 --scheduled
  python preflight.py --start-date 2019-01-01 --dispatch   # 人工補跑，放寬 freshness
"""

import argparse
import os
import re
import sys
from datetime import datetime, timezone, timedelta

import pandas as pd

from strategy.ai_strategy import fetch_panel_data
from twstk.data.contract import (
    validate_panel, build_manifest, freeze_snapshot, is_trading_day,
)

BENCHMARK_TICKER = "0050"  # regime filter 依賴，納入驗證與快照


def load_extended_tickers(path="ai_report.py"):
    """沿用 repo 慣例：以 regex 從 ai_report.py 取 EXTENDED_TICKERS，避免 import 副作用。"""
    with open(path, "r", encoding="utf-8") as f:
        src = f.read()
    m = re.search(r"EXTENDED_TICKERS\s*=\s*\[(.*?)\]", src, re.S)
    if not m:
        raise RuntimeError("無法從 ai_report.py 解析 EXTENDED_TICKERS")
    tickers = re.findall(r"'([0-9A-Za-z]+)'", m.group(1))
    # 去重、保序
    return list(dict.fromkeys(tickers))


def taiwan_today():
    return (datetime.now(timezone.utc) + timedelta(hours=8)).date()


def export_env(key, value):
    """寫入 GitHub Actions 的 $GITHUB_ENV 與 $GITHUB_OUTPUT（若存在）。"""
    for var in ("GITHUB_ENV", "GITHUB_OUTPUT"):
        path = os.environ.get(var)
        if path:
            with open(path, "a", encoding="utf-8") as f:
                f.write(f"{key}={value}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-date", default="2019-01-01")
    ap.add_argument("--end-date", default=None)
    ap.add_argument("--days", type=int, default=1200)
    ap.add_argument("--as-of", default=None, help="台灣日期，預設今天(UTC+8)")
    ap.add_argument("--scheduled", dest="scheduled", action="store_true", default=True)
    ap.add_argument("--dispatch", dest="scheduled", action="store_false",
                    help="人工補跑：放寬最後 bar 必須==今日 session")
    ap.add_argument("--out", default=None, help="snapshot 輸出目錄，預設 artifacts/snapshot_<date>")
    ap.add_argument("--min-completeness", type=float, default=0.90)
    args = ap.parse_args()

    as_of = pd.Timestamp(args.as_of) if args.as_of else pd.Timestamp(taiwan_today())
    print("=" * 60)
    print(f"🔒 Preflight 資料契約  as_of={as_of.date()}  mode={'scheduled' if args.scheduled else 'dispatch'}")
    print("=" * 60)

    # 排程模式下，若行事曆明確判定「今天不是交易日」→ 直接不發布（fail-closed 的另一面）。
    # 注意：is_trading_day 回傳 None 代表「行事曆無法判定」，此時也 fail-closed（不擅自假設是交易日）。
    if args.scheduled:
        trading = is_trading_day(as_of)
        if trading is False:
            print("📅 XTAI 行事曆：今天非交易日 → 不發布新報表（正常跳過）")
            return 0  # 正常跳過（非錯誤）；workflow 以 snapshot 是否產出決定是否續跑
        if trading is None:
            print("❌ 行事曆無法判定是否為交易日 → fail-closed，停止發布")
            return 2

    tickers = load_extended_tickers()
    fetch_list = list(dict.fromkeys([BENCHMARK_TICKER] + tickers))
    print(f"📥 前置下載 {len(fetch_list)} 檔（含 {BENCHMARK_TICKER}）...")

    try:
        close, open_, high, low, vol = fetch_panel_data(
            fetch_list, days=args.days,
            start_date=args.start_date, end_date=args.end_date,
        )
    except Exception as e:
        print(f"❌ 前置下載失敗：{e} → fail-closed")
        return 2

    panel = {"Close": close, "Open": open_, "High": high, "Low": low, "Volume": vol}

    result = validate_panel(
        panel, as_of,
        scheduled=args.scheduled,
        min_completeness=args.min_completeness,
        key_tickers=(BENCHMARK_TICKER,),
    )
    print(result.summary())

    if not result.ok:
        print("\n❌ 資料契約未通過 → 停止發布（保留前一份報表）")
        return 3

    out_dir = args.out or os.path.join("artifacts", f"snapshot_{as_of.strftime('%Y%m%d')}")
    manifest = build_manifest(panel, as_of, provider="yfinance",
                              auto_adjust=True, contract=result)
    freeze_snapshot(panel, out_dir, manifest)
    snap_pkl = os.path.join(out_dir, "panel.pkl")
    print(f"\n🧊 已凍結共享快照：{out_dir}")
    print(f"   hash={manifest['panel_sha256'][:16]}…  n_tickers={manifest['n_tickers']}  last={manifest['last_session']}")

    export_env("TWSTK_SNAPSHOT", snap_pkl)
    print(f"   → 已設 TWSTK_SNAPSHOT={snap_pkl}（四策略將共讀此份）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
