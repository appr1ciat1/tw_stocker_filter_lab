"""
run_final_acceptance.py — 動態候選池三門總驗收（Phase 2 收尾）

輸入：
  · 全市場成交額快取（決定覆蓋門與 ADV）
  · research_dynamic_pool_backtest.py 產出的 trial_returns.csv（統計門）
輸出：
  · 三門 scorecard（統計 / 容量 / 覆蓋腐化）
  · 容量天花板：這個池在多少資金下才開始撞流動性上限
    （小資金時容量門幾乎必過、不具約束力；真正的問題是「能撐多少錢」）
"""

import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pool_audit as A
from twstk.data.pool_generator import build_pointintime_pools, latest_pool, churn_stats
from pool_acceptance import run_acceptance, statistical_gate
from validation.capacity import capacity_gate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--returns", required=True, help="trial_returns.csv")
    ap.add_argument("--start", default="2019-01-01")
    ap.add_argument("--enter-rank", type=int, default=130)
    ap.add_argument("--exit-rank", type=int, default=170)
    ap.add_argument("--window", type=int, default=20)
    ap.add_argument("--min-adv", type=float, default=50_000_000)
    ap.add_argument("--min-history", type=int, default=60)
    ap.add_argument("--capital", type=float, default=200_000)
    ap.add_argument("--position-size", type=float, default=0.10)
    ap.add_argument("--top-k", type=int, default=7)
    args = ap.parse_args()

    # ── 資料 ──
    with open(args.cache, "rb") as f:
        cache = pickle.load(f)
    turn = pd.DataFrame.from_dict(cache, orient="index").sort_index()
    turn.index = pd.to_datetime(turn.index)
    turn = turn.loc[pd.Timestamp(args.start):]
    print(f"全市場成交額: {turn.shape}  {turn.index[0].date()} → {turn.index[-1].date()}")

    med = A.median_daily_value(turn, args.window)

    # ── 候選池（最後一日組成） ──
    res = build_pointintime_pools(turn, window=args.window, enter_rank=args.enter_rank,
                                  exit_rank=args.exit_rank, min_adv=args.min_adv,
                                  min_history=args.min_history)
    pool = latest_pool(turn, window=args.window, enter_rank=args.enter_rank,
                       exit_rank=args.exit_rank, min_adv=args.min_adv,
                       min_history=args.min_history)
    cs = churn_stats(res.mask)
    print(f"候選池: {len(pool)} 檔 | 平均池 {cs['avg_pool_size']:.0f} | "
          f"日均進 {cs['avg_daily_adds']:.2f}/出 {cs['avg_daily_drops']:.2f} | "
          f"單向換手/日 {cs['one_way_turnover_per_day']:.3f}")

    # ── 統計門輸入 ──
    rets = pd.read_csv(args.returns, index_col=0, parse_dates=True)
    dyn_cols = [c for c in rets.columns if c.startswith("dyn")]
    best = rets[dyn_cols].mean().idxmax() if dyn_cols else rets.columns[0]
    print(f"統計門：{rets.shape[0]} 個交易日、{rets.shape[1]} 個 trial，代表 trial={best}")

    # ── 容量門輸入（用池內每檔等權 position_size 當意圖部位） ──
    close_stub = pd.DataFrame(1.0, index=turn.index, columns=turn.columns)
    vol_stub = turn / 1.0  # turnover 已是「金額」，令 close=1 使 close*vol = 金額
    weights = {t: args.position_size for t in pool}

    report = run_acceptance(
        pool, med,
        weights=weights, capital=args.capital,
        close_df=close_stub, vol_df=vol_stub,
        returns_by_trial=rets, single_returns=rets[best],
        n_trials=rets.shape[1],
        present_codes=set(turn.columns),
        coverage_top=150, top_rank=300, floor_dollars=args.min_adv,
        min_coverage=0.80, max_corruption=0.10,
        max_pbo=0.5, min_dsr_prob=0.95,
        cap_kwargs=dict(lookback=args.window, max_participation=0.10,
                        max_days_to_exit=1.0),
    )
    print("\n" + "=" * 60)
    print(report.summary())
    print("=" * 60)

    # ── 容量天花板：多大資金才撞上限 ──
    print("\n📐 容量天花板（單筆 = 資金 × position_size，門檻 participation ≤ 10% ADV）")
    for cap in (2e5, 1e6, 1e7, 5e7, 1e8, 5e8, 1e9):
        g = capacity_gate(weights, cap, close_stub, vol_stub,
                          lookback=args.window, max_participation=0.10,
                          max_days_to_exit=1.0)
        n_bad = len(g.violations)
        print(f"   資金 NT${cap:>13,.0f}  單筆 NT${cap*args.position_size:>11,.0f}  "
              f"→ {'✅ 通過' if g.ok else f'❌ {n_bad} 檔超限'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
