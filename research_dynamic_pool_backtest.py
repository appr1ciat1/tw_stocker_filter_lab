"""
research_dynamic_pool_backtest.py — 動態池 vs 靜態 116 的同 run A/B 回測（研究線）

嚴格對齊 DYNAMIC_POOL_RESEARCH.md 的 Phase 2：
  · 候選池以 point-in-time 建構（pool_generator，shift(1) 無前視）
  · 兩邊都保留「內層 top-60 流動性篩選」，只讓『外層池』不同
    （靜態 116 vs 動態池）→ 差異單純歸因於池，符合「同 run off vs on」鐵律
  · 引擎參數固定為生產 v8.5 基準（不調參，調參止於 Phase 1）
  · 產出各 trial 的日報酬 → 餵進 pool_acceptance 統計門（PBO/DSR）

注意：本腳本只做研究，不寫任何生產報表、不碰四策略。
"""

import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from strategy.ai_strategy import fetch_panel_data, engineer_features, build_liquid_universe
from strategy.event_backtest import EventDrivenBacktester
from twstk.data.pool_generator import build_pointintime_pools, churn_stats
import pool_audit as A

# 生產 v8.5 基準（對齊 update_ai_report.yml 的 v8.5 步驟；其餘皆為引擎預設）
V85 = dict(regime_filter=True, initial_capital=200_000)
TOP_K = 7
THRESHOLD = 2.0

# PBO 需要多個 trial：用不同池參數當試驗組（合法的「選最好會不會過擬合」檢定）
VARIANTS = {
    "dyn_130_170_w20": dict(enter_rank=130, exit_rank=170, window=20),
    "dyn_100_140_w20": dict(enter_rank=100, exit_rank=140, window=20),
    "dyn_150_200_w20": dict(enter_rank=150, exit_rank=200, window=20),
    "dyn_130_170_w60": dict(enter_rank=130, exit_rank=170, window=60),
}


def load_turnover(cache_path) -> pd.DataFrame:
    with open(cache_path, "rb") as f:
        cache = pickle.load(f)
    df = pd.DataFrame.from_dict(cache, orient="index").sort_index()
    df.index = pd.to_datetime(df.index)
    return df


def run_engine(close, open_, high, low, vol, universe_mask, market_close=None):
    """跑一次 v8.5 基準回測，回傳 (equity_df, daily_returns)。"""
    total_score, ma_60, atr_df, short_ma = engineer_features(close, vol, universe_mask)
    bt = EventDrivenBacktester(**V85)
    trades, equity = bt.run(total_score, close, open_, high, low, ma_60,
                            top_k=TOP_K, threshold=THRESHOLD,
                            market_close=market_close, vol_df=vol,
                            universe_mask=universe_mask)
    eq = equity["Equity"].astype(float)
    ret = eq.pct_change().dropna()
    return equity, ret, trades


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, help="market_turnover_cache.pkl")
    ap.add_argument("--start", default="2019-01-01")
    ap.add_argument("--min-adv", type=float, default=50_000_000)
    ap.add_argument("--min-history", type=int, default=60)
    ap.add_argument("--inner-top", type=int, default=60, help="內層流動性篩選（兩邊一致）")
    ap.add_argument("--out-dir", default="artifacts/research_pool")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    turn = load_turnover(args.cache)
    turn = turn.loc[pd.Timestamp(args.start):]
    print(f"全市場成交額矩陣: {turn.shape}  ({turn.index[0].date()} → {turn.index[-1].date()})")

    # 1) 各 variant 的 point-in-time 池
    masks = {}
    for name, kw in VARIANTS.items():
        res = build_pointintime_pools(turn, min_adv=args.min_adv,
                                      min_history=args.min_history, **kw)
        masks[name] = res.mask
        cs = churn_stats(res.mask)
        print(f"  {name}: 平均池 {cs['avg_pool_size']:.0f} 檔, "
              f"日均進 {cs['avg_daily_adds']:.1f}/出 {cs['avg_daily_drops']:.1f}")

    # 2) 需要 OHLCV 的標的 = 所有 variant 曾入池者 ∪ 靜態 116
    static_pool = [c for l in A.parse_pool_with_sectors("ai_report.py").values() for c in l]
    union = sorted(set().union(*[set(m.columns[m.any()]) for m in masks.values()]) | set(static_pool))
    print(f"需抓 OHLCV 標的數: {len(union)}（動態池歷史成員 ∪ 靜態116）")

    # ★公平性關鍵：OHLCV 必須截到『成交額資料涵蓋範圍』為止。
    #   否則靜態池會跑滿全期、動態池只在有 mask 的期間有池 → 兩臂期間不同，比較無效。
    data_end = turn.index[-1]
    close, open_, high, low, vol = fetch_panel_data(union, start_date=args.start,
                                                    end_date=data_end)
    print(f"OHLCV: {close.shape}  (截至 {data_end.date()}，與成交額資料同範圍)")

    # 3) 大盤（regime filter 用）
    from strategy.ai_strategy import fetch_panel_data as _f
    mc, *_ = _f(["0050"], start_date=args.start, end_date=data_end)
    market_close = mc["0050"] if "0050" in mc.columns else None

    def align(mask):
        return mask.reindex(index=close.index, columns=close.columns).fillna(False).infer_objects(copy=False).astype(bool)

    results = {}

    # 4a) 靜態 116 基準（內層 top-60，與生產同）
    static_cols = [c for c in static_pool if c in close.columns]
    inner_static = build_liquid_universe(close[static_cols], vol[static_cols], top_n=args.inner_top)
    m_static = pd.DataFrame(False, index=close.index, columns=close.columns)
    m_static[static_cols] = inner_static.reindex(index=close.index, columns=static_cols).fillna(False)
    eq, ret, tr = run_engine(close, open_, high, low, vol, m_static, market_close)
    results["static_116"] = ret
    eq.to_csv(os.path.join(args.out_dir, "equity_static_116.csv"))
    print(f"  static_116: 交易 {len(tr)} 筆, 期末 {eq['Equity'].iloc[-1]:,.0f}")

    # 4b) 各動態池 variant（外層=動態池，內層同樣 top-60）
    for name, mask in masks.items():
        m = align(mask)
        vol_masked = vol.where(m)          # 池外設 NaN → 內層排名只在池內進行
        inner = build_liquid_universe(close, vol_masked, top_n=args.inner_top)
        m_final = (m & inner.reindex(index=close.index, columns=close.columns).fillna(False))
        eq, ret, tr = run_engine(close, open_, high, low, vol, m_final, market_close)
        results[name] = ret
        eq.to_csv(os.path.join(args.out_dir, f"equity_{name}.csv"))
        print(f"  {name}: 交易 {len(tr)} 筆, 期末 {eq['Equity'].iloc[-1]:,.0f}")

    # 5) 統一評估窗：從『動態池首次成形』起算，兩臂用完全相同期間比較。
    #    （更早的期間屬 window/min_history 暖機，動態池尚未成形，比了不公平。）
    any_mask = None
    for m in masks.values():
        am = align(m)
        any_mask = am if any_mask is None else (any_mask | am)
    populated = any_mask.sum(axis=1)
    ready = populated[populated >= 10]
    eval_start = ready.index[0] if len(ready) else close.index[0]
    print(f"\n📏 統一評估窗（動態池成形後）：{eval_start.date()} → {close.index[-1].date()}")

    rets = pd.DataFrame(results).loc[eval_start:]
    rets.to_csv(os.path.join(args.out_dir, "trial_returns.csv"))

    print("\n=== 同期間 A/B 摘要 ===")
    for name in rets.columns:
        r = rets[name].dropna()
        if len(r) < 2:
            print(f"  {name}: 樣本不足"); continue
        cum = float((1 + r).prod() - 1)
        ann = float((1 + cum) ** (252 / len(r)) - 1) if len(r) > 0 else float("nan")
        shp = float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else float("nan")
        eq = (1 + r).cumprod()
        mdd = float((eq / eq.cummax() - 1).min())
        print(f"  {name:<18} 累積 {cum:+7.1%}  年化 {ann:+7.1%}  Sharpe {shp:5.2f}  MDD {mdd:6.1%}")
    print(f"\n📁 已輸出 {args.out_dir}/trial_returns.csv  shape={rets.shape}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
