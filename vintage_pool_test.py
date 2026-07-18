"""
vintage_pool_test.py — 決定性檢驗：現行 116 的超額是不是「後見之明」？

問題：現行靜態池含 6547 高端疫苗、2618/2610/2603 航運、生技等
      『明顯是 2020-2021 事件之後才會被選進來』的標的。若這份清單是近年才
      編成的，那用它回測 2019 起就內嵌 selection look-ahead——
      回測的 +43% 年化可能是「早知道哪些會紅」的產物，而非可達成的預期。

檢驗：建一個嚴格 point-in-time 的「2019 年份(vintage)靜態池」——
      只用 2019 年初當下的流動性排名選出同樣檔數，然後『凍結不更新』
      （與現行池同為凍結清單），跑同一段、同引擎、同內層篩選的 A/B。

判讀：
  vintage ≈ 動態池(~30%)          → 現行池超額來自後見之明，績效預期應下修
  vintage 仍顯著優於動態池         → 「精選少數標的」本身有效，現行做法獲支持

★誠實性控制：vintage 池含後來下市者；若其 OHLCV 抓不到會被自動剔除，
  反而給 vintage 臂生存者偏誤的優勢。本腳本會量測並回報該比例。
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
import pool_audit as A

V85 = dict(regime_filter=True, initial_capital=200_000)
TOP_K, THRESHOLD = 7, 2.0


def run_engine(close, open_, high, low, vol, mask, market_close):
    total_score, ma_60, atr_df, short_ma = engineer_features(close, vol, mask)
    bt = EventDrivenBacktester(**V85)
    trades, equity = bt.run(total_score, close, open_, high, low, ma_60,
                            top_k=TOP_K, threshold=THRESHOLD,
                            market_close=market_close, vol_df=vol, universe_mask=mask)
    eq = equity["Equity"].astype(float)
    return equity, eq.pct_change().dropna(), trades


def stats(r):
    r = r.dropna()
    if len(r) < 2:
        return dict(cum=np.nan, ann=np.nan, sharpe=np.nan, mdd=np.nan)
    cum = float((1 + r).prod() - 1)
    ann = float((1 + cum) ** (252 / len(r)) - 1)
    shp = float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else np.nan
    eq = (1 + r).cumprod()
    mdd = float((eq / eq.cummax() - 1).min())
    return dict(cum=cum, ann=ann, sharpe=shp, mdd=mdd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--select-end", default="2019-04-10", help="選池窗結束（之後才開始評估）")
    ap.add_argument("--eval-start", default="2019-04-11")
    ap.add_argument("--select-window", type=int, default=40, help="選池用的中位數成交額窗")
    ap.add_argument("--inner-top", type=int, default=60)
    ap.add_argument("--out-dir", default="artifacts/vintage")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.cache, "rb") as f:
        cache = pickle.load(f)
    turn = pd.DataFrame.from_dict(cache, orient="index").sort_index()
    turn.index = pd.to_datetime(turn.index)
    print(f"全市場成交額: {turn.shape}  {turn.index[0].date()} → {turn.index[-1].date()}")

    # ── 1) 現行池 ──
    current = [c for l in A.parse_pool_with_sectors("ai_report.py").values() for c in l]
    n_pool = len(current)
    print(f"現行靜態池: {n_pool} 檔")

    # ── 2) vintage 池：只用 select_end 之前的資料 ──
    sel = turn.loc[:pd.Timestamp(args.select_end)]
    med = sel.tail(args.select_window).median(axis=0, skipna=True)
    # 需在選池窗內有足夠交易日（排除新上市/幾乎不交易者）
    active = sel.tail(args.select_window).notna().sum(axis=0) >= int(args.select_window * 0.8)
    med = med[active].dropna().sort_values(ascending=False)
    vintage = list(med.index[:n_pool])
    print(f"vintage 池: {len(vintage)} 檔（選池窗 {sel.index[-args.select_window].date()} → {sel.index[-1].date()}，"
          f"僅用當時資料）")

    overlap = sorted(set(vintage) & set(current))
    print(f"\n★ 重疊度: {len(overlap)}/{n_pool} = {len(overlap)/n_pool:.1%}")
    only_cur = sorted(set(current) - set(vintage))
    print(f"  現行池有、2019 選不到的 {len(only_cur)} 檔: {only_cur[:25]}{' ...' if len(only_cur)>25 else ''}")

    # ── 2b) 產業配額對齊的 vintage：分離『年份』與『選股方法論』兩個變因 ──
    #   純流動性 vintage 與現行池的產業結構差很多（金融 9vs22、航運 1vs9、電子零組件 19vs8），
    #   差距可能來自方法論而非年份。此臂用 2019 當時流動性，但在『每個產業內』
    #   各選與現行池相同檔數 → 若仍大幅落後，才是名單本身帶後見之明。
    from twstk.data import security_master as SM
    try:
        SM.load_master()
        ind_of = {t: (SM.describe(t).get("industry") or "其他") for t in med.index}
        quota = {}
        for t in current:
            k = SM.describe(t).get("industry") or "其他"
            quota[k] = quota.get(k, 0) + 1
        picked, used = [], set()
        for k, need in sorted(quota.items(), key=lambda x: -x[1]):
            cands = [t for t in med.index if ind_of.get(t) == k and t not in used]
            take = cands[:need]
            picked += take; used |= set(take)
        # 配額不足者用整體流動性序補滿
        for t in med.index:
            if len(picked) >= n_pool:
                break
            if t not in used:
                picked.append(t); used.add(t)
        vintage_sm = picked[:n_pool]
        ov_sm = set(vintage_sm) & set(current)
        print(f"  產業配額對齊 vintage: {len(vintage_sm)} 檔，與現行池重疊 {len(ov_sm)} ({len(ov_sm)/n_pool:.1%})")
    except Exception as e:
        print(f"  ⚠️ 產業配額 vintage 建立失敗（略過該臂）：{e}")
        vintage_sm = []

    # ── 3) 抓 OHLCV（各池聯集），並量測 vintage 的生存者偏誤 ──
    union = sorted(set(vintage) | set(current) | set(vintage_sm))
    print(f"\n抓取 OHLCV: {len(union)} 檔（vintage ∪ 現行）...")
    close, open_, high, low, vol = fetch_panel_data(union, start_date="2019-01-01",
                                                    end_date=turn.index[-1])
    print(f"OHLCV: {close.shape}")

    v_ok = [t for t in vintage if t in close.columns and close[t].notna().sum() > 100]
    c_ok = [t for t in current if t in close.columns and close[t].notna().sum() > 100]
    print(f"\n★ 生存者偏誤量測:")
    print(f"  vintage 池 {len(vintage)} 檔 → 有行情 {len(v_ok)} 檔（抓不到 {len(vintage)-len(v_ok)} 檔，"
          f"多為後來下市 → 這些被自動剔除，對 vintage 臂『有利』）")
    print(f"  現行池 {len(current)} 檔 → 有行情 {len(c_ok)} 檔")

    mc, *_ = fetch_panel_data(["0050"], start_date="2019-01-01", end_date=turn.index[-1])
    market_close = mc["0050"] if "0050" in mc.columns else None

    # ── 4) 同引擎、同內層 top-60，只有外層池不同 ──
    vsm_ok = [t for t in vintage_sm if t in close.columns and close[t].notna().sum() > 100]
    if vintage_sm:
        print(f"  產業配額 vintage {len(vintage_sm)} 檔 → 有行情 {len(vsm_ok)} 檔")

    arms = [("current_static", c_ok), ("vintage_2019", v_ok)]
    if vsm_ok:
        arms.append(("vintage_sector_matched", vsm_ok))

    results, meta = {}, {}
    for name, members in arms:
        cols = [t for t in members if t in close.columns]
        inner = build_liquid_universe(close[cols], vol[cols], top_n=args.inner_top)
        mask = pd.DataFrame(False, index=close.index, columns=close.columns)
        mask[cols] = inner.reindex(index=close.index, columns=cols).fillna(False)
        eq, ret, tr = run_engine(close, open_, high, low, vol, mask, market_close)
        eq.to_csv(os.path.join(args.out_dir, f"equity_{name}.csv"))
        results[name] = ret
        meta[name] = dict(n_members=len(cols), n_trades=len(tr))
        print(f"  {name}: {len(cols)} 檔可交易, {len(tr)} 筆交易")

    rets = pd.DataFrame(results).loc[pd.Timestamp(args.eval_start):]
    rets.to_csv(os.path.join(args.out_dir, "vintage_returns.csv"))

    print("\n" + "=" * 74)
    print(f"同期間 A/B（{rets.index[0].date()} → {rets.index[-1].date()}，{len(rets)} 交易日）")
    print("=" * 74)
    print(f"  {'臂':<18}{'累積':>11}{'年化':>10}{'Sharpe':>9}{'MDD':>9}   檔數/交易")
    for name in rets.columns:
        s = stats(rets[name])
        print(f"  {name:<18}{s['cum']:>+10.1%}{s['ann']:>+10.1%}{s['sharpe']:>9.2f}"
              f"{s['mdd']:>9.1%}   {meta[name]['n_members']}/{meta[name]['n_trades']}")

    print(f"\n  （對照：先前全期動態池結果 年化 +27.7%~+33.2%、Sharpe 0.98~1.12）")

    cs = stats(rets["current_static"])
    vs = stats(rets["vintage_2019"])
    print("\n" + "=" * 74)
    print("判讀")
    print("=" * 74)
    print(f"  現行池 {cs['ann']:+.1%} vs 純流動性 vintage {vs['ann']:+.1%}（差 {cs['ann']-vs['ann']:+.1%}）")
    if "vintage_sector_matched" in rets.columns:
        vm = stats(rets["vintage_sector_matched"])
        gap_m = cs["ann"] - vm["ann"]
        print(f"  現行池 {cs['ann']:+.1%} vs 產業配額 vintage {vm['ann']:+.1%}（差 {gap_m:+.1%}）")
        print()
        if abs(gap_m) < 0.06:
            print("  → 產業配額對齊後兩者接近：差距主要來自『選股方法論(產業結構)』而非年份，")
            print("     後見之明疑慮大幅降低，現行做法獲得支持。")
        elif gap_m >= 0.15:
            print("  → 即使產業配額對齊，現行池仍大幅領先：名單本身的『個股選擇』帶有後見之明成分，")
            print("     基於此回測調出的參數，其績效預期應下修。")
        else:
            print("  → 產業配額對齊後差距縮小但仍存在：後見之明與方法論兩者皆有貢獻，")
            print("     建議以較保守的預期看待回測績效。")
    print(f"\n  ⚠️ 生存者偏誤：vintage 臂有 {len(vintage)-len(v_ok)} 檔下市股抓不到行情而被剔除，")
    print(f"     其表現已被『高估』；真實 vintage 只會更差（此偏誤方向對結論是保守的）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
