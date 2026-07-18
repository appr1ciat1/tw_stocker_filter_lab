"""
s4_execution_slippage.py — S4：訊號→執行 的落差量測（唯讀）

背景（重要）：原本設想的「paper vs 回測追蹤誤差」在本專案『無法測量』——
  `build_confirmed_filter_pages.py` 與 `twstk/paper/tracker.py` 的 paper 都是
  用歷史資料重跑的『模擬』，不是實際下單記錄（paper_state.json 還被 gitignore、
  CI 每次重建）。拿模擬比回測 = 回測比回測，誤差恆為零。

因此改測真正會傷到績效的那一段：**訊號當下的參考價 → 隔日實際開盤成交價**。
回測假設 `model_entry_ref='next_open'` 進場，但你做決策時看到的是前一日收盤
(`reference_close`)。兩者之間的隔夜跳空，就是這套流程的真實執行成本來源，
也是 `--gap-filter` 想擋的東西。

輸入：artifacts/orders_*.json（系統真實發出過的訂單）
輸出：隔夜跳空分布、gap-filter 觸發率、尾端風險
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd


def load_orders(pattern):
    rows = []
    for f in sorted(glob.glob(pattern)):
        try:
            d = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        for o in d.get("orders", []):
            o["_file"] = os.path.basename(f)
            rows.append(o)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    for c in ("signal_date", "execution_date"):
        if c in df:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df


def fetch_ohlc(tickers, start, end):
    import yfinance as yf
    out = {}
    tl = sorted(set(tickers))
    for suffix in (".TW", ".TWO"):
        need = [t for t in tl if t not in out]
        if not need:
            break
        raw = yf.download([f"{t}{suffix}" for t in need], start=start, end=end,
                          progress=False, auto_adjust=True)
        if raw.empty:
            continue
        for t in need:
            sym = f"{t}{suffix}"
            try:
                sub = pd.DataFrame({
                    "open": raw[("Open", sym)], "high": raw[("High", sym)],
                    "low": raw[("Low", sym)], "close": raw[("Close", sym)],
                }).dropna()
                if not sub.empty:
                    out[t] = sub
            except Exception:
                pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orders", required=True, help="glob，如 '../old/artifacts/orders_*.json'")
    ap.add_argument("--gap-atr", type=float, default=1.5, help="gap-filter 門檻(ATR 倍數)")
    ap.add_argument("--atr-window", type=int, default=14)
    ap.add_argument("--out", default="artifacts/s4_execution_slippage.csv")
    args = ap.parse_args()

    od = load_orders(args.orders)
    if od.empty:
        print("找不到任何訂單"); return 1
    print(f"訂單 {len(od)} 筆，來自 {od['_file'].nunique()} 個檔案")
    print(f"signal_date 範圍：{od['signal_date'].min().date()} → {od['signal_date'].max().date()}")

    # 只留有完整欄位者
    need = {"ticker", "signal_date", "execution_date", "reference_close"}
    od = od.dropna(subset=[c for c in need if c in od.columns])
    od["ticker"] = od["ticker"].astype(str)

    start = (od["signal_date"].min() - pd.Timedelta(days=60)).strftime("%Y-%m-%d")
    end = (od["execution_date"].max() + pd.Timedelta(days=5)).strftime("%Y-%m-%d")
    print(f"抓取行情 {start} → {end}（{od['ticker'].nunique()} 檔）...")
    ohlc = fetch_ohlc(od["ticker"].unique(), start, end)
    print(f"取得 {len(ohlc)} 檔行情")

    recs = []
    for _, r in od.iterrows():
        t = r["ticker"]
        if t not in ohlc:
            continue
        px = ohlc[t]
        ed = pd.Timestamp(r["execution_date"]).normalize()
        # 找執行日（或其後第一個有交易的日）
        fut = px.index[px.index >= ed]
        if len(fut) == 0:
            continue
        d0 = fut[0]
        row = px.loc[d0]
        ref = float(r["reference_close"])
        if not np.isfinite(ref) or ref <= 0:
            continue
        open_ = float(row["open"])

        # ★方法修正：跳空必須用『同一次抓取』內的前一交易日收盤，兩邊還原基準才一致。
        #   若拿當年記錄的 reference_close 去比今天重抓的(已回溯還原)開盤價，
        #   除權息後的標的會產生假跳空（S1 已證實 yfinance 會回溯還原現金+股票股利）。
        prev_idx = px.index[px.index < d0]
        if len(prev_idx) == 0:
            continue
        prev_close = float(px.loc[prev_idx[-1], "close"])
        if not np.isfinite(prev_close) or prev_close <= 0:
            continue
        gap = open_ / prev_close - 1.0
        # 診斷：當年記錄價 vs 今日還原價的落差 = 期間累積的除權息還原幅度
        adj_drift = prev_close / ref - 1.0

        # 該股在訊號日前的 ATR%，用來換算 gap-filter 門檻
        hist = px.loc[:pd.Timestamp(r["signal_date"])].tail(args.atr_window + 1)
        atr_pct = np.nan
        if len(hist) >= 5:
            tr = (hist["high"] - hist["low"]).abs()
            atr_pct = float(tr.mean() / prev_close)

        recs.append(dict(
            ticker=t, signal_date=r["signal_date"].date(), exec_date=d0.date(),
            reference_close=ref, prev_close=prev_close, adj_drift=adj_drift,
            actual_open=open_, gap=gap,
            atr_pct=atr_pct,
            gap_over_atr=(abs(gap) / atr_pct) if (atr_pct and np.isfinite(atr_pct) and atr_pct > 0) else np.nan,
            day_high=float(row["high"]), day_low=float(row["low"]), day_close=float(row["close"]),
        ))

    df = pd.DataFrame(recs)
    if df.empty:
        print("無可比對訂單"); return 1

    print("\n" + "=" * 66)
    print(f"訊號→執行 隔夜跳空分析（{len(df)} 筆可比對訂單）")
    print("=" * 66)
    g = df["gap"]
    print(f"  中位數跳空      {g.median():+.3%}")
    print(f"  平均跳空        {g.mean():+.3%}")
    print(f"  標準差          {g.std():.3%}")
    print(f"  最好 / 最差     {g.max():+.2%} / {g.min():+.2%}")
    for thr in (0.005, 0.01, 0.02, 0.03):
        print(f"  |跳空| > {thr:.1%}      {(g.abs() > thr).mean():.1%}  ({int((g.abs()>thr).sum())} 筆)")
    print(f"\n  不利跳空(開高於參考價，買進成本上升) 佔比 {(g>0).mean():.1%}")
    print(f"  不利跳空的平均幅度 {g[g>0].mean():+.3%}" if (g > 0).any() else "")

    # gap-filter 觸發率
    valid = df["gap_over_atr"].notna()
    if valid.any():
        trig = df.loc[valid, "gap_over_atr"] > args.gap_atr
        print(f"\n  gap-filter({args.gap_atr}×ATR) 會取消 {trig.mean():.1%} 的訂單 "
              f"({int(trig.sum())}/{int(valid.sum())} 筆)")
        if trig.any():
            sub = df.loc[valid][trig.values]
            print(f"  被取消者的平均跳空 {sub['gap'].mean():+.2%}"
                  f"（未取消者 {df.loc[valid][~trig.values]['gap'].mean():+.2%}）")

    print("\n  最不利的 5 筆（前日收 → 當日開，同一還原基準）：")
    for _, r in df.nlargest(5, "gap").iterrows():
        print(f"    {r['ticker']} {r['signal_date']}→{r['exec_date']}  "
              f"前收 {r['prev_close']:.1f} → 開盤 {r['actual_open']:.1f}  ({r['gap']:+.2%})")

    # 診斷：當年記錄價與今日還原價的落差（= 期間累積除權息還原）
    ad = df["adj_drift"].dropna()
    if len(ad):
        big = (ad.abs() > 0.02)
        print(f"\n  【還原漂移診斷】當年 reference_close vs 今日還原前收：")
        print(f"    中位 {ad.median():+.2%} | |漂移|>2% 佔 {big.mean():.1%} ({int(big.sum())} 筆)")
        print(f"    ← 這是回溯還原造成的，不是執行成本；若用它算跳空會嚴重高估")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"\n📁 {args.out}")

    print("\n" + "=" * 66)
    print("解讀")
    print("=" * 66)
    print("· 這是『你看到訊號的價』與『實際能成交的價』之間的落差，")
    print("  回測以 next_open 進場，故此落差已在回測內；真正的風險是它的『尾端』。")
    print(f"· 若不利跳空平均 {g[g>0].mean():+.2%} 而單筆成本約 0.58%，")
    print("  跳空對報酬的影響與交易成本同量級，不可忽略。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
