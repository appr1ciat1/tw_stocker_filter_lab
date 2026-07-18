"""
s1_shadow_diff.py — S1：FinMind vs yfinance 雙源 shadow diff（唯讀量測）

目的：在實作 #6 雙資料源容錯『之前』，先用實據回答一個問題——
      兩個源能不能互換？若還原(adjust)基準不同，自動 fallback 會靜默污染動量訊號，
      比沒有 fallback 更糟。

方法：三方比對，才能分離「資料本身不一致」與「還原基準不一致」
  · yf_adj   = yfinance auto_adjust=True   （生產目前使用）
  · yf_raw   = yfinance auto_adjust=False  （未還原）
  · finmind  = FinMind TaiwanStockPrice     （據稱未還原）

診斷邏輯：
  finmind vs yf_raw 接近      → 兩源底層資料一致（差異純粹來自還原）
  yf_adj  vs yf_raw 有落差    → 還原幅度（除權息造成）
  finmind vs yf_adj 的落差    → 若直接 fallback 會吃到的誤差
  『日報酬』比『價位』更關鍵：動量策略吃的是報酬，報酬對不上才是致命的。

窗口刻意涵蓋台股除權息季（7–9 月），這是還原差異會現形的地方。
"""

import argparse
import sys
import time
import urllib.parse

import numpy as np
import pandas as pd

from pool_audit import _get_json  # 沿用含 curl 後援的取數（政府/第三方端點憑證問題）

FINMIND = "https://api.finmindtrade.com/api/v4/data"

# 取樣：涵蓋上市/上櫃、高低價、配息與不配息
SAMPLE = ["2330", "2317", "2454", "2412", "2881", "1301", "2603", "5274", "3529", "6547"]


def fetch_finmind(ticker, start, end, token=None):
    params = {"dataset": "TaiwanStockPrice", "data_id": ticker,
              "start_date": start, "end_date": end}
    if token:
        params["token"] = token
    url = FINMIND + "?" + urllib.parse.urlencode(params)
    d = _get_json(url)
    rows = d.get("data") or []
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date").sort_index()


def fetch_yf(tickers, start, end, adjust):
    import yfinance as yf
    syms = [f"{t}.TW" for t in tickers]
    raw = yf.download(syms, start=start, end=end, progress=False,
                      auto_adjust=adjust)
    if raw.empty:
        return {}, {}
    close, vol = {}, {}
    for t, s in zip(tickers, syms):
        try:
            close[t] = raw[("Close", s)].dropna()
            vol[t] = raw[("Volume", s)].dropna()
        except Exception:
            pass
    # 上櫃補抓
    missing = [t for t in tickers if t not in close or close[t].empty]
    if missing:
        raw2 = yf.download([f"{t}.TWO" for t in missing], start=start, end=end,
                           progress=False, auto_adjust=adjust)
        for t in missing:
            try:
                close[t] = raw2[("Close", f"{t}.TWO")].dropna()
                vol[t] = raw2[("Volume", f"{t}.TWO")].dropna()
            except Exception:
                pass
    return close, vol


def compare(a: pd.Series, b: pd.Series, label_a, label_b):
    """回傳價位與『日報酬』的比對統計（報酬才是策略吃的東西）。"""
    idx = a.index.intersection(b.index)
    if len(idx) < 5:
        return None
    x, y = a.loc[idx].astype(float), b.loc[idx].astype(float)
    lvl_ratio = float((x / y).median())
    lvl_maxdiff = float(((x - y).abs() / y).max())
    rx, ry = x.pct_change().dropna(), y.pct_change().dropna()
    ridx = rx.index.intersection(ry.index)
    rx, ry = rx.loc[ridx], ry.loc[ridx]
    ret_corr = float(rx.corr(ry)) if len(rx) > 2 else np.nan
    ret_maxdiff = float((rx - ry).abs().max())
    ret_big = int(((rx - ry).abs() > 0.005).sum())  # 日報酬差 >0.5% 的天數
    return dict(n=len(idx), lvl_ratio=lvl_ratio, lvl_maxdiff=lvl_maxdiff,
                ret_corr=ret_corr, ret_maxdiff=ret_maxdiff, ret_big_days=ret_big)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2025-06-01")
    ap.add_argument("--end", default="2026-07-17")
    ap.add_argument("--tickers", default=",".join(SAMPLE))
    ap.add_argument("--token", default=None, help="FinMind token（可選，提高額度）")
    args = ap.parse_args()

    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    print("=" * 72)
    print(f"S1 雙源 shadow diff  {args.start} → {args.end}  取樣 {len(tickers)} 檔")
    print("=" * 72)

    print("\n📥 yfinance (auto_adjust=True / False)...")
    yfa_c, yfa_v = fetch_yf(tickers, args.start, args.end, True)
    yfr_c, yfr_v = fetch_yf(tickers, args.start, args.end, False)

    print("📥 FinMind TaiwanStockPrice...")
    fm = {}
    for t in tickers:
        try:
            fm[t] = fetch_finmind(t, args.start, args.end, args.token)
        except Exception as e:
            print(f"   ⚠️ {t} FinMind 失敗: {e}")
            fm[t] = pd.DataFrame()
        time.sleep(0.4)

    rows = []
    for t in tickers:
        f = fm.get(t)
        if f is None or f.empty or t not in yfa_c or t not in yfr_c:
            rows.append(dict(ticker=t, note="資料缺失"))
            continue
        fc = f["close"].astype(float)
        fv = f["Trading_Volume"].astype(float) if "Trading_Volume" in f else None

        c1 = compare(fc, yfr_c[t], "finmind", "yf_raw")
        c2 = compare(yfa_c[t], yfr_c[t], "yf_adj", "yf_raw")
        c3 = compare(fc, yfa_c[t], "finmind", "yf_adj")
        vr = None
        if fv is not None and t in yfr_v:
            iv = fv.index.intersection(yfr_v[t].index)
            if len(iv) > 5:
                vr = float((fv.loc[iv] / yfr_v[t].loc[iv].astype(float)).median())

        rows.append(dict(
            ticker=t,
            n=c1["n"] if c1 else np.nan,
            fm_vs_raw_ratio=c1["lvl_ratio"] if c1 else np.nan,
            fm_vs_raw_retcorr=c1["ret_corr"] if c1 else np.nan,
            fm_vs_raw_retmax=c1["ret_maxdiff"] if c1 else np.nan,
            adj_vs_raw_ratio=c2["lvl_ratio"] if c2 else np.nan,
            adj_vs_raw_retmax=c2["ret_maxdiff"] if c2 else np.nan,
            adj_vs_raw_bigdays=c2["ret_big_days"] if c2 else np.nan,
            fm_vs_adj_ratio=c3["lvl_ratio"] if c3 else np.nan,
            fm_vs_adj_retmax=c3["ret_maxdiff"] if c3 else np.nan,
            fm_vs_adj_bigdays=c3["ret_big_days"] if c3 else np.nan,
            vol_ratio_fm_over_yf=vr,
        ))

    df = pd.DataFrame(rows).set_index("ticker")
    pd.set_option("display.width", 200)

    print("\n【A】FinMind vs yfinance(未還原) — 底層資料是否同源")
    print(df[["n", "fm_vs_raw_ratio", "fm_vs_raw_retcorr", "fm_vs_raw_retmax"]].to_string())
    print("\n【B】yfinance 還原 vs 未還原 — 還原幅度（除權息影響）")
    print(df[["adj_vs_raw_ratio", "adj_vs_raw_retmax", "adj_vs_raw_bigdays"]].to_string())
    print("\n【C】FinMind vs yfinance(還原) — 直接 fallback 會吃到的誤差")
    print(df[["fm_vs_adj_ratio", "fm_vs_adj_retmax", "fm_vs_adj_bigdays"]].to_string())
    print("\n【D】成交量單位比（FinMind / yfinance，1=同單位，1000=張vs股）")
    print(df[["vol_ratio_fm_over_yf"]].to_string())

    # 判讀
    print("\n" + "=" * 72)
    print("判讀")
    print("=" * 72)
    ok_same_source = (df["fm_vs_raw_retcorr"] > 0.999).sum()
    n_valid = df["fm_vs_raw_retcorr"].notna().sum()
    print(f"· 底層同源：{ok_same_source}/{n_valid} 檔的『未還原日報酬』相關係數 > 0.999")
    adj_days = df["adj_vs_raw_bigdays"].fillna(0).sum()
    print(f"· 還原造成的日報酬差異(>0.5%)總天數：{adj_days:.0f}")
    fb_days = df["fm_vs_adj_bigdays"].fillna(0).sum()
    print(f"· 若直接以 FinMind 取代生產(還原)資料，日報酬差異(>0.5%)總天數：{fb_days:.0f}")
    if fb_days > adj_days * 0.5:
        print("\n⚠️ 結論：兩源『還原基準不同』。#6 fallback 不可直接切換，"
              "必須先做還原對齊層，否則除權息日會產生假跳空、污染動量訊號。")
    else:
        print("\n✅ 結論：兩源可直接互換（還原基準一致）。")
    df.to_csv("artifacts/s1_shadow_diff.csv", encoding="utf-8-sig")
    print("\n📁 artifacts/s1_shadow_diff.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
