"""
pool_audit.py — 116 檔靜態池「腐化審計」（唯讀，不改任何既有程式）

目的：在動態池研究開跑前，量化現有池爛掉的程度，決定動態池的優先級。

四維度（方法論依權威來源）：
  A 流動性腐化 — 每檔近 N 日「中位數」日成交額（成交金額）
                 ① 全市場百分位排名（相對）② 絕對 NT$ 地板（絕對）。
                 用中位數而非平均：MSCI 流動性方法論明言中位數可「排除單日極端成交量」，
                 這對處置股/題材股單日爆量特別關鍵（平均會讓爛股看起來還活躍）。
  B 財務困難/接近下市（只抓最強訊號）— 已下市/停牌（存活者偏誤）：
                 個股從全市場近期資料中消失，或在 TWSE 終止上市清單內。
  C 時代印記/集中 — 依原始碼 EXTENDED_TICKERS 註解分組（半導體/電子/金融/傳產/航運/生技/其他），
                 各組家數 + 各組流動性中位數，找整組衰退者。
  D 覆蓋缺口 — 全市場 Top-150（依中位數成交額）有幾檔不在 116 內（漏掉的 alpha）。

決策：腐化率（A 未過 或 B 消失）與 覆蓋率（D）雙指標；任一差就支持動態池。

資料源（全部免費、唯讀）：
  上市 TWSE  MI_INDEX?date=YYYYMMDD&type=ALLBUT0999  （個股表：證券代號/成交金額/收盤）
  上櫃 TPEx  afterTrading/dailyQuotes?date=YYYY/MM/DD&type=EW （代號/成交金額(元)）
  下市 TWSE  openapi /v1/company/suspendListingCsvAndHtml

注意：本審計為唯讀分析，對外部端點採 fail-soft（某來源失敗 → 警告並續跑），
      與生產線 preflight 的 fail-closed 相反——因為它不產生任何交易訊號。
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import ssl
import sys
import time
import urllib.request
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd

TWSE_MI = "https://www.twse.com.tw/rwd/zh/afterTrading/MI_INDEX?date={ymd}&type=ALLBUT0999&response=json"
TPEX_DQ = "https://www.tpex.org.tw/www/zh-tw/afterTrading/dailyQuotes?date={y}/{m}/{d}&type=EW&response=json"
TWSE_DELISTED = "https://openapi.twse.com.tw/v1/company/suspendListingCsvAndHtml"

_UA = {"User-Agent": "Mozilla/5.0 (pool-audit read-only)"}


# ─────────────────── 股池解析（含原始碼註解分組）───────────────────

def parse_pool_with_sectors(path="ai_report.py"):
    """從 ai_report.py 的 EXTENDED_TICKERS 區塊解析 {sector: [tickers]}（依 # 註解）。"""
    src = open(path, "r", encoding="utf-8").read()
    m = re.search(r"EXTENDED_TICKERS\s*=\s*\[(.*?)\]", src, re.S)
    if not m:
        raise RuntimeError("無法解析 EXTENDED_TICKERS")
    block = m.group(1)
    sectors: dict[str, list[str]] = {}
    current = "未分類"
    for line in block.splitlines():
        cm = re.search(r"#\s*(.+?)\s*$", line)
        code_part = line.split("#")[0]
        if cm and not re.search(r"'[0-9A-Za-z]+'", code_part):
            current = cm.group(1).strip()
            sectors.setdefault(current, [])
            continue
        codes = re.findall(r"'([0-9A-Za-z]+)'", code_part)
        if codes:
            sectors.setdefault(current, []).extend(codes)
    # 去重（保序）
    seen = set()
    for s in sectors:
        uniq = []
        for c in sectors[s]:
            if c not in seen:
                seen.add(c); uniq.append(c)
        sectors[s] = uniq
    return sectors


# ─────────────────── 資料抓取（fail-soft）───────────────────

def _curl_json(url):
    """curl 後援：部分台灣政府端點憑證在較新 Python 的 OpenSSL 會解碼失敗，
    但 curl 可正常取得。純唯讀公開資料。"""
    import subprocess
    out = subprocess.run(["curl", "-s", "-m", "30", "-A", _UA["User-Agent"], url],
                         capture_output=True, text=True, encoding="utf-8")
    if out.returncode != 0 or not out.stdout:
        raise RuntimeError(f"curl 失敗 rc={out.returncode}")
    return json.loads(out.stdout)


def _get_json(url, retries=3, pause=0.4):
    last = None
    for i in range(retries):
        # 1) 正常驗證  2) 不驗證(憑證問題)  3) curl 後援
        for attempt in ("verify", "noverify", "curl"):
            try:
                if attempt == "curl":
                    return _curl_json(url)
                ctx = ssl._create_unverified_context() if attempt == "noverify" else None
                req = urllib.request.Request(url, headers=_UA)
                with urllib.request.urlopen(req, timeout=30, context=ctx) as r:
                    return json.loads(r.read().decode("utf-8"))
            except Exception as e:
                last = e
        time.sleep(pause * (i + 1))
    raise last


def _to_num(s):
    try:
        return float(str(s).replace(",", "").strip())
    except Exception:
        return np.nan


def _is_common_stock(code):
    """普通股：4 位純數字且非 0 開頭（排除 0050/0056 等 ETF、含字母之特別股/權證）。"""
    return len(code) == 4 and code.isdigit() and code[0] != "0"


def fetch_twse_day(ymd: str) -> dict:
    """上市單日 {code: 成交金額}。ymd='YYYYMMDD'。"""
    d = _get_json(TWSE_MI.format(ymd=ymd))
    if d.get("stat") != "OK":
        return {}
    for t in d.get("tables", []):
        f = t.get("fields") or []
        if "證券代號" in f and "成交金額" in f and "收盤價" in f:
            ci, vi = f.index("證券代號"), f.index("成交金額")
            out = {}
            for row in t.get("data", []):
                code = str(row[ci]).strip()
                if _is_common_stock(code):
                    out[code] = _to_num(row[vi])
            return out
    return {}


def fetch_tpex_day(dt: pd.Timestamp) -> dict:
    """上櫃單日 {code: 成交金額}。"""
    url = TPEX_DQ.format(y=dt.year, m=f"{dt.month:02d}", d=f"{dt.day:02d}")
    d = _get_json(url)
    tables = d.get("tables") or []
    if not tables:
        return {}
    t = tables[0]
    f = t.get("fields") or []
    # 代號 / 成交金額(元)
    ci = f.index("代號") if "代號" in f else 0
    vi = next((i for i, x in enumerate(f) if "成交金額" in str(x)), None)
    if vi is None:
        return {}
    out = {}
    for row in t.get("data", []):
        code = str(row[ci]).strip()
        if _is_common_stock(code):
            out[code] = _to_num(row[vi])
    return out


def fetch_market_history(sessions) -> pd.DataFrame:
    """對每個 session 抓上市+上櫃成交金額，回傳 (session x code) 的 DataFrame。fail-soft。"""
    rows = {}
    ok_days = 0
    for dt in sessions:
        ymd = dt.strftime("%Y%m%d")
        day = {}
        try:
            day.update(fetch_twse_day(ymd))
        except Exception as e:
            print(f"   ⚠️ TWSE {ymd} 抓取失敗（跳過）：{e}")
        try:
            day.update(fetch_tpex_day(dt))
        except Exception as e:
            print(f"   ⚠️ TPEx {ymd} 抓取失敗（跳過）：{e}")
        if day:
            rows[pd.Timestamp(dt).normalize()] = day
            ok_days += 1
            print(f"   · {dt.date()} 上市+上櫃 {len(day)} 檔")
        time.sleep(0.35)
    if ok_days == 0:
        raise RuntimeError("全市場資料全數抓取失敗")
    df = pd.DataFrame.from_dict(rows, orient="index").sort_index()
    return df


def fetch_delisted_codes() -> set:
    """TWSE 終止上市清單（fail-soft，抓不到回空集合）。"""
    try:
        d = _get_json(TWSE_DELISTED)
        codes = set()
        for rec in d:
            for k in ("Code", "公司代號", "code"):
                if k in rec and rec[k]:
                    codes.add(str(rec[k]).strip())
        return codes
    except Exception as e:
        print(f"   ⚠️ 終止上市清單抓取失敗（fail-soft）：{e}")
        return set()


# ─────────────────── 純計算核心（可單元測試）───────────────────

def median_daily_value(hist: pd.DataFrame, window: int) -> pd.Series:
    """每檔近 window 個 session 的『中位數』日成交額（排除單日爆量干擾）。"""
    return hist.tail(window).median(axis=0, skipna=True).rename(f"median_{window}d")


def audit_pool(pool_sectors: dict, med: pd.Series, *,
               top_rank: int = 300, coverage_top: int = 150,
               floor_dollars: float = 50_000_000,
               present_codes: set | None = None,
               delisted_codes: set | None = None) -> dict:
    """
    以全市場中位數成交額 med（index=code）對 116 檔做評估。
    回傳 dict：per-stock 明細 DataFrame + 匯總指標。
    """
    flat = [c for lst in pool_sectors.values() for c in lst]
    flat = list(dict.fromkeys(flat))
    present_codes = present_codes if present_codes is not None else set(med.dropna().index)
    delisted_codes = delisted_codes or set()

    # 全市場排名（成交額大 → rank 小）
    rank = med.rank(ascending=False, method="min")
    market_n = int(med.notna().sum())

    recs = []
    sector_of = {}
    for sec, lst in pool_sectors.items():
        for c in lst:
            sector_of[c] = sec

    for c in flat:
        in_market = c in present_codes and pd.notna(med.get(c, np.nan))
        mv = float(med.get(c, np.nan))
        rk = float(rank.get(c, np.nan))
        # 「下市/停牌」只認『近期全市場資料裡完全消失』這個硬訊號。
        # TWSE 歷史終止上市清單含『代碼重用』（如 2301 曾為舊公司、現為光寶科在市），
        # 不能拿來判定現況，故 delisted_codes 僅作旁註、不進判定。
        delisted = not in_market
        hist_delist = c in delisted_codes
        fail_rank = (not in_market) or (rk > top_rank)
        fail_floor = (not in_market) or (mv < floor_dollars)
        recs.append({
            "ticker": c,
            "sector": sector_of.get(c, "未分類"),
            "median_daily_value": mv,
            "market_rank": rk,
            "in_market": in_market,
            "delisted_or_suspended": delisted,
            "hist_delist_list": hist_delist,
            "fail_rank_top{}".format(top_rank): fail_rank,
            "fail_abs_floor": fail_floor,
            # 腐化 = 流動性未過(排名 或 地板) 或 已下市
            "corroded": bool(delisted or fail_rank or fail_floor),
        })
    detail = pd.DataFrame(recs).set_index("ticker")

    # 覆蓋率：全市場 Top-coverage_top 有幾檔不在池內
    top_codes = list(rank[rank <= coverage_top].sort_values().index)
    in_pool = set(flat)
    missing = [c for c in top_codes if c not in in_pool]
    coverage_hit = coverage_top - len(missing)

    # 各 sector 匯總
    sector_stats = []
    for sec, lst in pool_sectors.items():
        sub = detail.loc[[c for c in lst if c in detail.index]]
        sector_stats.append({
            "sector": sec,
            "n": len(sub),
            "corroded": int(sub["corroded"].sum()),
            "median_of_medians": float(sub["median_daily_value"].median(skipna=True)),
            "worst_rank": float(sub["market_rank"].max()),
        })
    sector_df = pd.DataFrame(sector_stats)

    n = len(flat)
    corruption_rate = float(detail["corroded"].mean())
    coverage_rate = coverage_hit / coverage_top
    return {
        "detail": detail,
        "sector": sector_df,
        "market_n": market_n,
        "pool_n": n,
        "corruption_rate": corruption_rate,
        "coverage_rate": coverage_rate,
        "coverage_missing": missing,
        "delisted_in_pool": [c for c in flat if detail.loc[c, "delisted_or_suspended"]],
        "params": dict(top_rank=top_rank, coverage_top=coverage_top,
                       floor_dollars=floor_dollars),
    }


# ─────────────────── 輸出 ───────────────────

def render_markdown(res: dict, window: int, as_of) -> str:
    d = res["detail"]
    p = res["params"]
    lines = []
    lines.append(f"# 116 檔靜態池 腐化審計  ({pd.Timestamp(as_of).date()})\n")
    lines.append(f"- 全市場普通股家數（近 {window} 日有量）：**{res['market_n']}**")
    lines.append(f"- 池內檔數：**{res['pool_n']}**")
    lines.append(f"- **腐化率**：**{res['corruption_rate']:.1%}**"
                 f"（流動性跌出 Top-{p['top_rank']} 或 < NT${p['floor_dollars']:,.0f}/日 或 已下市）")
    lines.append(f"- **覆蓋率**：**{res['coverage_rate']:.1%}**"
                 f"（全市場 Top-{p['coverage_top']} 中有 {len(res['coverage_missing'])} 檔不在池內）")
    if res["delisted_in_pool"]:
        lines.append(f"- ⚠️ **已下市/消失**：{', '.join(res['delisted_in_pool'])}")
    lines.append("\n## 各產業組\n")
    lines.append("| 組 | 檔數 | 腐化 | 中位數成交額(中位) | 最差排名 |")
    lines.append("|---|---|---|---|---|")
    for _, r in res["sector"].iterrows():
        mv = r["median_of_medians"]
        lines.append(f"| {r['sector']} | {r['n']} | {r['corroded']} | "
                     f"{('NT$%.0f萬' % (mv/1e4)) if pd.notna(mv) else '-'} | "
                     f"{('%.0f' % r['worst_rank']) if pd.notna(r['worst_rank']) else '-'} |")
    lines.append("\n## 腐化明細（僅列 corroded）\n")
    lines.append("| 代號 | 組 | 中位數日成交額 | 全市場排名 | 原因 |")
    lines.append("|---|---|---|---|---|")
    bad = d[d["corroded"]].sort_values("market_rank", na_position="last")
    for t, r in bad.iterrows():
        reason = []
        if r["delisted_or_suspended"]:
            reason.append("已下市/消失")
        else:
            if r.get(f"fail_rank_top{p['top_rank']}"):
                reason.append(f"排名>{p['top_rank']}")
            if r["fail_abs_floor"]:
                reason.append("低於地板")
        mv = r["median_daily_value"]
        lines.append(f"| {t} | {r['sector']} | "
                     f"{('NT$%.0f萬' % (mv/1e4)) if pd.notna(mv) else '-'} | "
                     f"{('%.0f' % r['market_rank']) if pd.notna(r['market_rank']) else '-'} | "
                     f"{'/'.join(reason)} |")
    lines.append("\n## 覆蓋缺口：全市場 Top-{} 但不在池內（前 30）\n".format(p["coverage_top"]))
    lines.append(", ".join(res["coverage_missing"][:30]) or "（無）")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="116 檔靜態池腐化審計（唯讀）")
    ap.add_argument("--window", type=int, default=20, help="中位數回溯 session 數")
    ap.add_argument("--top-rank", type=int, default=300, help="流動性相對門檻：跌出全市場 Top-N 即腐化")
    ap.add_argument("--coverage-top", type=int, default=150, help="覆蓋率基準：全市場 Top-N")
    ap.add_argument("--floor", type=float, default=50_000_000, help="絕對地板：中位數日成交額(元)")
    ap.add_argument("--pool", default="ai_report.py")
    ap.add_argument("--out-dir", default="artifacts")
    args = ap.parse_args()

    try:
        import exchange_calendars as xcals
        cal = xcals.get_calendar("XTAI")
        today = pd.Timestamp((datetime.now(timezone.utc) + timedelta(hours=8)).date())
        sess = cal.sessions_in_range(today - pd.Timedelta(days=int(args.window * 2.2) + 15), today)
        sessions = [pd.Timestamp(s).normalize() for s in sess][-args.window:]
    except Exception as e:
        print(f"⚠️ 行事曆不可用，改用近似工作日：{e}")
        today = pd.Timestamp((datetime.now(timezone.utc) + timedelta(hours=8)).date())
        sessions = [d for d in pd.bdate_range(end=today, periods=args.window)]

    print("=" * 60)
    print(f"🔍 116 檔靜態池腐化審計  window={args.window} sessions "
          f"({sessions[0].date()} → {sessions[-1].date()})")
    print("=" * 60)

    pool = parse_pool_with_sectors(args.pool)
    print("池分組：" + ", ".join(f"{k}={len(v)}" for k, v in pool.items()))

    print("\n📥 抓全市場成交額（上市+上櫃）...")
    hist = fetch_market_history(sessions)
    med = median_daily_value(hist, args.window)
    delisted = fetch_delisted_codes()

    res = audit_pool(pool, med, top_rank=args.top_rank,
                     coverage_top=args.coverage_top, floor_dollars=args.floor,
                     present_codes=set(hist.columns), delisted_codes=delisted)

    md = render_markdown(res, args.window, sessions[-1])
    os.makedirs(args.out_dir, exist_ok=True)
    date_str = sessions[-1].strftime("%Y%m%d")
    md_path = os.path.join(args.out_dir, f"pool_audit_{date_str}.md")
    csv_path = os.path.join(args.out_dir, f"pool_audit_{date_str}.csv")
    open(md_path, "w", encoding="utf-8").write(md)
    res["detail"].to_csv(csv_path, encoding="utf-8-sig")

    print("\n" + md)
    print(f"\n📁 已輸出：{md_path} / {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
