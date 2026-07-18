"""
forward_record.py — 從 git 歷史還原「真實前瞻紀錄」並評估實際績效（唯讀）

為什麼這是真前瞻：每個交易日的 HTML 報表都在**當天收盤後、結果揭曉前**
產生並 commit 進 repo。git 歷史不可竄改，因此逐日 commit 中的
「🟢 建議買進 #N」清單，就是系統當時真正發出的訊號——
不是事後重跑的模擬（paper 權益頁才是重跑的）。

流程：
  1. 走訪每個每日報表 commit，從當時的 HTML 解析當日建議清單
     （代號 / 分數 / 參考價 / 停利 / 停損 / 排名）
  2. 依既有規則模擬執行：**隔一交易日開盤進場**，
     觸及停利或停損即出場，否則持有上限天數後出場
  3. 計入實際成本（買 0.143% / 賣 0.443%）
  4. 產出真實前瞻報酬，與回測宣稱值對照

注意：這量測的是「訊號品質 + 執行假設」，未計入你實際下單的時點差異。
"""

import argparse
import os
import re
import subprocess
import sys

import numpy as np
import pandas as pd

BUY_RE = re.compile(
    r"<tr[^>]*><td>(\d{4})</td><td>([\d.]+)</td><td>([\d.]+)</td><td>(.*?)</td><td>(.*?)</td>")
TP_RE = re.compile(r"停利:\s*([\d.]+)")
SL_RE = re.compile(r"停損:\s*([\d.]+)")


def git(repo, *args):
    return subprocess.run(["git", "-C", repo, *args], capture_output=True,
                          text=True, encoding="utf-8", errors="replace").stdout


def daily_commits(repo, since=None):
    out = git(repo, "log", "--format=%H|%ad", "--date=short", "--grep=AI Report", "--reverse")
    rows = []
    for line in out.strip().splitlines():
        if "|" not in line:
            continue
        h, d = line.split("|", 1)
        if since and d < since:
            continue
        rows.append((h, d))
    return rows


def parse_report(html):
    """回傳當日 [(ticker, score, ref_price, rank, tp, sl)]，僅取 🟢 建議買進。"""
    out = []
    for m in BUY_RE.finditer(html):
        tk, sc, px, status, plan = m.groups()
        st = re.sub(r"<[^>]+>", "", status)
        if "建議買進" not in st:
            continue
        rank = re.search(r"#(\d+)", st)
        # ★停利/停損數值被 <b>/<span> 夾住，必須先去標籤再抓數字
        plan_txt = re.sub(r"<[^>]+>", " ", plan)
        tp = TP_RE.search(plan_txt)
        sl = SL_RE.search(plan_txt)
        out.append(dict(ticker=tk, score=float(sc), ref_price=float(px),
                        rank=int(rank.group(1)) if rank else None,
                        tp=float(tp.group(1)) if tp else np.nan,
                        sl=float(sl.group(1)) if sl else np.nan))
    return out


def extract(repo, report_file, since):
    recs = []
    commits = daily_commits(repo, since)
    print(f"每日報表 commit: {len(commits)} 個（{commits[0][1]} → {commits[-1][1]}）")
    seen = set()
    # 早期 commit 尚無 report_v85.html（四策略分頁是後來才有），回退主報表
    candidates = [report_file] + [f for f in ("stock_report.html",) if f != report_file]
    for h, d in commits:
        html = ""
        for f in candidates:
            html = git(repo, "show", f"{h}:{f}")
            if html:
                break
        if not html:
            continue
        rows = parse_report(html)
        if not rows:
            continue
        # 同一天可能有多個 commit，只取第一份
        if d in seen:
            continue
        seen.add(d)
        for r in rows:
            r["signal_date"] = d
            recs.append(r)
    df = pd.DataFrame(recs)
    if df.empty:
        return df
    df["signal_date"] = pd.to_datetime(df["signal_date"])

    # ★去重：週末 commit、國定假日、颱風休市日（實例：2026-07-10）會把前一交易日的
    #   訊號原封不動再發布一次。指紋 = 當日(代號,參考價) 集合；與前一日相同即視為
    #   重複發布，只保留最早那次。這同時修掉「用 commit 日期當訊號日」的偏差。
    fp = (df.groupby("signal_date")
            .apply(lambda g: tuple(sorted(zip(g["ticker"], g["ref_price"]))), include_groups=False))
    keep, prev = [], None
    for d, sig in fp.sort_index().items():
        if sig != prev:
            keep.append(d)
        prev = sig
    dropped = len(fp) - len(keep)
    if dropped:
        print(f"  ⚠️ 去重：剔除 {dropped} 個重複發布日（休市/週末重發前日訊號）")
    return df[df["signal_date"].isin(keep)].reset_index(drop=True)


def fetch_ohlc(tickers, start, end):
    import yfinance as yf
    out = {}
    tl = sorted(set(tickers))
    for sfx in (".TW", ".TWO"):
        need = [t for t in tl if t not in out]
        if not need:
            break
        raw = yf.download([f"{t}{sfx}" for t in need], start=start, end=end,
                          progress=False, auto_adjust=True)
        if raw.empty:
            continue
        for t in need:
            s = f"{t}{sfx}"
            try:
                sub = pd.DataFrame({"open": raw[("Open", s)], "high": raw[("High", s)],
                                    "low": raw[("Low", s)], "close": raw[("Close", s)]}).dropna()
                if not sub.empty:
                    out[t] = sub
            except Exception:
                pass
    return out


def simulate(df, ohlc, hold_days, buy_cost, sell_cost):
    """隔日開盤進場；觸價停利/停損出場；否則持有上限天數以收盤出場。"""
    trades = []
    for _, r in df.iterrows():
        t = r["ticker"]
        px = ohlc.get(t)
        if px is None:
            continue
        fut = px.index[px.index > pd.Timestamp(r["signal_date"])]
        if len(fut) == 0:
            continue
        d0 = fut[0]
        entry = float(px.loc[d0, "open"])
        if not np.isfinite(entry) or entry <= 0:
            continue
        win = px.loc[d0:].head(hold_days + 1)
        exit_px, exit_d, reason = None, None, "hold"
        for d, row in win.iloc[1:].iterrows():   # 進場當日不判出場
            if np.isfinite(r["sl"]) and float(row["low"]) <= r["sl"]:
                exit_px, exit_d, reason = float(r["sl"]), d, "SL"; break
            if np.isfinite(r["tp"]) and float(row["high"]) >= r["tp"]:
                exit_px, exit_d, reason = float(r["tp"]), d, "TP"; break
        closed = True
        if exit_px is None:
            last = win.iloc[-1]
            exit_px, exit_d = float(last["close"]), win.index[-1]
            # ★持有期尚未走完就撞到資料末端 → 這筆『還沒結束』，不可計入已實現績效
            closed = (len(win) - 1) >= hold_days
            reason = "hold" if closed else "OPEN(未到期)"
        gross = exit_px / entry - 1.0
        net = (exit_px * (1 - sell_cost)) / (entry * (1 + buy_cost)) - 1.0
        trades.append(dict(ticker=t, signal_date=r["signal_date"].date(), entry_date=d0.date(),
                           exit_date=exit_d.date(), rank=r["rank"], score=r["score"],
                           entry=entry, exit=exit_px, reason=reason, closed=closed,
                           gross=gross, net=net,
                           days=int((exit_d - d0).days)))
    return pd.DataFrame(trades)


LOG_COLS = ["signal_date", "ticker", "score", "ref_price", "rank", "tp", "sl"]


def load_log(path):
    if not os.path.exists(path):
        return pd.DataFrame(columns=LOG_COLS)
    df = pd.read_csv(path)
    df["signal_date"] = pd.to_datetime(df["signal_date"])
    return df


def save_log(df, path):
    out = df[LOG_COLS].copy()
    out["signal_date"] = pd.to_datetime(out["signal_date"]).dt.strftime("%Y-%m-%d")
    out.sort_values(["signal_date", "rank"]).to_csv(path, index=False, encoding="utf-8-sig")


def append_today(report_file, log_path, as_of):
    """CI 每日呼叫：把當日報表的建議追加進日誌（不需 git 歷史，配合淺 checkout）。"""
    if not os.path.exists(report_file):
        print(f"找不到 {report_file}"); return 1
    rows = parse_report(open(report_file, encoding="utf-8", errors="replace").read())
    if not rows:
        print("當日無建議買進訊號，不追加"); return 0
    new = pd.DataFrame(rows)
    new["signal_date"] = pd.Timestamp(as_of)
    log = load_log(log_path)
    if not log.empty:
        last_day = log["signal_date"].max()
        prev = log[log["signal_date"] == last_day]
        same = (sorted(zip(prev["ticker"].astype(str), prev["ref_price"])) ==
                sorted(zip(new["ticker"].astype(str), new["ref_price"])))
        # 休市日重發前日訊號 → 不追加（資料契約已擋大部分，這是第二道保險）
        if same and pd.Timestamp(as_of) != last_day:
            print(f"⚠️ 當日訊號與 {last_day.date()} 完全相同（疑似休市重發）→ 不追加")
            return 0
        log = log[log["signal_date"] != pd.Timestamp(as_of)]   # 同日重跑則覆蓋
    save_log(pd.concat([log, new], ignore_index=True), log_path)
    print(f"✅ 已追加 {len(new)} 筆（{as_of}）→ {log_path}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", help="git repo 路徑（--backfill 用）")
    ap.add_argument("--report", default="report_v85.html")
    ap.add_argument("--since", default="2026-04-04", help="策略/股池定案日")
    ap.add_argument("--hold-days", type=int, default=20)
    ap.add_argument("--buy-cost", type=float, default=0.001425)
    ap.add_argument("--sell-cost", type=float, default=0.004425)
    ap.add_argument("--out", default="artifacts/forward_record.csv")
    ap.add_argument("--log", default="forward_log.csv", help="累積式前瞻訊號日誌")
    ap.add_argument("--backfill", action="store_true", help="一次性：從 git 歷史回填日誌")
    ap.add_argument("--append", action="store_true", help="每日：把當前報表追加進日誌（CI 用）")
    ap.add_argument("--as-of", default=None)
    args = ap.parse_args()

    if args.append:
        return append_today(args.report, args.log,
                            args.as_of or pd.Timestamp.today().strftime("%Y-%m-%d"))

    print("=" * 74)
    print(f"真實前瞻紀錄評估（日誌：{args.log}）")
    print("=" * 74)
    if args.backfill:
        if not args.repo:
            print("--backfill 需要 --repo"); return 1
        sig = extract(args.repo, args.report, args.since)
        if sig.empty:
            print("未解析到任何訊號"); return 1
        save_log(sig, args.log)
        print(f"✅ 回填 {len(sig)} 筆訊號 → {args.log}")
    else:
        sig = load_log(args.log)
        if sig.empty:
            print("日誌為空，請先執行 --backfill"); return 1
    print(f"訊號 {len(sig)} 筆，涵蓋 {sig['signal_date'].nunique()} 個交易日")
    print(f"訊號日期: {sig['signal_date'].min().date()} → {sig['signal_date'].max().date()}")

    top = sig["ticker"].value_counts()
    print(f"\n出現最多次的標的（你人工挑選的依據）：")
    for t, n in top.head(8).items():
        print(f"   {t}: {n} 次")

    print(f"\n抓取行情（{sig['ticker'].nunique()} 檔）...")
    ohlc = fetch_ohlc(sig["ticker"].unique(),
                      (sig["signal_date"].min() - pd.Timedelta(days=10)).strftime("%Y-%m-%d"),
                      "2026-07-19")
    print(f"取得 {len(ohlc)} 檔")

    tr = simulate(sig, ohlc, args.hold_days, args.buy_cost, args.sell_cost)
    if tr.empty:
        print("無可評估交易"); return 1

    done = tr[tr["closed"]]
    openp = tr[~tr["closed"]]
    print("\n" + "=" * 74)
    print(f"真實前瞻績效（訊號 {len(tr)} 筆：已結束 {len(done)}、尚未到期 {len(openp)}）")
    print("=" * 74)
    for label, d in (("★已結束(可信)", done), ("未到期(僅參考)", openp)):
        if d.empty:
            print(f"  [{label}] 無"); continue
        print(f"  [{label}] 筆數 {len(d):<4} 勝率 {(d['net']>0).mean():.1%}  "
              f"平均淨報酬 {d['net'].mean():+.2%}  中位 {d['net'].median():+.2%}")

    # 基準對照：同期間 0050，用來分辨是策略還是大盤
    try:
        import yfinance as yf
        b = yf.download(["0050.TW"], start=str(tr["entry_date"].min()),
                        end="2026-07-19", progress=False, auto_adjust=True)
        bc = b[("Close", "0050.TW")].dropna()
        if len(bc) > 1:
            print(f"\n  📊 同期間 0050：{bc.iloc[-1]/bc.iloc[0]-1:+.2%}"
                  f"（{bc.index[0].date()} → {bc.index[-1].date()}）← 用來分辨策略 vs 大盤")
    except Exception as e:
        print(f"  （基準抓取失敗：{e}）")
    print(f"\n  出場原因: {dict(tr['reason'].value_counts())}")
    print(f"  平均持有天數: {tr['days'].mean():.1f}")

    # ★以下分項一律只用『已結束』交易。未到期部位是在期末被強制標價、系統性偏負，
    #   混入會嚴重扭曲（曾因此得出與多頭市況矛盾的 -8%／勝率22% 錯誤結論）。
    print("\n  依訊號排名（僅已結束）:")
    for rk in sorted(done["rank"].dropna().unique()):
        d = done[done["rank"] == rk]
        print(f"    #{int(rk)}: {len(d):>3} 筆  平均 {d['net'].mean():+.2%}  勝率 {(d['net']>0).mean():.0%}")

    print("\n  出現次數 vs 表現（僅已結束）:")
    fr = tr.groupby("ticker").size().rename("freq")
    dd = done.join(fr, on="ticker")
    try:
        lab = pd.qcut(dd["freq"], 3, labels=["少", "中", "多"], duplicates="drop")
        for q, sub in dd.groupby(lab, observed=True):
            print(f"    出現次數{q}: {len(sub):>3} 筆  平均 {sub['net'].mean():+.2%}  "
                  f"勝率 {(sub['net']>0).mean():.0%}  (freq {sub['freq'].min()}-{sub['freq'].max()})")
    except Exception:
        pass
    r1 = dd[dd["rank"] == 1]
    if len(r1):
        hi = r1[r1["freq"] >= fr.median()]
        print(f"    ★#1 且出現次數≥中位數: {len(hi):>3} 筆  平均 {hi['net'].mean():+.2%}  "
              f"勝率 {(hi['net']>0).mean():.0%}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    tr.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"\n📁 {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
