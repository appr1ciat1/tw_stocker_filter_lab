"""
forward_stats_widget.py — paper 頁共用區塊：持續掛榜警示 + 挑選方式績效對照

資料來源：`forward_stats.json`（由 forward_record.py 產出，讀『真實前瞻紀錄』
—— 每日報表在結果揭曉前 commit 進 repo 的實際訊號，不是重跑的模擬）。

設計原則：
  · 數字不寫死在頁面，一律讀 JSON 並標明 as_of，避免顯示過時統計。
  · 「持續掛榜」以**警示**呈現，不是買進建議——實測該狀態下進場
    停損出場 72%、停利僅 12%，平均 -12.8%（基準 +5.6%）。
  · 樣本警語必須同時顯示，避免讀者把 3.5 個月多頭窗的數字當長期預期。
"""

from __future__ import annotations

import json
import os

STATS_FILE = "forward_stats.json"

_CSS = """
<style>
.fw-wrap{margin:18px 0}
.fw-warn{background:#3f1d1d;border:1px solid #7f1d1d;border-radius:8px;padding:12px 14px;margin:10px 0}
.fw-warn h4{margin:0 0 6px;color:#fca5a5;font-size:.95rem}
.fw-chips{display:flex;flex-wrap:wrap;gap:6px;margin:8px 0}
.fw-chip{background:#7f1d1d;color:#fecaca;border-radius:5px;padding:3px 9px;font-weight:600;font-size:.9rem}
.fw-tbl{width:100%;border-collapse:collapse;margin:8px 0;font-size:.9rem}
.fw-tbl th,.fw-tbl td{border-bottom:1px solid #334155;padding:6px 8px;text-align:right}
.fw-tbl th:first-child,.fw-tbl td:first-child{text-align:left}
.fw-tbl thead th{color:#94a3b8;font-weight:600}
.fw-pos{color:#34d399;font-weight:600}.fw-neg{color:#f87171;font-weight:600}
.fw-base{color:#94a3b8}
.fw-note{color:#94a3b8;font-size:.82rem;line-height:1.6;margin-top:8px}
</style>
"""


def load_stats(path: str = STATS_FILE) -> dict | None:
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _fmt(v: float, pct=True) -> str:
    cls = "fw-pos" if v > 0 else ("fw-neg" if v < 0 else "fw-base")
    txt = f"{v:+.2%}" if pct else f"{v:.0%}"
    return f"<span class='{cls}'>{txt}</span>"


def render(stats: dict | None = None, path: str = STATS_FILE,
           persist_today: list | None = None, link_fn=None) -> str:
    """
    回傳可直接插入 paper 頁的 HTML。stats 為 None 時自動讀檔；讀不到則回空字串
    （顯示層 fail-soft：缺統計不該讓整頁生不出來）。
    persist_today 可覆寫 JSON 內的警示名單（頁面若有更即時的資料時使用）。
    """
    stats = stats or load_stats(path)
    if not stats or not stats.get("blocks"):
        return ""

    today = persist_today if persist_today is not None else stats.get("persist_today") or []
    lf = link_fn or (lambda c: str(c))

    warn = ""
    if today:
        chips = "".join(f"<span class='fw-chip'>{lf(c)}</span>" for c in today)
        warn = (
            "<div class='fw-warn'>"
            f"<h4>⚠️ 持續掛榜未兌現（{stats.get('persist_rule','')}）</h4>"
            f"<div class='fw-chips'>{chips}</div>"
            "<div class='fw-note'>這些標的訊號持續出現卻走不出趨勢。實測在此狀態進場："
            "<b>停損出場 72%、停利僅 12%</b>。<b>這是排除清單，不是買進建議。</b>"
            "注意：這是一個<b>狀態</b>而非「壞股票」——同一檔在非此狀態時表現正常。</div>"
            "</div>"
        )
    else:
        warn = ("<div class='fw-note'>目前無標的處於「持續掛榜未兌現」狀態。</div>")

    # ── 今日訊號分級：把所有挑選方式套用到當日實際訊號，一檔一判定 ──
    today_tbl = ""
    td = stats.get("today") or {}
    if td.get("picks"):
        TONE = {"best": "#34d399", "bad": "#f87171", "warn": "#fbbf24", "neutral": "#94a3b8"}
        trows = ""
        for p in td["picks"]:
            c = TONE.get(p.get("tone"), "#94a3b8")
            trows += (
                f"<tr><td>{lf(p['ticker'])}</td>"
                f"<td>#{p.get('rank','-')}</td>"
                f"<td>{p.get('freq','-')} 次（{p.get('freq_tier','-')}）</td>"
                f"<td>{'⚠️ 是' if p.get('persistent') else '否'}</td>"
                f"<td style='text-align:left;color:{c};font-weight:600'>{p.get('verdict','')}</td></tr>"
            )
        today_tbl = (
            f"<h4 style='margin:14px 0 4px;color:#fcd34d;font-size:.95rem'>"
            f"今日訊號分級（{td.get('date','')}）</h4>"
            "<table class='fw-tbl'><thead><tr><th>股票</th><th>排名</th><th>出現次數</th>"
            "<th>持續掛榜</th><th style='text-align:left'>綜合判定</th></tr></thead>"
            f"<tbody>{trows}</tbody></table>"
            "<div class='fw-note'>判定同時套用所有挑選方式；"
            "<b>規則互相矛盾時明示衝突</b>（例：排名 #1 但出現次數偏少），不擇優呈現。</div>"
        )

    rows = ""
    for b in stats["blocks"]:
        base = "全部訊號" in b["name"]
        style = " style='background:#1e293b'" if base else ""
        rows += (f"<tr{style}><td>{b['name']}</td><td>{b['n']}</td>"
                 f"<td>{_fmt(b['avg'])}</td><td>{_fmt(b['wr'], pct=False)}</td></tr>")

    tbl = (
        "<table class='fw-tbl'><thead><tr><th>挑選方式</th><th>筆數</th>"
        "<th>平均淨報酬</th><th>勝率</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
    )

    note = (
        f"<div class='fw-note'>"
        f"資料：<b>真實前瞻紀錄</b>（每日報表於結果揭曉前 commit，非重跑模擬）"
        f"，僅計<b>已結束</b>交易，含成本（買 0.1425%／賣 0.4425%）。"
        f"區間 {stats.get('window','')}，as_of {stats.get('as_of','')}。<br>"
        f"⚠️ 樣本僅約 3.5 個月且同期 0050 +33%（<b>強多頭，未經完整回檔</b>），"
        f"樣本數少的分組請謹慎解讀；此非長期預期。"
        f"</div>"
    )
    return f"{_CSS}<div class='fw-wrap'>{warn}{today_tbl}{tbl}{note}</div>"


def persistent_from_rounds(rounds, min_hits: int = 2) -> list:
    """
    從 recent_buy_signal_rounds() 的輸出算出「每一輪都 ≥min_hits 次」的交集。
    rounds 內每筆 stocks 為 (code, name, count)。
    """
    if not rounds:
        return []
    sets = []
    for rd in rounds:
        sets.append({str(c) for c, _n, cnt in rd.get("stocks", []) if cnt >= min_hits})
    if not sets:
        return []
    out = set.intersection(*sets) if len(sets) > 1 else sets[0]
    return sorted(out)
