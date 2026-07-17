#!/usr/bin/env python3
"""Build the alert-only SURGE PRO capital-rotation monitor."""

import argparse
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

from strategies.registry import get_strategy
from strategies.rotation_exit import (
    active_rotation_alerts,
    compute_sector_rotation,
    extract_rotation_events,
)
from strategy.sector_flow import SECTOR_MAP
from twstk.backtest.engine import RunConfig, build_market_data


HTML_PATH = Path("capital_rotation_alert.html")
JSON_PATH = Path("capital_rotation_alert_latest.json")
HISTORY_PATH = Path("capital_rotation_history_10y.json")


def _value(value, pct=False):
    if value is None or not np.isfinite(float(value)):
        return "-"
    return f"{float(value) * 100:+.1f}%" if pct else f"{float(value):.3f}"


def _render_rows(alerts, labels):
    if alerts.empty:
        return '<tr><td colspan="12">目前沒有三日確認中的資金輪動警報</td></tr>'
    rows = []
    for _, row in alerts.iterrows():
        source = labels.get(row["source_sector"], row["source_sector"])
        destination = labels.get(row["destination_sector"], row["destination_sector"])
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(source))}</td>"
            f"<td>{html.escape(str(destination))}</td>"
            f"<td>{int(row['confirmation_days'])}</td>"
            f"<td>{_value(row['source_score'])}</td>"
            f"<td>{_value(row['destination_score'])}</td>"
            f"<td>{_value(row['score_spread'])}</td>"
            f"<td>{_value(row['source_return_5d'], True)}</td>"
            f"<td>{_value(row['destination_return_5d'], True)}</td>"
            f"<td>{_value(row['source_turnover_acceleration'], True)}</td>"
            f"<td>{_value(row['source_breadth_above_20d'], True)}</td>"
            f"<td>{_value(row['source_institutional_flow'])}</td>"
            f"<td>{_value(row['average_correlation_20d'])}</td>"
            "</tr>"
        )
    return "".join(rows)


def _records(frame):
    if frame.empty:
        return []
    clean = frame.replace([np.inf, -np.inf], np.nan).copy()
    for column in clean.columns:
        if pd.api.types.is_datetime64_any_dtype(clean[column]):
            clean[column] = clean[column].dt.strftime("%Y-%m-%d")
    return clean.where(clean.notna(), None).to_dict("records")


def _history_value(value, pct=False, digits=1):
    if value is None:
        return "-"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return html.escape(str(value))
    if not np.isfinite(number):
        return "-"
    if pct:
        return f"{number * 100:.{digits}f}%"
    return f"{number:.{digits}f}"


def _history_rows(history):
    events = history.get("events", [])
    if not events:
        return '<tr><td colspan="16">完整十年歷史檔不存在，請先執行 capital_rotation_10y.py</td></tr>'
    rows = []
    feature_labels = {
        "source_score": "流出領域分數",
        "destination_score": "流入領域分數",
        "score_spread": "分數差距",
        "source_score_change_5d": "流出分數5日變化",
        "destination_score_change_5d": "流入分數5日變化",
        "source_return_5d": "流出領域5日報酬",
        "destination_return_5d": "流入領域5日報酬",
        "source_return_20d": "流出領域20日報酬",
        "destination_return_20d": "流入領域20日報酬",
        "source_turnover_acceleration": "流出成交額加速",
        "destination_turnover_acceleration": "流入成交額加速",
        "source_breadth_above_20d": "流出20日線上廣度",
        "destination_breadth_above_20d": "流入20日線上廣度",
        "source_institutional_flow": "流出法人流",
        "destination_institutional_flow": "流入法人流",
        "average_correlation_20d": "20日平均相關",
    }
    pct_features = {
        "source_return_5d", "destination_return_5d",
        "source_return_20d", "destination_return_20d",
        "source_turnover_acceleration", "destination_turnover_acceleration",
        "source_breadth_above_20d", "destination_breadth_above_20d",
    }
    for item in reversed(events):
        event_id = int(item["event_id"])
        signal = item.get("signal", {})
        outcome = item.get("outcome", {})
        source = signal.get("source_sector_label") or signal.get("source_sector", "-")
        destination = (
            signal.get("destination_sector_label")
            or signal.get("destination_sector", "-")
        )
        tested = int(outcome.get("stocks_tested") or 0)
        hits = int(outcome.get("stocks_hit_20pct") or 0)
        search = html.escape(
            f"{event_id} {signal.get('signal_date', '')} {source} {destination}".lower(),
            quote=True,
        )
        rows.append(
            f'<tr class="history-main" data-search="{search}" data-event="{event_id}">'
            f'<td><button class="detail-btn" onclick="toggleEvent({event_id})">展開</button></td>'
            f"<td>{event_id}</td><td>{html.escape(str(signal.get('signal_date', '-')))}</td>"
            f"<td>{html.escape(str(source))}</td><td>{html.escape(str(destination))}</td>"
            f"<td>{int(signal.get('confirmation_days') or 0)}</td>"
            f"<td>{_history_value(signal.get('score_spread'), digits=3)}</td>"
            f"<td>{tested}</td><td>{hits}</td>"
            f"<td>{_history_value(outcome.get('hit_rate'), pct=True)}</td>"
            f"<td>{html.escape(str(outcome.get('earliest_drawdown_date') or '-'))}</td>"
            f"<td>{_history_value(outcome.get('earliest_observed_drawdown_days'), digits=0)}</td>"
            f"<td>{_history_value(outcome.get('median_lead_days'), digits=0)}</td>"
            f"<td>{_history_value(outcome.get('p80_lead_days'), digits=0)}</td>"
            f"<td>{_history_value(outcome.get('max_lead_days'), digits=0)}</td>"
            f"<td>{html.escape(str(outcome.get('last_session_before_earliest_drawdown') or '-'))}</td>"
            "</tr>"
        )
        feature_html = "".join(
            f"<div><span>{html.escape(label)}</span><b>{_history_value(signal.get(key), pct=key in pct_features, digits=2)}</b></div>"
            for key, label in feature_labels.items()
        )
        rows.append(
            f'<tr class="history-detail" id="event-detail-{event_id}" hidden>'
            f'<td colspan="16"><div class="feature-grid">{feature_html}</div>'
            f'<div class="stock-detail" id="stock-detail-{event_id}">載入 {tested} 檔股票明細中…</div>'
            "</td></tr>"
        )
    return "".join(rows)


def _history_section(history):
    if not history:
        return '<div class="card"><h2>十年完整回測歷史</h2><p>尚未產生完整歷史檔。</p></div>'
    headline = history.get("headline", {})
    period = history.get("period", {})
    files = history.get("files", {})
    downloads = "".join(
        f'<li><a href="{html.escape(filename)}">{html.escape(filename)}</a>'
        f' — {int(info.get("rows", 0)):,} 列'
        f' — SHA-256 <code>{html.escape(str(info.get("sha256", "")))}</code></li>'
        for filename, info in files.items()
    )
    return f"""
<div class="card">
 <h2>十年完整回測歷史：逐筆可核對</h2>
 <p class="note">期間 {html.escape(str(period.get('start', '-')))} ～ {html.escape(str(period.get('end', '-')))}；
 三日確認事件 {int(history.get('event_count', 0)):,} 筆；受測股票結果 {int(history.get('stock_test_count', 0)):,} 筆。
 下表不是抽樣，完整列出每一個事件。點「展開」可查看該事件的全部訊號參數與每一檔股票結果。</p>
 <div class="kpis">
  <div><span>事件數</span><b>{int(headline.get('events', 0)):,}</b></div>
  <div><span>股票明細</span><b>{int(headline.get('stocks_tested', 0)):,}</b></div>
  <div><span>股票20%回撤率</span><b>{_history_value(headline.get('stock_hit_rate'), pct=True, digits=2)}</b></div>
  <div><span>配對基準</span><b>{_history_value(headline.get('matched_control_stock_hit_rate'), pct=True, digits=2)}</b></div>
  <div><span>差異</span><b>{_history_value(headline.get('stock_hit_rate_lift'), pct=True, digits=2)}</b></div>
  <div><span>命中中位間隔</span><b>{_history_value(headline.get('median_lead_days'), digits=0)} 日</b></div>
 </div>
 <label class="search">篩選事件：<input id="history-search" oninput="filterHistory()" placeholder="日期、事件ID、流出或流入領域"></label>
 <div class="history-wrap"><table class="history-table"><thead><tr>
  <th>明細</th><th>ID</th><th>確認日</th><th>流出</th><th>流入</th><th>連續日</th><th>分數差</th>
  <th>測試檔數</th><th>命中檔數</th><th>命中率</th><th>最早回撤日</th><th>最早間隔</th>
  <th>中位間隔</th><th>P80</th><th>最晚命中</th><th>最早回撤前一交易日</th>
 </tr></thead><tbody>{_history_rows(history)}</tbody></table></div>
</div>
<div class="card note">
 <h2>原始資料與可重現性</h2>
 <p>「回撤前一交易日」只表示歷史資料中第一次收盤跌至參考峰值 80% 以下之前的最後一個觀察交易日，
 不代表當時能預知、也不是建議賣出日。事件命中率是「該事件至少一檔命中」，不能與逐股命中率混用。</p>
 <ul>{downloads}
  <li><a href="{HISTORY_PATH.name}">{HISTORY_PATH.name}</a> — 事件與 5,596 筆逐股結果的完整 JSON</li>
 </ul>
</div>"""


def main(argv=None):
    parser = argparse.ArgumentParser(description="建立 SURGE PRO 資金輪動警報頁")
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end")
    parser.add_argument("--confirm-days", type=int, default=3)
    args = parser.parse_args(argv)

    strategy = get_strategy("mom_surge_pro_rotation_alert")
    cfg = RunConfig(
        start_date=args.start, end_date=args.end, days=1000,
        universe_size=60, initial_capital=1_000_000, use_inst_flow=True,
    )
    data = build_market_data(cfg, strategy)
    model = compute_sector_rotation(
        data.close, data.volume, inst_flow=data.inst_flow_df,
        universe_mask=data.universe_mask, confirm_days=args.confirm_days,
    )
    as_of = data.close.index[-1]
    alerts = active_rotation_alerts(model, as_of=as_of)
    events = extract_rotation_events(model)
    recent_start = data.close.index[max(0, len(data.close.index) - 20)]
    recent = events[events["signal_date"] >= recent_start] if len(events) else events
    labels = {key: value.get("label", key) for key, value in SECTOR_MAP.items()}
    history = {}
    if HISTORY_PATH.exists():
        history = json.loads(HISTORY_PATH.read_text(encoding="utf-8"))

    payload = {
        "as_of_close": str(as_of.date()),
        "confirmation_days": args.confirm_days,
        "warning_only": True,
        "automatic_trade_action": False,
        "active_alert_count": int(len(alerts)),
        "active_alerts": _records(alerts),
        "recent_new_alerts_20_sessions": _records(recent),
        "historical_backtest": {
            "file": HISTORY_PATH.name,
            "period": history.get("period"),
            "event_count": history.get("event_count", 0),
            "stock_test_count": history.get("stock_test_count", 0),
            "complete_event_history_published": bool(history),
        },
    }
    JSON_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )

    status = "目前有警報" if len(alerts) else "目前無警報"
    status_class = "warn" if len(alerts) else "ok"
    page = f"""<!doctype html>
<html lang="zh-Hant"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SURGE PRO 資金輪動警報</title><style>
body{{font-family:system-ui,"Noto Sans TC",sans-serif;background:#0f172a;color:#e2e8f0;margin:0;padding:24px}}
.wrap{{max-width:1500px;margin:auto}}.card{{background:#1e293b;border:1px solid #334155;border-radius:14px;padding:18px;margin:16px 0;overflow:auto}}
.badge{{display:inline-block;padding:7px 12px;border-radius:999px;font-weight:800}}.ok{{background:#14532d;color:#86efac}}.warn{{background:#78350f;color:#fde68a}}
table{{width:100%;border-collapse:collapse;min-width:1050px}}th,td{{padding:9px;border-bottom:1px solid #334155;text-align:right}}th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){{text-align:left}}
.note{{color:#cbd5e1;line-height:1.7}}a{{color:#93c5fd}}code{{color:#fcd34d;font-size:.75rem;word-break:break-all}}
.kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin:16px 0}}.kpis div{{background:#111c30;border-radius:10px;padding:12px}}.kpis span{{display:block;color:#94a3b8;font-size:.78rem}}.kpis b{{font-size:1.15rem}}
.search{{display:block;margin:14px 0;color:#cbd5e1}}.search input{{width:min(420px,90%);margin-left:8px;padding:9px 12px;border:1px solid #475569;border-radius:8px;background:#0f172a;color:#e2e8f0}}
.history-wrap{{overflow:auto;max-height:72vh;border:1px solid #334155;border-radius:10px}}.history-table{{min-width:1900px}}.history-table thead{{position:sticky;top:0;background:#111827;z-index:2}}
.detail-btn{{border:1px solid #60a5fa;background:#172554;color:#bfdbfe;border-radius:7px;padding:5px 9px;cursor:pointer}}.history-detail td{{text-align:left;background:#111827;padding:16px}}
.feature-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:8px;margin-bottom:14px}}.feature-grid div{{background:#1e293b;border-radius:8px;padding:9px}}.feature-grid span{{display:block;color:#94a3b8;font-size:.72rem}}.feature-grid b{{display:block;margin-top:3px}}
.stock-table{{min-width:900px}}.hit{{color:#fca5a5;font-weight:700}}.miss{{color:#86efac}}ul{{padding-left:20px}}
</style></head><body><div class="wrap">
<h1>SURGE PRO · 資金輪動警報</h1>
<p><span class="badge {status_class}">{status}</span>　資料截至 {as_of.date()} 收盤｜三日同一來源→目的領域才成立</p>
<div class="card note"><b>用途：</b>提高監控頻率與人工複核，不預測崩盤、不自動減碼、不改變 SURGE PRO 母策略持倉。警報在收盤確認後，下一交易時段可見。<br>
十年校準：三日確認 192 次；個股 120 日內達 20% 回撤比例 43.73%，配對基準 44.35%，尚無預測增益。中位觀察間隔 20 個交易日只描述歷史結果，不是賣出期限。</div>
<div class="card"><h2>目前有效警報</h2><table><thead><tr>
<th>資金流出領域</th><th>資金流入領域</th><th>連續日</th><th>來源分數</th><th>目的分數</th><th>差距</th><th>來源5日</th><th>目的5日</th><th>來源成交額加速</th><th>來源20日線上廣度</th><th>來源法人流</th><th>20日平均相關</th>
</tr></thead><tbody>{_render_rows(alerts, labels)}</tbody></table></div>
{_history_section(history)}
<div class="card note"><b>下一階段預測研究候選：</b>外資／投信／自營商拆分、融資融券與借券、ETF申贖、期貨基差、選擇權波動率曲面、匯率利率、全球龍頭與產業 lead-lag。必須用走勢外測試與機率校準後，才可升級為預測模型。</div>
<p><a href="index.html">返回策略選單</a> · <a href="capital_rotation_alert_latest.json">檢視原始警報 JSON</a></p>
</div><script>
let historyPromise;
function esc(value) {{
  return String(value ?? '-').replace(/[&<>"']/g, c => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));
}}
function historyData() {{
  if (!historyPromise) {{
    historyPromise = fetch({json.dumps(HISTORY_PATH.name)}).then(response => {{
      if (!response.ok) throw new Error(`HTTP ${{response.status}}`);
      return response.json();
    }});
  }}
  return historyPromise;
}}
async function toggleEvent(eventId) {{
  const row = document.getElementById(`event-detail-${{eventId}}`);
  const button = document.querySelector(`tr[data-event="${{eventId}}"] .detail-btn`);
  row.hidden = !row.hidden;
  button.textContent = row.hidden ? '展開' : '收合';
  if (row.hidden || row.dataset.loaded) return;
  const target = document.getElementById(`stock-detail-${{eventId}}`);
  try {{
    const data = await historyData();
    const event = data.events.find(item => Number(item.event_id) === Number(eventId));
    const stocks = event ? event.stocks : [];
    const body = stocks.map(item => `<tr>
      <td>${{esc(item.ticker)}}</td><td>${{Number(item.reference_peak).toLocaleString(undefined, {{maximumFractionDigits:2}})}}</td>
      <td class="${{item.hit_20pct ? 'hit' : 'miss'}}">${{item.hit_20pct ? '是' : '否'}}</td>
      <td>${{esc(item.drawdown_date)}}</td><td>${{esc(item.lead_trading_days)}}</td>
      <td>${{esc(item.last_session_before_drawdown)}}</td></tr>`).join('');
    target.innerHTML = `<div style="overflow:auto"><table class="stock-table"><thead><tr>
      <th>股票</th><th>訊號前20日參考峰值</th><th>120日內達20%回撤</th><th>首次回撤日</th>
      <th>相隔交易日</th><th>首次回撤前一交易日</th></tr></thead><tbody>${{body}}</tbody></table></div>`;
    row.dataset.loaded = 'true';
  }} catch (error) {{
    target.textContent = `明細載入失敗：${{error.message}}`;
  }}
}}
function filterHistory() {{
  const query = document.getElementById('history-search').value.trim().toLowerCase();
  document.querySelectorAll('.history-main').forEach(row => {{
    const visible = !query || row.dataset.search.includes(query);
    row.hidden = !visible;
    const detail = document.getElementById(`event-detail-${{row.dataset.event}}`);
    if (!visible) detail.hidden = true;
  }});
}}
</script></body></html>"""
    HTML_PATH.write_text(page, encoding="utf-8")
    print(f"✅ {HTML_PATH}｜{as_of.date()}｜active alerts={len(alerts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
