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

    payload = {
        "as_of_close": str(as_of.date()),
        "confirmation_days": args.confirm_days,
        "warning_only": True,
        "automatic_trade_action": False,
        "active_alert_count": int(len(alerts)),
        "active_alerts": _records(alerts),
        "recent_new_alerts_20_sessions": _records(recent),
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
.wrap{{max-width:1250px;margin:auto}}.card{{background:#1e293b;border:1px solid #334155;border-radius:14px;padding:18px;margin:16px 0;overflow:auto}}
.badge{{display:inline-block;padding:7px 12px;border-radius:999px;font-weight:800}}.ok{{background:#14532d;color:#86efac}}.warn{{background:#78350f;color:#fde68a}}
table{{width:100%;border-collapse:collapse;min-width:1050px}}th,td{{padding:9px;border-bottom:1px solid #334155;text-align:right}}th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){{text-align:left}}
.note{{color:#cbd5e1;line-height:1.7}}a{{color:#93c5fd}}code{{color:#fcd34d}}
</style></head><body><div class="wrap">
<h1>SURGE PRO · 資金輪動警報</h1>
<p><span class="badge {status_class}">{status}</span>　資料截至 {as_of.date()} 收盤｜三日同一來源→目的領域才成立</p>
<div class="card note"><b>用途：</b>提高監控頻率與人工複核，不預測崩盤、不自動減碼、不改變 SURGE PRO 母策略持倉。警報在收盤確認後，下一交易時段可見。<br>
十年校準：三日確認 192 次；個股 120 日內達 20% 回撤比例 43.73%，配對基準 44.35%，尚無預測增益。中位觀察間隔 20 個交易日只描述歷史結果，不是賣出期限。</div>
<div class="card"><h2>目前有效警報</h2><table><thead><tr>
<th>資金流出領域</th><th>資金流入領域</th><th>連續日</th><th>來源分數</th><th>目的分數</th><th>差距</th><th>來源5日</th><th>目的5日</th><th>來源成交額加速</th><th>來源20日線上廣度</th><th>來源法人流</th><th>20日平均相關</th>
</tr></thead><tbody>{_render_rows(alerts, labels)}</tbody></table></div>
<div class="card note"><b>下一階段預測研究候選：</b>外資／投信／自營商拆分、融資融券與借券、ETF申贖、期貨基差、選擇權波動率曲面、匯率利率、全球龍頭與產業 lead-lag。必須用走勢外測試與機率校準後，才可升級為預測模型。</div>
<p><a href="index.html">返回策略選單</a> · <a href="capital_rotation_alert_latest.json">檢視原始警報 JSON</a></p>
</div></body></html>"""
    HTML_PATH.write_text(page, encoding="utf-8")
    print(f"✅ {HTML_PATH}｜{as_of.date()}｜active alerts={len(alerts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
