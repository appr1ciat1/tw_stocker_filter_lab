#!/usr/bin/env python3
"""Build daily pages for the two independent entry-confirmation variants."""

import argparse
import datetime as dt
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

from strategies.base import ExecConfig
from strategies.registry import get_strategy
from strategy.sector_flow import SECTOR_MAP, classify_sector
from twstk.backtest.engine import RunConfig, build_market_data
from twstk.backtest.metrics import compute_risk_metrics


CAPITAL = 1_000_000
VERSIONS = [
    {
        "name": "v8.5 CONFIRMED FILTER",
        "strategy": "momentum_v85_confirmed",
        "html": Path("paper_trading_v85_confirmed.html"),
        "json": Path("paper_trading_v85_confirmed.json"),
        "color": "#60a5fa",
        "role": "v8.5 的獨立進場濾網版",
    },
    {
        "name": "SURGE PRO CONFIRMED FILTER",
        "strategy": "mom_surge_pro_confirmed",
        "html": Path("paper_trading_surge_pro_confirmed.html"),
        "json": Path("paper_trading_surge_pro_confirmed.json"),
        "color": "#fb7185",
        "role": "SURGE PRO 的獨立進場濾網研究版",
    },
]

OVERNIGHT_LABELS = [
    ("S&P 500", "spx"),
    ("費城半導體", "sox"),
    ("台積電 ADR", "tsm_adr"),
]


def _finite(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _pct(value, signed=False):
    number = _finite(value)
    if number is None:
        return "-"
    sign = "+" if signed else ""
    return f"{number:{sign}.2%}"


def _clean(value):
    """Recursively convert pandas/numpy values to strict JSON values."""
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return str(pd.Timestamp(value).date())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return _finite(value)
    if pd.isna(value):
        return None
    return value


def _overnight_snapshot(data, as_of):
    context = getattr(data, "global_context", None)
    overnight = getattr(context, "overnight", None)
    rows = []
    if overnight is None or as_of not in overnight.index:
        return rows
    values = overnight.loc[as_of]
    ordinal = _finite(values.get("completed_us_session_ordinal"))
    session_date = (
        dt.date.fromordinal(int(ordinal)).isoformat() if ordinal is not None else None
    )
    for label, prefix in OVERNIGHT_LABELS:
        rows.append({
            "market": label,
            "us_session_date": session_date,
            "open_return": _finite(values.get(f"{prefix}_open_return")),
            "mid_return": _finite(values.get(f"{prefix}_mid_return")),
            "close_return": _finite(values.get(f"{prefix}_close_return")),
            "intraday_return": _finite(values.get(f"{prefix}_intraday_return")),
        })
    return rows


def _validate_daily_inputs(data):
    """Fail closed instead of publishing stale or silently neutral data."""
    as_of = pd.Timestamp(data.close.index[-1]).normalize()
    context = getattr(data, "global_context", None)
    overnight = getattr(context, "overnight", None)
    if overnight is None or as_of not in overnight.index:
        raise RuntimeError("缺少隔夜美股／全球龍頭資料，停止發布而非採中性值")
    row = overnight.loc[as_of]
    required = [
        "completed_us_session_ordinal",
        "spx_open_return", "spx_mid_return", "spx_close_return",
        "sox_open_return", "sox_mid_return", "sox_close_return",
        "tsm_adr_open_return", "tsm_adr_mid_return", "tsm_adr_close_return",
    ]
    missing = [column for column in required if column not in row or pd.isna(row[column])]
    if missing:
        raise RuntimeError(f"隔夜核心欄位不完整，停止發布: {', '.join(missing)}")
    session_date = pd.Timestamp(
        dt.date.fromordinal(int(row["completed_us_session_ordinal"]))
    )
    calendar_lag = int((as_of - session_date).days)
    if calendar_lag < 1:
        raise RuntimeError("隔夜資料日期未早於台股日期，可能有前視偏誤")
    if calendar_lag > 4:
        raise RuntimeError(
            f"隔夜資料過舊：台股 {as_of.date()} / 美股 {session_date.date()}"
        )

    required_frames = {
        "三大法人": data.inst_flow_df,
        "融資餘額": data.margin_balance_df,
        "融券餘額": data.margin_short_df,
        "借券餘額": data.short_sale_df,
    }
    minimum_coverage = max(10, int(len(data.close.columns) * 0.5))
    for label, frame in required_frames.items():
        if frame is None or frame.empty:
            raise RuntimeError(f"{label}資料缺失，停止發布")
        recent_coverage = int(
            frame.reindex(index=data.close.index, columns=data.close.columns)
            .tail(5).notna().any(axis=0).sum()
        )
        if recent_coverage < minimum_coverage:
            raise RuntimeError(
                f"{label}近五個交易日僅覆蓋 {recent_coverage} 檔，低於 {minimum_coverage}"
            )
    return session_date


def _positions(strategy, data, equity):
    as_of = equity.index[-1]
    last_equity = float(equity["Equity"].iloc[-1])
    layers = getattr(strategy, "last_confirmation_layers", None)
    context = getattr(data, "global_context", None)
    leader_map = getattr(context, "leader_symbol_by_ticker", {}) if context else {}
    leader_return = getattr(context, "leader_return", None) if context else None
    labels = {key: value.get("label", key) for key, value in SECTOR_MAP.items()}
    rows = []
    for ticker, position in getattr(strategy, "last_positions", {}).items():
        ticker = str(ticker)
        price = data.close.at[as_of, ticker] if ticker in data.close.columns else np.nan
        if not np.isfinite(price):
            price = float(position["entry_price"])
        value = float(position["shares"]) * float(price)
        leader = leader_map.get(ticker, "-")
        leader_day_return = None
        if leader_return is not None and ticker in leader_return.columns and as_of in leader_return.index:
            leader_day_return = _finite(leader_return.at[as_of, ticker])
        gate = None
        scale = None
        support_days = None
        required_days = None
        chip_score = None
        reward_risk = None
        if layers is not None:
            if ticker in layers.entry_gate.columns:
                gate = bool(layers.entry_gate.at[as_of, ticker])
                scale = _finite(layers.entry_scale.at[as_of, ticker])
            diagnostics = layers.diagnostics
            for key, target in (
                ("support_days", "support_days"),
                ("required_days", "required_days"),
                ("chip_score", "chip_score"),
                ("reward_risk", "reward_risk"),
            ):
                frame = diagnostics.get(key)
                if frame is not None and ticker in frame.columns and as_of in frame.index:
                    value_at_date = _finite(frame.at[as_of, ticker])
                    if target == "support_days":
                        support_days = value_at_date
                    elif target == "required_days":
                        required_days = value_at_date
                    elif target == "chip_score":
                        chip_score = value_at_date
                    else:
                        reward_risk = value_at_date
        rows.append({
            "ticker": ticker,
            "sector": labels.get(classify_sector(ticker), classify_sector(ticker)),
            "shares": _finite(position["shares"]),
            "entry_price": _finite(position["entry_price"]),
            "price": _finite(price),
            "market_value": value,
            "weight": value / last_equity if last_equity else None,
            "unrealized_return": float(price / position["entry_price"] - 1),
            "global_leader": leader,
            "global_leader_return": leader_day_return,
            "entry_gate_as_of": gate,
            "entry_scale_as_of": scale,
            "support_days": support_days,
            "required_days": required_days,
            "chip_score": chip_score,
            "reward_risk": reward_risk,
        })
    return sorted(rows, key=lambda row: row["market_value"], reverse=True)


def _render(version, metrics, equity, positions, cash, overnight):
    as_of = equity.index[-1]
    last_equity = float(equity["Equity"].iloc[-1])
    dates = [str(value.date()) for value in equity.index]
    values = [round(float(value), 2) for value in equity["Equity"]]
    step = max(1, len(dates) // 900)
    dates, values = dates[::step], values[::step]

    overnight_rows = "".join(
        "<tr>"
        f"<td>{html.escape(row['market'])}</td>"
        f"<td>{html.escape(str(row['us_session_date'] or '-'))}</td>"
        f"<td>{_pct(row['open_return'], True)}</td>"
        f"<td>{_pct(row['mid_return'], True)}</td>"
        f"<td>{_pct(row['intraday_return'], True)}</td>"
        f"<td>{_pct(row['close_return'], True)}</td>"
        "</tr>"
        for row in overnight
    ) or '<tr><td colspan="6">隔夜資料不可用；嚴格模式不會發布此頁</td></tr>'

    position_rows = "".join(
        "<tr>"
        f"<td><a target='_blank' href='https://tw.stock.yahoo.com/quote/{html.escape(row['ticker'])}.TW'>{html.escape(row['ticker'])}</a></td>"
        f"<td>{html.escape(row['sector'])}</td>"
        f"<td>{html.escape(str(row['global_leader']))}</td>"
        f"<td>{_pct(row['global_leader_return'], True)}</td>"
        f"<td>{row['shares']:,.2f}</td>"
        f"<td>{row['price']:,.2f}</td>"
        f"<td>{row['market_value']:,.0f}</td>"
        f"<td>{_pct(row['weight'])}</td>"
        f"<td class='{'up' if row['unrealized_return'] >= 0 else 'down'}'>{_pct(row['unrealized_return'], True)}</td>"
        f"<td>{'通過' if row['entry_gate_as_of'] else '等待'}</td>"
        f"<td>{row['entry_scale_as_of']:.2f}×</td>"
        "</tr>"
        for row in positions
    ) or '<tr><td colspan="11">目前無持倉，現金等待完整確認訊號</td></tr>'

    page = f"""<!doctype html>
<html lang="zh-Hant"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(version['name'])}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
body{{font-family:system-ui,"Noto Sans TC",sans-serif;background:#0f172a;color:#e2e8f0;margin:0;padding:24px}}
.wrap{{max-width:1220px;margin:auto}}.card{{background:#1e293b;border:1px solid #334155;border-radius:14px;padding:18px;margin:16px 0;overflow:auto}}
.kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(145px,1fr));gap:10px}}.k{{background:#111c30;padding:12px;border-radius:10px}}
.lab{{font-size:.75rem;color:#94a3b8}}.val{{font-size:1.15rem;font-weight:750;margin-top:3px}}
table{{width:100%;border-collapse:collapse;min-width:760px}}th,td{{padding:9px;border-bottom:1px solid #334155;text-align:right;white-space:nowrap}}
th:first-child,td:first-child,th:nth-child(2),td:nth-child(2),th:nth-child(3),td:nth-child(3){{text-align:left}}
a{{color:#93c5fd}}.up{{color:#4ade80}}.down{{color:#f87171}}.note{{color:#cbd5e1;line-height:1.7}}.warn{{color:#fbbf24}}
</style></head><body><div class="wrap">
<h1 style="color:{version['color']}">{html.escape(version['name'])}</h1>
<p class="note">{html.escape(version['role'])}｜資料截至台股 {as_of.date()} 收盤。隔夜欄位嚴格使用該台股交易日前一個已完整收盤的美股交易日。</p>
<div class="card kpis">
 <div class="k"><div class="lab">模擬權益</div><div class="val">NT${last_equity:,.0f}</div></div>
 <div class="k"><div class="lab">現金</div><div class="val">NT${cash:,.0f}</div></div>
 <div class="k"><div class="lab">年化報酬</div><div class="val">{_pct(metrics['ann_return'])}</div></div>
 <div class="k"><div class="lab">Sharpe</div><div class="val">{metrics['sharpe']:.2f}</div></div>
 <div class="k"><div class="lab">最大回撤</div><div class="val">{_pct(metrics['max_drawdown_pct'])}</div></div>
 <div class="k"><div class="lab">已完成交易</div><div class="val">{int(metrics['total_trades']):,}</div></div>
</div>
<div class="card"><canvas id="equity" height="95"></canvas></div>
<div class="card"><h2>美股隔夜三段資訊</h2><table><thead><tr><th>市場</th><th>美股日期</th><th>開盤/前收</th><th>盤中代理/開盤</th><th>收盤/開盤</th><th>收盤/前收</th></tr></thead><tbody>{overnight_rows}</tbody></table>
<p class="note">「盤中」由日 K 高低價中點估計，只描述盤中路徑，不假裝知道逐筆先後。</p></div>
<div class="card"><h2>目前持倉與訊號日進場濾網狀態</h2><table><thead><tr><th>代號</th><th>領域</th><th>全球龍頭</th><th>龍頭隔夜</th><th>股數</th><th>現價</th><th>市值</th><th>權重</th><th>未實現</th><th>訊號日進場</th><th>縮放</th></tr></thead><tbody>{position_rows}</tbody></table></div>
<div class="card note"><b>濾網層：</b>SPX／SOX／TSM ADR 開盤、盤中代理與收盤；台股個股對應全球龍頭；三大法人、融資餘額、融券餘額與借券餘額數量；相對低點、反彈與預估風報比；多頭可 1 日確認、一般趨勢 2 日、弱勢／恢復期 3 日確認。<br>
<span class="warn">這是規則型 paper／研究輸出，不是獲利保證，也不構成個人化投資建議。SURGE PRO CONFIRMED 尚未勝過母策略，只應作對照研究。</span></div>
<p><a href="index.html">返回策略選單</a> · <a href="{version['json'].name}">嚴格 JSON</a></p>
</div><script>
new Chart(document.getElementById('equity'),{{type:'line',data:{{labels:{json.dumps(dates)},datasets:[{{label:{json.dumps(version['name'])},data:{json.dumps(values)},borderColor:{json.dumps(version['color'])},pointRadius:0,borderWidth:1.6}}]}},options:{{responsive:true,plugins:{{legend:{{labels:{{color:'#cbd5e1'}}}}}},scales:{{x:{{ticks:{{color:'#64748b',maxTicksLimit:10}}}},y:{{ticks:{{color:'#94a3b8'}}}}}}}}}});
</script></body></html>"""
    version["html"].write_text(page, encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description="建立兩個獨立 CONFIRMED FILTER 每日頁")
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--end")
    args = parser.parse_args(argv)

    cfg = RunConfig(
        start_date=args.start,
        end_date=args.end,
        days=1000,
        universe_size=60,
        initial_capital=CAPITAL,
        top_k=7,
        threshold=2.0,
        use_inst_flow=True,
        refresh_latest=True,
    )
    data = build_market_data(cfg, get_strategy("momentum_v85_confirmed"))
    session_date = _validate_daily_inputs(data)
    print(f"✅ 資料品質檢查通過：台股 {data.close.index[-1].date()} / 美股 {session_date.date()}")
    exec_cfg = ExecConfig(initial_capital=CAPITAL, top_k=7, threshold=2.0)
    for version in VERSIONS:
        strategy = get_strategy(version["strategy"])
        trades, equity = strategy.run_engine(data, exec_cfg)
        metrics = compute_risk_metrics(equity, trades, CAPITAL)
        as_of = equity.index[-1]
        positions = _positions(strategy, data, equity)
        overnight = _overnight_snapshot(data, as_of)
        cash = float(getattr(strategy, "last_cash", CAPITAL))
        payload = _clean({
            "version": version["strategy"],
            "as_of_close": as_of,
            "initial_capital": CAPITAL,
            "cash": cash,
            "equity": float(equity["Equity"].iloc[-1]),
            "metrics": {
                "ann_return": metrics["ann_return"],
                "sharpe": metrics["sharpe"],
                "max_drawdown_pct": metrics["max_drawdown_pct"],
                "total_trades": metrics["total_trades"],
            },
            "overnight_completed_us_session": overnight,
            "positions": positions,
        })
        version["json"].write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        _render(version, metrics, equity, positions, cash, overnight)
        print(f"✅ {version['html']}｜{as_of.date()}｜持倉 {len(positions)} 檔")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
