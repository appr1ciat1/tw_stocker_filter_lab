#!/usr/bin/env python3
"""Build two independent TWD 300,000 paper-trading pages."""

import argparse
import html
import json

import numpy as np
import pandas as pd

from strategies.base import ExecConfig
from strategies.registry import get_strategy
from strategy.sector_flow import SECTOR_MAP, classify_sector
from twstk.backtest.engine import RunConfig, build_market_data
from twstk.backtest.metrics import compute_risk_metrics


CAPITAL = 300_000
VERSIONS = [
    {
        "name": "v8.5 300K", "strategy": "momentum_v85_300k",
        "file": "paper_trading_v85_300k.html", "color": "#60a5fa",
        "policy": "現金至少2%（另有整股自然餘額）｜單檔上限15%｜相關聚落最多2檔｜不使用第一件事濾網",
    },
    {
        "name": "SURGE PRO 300K", "strategy": "mom_surge_pro_300k",
        "file": "paper_trading_surge_pro_300k.html", "color": "#fb7185",
        "policy": "現金至少2%（另有整股自然餘額）｜單檔上限20%｜保留強勢聚落 alpha｜不使用第一件事濾網",
    },
]


def _pct(value):
    return f"{float(value) * 100:.1f}%" if value is not None and pd.notna(value) else "-"


def _positions(strategy, data, equity):
    positions = getattr(strategy, "last_positions", {})
    last_date = equity.index[-1]
    last_equity = float(equity["Equity"].iloc[-1])
    rows = []
    for ticker, position in positions.items():
        price = data.close.at[last_date, ticker] if ticker in data.close else np.nan
        if pd.isna(price):
            price = position["entry_price"]
        paper_value = float(position["shares"] * price)
        target_weight = paper_value / last_equity
        target_shares = int(np.floor(CAPITAL * target_weight / float(price)))
        target_value = float(target_shares * price)
        rows.append({
            "ticker": str(ticker), "sector": classify_sector(str(ticker)),
            "shares": target_shares,
            "entry": float(position["entry_price"]), "price": float(price),
            "value": target_value, "target_weight": target_weight,
            "weight": target_value / CAPITAL,
            "unrealized": float(price / position["entry_price"] - 1),
        })
    rows = sorted(rows, key=lambda row: row["target_weight"], reverse=True)
    target_cash = CAPITAL - sum(row["value"] for row in rows)
    return rows, target_cash


def _render(version, metrics, equity, positions, cash):
    name, color = version["name"], version["color"]
    last_equity = float(equity["Equity"].iloc[-1])
    total_return = last_equity / CAPITAL - 1
    as_of = str(equity.index[-1].date())
    dates = [str(d.date()) for d in equity.index]
    values = [round(float(v), 2) for v in equity["Equity"]]
    step = max(1, len(dates) // 900)
    dates, values = dates[::step], values[::step]
    labels = {key: value.get("label", key) for key, value in SECTOR_MAP.items()}
    pos_rows = "".join(
        "<tr>"
        f"<td><a href='https://tw.stock.yahoo.com/quote/{html.escape(row['ticker'])}.TW' "
        f"target='_blank'>{html.escape(row['ticker'])}</a></td>"
        f"<td>{html.escape(labels.get(row['sector'], row['sector']))}</td>"
        f"<td>{row['shares']:,}</td><td>{row['price']:,.2f}</td>"
        f"<td>{row['value']:,.0f}</td><td>{row['target_weight'] * 100:.1f}%</td>"
        f"<td>{row['weight'] * 100:.1f}%</td>"
        f"<td class='{'up' if row['unrealized'] >= 0 else 'down'}'>{row['unrealized'] * 100:+.1f}%</td>"
        "</tr>" for row in positions
    ) or "<tr><td colspan='8'>目前無持倉，現金等待訊號</td></tr>"
    ann = metrics.get("ann_return", metrics.get("annual_return", np.nan))
    mdd = metrics.get("max_drawdown_pct", np.nan)
    sharpe = metrics.get("sharpe", np.nan)
    html_text = f"""<!doctype html>
<html lang="zh-Hant"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{name} Paper Trading</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
body{{font-family:system-ui,"Noto Sans TC",sans-serif;background:#0f172a;color:#e2e8f0;margin:0;padding:24px}}
.wrap{{max-width:1000px;margin:auto}} .card{{background:#1e293b;border:1px solid #334155;border-radius:14px;padding:18px;margin:16px 0}}
.kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:10px}} .k{{background:#111c30;padding:12px;border-radius:10px}}
.lab{{font-size:.75rem;color:#94a3b8}} .val{{font-size:1.2rem;font-weight:700;margin-top:3px}} table{{width:100%;border-collapse:collapse}}
th,td{{padding:9px;border-bottom:1px solid #334155;text-align:right}} th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){{text-align:left}}
a{{color:#93c5fd}} .up{{color:#4ade80}} .down{{color:#f87171}} .note{{color:#cbd5e1;line-height:1.7}} .warn{{color:#fbbf24}}
</style></head><body><div class="wrap">
<h1 style="color:{color}">{name} · 獨立 Paper</h1>
<p class="note">初始投入本金上限 NT$300,000；{version['policy']}。下表是依最新訊號換算的今日 30 萬可執行股數；歷史曲線則允許獲利複利再投入。</p>
<div class="card kpis">
 <div class="k"><div class="lab">30萬配置現金</div><div class="val">NT${cash:,.0f}</div></div>
 <div class="k"><div class="lab">歷史模擬權益</div><div class="val">NT${last_equity:,.0f}</div></div>
 <div class="k"><div class="lab">歷史累積報酬</div><div class="val">{total_return * 100:+.1f}%</div></div>
 <div class="k"><div class="lab">年化報酬</div><div class="val">{_pct(ann)}</div></div>
 <div class="k"><div class="lab">Sharpe</div><div class="val">{sharpe:.2f}</div></div>
 <div class="k"><div class="lab">最大回撤</div><div class="val">{_pct(mdd)}</div></div>
</div>
<div class="card"><canvas id="equity" height="105"></canvas></div>
<div class="card"><h2>30 萬今日可執行配置（訊號日 {as_of}）</h2><table><thead><tr><th>代號</th><th>領域</th><th>股數</th><th>現價</th><th>投入金額</th><th>目標權重</th><th>整股後權重</th><th>訊號持倉損益</th></tr></thead><tbody>{pos_rows}</tbody></table></div>
<div class="card note"><b>配置原則</b><br>
選股仍由原策略排名決定，配置只加入整股成交、現金緩衝、單檔上限；v8.5 另限制相關聚落。測試過排名加權、跳空縮碼和較高現金，但它們把年化壓到 35% 以下，故未採用。30 萬版本不以最低波動為唯一目標，因那會系統性刪除動量贏家。<br>
<span class="warn">回測與 paper 不保證未來報酬，亦不構成個人化投資建議。</span></div>
<p><a href="index.html">返回策略選單</a> · <a href="paper_trading_v85_300k.html">v8.5 300K</a> · <a href="paper_trading_surge_pro_300k.html">SURGE PRO 300K</a></p>
</div><script>
new Chart(document.getElementById('equity'),{{type:'line',data:{{labels:{json.dumps(dates)},datasets:[{{label:{json.dumps(name)},data:{json.dumps(values)},borderColor:{json.dumps(color)},pointRadius:0,borderWidth:1.6}}]}},options:{{responsive:true,plugins:{{legend:{{labels:{{color:'#cbd5e1'}}}}}},scales:{{x:{{ticks:{{color:'#64748b',maxTicksLimit:10}}}},y:{{ticks:{{color:'#94a3b8'}}}}}}}}}});
</script></body></html>"""
    with open(version["file"], "w", encoding="utf-8") as f:
        f.write(html_text)


def main(argv=None):
    parser = argparse.ArgumentParser(description="建立 v8.5 / SURGE PRO 30萬獨立 paper")
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--end")
    args = parser.parse_args(argv)
    cfg = RunConfig(
        start_date=args.start, end_date=args.end, days=3000, universe_size=60,
        initial_capital=CAPITAL, top_k=7, threshold=2.0,
    )
    data = build_market_data(cfg, get_strategy("momentum_v85_300k"))
    exec_cfg = ExecConfig(initial_capital=CAPITAL, top_k=7, threshold=2.0)
    for version in VERSIONS:
        strategy = get_strategy(version["strategy"])
        trades, equity = strategy.run_engine(data, exec_cfg)
        metrics = compute_risk_metrics(equity, trades, CAPITAL)
        positions, target_cash = _positions(strategy, data, equity)
        _render(version, metrics, equity, positions, target_cash)
        print(f"✅ {version['file']} ({len(positions)} 檔持倉)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
