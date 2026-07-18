"""
validation.capacity — 容量 / 流動性驗收閘門 (Capacity Gate)

統計驗證（PBO、Deflated Sharpe、nested walk-forward）擋得住「過擬合」，
但擋不住「流動性幻覺」：回測用收盤價成交、不限量，實盤卻買不到那麼多。
一旦動態池納入更小的票，這個問題會惡化，且統計指標完全看不出來。

這個模組是第二道門：給定「意圖權重 / 部位」，用每檔的 ADV（近 N 日平均成交額）
估算「單日下單佔 ADV 比例 (participation)」與「完全建/出倉所需天數」，
超過門檻即 FAIL。任何股池變更（尤其動態池）必須同時通過統計門與此容量門。

純函式，不依賴交易引擎；可對「靜態池權重」或「動態池候選權重」直接驗收。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


def average_daily_dollar_volume(close_df: pd.DataFrame, vol_df: pd.DataFrame,
                                lookback: int = 20, as_of=None) -> pd.Series:
    """
    每檔的近 lookback 日平均成交額（元）= mean(close × volume)。
    as_of 為 None 時取最後一根 bar 為基準往前算。
    """
    turnover = close_df * vol_df
    if as_of is not None:
        turnover = turnover.loc[:pd.Timestamp(as_of)]
    adv = turnover.tail(lookback).mean(axis=0)
    return adv.rename("adv_dollars")


@dataclass
class GateResult:
    ok: bool
    table: pd.DataFrame
    violations: list[str] = field(default_factory=list)
    params: dict = field(default_factory=dict)

    def summary(self) -> str:
        head = "✅ 容量門通過" if self.ok else "❌ 容量門未通過（流動性幻覺風險）"
        lines = [head, f"   參數: {self.params}"]
        if not self.table.empty:
            worst = self.table.sort_values("participation", ascending=False).head(5)
            lines.append("   最吃流動性的部位：")
            for t, row in worst.iterrows():
                lines.append(
                    f"     {t}: 部位 {row['position_dollars']:,.0f} 元 / ADV {row['adv_dollars']:,.0f} 元"
                    f" → 佔比 {row['participation']:.1%}，出清約 {row['days_to_exit']:.1f} 日"
                )
        for v in self.violations:
            lines.append(f"   ⚠️ {v}")
        return "\n".join(lines)


def _normalize_weights(weights) -> dict:
    """接受 dict{ticker:weight}、list[ticker]（等權）、或 orders JSON 的 list[dict]。"""
    if isinstance(weights, dict):
        return {str(k): float(v) for k, v in weights.items()}
    if isinstance(weights, (list, tuple)):
        if len(weights) == 0:
            return {}
        first = weights[0]
        if isinstance(first, dict):
            # orders JSON：list[{'ticker':..., 'weight'?:...}]，無 weight 則等權
            has_w = all(("weight" in o and o["weight"] is not None) for o in weights)
            if has_w:
                return {str(o["ticker"]): float(o["weight"]) for o in weights}
            n = len(weights)
            return {str(o["ticker"]): 1.0 / n for o in weights}
        # list[ticker] → 等權
        n = len(weights)
        return {str(t): 1.0 / n for t in weights}
    raise TypeError("weights 需為 dict / list[ticker] / orders list[dict]")


def participation_table(weights, capital: float,
                        close_df: pd.DataFrame, vol_df: pd.DataFrame,
                        *, lookback: int = 20, as_of=None,
                        adv_participation_per_day: float = 0.10) -> pd.DataFrame:
    """
    回傳每檔的容量表：weight / position_dollars / adv_dollars / participation / days_to_exit。

    participation           = 部位金額 / ADV（單日就想全部成交需吃掉的 ADV 比例）
    days_to_exit            = participation / adv_participation_per_day
                              （每日最多只吃 ADV 的 adv_participation_per_day，需幾天建/出完）
    """
    w = _normalize_weights(weights)
    adv = average_daily_dollar_volume(close_df, vol_df, lookback=lookback, as_of=as_of)

    rows = {}
    for t, wt in w.items():
        pos = float(capital) * float(wt)
        a = float(adv.get(t, np.nan))
        part = pos / a if (a and np.isfinite(a) and a > 0) else np.inf
        rows[t] = {
            "weight": wt,
            "position_dollars": pos,
            "adv_dollars": a,
            "participation": part,
            "days_to_exit": part / adv_participation_per_day if np.isfinite(part) else np.inf,
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def capacity_gate(weights, capital: float,
                  close_df: pd.DataFrame, vol_df: pd.DataFrame,
                  *, lookback: int = 20, as_of=None,
                  max_participation: float = 0.10,
                  max_days_to_exit: float = 1.0,
                  adv_participation_per_day: float = 0.10,
                  min_adv_dollars: float | None = None) -> GateResult:
    """
    容量驗收。任一條件不滿足即 FAIL：
      - 單檔 participation（部位/ADV）> max_participation
      - 單檔 days_to_exit > max_days_to_exit
      - （選用）單檔 ADV < min_adv_dollars（絕對流動性地板）

    預設門檻偏保守：單筆部位不超過該股 ADV 的 10%，且一日內可出清。
    對「動態池候選」用意圖權重跑一次；FAIL 代表回測報酬有一部分建立在買不到的量上。
    """
    table = participation_table(
        weights, capital, close_df, vol_df,
        lookback=lookback, as_of=as_of,
        adv_participation_per_day=adv_participation_per_day,
    )
    violations: list[str] = []

    if table.empty:
        return GateResult(True, table, ["（無部位可檢查）"],
                          params=dict(max_participation=max_participation))

    over_part = table[table["participation"] > max_participation]
    for t, row in over_part.iterrows():
        violations.append(
            f"{t}: participation {row['participation']:.1%} > {max_participation:.0%}"
            + ("（ADV 為 0/缺值）" if not np.isfinite(row["adv_dollars"]) or row["adv_dollars"] <= 0 else "")
        )

    over_days = table[table["days_to_exit"] > max_days_to_exit]
    for t, row in over_days.iterrows():
        if t in over_part.index:
            continue
        violations.append(f"{t}: 出清需 {row['days_to_exit']:.1f} 日 > {max_days_to_exit} 日")

    if min_adv_dollars is not None:
        thin = table[~(table["adv_dollars"] >= min_adv_dollars)]
        for t, row in thin.iterrows():
            violations.append(f"{t}: ADV {row['adv_dollars']:,.0f} 元 < 地板 {min_adv_dollars:,.0f} 元")

    return GateResult(
        ok=(len(violations) == 0),
        table=table,
        violations=violations,
        params=dict(capital=capital, lookback=lookback,
                    max_participation=max_participation,
                    max_days_to_exit=max_days_to_exit,
                    adv_participation_per_day=adv_participation_per_day,
                    min_adv_dollars=min_adv_dollars),
    )


def _cli():
    """
    動態池容量驗收 CLI。給一份 snapshot + 意圖權重，輸出是否通過容量門。
    exit code 0=通過、1=未通過（可接進研究流程當第二道門）。

    範例：
      python -m validation.capacity --snapshot artifacts/snapshot_20260715 \\
          --weights pool_weights.json --capital 1000000 --max-participation 0.10
    """
    import argparse
    import json
    import sys

    from twstk.data.contract import load_snapshot

    ap = argparse.ArgumentParser(description="動態池容量/流動性驗收門")
    ap.add_argument("--snapshot", required=True, help="snapshot 目錄或 panel.pkl")
    ap.add_argument("--weights", required=True,
                    help="JSON：{ticker:weight}、[ticker,...] 或 orders list[dict]")
    ap.add_argument("--capital", type=float, required=True)
    ap.add_argument("--lookback", type=int, default=20)
    ap.add_argument("--max-participation", type=float, default=0.10)
    ap.add_argument("--max-days-to-exit", type=float, default=1.0)
    ap.add_argument("--adv-participation-per-day", type=float, default=0.10)
    ap.add_argument("--min-adv-dollars", type=float, default=None)
    args = ap.parse_args()

    with open(args.weights, "r", encoding="utf-8") as f:
        weights = json.load(f)
    # orders JSON 常包成 {"orders": [...]}
    if isinstance(weights, dict) and "orders" in weights:
        weights = weights["orders"]

    close, _o, _h, _l, vol = load_snapshot(args.snapshot)
    res = capacity_gate(
        weights, args.capital, close, vol,
        lookback=args.lookback,
        max_participation=args.max_participation,
        max_days_to_exit=args.max_days_to_exit,
        adv_participation_per_day=args.adv_participation_per_day,
        min_adv_dollars=args.min_adv_dollars,
    )
    print(res.summary())
    sys.exit(0 if res.ok else 1)


if __name__ == "__main__":
    _cli()
