"""
pool_acceptance.py — 動態候選池「三門驗收基準」

任何候選池要取代現行 116 檔靜態池，必須『同時』通過三道門（全部走研究流程，
碰不到生產四策略）。統計驗證擋過擬合，容量/覆蓋擋流動性幻覺與漏贏家：

  ① 統計門（擋過擬合）— 重用 validation/：
       PBO/CSCV ≤ 0.5（多組候選的 trial 報酬）＋ Deflated Sharpe 機率 ≥ 0.95。
       nested walk-forward（walk_forward_nested.py）較重，作為外部 boolean 輸入。
  ② 容量門（擋流動性幻覺）— 重用 validation/capacity.py：
       候選池意圖部位 participation ≤ ADV 10%、出清 ≤ 1 日。
  ③ 覆蓋/腐化門（擋漏贏家＋擋納垃圾）— 重用 pool_audit.py：
       候選池對全市場 Top-150 覆蓋率 ≥ 80% 且自身腐化率 ≤ 10%。

三門皆過才 PASS。門檻皆可調（風險旋鈕）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from pool_audit import audit_pool
from validation.capacity import capacity_gate
from validation.pbo_cscv import compute_pbo
from validation.deflated_sharpe import compute_deflated_sharpe


@dataclass
class GateOutcome:
    name: str
    passed: bool
    detail: dict = field(default_factory=dict)

    def line(self) -> str:
        return f"[{'PASS' if self.passed else 'FAIL'}] {self.name} — " + \
               "; ".join(f"{k}={v}" for k, v in self.detail.items())


# ─────────────────── ① 統計門 ───────────────────

def statistical_gate(returns_by_trial: pd.DataFrame | dict | None = None,
                     single_returns=None,
                     *, max_pbo: float = 0.5, n_trials: int = 1,
                     min_dsr_prob: float = 0.95,
                     nested_wf_pass: bool | None = None) -> GateOutcome:
    detail = {}
    ok = True
    if returns_by_trial is not None:
        pbo = compute_pbo(returns_by_trial)
        detail["pbo"] = None if np.isnan(pbo.pbo) else round(pbo.pbo, 3)
        detail["n_trials"] = pbo.n_trials
        if not np.isnan(pbo.pbo):
            ok = ok and (pbo.pbo <= max_pbo)
    if single_returns is not None:
        dsr = compute_deflated_sharpe(single_returns, n_trials=n_trials)
        detail["dsr"] = round(dsr.deflated_sharpe, 3)
        detail["dsr_prob"] = round(dsr.probability, 3)
        ok = ok and (dsr.probability >= min_dsr_prob)
    if nested_wf_pass is not None:
        detail["nested_wf"] = nested_wf_pass
        ok = ok and bool(nested_wf_pass)
    if not detail:
        detail["note"] = "未提供任何統計輸入（跳過，視為未驗證）"
        ok = False
    return GateOutcome("① 統計門(過擬合)", ok, detail)


# ─────────────────── ② 容量門 ───────────────────

def capacity_gate_for_pool(weights, capital: float,
                           close_df: pd.DataFrame, vol_df: pd.DataFrame,
                           **kwargs) -> GateOutcome:
    res = capacity_gate(weights, capital, close_df, vol_df, **kwargs)
    worst = None
    if not res.table.empty:
        w = res.table.sort_values("participation", ascending=False).iloc[0]
        worst = f"{res.table['participation'].idxmax()}={w['participation']:.1%}"
    return GateOutcome("② 容量門(流動性幻覺)", res.ok,
                       {"violations": len(res.violations), "worst_participation": worst})


# ─────────────────── ③ 覆蓋/腐化門 ───────────────────

def coverage_corrosion_gate(candidate_codes, market_median: pd.Series,
                            *, coverage_top: int = 150, top_rank: int = 300,
                            floor_dollars: float = 50_000_000,
                            min_coverage: float = 0.80,
                            max_corruption: float = 0.10,
                            present_codes: set | None = None) -> GateOutcome:
    codes = list(dict.fromkeys(str(c) for c in candidate_codes))
    res = audit_pool({"candidate": codes}, market_median,
                     top_rank=top_rank, coverage_top=coverage_top,
                     floor_dollars=floor_dollars,
                     present_codes=present_codes, delisted_codes=set())
    cov = res["coverage_rate"]
    cor = res["corruption_rate"]
    ok = (cov >= min_coverage) and (cor <= max_corruption)
    return GateOutcome("③ 覆蓋/腐化門(漏贏家+納垃圾)", ok,
                       {"coverage": f"{cov:.1%}(≥{min_coverage:.0%})",
                        "corruption": f"{cor:.1%}(≤{max_corruption:.0%})",
                        "pool_n": res["pool_n"]})


# ─────────────────── 匯總 ───────────────────

@dataclass
class AcceptanceReport:
    outcomes: list[GateOutcome]

    @property
    def passed(self) -> bool:
        return all(o.passed for o in self.outcomes)

    def summary(self) -> str:
        # 先研究後投資：三門通過僅代表『通過研究驗收』，下一步是影子/紙上追蹤，
        # 尚『不可』直接投入資金部署（見 DYNAMIC_POOL_RESEARCH.md 的階段閘門）。
        head = "✅ 候選池通過三門研究驗收 → 進入影子/紙上追蹤期（尚不可直接部署）" if self.passed \
            else "❌ 候選池未通過研究驗收（任一門失敗即擋，不得取代現行池）"
        return "\n".join([head] + ["  " + o.line() for o in self.outcomes])


def run_acceptance(candidate_codes, market_median, *,
                   weights=None, capital=None, close_df=None, vol_df=None,
                   returns_by_trial=None, single_returns=None, n_trials=1,
                   nested_wf_pass=None, present_codes=None,
                   coverage_top=150, top_rank=300, floor_dollars=50_000_000,
                   min_coverage=0.80, max_corruption=0.10,
                   max_pbo=0.5, min_dsr_prob=0.95,
                   cap_kwargs=None) -> AcceptanceReport:
    """跑三門驗收。缺輸入的門會標記為未驗證（FAIL），確保『沒驗證 ≠ 通過』。"""
    outcomes = [
        statistical_gate(returns_by_trial, single_returns, max_pbo=max_pbo,
                         n_trials=n_trials, min_dsr_prob=min_dsr_prob,
                         nested_wf_pass=nested_wf_pass),
        coverage_corrosion_gate(candidate_codes, market_median,
                                coverage_top=coverage_top, top_rank=top_rank,
                                floor_dollars=floor_dollars,
                                min_coverage=min_coverage, max_corruption=max_corruption,
                                present_codes=present_codes),
    ]
    if weights is not None and capital is not None and close_df is not None:
        outcomes.insert(1, capacity_gate_for_pool(weights, capital, close_df, vol_df,
                                                   **(cap_kwargs or {})))
    else:
        outcomes.insert(1, GateOutcome("② 容量門(流動性幻覺)", False,
                                       {"note": "未提供部位/資金/量能 → 未驗證"}))
    return AcceptanceReport(outcomes)
