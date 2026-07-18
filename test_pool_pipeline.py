"""
pool_generator + pool_acceptance 純邏輯測試（合成資料，不連網）。
python test_pool_pipeline.py
"""
import numpy as np
import pandas as pd

from twstk.data.pool_generator import (
    build_pointintime_pools, trailing_median_turnover, latest_pool, churn_stats,
)
from pool_acceptance import (
    statistical_gate, coverage_corrosion_gate, run_acceptance,
)

PASS, FAIL = [], []
def check(name, cond):
    (PASS if cond else FAIL).append(name)
    print(f"  {'✅' if cond else '❌'} {name}")

rng = np.random.default_rng(0)
idx = pd.bdate_range(end="2026-07-15", periods=80)
codes = [f"C{i:03d}" for i in range(50)]
# 讓每檔有穩定但不同的成交額水位（C000 最大 … C049 最小）
base = np.linspace(1e9, 1e6, len(codes))
turn = pd.DataFrame(base * (1 + 0.05*rng.standard_normal((len(idx), len(codes)))),
                    index=idx, columns=codes).abs()

print("── pool_generator：point-in-time 無前視 ──")
res = build_pointintime_pools(turn, window=20, enter_rank=10, exit_rank=15,
                              min_adv=0, min_history=20)
# 只改『最後一天』的成交額 → 因 shift(1) 排除當日，整個 mask 不該有任何改變
turn2 = turn.copy(); turn2.iloc[-1] = turn2.iloc[-1] * 100
res2 = build_pointintime_pools(turn2, window=20, enter_rank=10, exit_rank=15,
                               min_adv=0, min_history=20)
check("只改最後一天 → mask 完全不變(無前視)", res.mask.equals(res2.mask))
# 改『早期一段持續期間』→ 之後的 mask 應受影響（對照組，證明確實有在用歷史）。
# 注意：中位數本就抵抗單日爆量，所以需持續 20 天才會移動 median（此即想要的穩健性）。
turn3 = turn.copy(); turn3.iloc[25:45, turn3.columns.get_loc("C049")] = 1e12
res3 = build_pointintime_pools(turn3, window=20, enter_rank=10, exit_rank=15,
                               min_adv=0, min_history=20)
check("改早期持續段 → 後續 mask 改變(有用歷史)", not res.mask.equals(res3.mask))

print("── min_history / min_adv 過濾 ──")
# min_history=30 → 前 30 天內不可能有人在池
early = res.mask.iloc[:20]
check("暖機期(不足window)池為空", bool((~res.mask.iloc[:20]).all().all()))
res_floor = build_pointintime_pools(turn, window=20, enter_rank=40, exit_rank=45,
                                    min_adv=5e8, min_history=20)
# 地板 5e8 → 只有前段大票（base>5e8 約前 25 檔）可能入選
last_pool_codes = latest_pool(turn, window=20, enter_rank=40, exit_rank=45,
                              min_adv=5e8, min_history=20)
check("絕對地板擋掉小額票", all(turn[c].tail(20).median() >= 5e8 for c in last_pool_codes))

print("── hysteresis 降低換手 ──")
noisy = pd.DataFrame(base * (1 + 0.35*rng.standard_normal((len(idx), len(codes)))),
                     index=idx, columns=codes).abs()
no_hyst = build_pointintime_pools(noisy, window=10, enter_rank=12, exit_rank=12,
                                  min_adv=0, min_history=10)
hyst = build_pointintime_pools(noisy, window=10, enter_rank=12, exit_rank=20,
                               min_adv=0, min_history=10)
c0 = churn_stats(no_hyst.mask)["one_way_turnover_per_day"]
c1 = churn_stats(hyst.mask)["one_way_turnover_per_day"]
print(f"    churn 無緩衝={c0:.3f} vs 有緩衝={c1:.3f}")
check("hysteresis 降低日均換手", c1 < c0)

print("── pool_acceptance：三門 ──")
# 覆蓋/腐化門：候選=全市場前 120（覆蓋足、腐化低）→ PASS
med = turn.tail(20).median(axis=0)
big_market = pd.Series({**{c: float(med[c]) for c in codes},
                        **{f"M{i}": 1e9 - i*1e6 for i in range(200)}})
good_candidate = list(big_market.sort_values(ascending=False).index[:160])
g = coverage_corrosion_gate(good_candidate, big_market, coverage_top=150,
                            min_coverage=0.80, max_corruption=0.10,
                            present_codes=set(big_market.index))
check("覆蓋門：好候選(覆蓋高)PASS", g.passed)
bad_candidate = list(big_market.sort_values().index[:30])  # 全是最小額
gb = coverage_corrosion_gate(bad_candidate, big_market, coverage_top=150,
                             min_coverage=0.80, max_corruption=0.10,
                             present_codes=set(big_market.index))
check("覆蓋門：爛候選 FAIL", not gb.passed)

# 統計門：強報酬單序列 → DSR 機率高 PASS；純噪音 → FAIL
strong = pd.Series(0.002 + 0.005*rng.standard_normal(500))
weak = pd.Series(0.0 + 0.02*rng.standard_normal(500))
sg = statistical_gate(single_returns=strong, n_trials=1, min_dsr_prob=0.90)
sgw = statistical_gate(single_returns=weak, n_trials=50, min_dsr_prob=0.95)
check("統計門：強序列 PASS", sg.passed)
check("統計門：噪音+多試驗 FAIL", not sgw.passed)
check("統計門：無輸入視為未驗證FAIL", not statistical_gate().passed)

# run_acceptance：缺容量輸入 → 該門未驗證 → 整體 FAIL（沒驗證≠通過）
rep = run_acceptance(good_candidate, big_market, single_returns=strong, n_trials=1,
                     present_codes=set(big_market.index), min_dsr_prob=0.90,
                     min_coverage=0.80, max_corruption=0.10)
check("整體：缺容量門 → 未通過(沒驗證≠通過)", not rep.passed)
# 補上容量輸入（好票、部位小）→ 三門齊過
cc = pd.DataFrame(100.0, index=idx, columns=good_candidate)
vv = pd.DataFrame(5e6, index=idx, columns=good_candidate)
rep2 = run_acceptance(good_candidate, big_market, single_returns=strong, n_trials=1,
                      weights={c: 1.0/len(good_candidate) for c in good_candidate},
                      capital=1_000_000, close_df=cc, vol_df=vv,
                      present_codes=set(big_market.index), min_dsr_prob=0.90,
                      min_coverage=0.80, max_corruption=0.10,
                      cap_kwargs=dict(max_participation=0.10))
print("   ", rep2.summary().replace("\n", "\n    "))
check("整體：三門齊備且達標 → PASS", rep2.passed)

print(f"\n通過 {len(PASS)} / 失敗 {len(FAIL)}")
if FAIL:
    print("失敗：", FAIL); raise SystemExit(1)
print("全部通過 ✅")
