"""
pool_audit 純核心邏輯測試（合成資料，不連網）。
python test_pool_audit.py
"""
import numpy as np
import pandas as pd

import pool_audit as A

PASS, FAIL = [], []
def check(name, cond):
    (PASS if cond else FAIL).append(name)
    print(f"  {'✅' if cond else '❌'} {name}")

# 1) 分組解析：應得 115 檔、7 組（2888 已下市併入 2887，已自池中剔除）
sec = A.parse_pool_with_sectors("ai_report.py")
flat = [c for l in sec.values() for c in l]
check("解析 115 檔", len(flat) == 115)
check("含 7 個產業組", len([k for k in sec if sec[k]]) == 7)
check("已下市 2888 不在池內", "2888" not in flat)
check("存續公司 2887 仍在池內", "2887" in flat)

# 2) 中位數排除單日爆量（處置股情境）
idx = pd.bdate_range(end="2026-07-15", periods=20)
hist = pd.DataFrame({"SPIKE": [1e6]*19 + [1e9], "STEADY": [5e8]*20}, index=idx)
med = A.median_daily_value(hist, 20)
check("中位數排除爆量(SPIKE~1e6)", med["SPIKE"] < 2e6 and med["STEADY"] > 4e8)

# 3) audit_pool：腐化(地板)、已下市(消失)、覆蓋率
pool = {"半導體": ["AAA", "BBB"], "生技": ["DEAD", "THIN"]}
mk = {"AAA": 1e9, "BBB": 5e8, "THIN": 1e6}
for i in range(400):
    mk[f"X{i}"] = 1e9 - i * 1e6
med2 = pd.Series(mk)
res = A.audit_pool(pool, med2, top_rank=300, coverage_top=150,
                   floor_dollars=5e7, present_codes=set(mk), delisted_codes=set())
d = res["detail"]
check("DEAD 消失→下市+腐化", bool(d.loc["DEAD", "delisted_or_suspended"]) and bool(d.loc["DEAD", "corroded"]))
check("THIN 低於地板→腐化", bool(d.loc["THIN", "corroded"]))
check("AAA 大量→不腐化", not bool(d.loc["AAA", "corroded"]))

# 4) 歷史終止上市清單『不』影響判定（代碼重用防呆）：AAA 在活躍卻在歷史清單 → 不算下市
res2 = A.audit_pool(pool, med2, top_rank=300, coverage_top=150, floor_dollars=5e7,
                    present_codes=set(mk), delisted_codes={"AAA"})
check("活躍股在歷史下市清單仍不誤判", not bool(res2["detail"].loc["AAA", "delisted_or_suspended"]))

# 5) ETF/權證過濾
check("0050 不算普通股", not A._is_common_stock("0050"))
check("2330 是普通股", A._is_common_stock("2330"))
check("6 位權證不算普通股", not A._is_common_stock("030001"))

print(f"\n通過 {len(PASS)} / 失敗 {len(FAIL)}")
if FAIL:
    print("失敗：", FAIL); raise SystemExit(1)
print("全部通過 ✅")
