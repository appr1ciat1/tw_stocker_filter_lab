"""
security_master 測試。使用已提交的 security_master.json（不連網）；
若快取不存在則跳過需要資料的檢查。
python test_security_master.py
"""
import os
import time

from twstk.data import security_master as S
import pool_audit as A

PASS, FAIL = [], []
def check(name, cond):
    (PASS if cond else FAIL).append(name)
    print(f"  {'✅' if cond else '❌'} {name}")

has_cache = os.path.exists(S.DEFAULT_PATH)
check("security_master.json 存在", has_cache)

if has_cache:
    m = S.load_master()          # 未過期 → 讀快取，不連網
    check("master 載入且筆數合理(>3000)", len(m) > 3000)
    check("2330 → 台積電", S.get_name("2330") == "台積電")
    check("label 格式 '代號 名稱'", S.label("2330") == "2330 台積電")

    d = S.describe("5274")
    check("describe 帶 industry/market", bool(d.get("industry")) and d.get("market") in ("twse", "tpex", "emerging"))

    # 生產池全覆蓋（人工下單防看錯代號的關鍵）
    pool = [c for l in A.parse_pool_with_sectors("ai_report.py").values() for c in l]
    miss = [t for t in pool if S.get_name(t) == t]
    check(f"生產池 {len(pool)} 檔全數有名稱", not miss)

    # 快取重用：第二次載入不得重抓（檔案 mtime 不變、且要快）
    mt = os.path.getmtime(S.DEFAULT_PATH)
    S._CACHE = None
    t0 = time.time(); S.load_master(); dt = time.time() - t0
    check("二次載入走快取(不重抓)", dt < 2.0 and os.path.getmtime(S.DEFAULT_PATH) == mt)

    # 未知代號優雅降級（顯示層 fail-soft，不可拋錯擋住報表）
    check("未知代號回傳代號本身", S.get_name("9999") == "9999")
    check("enrich 批次可用", set(S.enrich(["2330", "9999"]).keys()) == {"2330", "9999"})

print(f"\n通過 {len(PASS)} / 失敗 {len(FAIL)}")
if FAIL:
    print("失敗：", FAIL); raise SystemExit(1)
print("全部通過 ✅")
