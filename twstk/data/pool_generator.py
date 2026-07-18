"""
twstk.data.pool_generator — Point-in-time 動態候選池生成器（純邏輯）

從「全市場每日成交額」建構逐日候選池，供動態池研究使用。核心鐵律：
  1. Point-in-time：第 t 日的池只能用 t-1（含）之前的資料決定，禁用未來成交額
     （否則成交額與過去漲幅高度相關 → 回測前視偏誤 → 覆蓋率/報酬虛高）。
  2. 進出緩衝 (hysteresis)：rank ≤ enter_rank 才進榜；已在榜者要跌出 exit_rank 才剔除，
     壓低邊界股頻繁進出造成的無謂換手/滑價。
  3. 最短上市天數 (min_history) + 絕對流動性地板 (min_adv)：排除新股/殭屍股。

純函式：輸入 turnover_hist (session x code)，不含任何抓取邏輯（資料源見 pool_audit.py）。
與既有 strategy.ai_strategy.build_liquid_universe（在 116 檔靜態池內選 Top-N）不同：
這裡是「全市場」版，是 [[pool-corrosion-audit]] 覆蓋率問題的解方候選。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def trailing_median_turnover(turnover_hist: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    每個 session t 的『截至 t-1』trailing 中位數日成交額（point-in-time，不含當日）。
    用中位數排除單日爆量（處置股/題材股），與審計同口徑。
    回傳 (session x code)；前 window 列因資料不足為 NaN。
    """
    h = turnover_hist.sort_index()
    # shift(1)：第 t 列只看到 t-1 及更早 → 嚴格排除當日，杜絕前視。
    return h.shift(1).rolling(window, min_periods=window).median()


def trailing_history_count(turnover_hist: pd.DataFrame) -> pd.DataFrame:
    """截至 t-1，每檔已累積的『有量交易日數』（近似上市天數，point-in-time）。"""
    h = turnover_hist.sort_index()
    return h.notna().shift(1).fillna(False).cumsum()


@dataclass
class PoolBuildResult:
    mask: pd.DataFrame          # (session x code) bool：當日是否在候選池
    median_turnover: pd.DataFrame
    rank: pd.DataFrame


def build_pointintime_pools(
    turnover_hist: pd.DataFrame,
    *,
    window: int = 20,
    enter_rank: int = 130,
    exit_rank: int = 170,
    min_adv: float = 50_000_000,
    min_history: int = 60,
) -> PoolBuildResult:
    """
    逐日建構 point-in-time 候選池（含 hysteresis）。

    決策（第 t 日，全用 ≤ t-1 資料）：
      · med = trailing window 中位數成交額
      · rank = 全市場排名（med 大→rank 小）
      · 合格門檻 = med ≥ min_adv 且 history ≥ min_history
      · 進榜：合格 且 rank ≤ enter_rank
      · 續留：前一日在榜 且 合格 且 rank ≤ exit_rank
    """
    if enter_rank > exit_rank:
        raise ValueError("enter_rank 應 ≤ exit_rank（進榜比續留嚴）")

    med = trailing_median_turnover(turnover_hist, window)
    hist = trailing_history_count(turnover_hist)
    rank = med.rank(axis=1, ascending=False, method="min")

    codes = list(turnover_hist.columns)
    sessions = list(med.index)
    mask = pd.DataFrame(False, index=sessions, columns=codes)

    eligible = (med >= min_adv) & (hist >= min_history) & med.notna()
    prev = pd.Series(False, index=codes)
    for t in sessions:
        rk = rank.loc[t]
        elig = eligible.loc[t]
        enter = elig & (rk <= enter_rank)
        stay = prev & elig & (rk <= exit_rank)
        cur = (enter | stay).fillna(False)
        mask.loc[t] = cur
        prev = cur
    return PoolBuildResult(mask=mask, median_turnover=med, rank=rank)


def latest_pool(turnover_hist: pd.DataFrame, **kwargs) -> list[str]:
    """回傳最後一個 session 的候選池代號清單（依中位數成交額由大到小）。"""
    res = build_pointintime_pools(turnover_hist, **kwargs)
    last = res.mask.index[-1]
    sel = res.mask.loc[last]
    codes = list(sel[sel].index)
    med_last = res.median_turnover.loc[last]
    return list(med_last[codes].sort_values(ascending=False).index)


def churn_stats(mask: pd.DataFrame) -> dict:
    """池換手統計：日均進/出檔數、日均池大小（評估 hysteresis 是否壓住 churn）。"""
    m = mask.astype(int)
    diff = m.diff()
    adds = (diff == 1).sum(axis=1)
    drops = (diff == -1).sum(axis=1)
    size = m.sum(axis=1)
    return {
        "avg_pool_size": float(size.mean()),
        "avg_daily_adds": float(adds.iloc[1:].mean()) if len(adds) > 1 else 0.0,
        "avg_daily_drops": float(drops.iloc[1:].mean()) if len(drops) > 1 else 0.0,
        "one_way_turnover_per_day": float((adds.iloc[1:].mean() / size.mean())) if size.mean() else 0.0,
    }
