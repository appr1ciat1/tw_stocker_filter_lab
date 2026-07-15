"""Independent TWD 300,000 execution variants for v8.5 and SURGE PRO.

These versions intentionally do not include the first-task confirmation layer.
They isolate the effect of small-account allocation: cash buffer, whole-share
odd-lot execution and risk concentration limits.  Rank/gap sizing were tested
and rejected because they cut annual return below the user's 35% floor.
"""

from dataclasses import replace

from strategies.base import EngineStrategy, ExecConfig, MarketData
from strategies.momentum_v85 import MomentumV85
from strategies.optimized_v85 import SURGE_PRO_PARAMS, _build_engine
from strategies.registry import register
from strategy.event_backtest import EventDrivenBacktester


CAPITAL_CAP = 300_000

V85_300K_PARAMS = dict(
    position_size=0.10,
    cash_reserve_pct=0.02,
    max_position_pct=0.15,
    max_portfolio_heat=1.0,
    rank_weighted=False,
    gap_aware_sizing=False,
    corr_select_max=0.72,
    corr_select_window=60,
    corr_select_cap=2,
    sector_max_pct=0.75,
    integer_shares=True,
    min_trade_amount=0,
)

SURGE_PRO_300K_PARAMS = {
    **SURGE_PRO_PARAMS,
    "position_size": 0.10,
    "cash_reserve_pct": 0.02,
    "max_position_pct": 0.20,
    "max_portfolio_heat": 1.0,
    "rank_weighted": False,
    "gap_aware_sizing": False,
    # Prior validation found correlation selection destroys SURGE's cluster
    # alpha, so the 300k variant controls concentration via cash/position caps.
    "sector_max_pct": 0.75,
    "integer_shares": True,
    "min_trade_amount": 0,
}


def _capped_exec(exec_cfg: ExecConfig) -> ExecConfig:
    return replace(exec_cfg, initial_capital=min(exec_cfg.initial_capital, CAPITAL_CAP))


@register("momentum_v85_300k")
class MomentumV85300K(EngineStrategy):
    description = "v8.5 300K｜30萬上限、最低2%現金、單檔15%、整股零股、相關聚落最多2檔"
    capital_cap = CAPITAL_CAP

    def run_engine(self, data: MarketData, exec_cfg: ExecConfig):
        cfg = _capped_exec(exec_cfg)
        bundle = MomentumV85().prepare(data)
        p = {**V85_300K_PARAMS, **self.params}
        bt = EventDrivenBacktester(
            tp_sl_mode="atr", tp_atr_mult=4.0, sl_atr_mult=3.0,
            max_hold_days=20, initial_capital=cfg.initial_capital,
            position_size=p["position_size"], regime_filter=True,
            gap_filter_atr=1.5, hybrid_tiered=False,
            buy_cost=cfg.buy_cost, sell_cost=cfg.sell_cost, slippage=cfg.slippage,
            cash_reserve_pct=p["cash_reserve_pct"],
            max_position_pct=p["max_position_pct"],
            max_portfolio_heat=p["max_portfolio_heat"],
            rank_weighted=p["rank_weighted"],
            gap_aware_sizing=p["gap_aware_sizing"],
            corr_select_max=p["corr_select_max"],
            corr_select_window=p["corr_select_window"],
            corr_select_cap=p["corr_select_cap"],
            sector_max_pct=p["sector_max_pct"],
            integer_shares=p["integer_shares"],
            min_trade_amount=p["min_trade_amount"],
        )
        result = bt.run(
            bundle.total_score, data.close, data.open, data.high, data.low,
            bundle.ma_long, top_k=cfg.top_k, threshold=cfg.threshold,
            market_close=data.market_close, vol_df=data.volume,
            universe_mask=data.universe_mask,
        )
        self.last_positions = getattr(bt, "last_positions", {})
        self.last_cash = getattr(bt, "last_cash", cfg.initial_capital)
        return result


@register("mom_surge_pro_300k")
class MomSurgePro300K(EngineStrategy):
    description = "SURGE PRO 300K｜30萬上限、最低2%現金、單檔20%、整股零股、保留強勢聚落alpha"
    capital_cap = CAPITAL_CAP

    def run_engine(self, data: MarketData, exec_cfg: ExecConfig):
        cfg = _capped_exec(exec_cfg)
        bundle = MomentumV85().prepare(data)
        bt = _build_engine({**SURGE_PRO_300K_PARAMS, **self.params}, cfg)
        result = bt.run(
            bundle.total_score, data.close, data.open, data.high, data.low,
            bundle.ma_long, top_k=cfg.top_k, threshold=cfg.threshold,
            market_close=data.market_close, vol_df=data.volume,
            universe_mask=data.universe_mask,
        )
        self.last_positions = getattr(bt, "last_positions", {})
        self.last_cash = getattr(bt, "last_cash", cfg.initial_capital)
        return result


__all__ = [
    "CAPITAL_CAP", "V85_300K_PARAMS", "SURGE_PRO_300K_PARAMS",
    "MomentumV85300K", "MomSurgePro300K",
]
