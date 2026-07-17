"""Overnight US context and Taiwan-to-global-leader mapping.

The important timing rule is explicit: the Taiwan open on date ``T`` may only
use a completed US session whose exchange date is strictly earlier than ``T``.
For example, the 2026-07-16 Taiwan open uses the 2026-07-15 US session.  This
module keeps open, intraday and close information separate so a gap-up that
fades into the close is not treated the same as a strong close.
"""

from dataclasses import dataclass
import os
import pickle
from typing import Dict, Iterable

import numpy as np
import pandas as pd
import yfinance as yf


GLOBAL_CONTEXT_CACHE = os.path.join("artifacts", "global_context_ohlc.pkl")


US_CORE_SYMBOLS = {
    "spx": "^GSPC",
    "nasdaq": "^IXIC",
    "dow": "^DJI",
    "sox": "^SOX",
    "tsm_adr": "TSM",
    "vix": "^VIX",
}

# Individual mappings are deliberately limited to economically meaningful
# peers.  Unlisted names fall back to a liquid global sector leader/ETF.
GLOBAL_LEADER_BY_TW: Dict[str, str] = {
    # Semiconductors
    "2330": "TSM", "2454": "QCOM", "2303": "UMC", "3711": "ASX",
    "2379": "MCHP", "6770": "NVDA", "3034": "AVGO", "2449": "AMAT",
    "5274": "NVDA", "3529": "NVDA", "2408": "MU", "3443": "NVDA",
    "3035": "MRVL", "6415": "AMAT", "6525": "NVDA", "3661": "AMD",
    "3037": "AMAT", "2344": "MU", "6547": "ASML",
    # Electronics / computing / communications
    "2317": "AAPL", "2382": "DELL", "2308": "ETN", "2301": "CSCO",
    "2357": "DELL", "2376": "NVDA", "2395": "HON", "3231": "SMCI",
    "2474": "AAPL", "2353": "HPQ", "3481": "GLW", "3017": "NVDA",
    "2345": "ANET", "2383": "JBL", "2356": "DELL", "3044": "JBL",
    "2327": "APH", "3036": "MCHP", "2324": "HPQ", "2377": "DELL",
    "2385": "LOGI", "2360": "ETN", "2404": "GLW", "2412": "VZ",
    "2459": "GLW", "2458": "APH", "3045": "VZ", "3023": "APH",
    "3706": "CSCO", "3533": "APH", "2368": "JBL", "4904": "CIEN",
    "4938": "JBL", "6669": "ENPH",
    # Selected non-tech leaders
    "6505": "XOM", "1301": "DOW", "1303": "DOW", "1326": "DOW",
    "2002": "NUE", "1101": "VMC", "1102": "VMC", "2207": "TM",
    "2201": "TM", "2204": "TM", "2912": "WMT", "1216": "KHC",
    "2105": "B", "2603": "ZIM", "2609": "ZIM", "2615": "ZIM",
    "2606": "IYT", "2618": "DAL", "2610": "DAL", "2637": "BDRY",
    "4142": "NVO", "1760": "NVO", "6446": "REGN", "1707": "JNJ",
    "4743": "NVO",
}

SECTOR_FALLBACK_LEADER = {
    "semiconductor": "SOXX",
    "electronics": "XLK",
    "computing": "QQQ",
    "finance": "JPM",
    "traditional": "XLI",
    "shipping": "IYT",
    "biotech": "IBB",
}


@dataclass
class GlobalContext:
    """Data aligned to Taiwan trading dates and safe at the Taiwan open."""

    overnight: pd.DataFrame
    leader_score: pd.DataFrame
    leader_return: pd.DataFrame
    leader_symbol_by_ticker: Dict[str, str]


def _download_ohlc(symbol: str, start_date, end_date) -> pd.DataFrame:
    """Download one symbol and normalize yfinance's changing column schema."""
    start = pd.Timestamp(start_date) - pd.Timedelta(days=100)
    # yfinance's end is exclusive.  Add two calendar days to cover the last
    # completed US session when Taiwan has already moved to the next date.
    end = pd.Timestamp(end_date) + pd.Timedelta(days=2)
    raw = yf.download(
        symbol, start=start, end=end, progress=False, auto_adjust=False,
        actions=False, threads=False,
    )
    if raw is None or raw.empty:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close"])
    if isinstance(raw.columns, pd.MultiIndex):
        # New yfinance commonly returns (Price, Ticker).
        if symbol in raw.columns.get_level_values(-1):
            raw = raw.xs(symbol, axis=1, level=-1, drop_level=True)
        else:
            raw.columns = raw.columns.get_level_values(0)
    cols = [c for c in ("Open", "High", "Low", "Close") if c in raw.columns]
    out = raw[cols].copy()
    for c in ("Open", "High", "Low", "Close"):
        if c not in out:
            out[c] = np.nan
    out = out[["Open", "High", "Low", "Close"]].astype(float)
    out.index = pd.DatetimeIndex(out.index).tz_localize(None).normalize()
    return out[~out.index.duplicated(keep="last")].sort_index()


def _session_features(ohlc: pd.DataFrame, prefix: str) -> pd.DataFrame:
    prev_close = ohlc["Close"].shift(1)
    span = (ohlc["High"] - ohlc["Low"]).replace(0, np.nan)
    feat = pd.DataFrame(index=ohlc.index)
    feat[f"{prefix}_open"] = ohlc["Open"]
    feat[f"{prefix}_high"] = ohlc["High"]
    feat[f"{prefix}_low"] = ohlc["Low"]
    feat[f"{prefix}_close"] = ohlc["Close"]
    feat[f"{prefix}_open_return"] = ohlc["Open"] / prev_close - 1
    # The mid-session proxy retains the high/low path without pretending that
    # daily bars reveal the exact intraday sequence.
    feat[f"{prefix}_mid_return"] = ((ohlc["High"] + ohlc["Low"]) / 2) / ohlc["Open"] - 1
    feat[f"{prefix}_intraday_return"] = ohlc["Close"] / ohlc["Open"] - 1
    feat[f"{prefix}_close_return"] = ohlc["Close"] / prev_close - 1
    feat[f"{prefix}_range"] = span / prev_close
    feat[f"{prefix}_close_location"] = ((ohlc["Close"] - ohlc["Low"]) / span).clip(0, 1)
    vol20 = feat[f"{prefix}_close_return"].rolling(20, min_periods=10).std()
    directional = (
        0.25 * feat[f"{prefix}_open_return"]
        + 0.25 * feat[f"{prefix}_mid_return"]
        + 0.50 * feat[f"{prefix}_intraday_return"]
    )
    feat[f"{prefix}_session_score"] = np.tanh(
        directional / vol20.replace(0, np.nan)
    ).fillna(0.0)
    return feat


def align_completed_us_session(frame: pd.DataFrame, tw_dates: Iterable) -> pd.DataFrame:
    """Align the latest *strictly previous* US exchange date to each TW date."""
    tw_index = pd.DatetimeIndex(tw_dates).tz_localize(None).normalize()
    if frame is None or frame.empty:
        return pd.DataFrame(index=tw_index, columns=getattr(frame, "columns", None))
    source = frame.copy()
    source.index = pd.DatetimeIndex(source.index).tz_localize(None).normalize()
    source = source[~source.index.duplicated(keep="last")].sort_index()
    positions = source.index.searchsorted(tw_index, side="left") - 1
    valid = positions >= 0
    out = pd.DataFrame(np.nan, index=tw_index, columns=source.columns, dtype=object)
    if valid.any():
        out.iloc[np.flatnonzero(valid)] = source.iloc[positions[valid]].to_numpy()
    # All market features in this module are numeric.
    return out.apply(pd.to_numeric, errors="coerce")


def _leader_for_ticker(ticker: str) -> str:
    code = str(ticker).split(".")[0]
    if code in GLOBAL_LEADER_BY_TW:
        return GLOBAL_LEADER_BY_TW[code]
    try:
        from strategy.sector_flow import classify_sector
        sector = classify_sector(code)
    except Exception:
        sector = "traditional"
    return SECTOR_FALLBACK_LEADER.get(sector, "ACWI")


def fetch_global_context(
    tw_tickers: Iterable[str], tw_dates: Iterable, start_date=None, end_date=None,
    verbose: bool = True,
) -> GlobalContext:
    """Fetch overnight indices and each Taiwan stock's global leader context."""
    tw_dates = pd.DatetimeIndex(tw_dates).tz_localize(None).normalize()
    if len(tw_dates) == 0:
        empty = pd.DataFrame(index=tw_dates)
        return GlobalContext(empty, empty, empty, {})
    start_date = start_date or tw_dates.min()
    end_date = end_date or tw_dates.max()
    leader_map = {str(t): _leader_for_ticker(str(t)) for t in tw_tickers}
    symbols = list(dict.fromkeys(list(US_CORE_SYMBOLS.values()) + list(leader_map.values())))
    if verbose:
        print(f"🌍 隔夜確認：下載 {len(symbols)} 個美股指標／全球龍頭...")

    raw_cache = {}
    if os.path.exists(GLOBAL_CONTEXT_CACHE):
        try:
            with open(GLOBAL_CONTEXT_CACHE, "rb") as f:
                loaded = pickle.load(f)
            if isinstance(loaded, dict):
                raw_cache = loaded
        except Exception:
            raw_cache = {}

    feature_by_symbol: Dict[str, pd.DataFrame] = {}
    failed = []
    cache_changed = False
    for symbol in symbols:
        cached = raw_cache.get(symbol)
        need_download = (
            cached is None or cached.empty
            or cached.index.min() > pd.Timestamp(start_date) - pd.Timedelta(days=80)
            # Taiwan date T may use the last completed US session strictly
            # before T. Refresh when that prior session can be missing.
            or cached.index.max() < pd.Timestamp(end_date) - pd.Timedelta(days=1)
        )
        try:
            if need_download:
                fetch_start = (
                    pd.Timestamp(start_date) if cached is None or cached.empty
                    else cached.index.max() - pd.Timedelta(days=10)
                )
                fresh = _download_ohlc(symbol, fetch_start, end_date)
                bars = fresh if cached is None or cached.empty else fresh.combine_first(cached).sort_index()
                if not bars.empty:
                    raw_cache[symbol] = bars
                    cache_changed = True
            else:
                bars = cached
        except Exception:
            bars = cached if cached is not None else pd.DataFrame()
        if bars.empty:
            failed.append(symbol)
            continue
        # Old cache snapshots can carry object dtype after pickle/schema
        # changes.  Force numeric OHLC before rolling volatility and np.tanh.
        bars = bars[["Open", "High", "Low", "Close"]].apply(
            pd.to_numeric, errors="coerce"
        ).astype(float).dropna(how="all")
        if bars.empty:
            failed.append(symbol)
            continue
        feature_by_symbol[symbol] = _session_features(bars, symbol.replace("^", "").lower())

    if cache_changed:
        try:
            os.makedirs(os.path.dirname(GLOBAL_CONTEXT_CACHE), exist_ok=True)
            with open(GLOBAL_CONTEXT_CACHE, "wb") as f:
                pickle.dump(raw_cache, f)
        except Exception:
            pass

    overnight_parts = []
    for label, symbol in US_CORE_SYMBOLS.items():
        feat = feature_by_symbol.get(symbol)
        if feat is None:
            continue
        old_prefix = symbol.replace("^", "").lower()
        renamed = feat.rename(columns=lambda c: c.replace(old_prefix, label, 1))
        overnight_parts.append(renamed)
    raw_overnight = pd.concat(overnight_parts, axis=1).sort_index() if overnight_parts else pd.DataFrame()
    if not raw_overnight.empty:
        raw_overnight["completed_us_session_ordinal"] = [
            value.toordinal() for value in raw_overnight.index
        ]
    overnight = align_completed_us_session(raw_overnight, tw_dates)

    # Composite emphasizes semiconductors because Taiwan's index earnings and
    # the current strategy universe are technology-heavy.
    components = []
    weights = []
    # NASDAQ/Dow are retained in the report, but the tradable risk composite
    # deliberately uses SPX + SOX + TSM: adding highly collinear US indices
    # triple-counts the same overnight move and degraded the Taiwan OOS test.
    for label, weight in (("spx", 0.35), ("sox", 0.40), ("tsm_adr", 0.25)):
        col = f"{label}_session_score"
        if col in overnight:
            components.append(overnight[col].astype(float) * weight)
            weights.append(weight)
    if components:
        overnight["global_risk_score"] = sum(components) / sum(weights)
    else:
        overnight["global_risk_score"] = 0.0
    broad_parts = [
        ("spx_session_score", 0.55),
        ("nasdaq_session_score", 0.30),
        ("dow_session_score", 0.15),
    ]
    if all(col in overnight for col, _ in broad_parts):
        overnight["broad_us_score"] = sum(overnight[col] * weight for col, weight in broad_parts)
    if "vix_close_return" in overnight:
        overnight["global_risk_score"] = (
            overnight["global_risk_score"]
            - 0.20 * np.tanh(overnight["vix_close_return"].astype(float) / 0.05)
        ).clip(-1, 1)

    leader_score = pd.DataFrame(index=tw_dates, columns=[str(t) for t in tw_tickers], dtype=float)
    leader_return = leader_score.copy()
    for ticker, symbol in leader_map.items():
        feat = feature_by_symbol.get(symbol)
        if feat is None:
            continue
        prefix = symbol.replace("^", "").lower()
        aligned = align_completed_us_session(feat, tw_dates)
        leader_score[ticker] = aligned.get(f"{prefix}_session_score", 0.0)
        leader_return[ticker] = aligned.get(f"{prefix}_close_return", np.nan)
    if verbose:
        ok = len(symbols) - len(failed)
        suffix = f"；失敗 {','.join(failed)}" if failed else ""
        print(f"   ✅ 隔夜／龍頭資料 {ok}/{len(symbols)} 個成功{suffix}")
    return GlobalContext(
        overnight=overnight,
        leader_score=leader_score.fillna(0.0),
        leader_return=leader_return,
        leader_symbol_by_ticker=leader_map,
    )


__all__ = [
    "GlobalContext", "GLOBAL_LEADER_BY_TW", "SECTOR_FALLBACK_LEADER",
    "align_completed_us_session", "fetch_global_context",
]
