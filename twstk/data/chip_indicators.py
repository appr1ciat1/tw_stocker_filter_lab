"""Margin financing, margin short and securities-borrowing balances.

All matrices contain reported *quantities*, not only ratios.  Strategy code is
responsible for converting them into lagged changes/ranks so information is not
used before it was available.
"""

from dataclasses import dataclass
from datetime import date
import json
import os
import pickle
import ssl
import time
import urllib.parse
import urllib.request
from typing import Iterable

import numpy as np
import pandas as pd


FINMIND_URL = "https://api.finmindtrade.com/api/v4/data"
MARGIN_DATASET = "TaiwanStockMarginPurchaseShortSale"
SHORT_DATASET = "TaiwanDailyShortSaleBalances"
MARGIN_CACHE = os.path.join("artifacts", "margin_cache.pkl")
SBL_DEEP_CACHE = os.path.join("artifacts", "sbl_7yr_cache.pkl")
SBL_CACHE = os.path.join("artifacts", "sbl_balances_cache.pkl")

_CTX = ssl.create_default_context()
_CTX.verify_flags &= ~ssl.VERIFY_X509_STRICT


@dataclass
class ChipIndicators:
    margin_balance: pd.DataFrame
    margin_short_balance: pd.DataFrame
    sbl_balance: pd.DataFrame


def _empty(index, columns):
    return pd.DataFrame(np.nan, index=index, columns=columns, dtype=float)


def _load_margin_cache():
    if not os.path.exists(MARGIN_CACHE):
        return None, None
    try:
        with open(MARGIN_CACHE, "rb") as f:
            cached = pickle.load(f)
        if isinstance(cached, tuple) and len(cached) == 2:
            return cached
        if isinstance(cached, dict):
            return cached.get("margin"), cached.get("short")
    except Exception:
        pass
    return None, None


def _load_sbl_cache():
    # Prefer the validated deep-history cache, then fill newer observations
    # from the incremental production cache.
    deep = recent = None
    for path, name in ((SBL_DEEP_CACHE, "deep"), (SBL_CACHE, "recent")):
        if not os.path.exists(path):
            continue
        try:
            with open(path, "rb") as f:
                value = pickle.load(f)
            if isinstance(value, pd.DataFrame):
                if name == "deep":
                    deep = value
                else:
                    recent = value
        except Exception:
            continue
    if deep is None:
        return recent
    if recent is None:
        return deep
    return recent.combine_first(deep).sort_index()


def _fetch_rows(dataset: str, ticker: str, start_date: str, end_date: str):
    params = {
        "dataset": dataset, "data_id": ticker,
        "start_date": start_date, "end_date": end_date,
    }
    url = FINMIND_URL + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "twstk/2.0"})
    try:
        with urllib.request.urlopen(req, context=_CTX, timeout=30) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        return payload.get("data", []) if payload.get("status") == 200 else []
    except Exception:
        return []


def _needs_refresh(frame: pd.DataFrame, end_date) -> bool:
    if frame is None or frame.empty:
        return True
    end = pd.Timestamp(end_date)
    # Exchange data may lag one or two business days.  Do not refetch merely
    # because today is a weekend/holiday.
    return frame.index.max() < end - pd.Timedelta(days=5)


def _incremental_refresh(tickers, margin, short, sbl, end_date, verbose):
    end = pd.Timestamp(end_date).strftime("%Y-%m-%d")
    starts = []
    for frame in (margin, short, sbl):
        if frame is not None and not frame.empty:
            starts.append(frame.index.max() - pd.Timedelta(days=7))
    start = (min(starts) if starts else pd.Timestamp(end) - pd.Timedelta(days=400))
    start = start.strftime("%Y-%m-%d")
    if verbose:
        print(f"💳 更新融資／融券／借券數量 {start} → {end}（{len(tickers)} 檔）...")

    md, sd, bd = {}, {}, {}
    ok = 0
    for i, ticker in enumerate(tickers):
        mrows = _fetch_rows(MARGIN_DATASET, ticker, start, end)
        brows = _fetch_rows(SHORT_DATASET, ticker, start, end)
        if mrows or brows:
            ok += 1
        for row in mrows:
            dt = pd.Timestamp(row.get("date"))
            md.setdefault(dt, {})[ticker] = row.get("MarginPurchaseTodayBalance", np.nan)
            sd.setdefault(dt, {})[ticker] = row.get("ShortSaleTodayBalance", np.nan)
        for row in brows:
            dt = pd.Timestamp(row.get("date"))
            bd.setdefault(dt, {})[ticker] = row.get("SBLShortSalesCurrentDayBalance", np.nan)
        if verbose and (i + 1) % 25 == 0:
            print(f"   {i + 1}/{len(tickers)}（有效 {ok}）")
        time.sleep(0.12)

    def merge(old, new_dict):
        new = pd.DataFrame.from_dict(new_dict, orient="index").sort_index()
        if old is None or old.empty:
            return new
        return new.combine_first(old).sort_index()

    margin = merge(margin, md)
    short = merge(short, sd)
    sbl = merge(sbl, bd)
    os.makedirs(os.path.dirname(MARGIN_CACHE), exist_ok=True)
    with open(MARGIN_CACHE, "wb") as f:
        pickle.dump((margin, short), f)
    with open(SBL_DEEP_CACHE, "wb") as f:
        pickle.dump(sbl, f)
    if verbose:
        print(f"   ✅ 籌碼數量更新完成，有效 {ok}/{len(tickers)} 檔")
    return margin, short, sbl


def fetch_chip_indicators(
    tickers: Iterable[str], index: Iterable, start_date=None, end_date=None,
    refresh_latest: bool = False, verbose: bool = True,
) -> ChipIndicators:
    """Load or update the three balance matrices and align to a price index."""
    tickers = [str(t) for t in tickers]
    index = pd.DatetimeIndex(index).tz_localize(None).normalize()
    if len(index) == 0:
        empty = _empty(index, tickers)
        return ChipIndicators(empty, empty.copy(), empty.copy())
    start_date = pd.Timestamp(start_date or index.min())
    end_date = pd.Timestamp(end_date or index.max())
    margin, short = _load_margin_cache()
    sbl = _load_sbl_cache()

    should_refresh = refresh_latest and (
        _needs_refresh(margin, end_date) or _needs_refresh(sbl, end_date)
    )
    if should_refresh:
        margin, short, sbl = _incremental_refresh(
            tickers, margin, short, sbl, end_date, verbose,
        )

    def aligned(frame):
        if frame is None or frame.empty:
            return _empty(index, tickers)
        out = frame.copy()
        out.index = pd.DatetimeIndex(out.index).tz_localize(None).normalize()
        return out.reindex(index=index, columns=tickers)

    result = ChipIndicators(aligned(margin), aligned(short), aligned(sbl))
    if verbose:
        cover = {
            "融資": int(result.margin_balance.notna().any().sum()),
            "融券": int(result.margin_short_balance.notna().any().sum()),
            "借券": int(result.sbl_balance.notna().any().sum()),
        }
        print(f"   ✅ 籌碼數量覆蓋：{cover}")
    return result


__all__ = ["ChipIndicators", "fetch_chip_indicators"]
