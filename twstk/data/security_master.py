"""
twstk.data.security_master — 代號 ↔ 名稱／產業 對照（本地 security master）

用途：報表與訂單只顯示數字代號，人工下單容易看錯行（2317 vs 2371）。
      這一層提供 代號→中文名稱／產業／市場別，純顯示用，不參與任何交易決策。

設計（依既有慣例）：
  · 建『本地 master 檔』並定期更新，不要每次產報表就即時打 API。
  · 主來源 FinMind `TaiwanStockInfo`（免 token，約 4,300 檔，含上市/上櫃/興櫃）。
  · 快取過期或抓取失敗時：沿用既有快取並告警（顯示層 fail-soft，
    與生產資料層的 fail-closed 相反——名稱缺失不該擋住交易報表）。
"""

from __future__ import annotations

import json
import os
import urllib.parse
from datetime import datetime, timezone

DEFAULT_PATH = os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), "security_master.json")
FINMIND = "https://api.finmindtrade.com/api/v4/data"
DEFAULT_MAX_AGE_DAYS = 30

_CACHE: dict | None = None


def _fetch_finmind(token: str | None = None) -> dict:
    from pool_audit import _get_json  # 沿用含 curl 後援的取數
    params = {"dataset": "TaiwanStockInfo"}
    if token:
        params["token"] = token
    d = _get_json(FINMIND + "?" + urllib.parse.urlencode(params))
    rows = d.get("data") or []
    if not rows:
        raise RuntimeError("TaiwanStockInfo 回應為空")
    out = {}
    for r in rows:
        sid = str(r.get("stock_id", "")).strip()
        if not sid:
            continue
        # 同一代號可能多筆（不同日期），保留最後一筆即可
        out[sid] = {
            "name": r.get("stock_name"),
            "industry": r.get("industry_category"),
            "market": r.get("type"),
        }
    return out


def _age_days(meta: dict) -> float:
    try:
        t = datetime.fromisoformat(meta["updated_at"])
        if t.tzinfo is None:
            t = t.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - t).total_seconds() / 86400
    except Exception:
        return 1e9


def load_master(path: str = DEFAULT_PATH, *, max_age_days: int = DEFAULT_MAX_AGE_DAYS,
                refresh: bool = False, token: str | None = None,
                verbose: bool = False) -> dict:
    """
    載入 security master。過期或 refresh=True 時重新抓取並寫回快取。
    抓取失敗但有舊快取 → 沿用舊快取並告警（fail-soft）。
    """
    global _CACHE
    disk = None
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                disk = json.load(f)
        except Exception:
            disk = None

    stale = disk is None or _age_days(disk.get("_meta", {})) > max_age_days
    if not refresh and not stale and disk:
        _CACHE = disk.get("securities", {})
        return _CACHE

    try:
        sec = _fetch_finmind(token)
        payload = {"_meta": {"updated_at": datetime.now(timezone.utc).isoformat(),
                             "source": "FinMind TaiwanStockInfo",
                             "count": len(sec)},
                   "securities": sec}
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=1, sort_keys=True)
        os.replace(tmp, path)
        if verbose:
            print(f"✅ security master 更新：{len(sec)} 檔 → {path}")
        _CACHE = sec
        return sec
    except Exception as e:
        if disk:
            print(f"⚠️ security master 更新失敗（沿用既有快取，{_age_days(disk.get('_meta', {})):.0f} 天前）：{e}")
            _CACHE = disk.get("securities", {})
            return _CACHE
        raise


def get_name(ticker, default: str | None = None, **kw) -> str:
    """代號 → 中文名稱；查無時回 default（預設為代號本身）。"""
    m = _CACHE if _CACHE is not None else load_master(**kw)
    rec = m.get(str(ticker))
    if rec and rec.get("name"):
        return rec["name"]
    return default if default is not None else str(ticker)


def describe(ticker, **kw) -> dict:
    """代號 → {name, industry, market}；查無時各欄為 None。"""
    m = _CACHE if _CACHE is not None else load_master(**kw)
    return m.get(str(ticker), {"name": None, "industry": None, "market": None})


def label(ticker, **kw) -> str:
    """報表用標籤，如 '2330 台積電'；查無名稱時退回純代號。"""
    n = get_name(ticker, default="", **kw)
    return f"{ticker} {n}".strip()


def enrich(tickers, **kw) -> dict:
    """批次：{ticker: {name, industry, market}}。"""
    m = _CACHE if _CACHE is not None else load_master(**kw)
    return {str(t): m.get(str(t), {"name": None, "industry": None, "market": None})
            for t in tickers}


def _cli():
    import argparse
    ap = argparse.ArgumentParser(description="建立/更新 security master")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--path", default=DEFAULT_PATH)
    ap.add_argument("--token", default=None)
    ap.add_argument("--show", default=None, help="查詢代號，逗號分隔")
    args = ap.parse_args()
    m = load_master(args.path, refresh=args.refresh, token=args.token, verbose=True)
    print(f"master 共 {len(m)} 檔")
    if args.show:
        for t in args.show.split(","):
            print(f"  {label(t.strip())}  {describe(t.strip())}")


if __name__ == "__main__":
    _cli()
