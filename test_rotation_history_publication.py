import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
HISTORY = ROOT / "capital_rotation_history_10y.json"


def _sha256(path):
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def test_complete_rotation_history_is_internally_consistent():
    history = json.loads(
        HISTORY.read_text(encoding="utf-8"),
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )
    assert history["period"] == {"start": "2016-07-01", "end": "2026-07-07"}
    assert history["event_count"] == 192 == len(history["events"])
    assert history["stock_test_count"] == 5596
    assert [event["event_id"] for event in history["events"]] == list(range(192))

    stocks = [stock for event in history["events"] for stock in event["stocks"]]
    assert len(stocks) == history["stock_test_count"]
    hits = [stock for stock in stocks if stock["hit_20pct"]]
    assert len(hits) / len(stocks) == history["headline"]["stock_hit_rate"]
    for stock in hits:
        assert stock["drawdown_date"]
        assert 1 <= float(stock["lead_trading_days"]) <= 120
        assert stock["last_session_before_drawdown"] < stock["drawdown_date"]
    for event in history["events"]:
        assert len(event["stocks"]) == int(event["outcome"]["stocks_tested"])
        assert sum(stock["hit_20pct"] for stock in event["stocks"]) == int(
            event["outcome"]["stocks_hit_20pct"]
        )


def test_published_file_hashes_and_html_event_count():
    history = json.loads(HISTORY.read_text(encoding="utf-8"))
    for filename, info in history["files"].items():
        path = ROOT / filename
        assert path.exists()
        assert _sha256(path) == info["sha256"]

    page = (ROOT / "capital_rotation_alert.html").read_text(encoding="utf-8")
    assert page.count('class="history-main"') == history["event_count"]
    assert page.count('class="history-detail"') == history["event_count"]
    assert "十年完整回測歷史：逐筆可核對" in page
    assert HISTORY.name in page


def test_latest_alert_points_to_complete_history():
    latest = json.loads(
        (ROOT / "capital_rotation_alert_latest.json").read_text(encoding="utf-8")
    )
    backtest = latest["historical_backtest"]
    assert backtest["complete_event_history_published"] is True
    assert backtest["event_count"] == 192
    assert backtest["stock_test_count"] == 5596
    assert backtest["file"] == HISTORY.name
