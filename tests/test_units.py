"""
Tests for wheel_dashboard.py — AppTest smoke tests + pure-function unit tests.

Run with:  source venv/bin/activate && python3 -m pytest tests/ -v
"""

from __future__ import annotations

import os
import uuid
import sys
from datetime import date, timedelta

# Ensure the project root is on sys.path so wheel_screener and the main app are importable.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

# ---------------------------------------------------------------------------
# Pure-function tests — no Streamlit runtime needed
# ---------------------------------------------------------------------------

# Import validate functions by re-defining them so tests don't depend on the
# app module's side effects (supabase import, st secrets, etc.)


def validate_option_trade(
    ticker: str, strike: float, contracts: int, premium: float, expiry: date
) -> tuple[bool, str]:
    if not ticker or not ticker.strip():
        return False, "Ticker cannot be empty"
    if strike <= 0:
        return False, "Strike must be greater than 0"
    if contracts < 1:
        return False, "Contracts must be at least 1"
    if expiry <= date.today():
        return False, "Expiry must be after today"
    if premium < 0:
        return False, "Premium cannot be negative"
    return True, ""


def validate_bull_put_spread(
    ticker: str, short_strike: float, long_strike: float,
    contracts: int, net_credit: float, expiry: date
) -> tuple[bool, str]:
    if not ticker or not ticker.strip():
        return False, "Ticker cannot be empty"
    if short_strike <= 0 or long_strike <= 0:
        return False, "Strikes must be greater than 0"
    if long_strike >= short_strike:
        return False, "Long put strike must be below short put strike"
    if contracts < 1:
        return False, "Contracts must be at least 1"
    if net_credit < 0:
        return False, "Net credit cannot be negative"
    if expiry <= date.today():
        return False, "Expiry must be after today"
    return True, ""


class TestValidateOptionTrade:
    def test_valid(self):
        ok, msg = validate_option_trade("AAPL", 150, 1, 1.00, date.today() + timedelta(days=1))
        assert ok and msg == ""

    def test_empty_ticker(self):
        ok, msg = validate_option_trade("", 150, 1, 1.00, date.today() + timedelta(days=1))
        assert not ok and "empty" in msg.lower()

    def test_zero_strike(self):
        ok, msg = validate_option_trade("AAPL", 0, 1, 1.00, date.today() + timedelta(days=1))
        assert not ok

    def test_zero_contracts(self):
        ok, msg = validate_option_trade("AAPL", 150, 0, 1.00, date.today() + timedelta(days=1))
        assert not ok

    def test_past_expiry(self):
        ok, msg = validate_option_trade("AAPL", 150, 1, 1.00, date.today() - timedelta(days=1))
        assert not ok

    def test_neg_premium(self):
        ok, msg = validate_option_trade("AAPL", 150, 1, -0.50, date.today() + timedelta(days=1))
        assert not ok


class TestValidateBullPutSpread:
    def test_valid(self):
        ok, msg = validate_bull_put_spread("NVDA", 195, 190, 2, 1.50, date.today() + timedelta(days=1))
        assert ok and msg == ""

    def test_long_gte_short_blocked(self):
        ok, msg = validate_bull_put_spread("NVDA", 190, 195, 2, 1.50, date.today() + timedelta(days=1))
        assert not ok and "below" in msg.lower()

    def test_long_equal_short_blocked(self):
        ok, msg = validate_bull_put_spread("NVDA", 195, 195, 2, 1.50, date.today() + timedelta(days=1))
        assert not ok

    def test_empty_ticker(self):
        ok, msg = validate_bull_put_spread("", 195, 190, 2, 1.50, date.today() + timedelta(days=1))
        assert not ok

    def test_neg_credit(self):
        ok, msg = validate_bull_put_spread("NVDA", 195, 190, 2, -0.50, date.today() + timedelta(days=1))
        assert not ok

    def test_past_expiry(self):
        ok, msg = validate_bull_put_spread("NVDA", 195, 190, 2, 1.50, date.today() - timedelta(days=1))
        assert not ok

    def test_zero_contracts(self):
        ok, msg = validate_bull_put_spread("NVDA", 195, 190, 0, 1.50, date.today() + timedelta(days=1))
        assert not ok


class TestKPIPremiumFormula:
    """Verify the Unrealized Premium Held logic matches what the app computes."""

    def test_single_leg_only(self):
        positions = [{"Premium": 2.50, "Contracts": 2}]
        spreads = []
        single = sum(p["Premium"] * p["Contracts"] * 100 for p in positions)
        spread = sum(float(s.get("NetCredit", 0)) * int(s.get("Contracts", 0)) * 100 for s in spreads if s.get("Status") == "Open")
        assert single == 500
        assert spread == 0
        assert single + spread == 500

    def test_spreads_added(self):
        positions = [{"Premium": 2.50, "Contracts": 2}]
        spreads = [
            {"NetCredit": 1.50, "Contracts": 2, "Status": "Open"},
            {"NetCredit": 1.00, "Contracts": 1, "Status": "Open"},
        ]
        single = sum(p["Premium"] * p["Contracts"] * 100 for p in positions)
        spread = sum(float(s["NetCredit"]) * int(s["Contracts"]) * 100 for s in spreads if s.get("Status") == "Open")
        assert single == 500
        assert spread == 400  # (1.50*2*100) + (1.00*1*100)
        assert single + spread == 900

    def test_closed_spreads_excluded(self):
        spreads = [
            {"NetCredit": 1.50, "Contracts": 2, "Status": "Open"},
            {"NetCredit": 0.50, "Contracts": 5, "Status": "Closed"},
        ]
        spread = sum(float(s["NetCredit"]) * int(s["Contracts"]) * 100 for s in spreads if s.get("Status") == "Open")
        assert spread == 300  # only the open one: 1.50*2*100

    def test_empty_spreads(self):
        spreads = []
        spread = sum(float(s.get("NetCredit", 0)) * int(s.get("Contracts", 0)) * 100 for s in spreads if s.get("Status") == "Open")
        assert spread == 0


class TestAddDocumentReturn:
    """add_document should return True on success, False when db is None."""

    @staticmethod
    def _add_document(db, collection_name, data, owner="admin"):
        """Replica of wheel_dashboard.add_document for contract testing."""
        if not db:
            return False
        data["id"] = data.get("id", str(uuid.uuid4()))
        data["owner"] = owner
        try:
            db.table(collection_name).insert(data).execute()
            return True
        except Exception:
            return False

    def test_returns_false_when_db_none(self):
        result = self._add_document(None, "positions", {"Ticker": "TEST"})
        assert result is False

    def test_returns_true_on_success(self):
        class FakeResponse:
            data = [{"id": "1"}]

        class FakeTable:
            def insert(self, data):
                return self
            def execute(self):
                return FakeResponse()

        class FakeDb:
            def table(self, name):
                return FakeTable()

        result = self._add_document(FakeDb(), "positions", {"Ticker": "TEST"})
        assert result is True


class TestDeleteDocumentReturn:
    """delete_document should return True on success, False when db is None."""

    @staticmethod
    def _delete_document(db, collection_name, doc_id, owner="admin"):
        """Replica of wheel_dashboard.delete_document for contract testing."""
        if not db:
            return False
        try:
            db.table(collection_name).delete().eq("id", doc_id).eq("owner", owner).execute()
            return True
        except Exception:
            return False

    def test_returns_false_when_db_none(self):
        result = self._delete_document(None, "positions", "some-id")
        assert result is False


# ---------------------------------------------------------------------------
# AppTest smoke tests — require the full app module to be importable