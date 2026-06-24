# -*- coding: utf-8 -*-
"""Parsed-layer assertions on the frozen main table.
"""
import pandas as pd
import pytest

from informed_order_flow.data import validate
from informed_order_flow.data.schemas import (
    DATA_DIR,
    EXPECTED_TRADE_ROWS,
    N_MARKETS,
    N_TOKENS,
    TRADES_EVENT_LEVEL_COLUMNS,
)

PROCESSED = DATA_DIR / "processed" / "trades_event_level.parquet"
META = DATA_DIR / "interim" / "market_metadata.parquet"


@pytest.fixture(scope="module")
def trades() -> pd.DataFrame:
    if not PROCESSED.exists():
        pytest.skip("processed table not present; run scripts/02 first")
    return pd.read_parquet(PROCESSED)


@pytest.fixture(scope="module")
def token_ids() -> set[str]:
    if not META.exists():
        pytest.skip("market metadata not present; run scripts/02 first")
    return validate.token_id_set(pd.read_parquet(META))


def test_row_count(trades):
    assert len(trades) == EXPECTED_TRADE_ROWS


def test_columns_match_schema(trades):
    assert list(trades.columns) == TRADES_EVENT_LEVEL_COLUMNS


def test_unique_transaction_hash(trades):
    assert trades["transaction_hash"].is_unique


def test_prices_in_unit_interval(trades):
    assert trades["gross_price"].between(0, 1).all()
    assert trades["yes_price"].between(0, 1).all()


def test_positive_sizes(trades):
    assert trades["gross_shares"].gt(0).all()
    assert trades["gross_cash"].ge(0).all()


def test_tokens_belong_to_event(trades, token_ids):
    assert len(token_ids) == N_TOKENS
    assert trades["token_id"].isin(token_ids).all()


def test_full_assertion_suite(trades, token_ids):
    # Should not raise.
    validate.assert_trades_event_level(trades, token_ids)


def test_six_markets(trades):
    assert trades["condition_id"].nunique() == N_MARKETS
