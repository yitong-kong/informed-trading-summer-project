# -*- coding: utf-8 -*-
"""Integrity checks for the two data stages.

- raw layer: acts on the Goldsky OrderFilledEvent table at download time;
- parsed layer: acts on the converted trades_event_level table.

"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .schemas import (
    DATA_DIR,
    EXCHANGE_ADDRESS,
    EXPECTED_TRADE_ROWS,
    N_MARKETS,
    N_TOKENS,
    TRADES_EVENT_LEVEL_COLUMNS,
)


def token_id_set(meta: pd.DataFrame) -> set[str]:
    """The 12 outcome token ids of this event, from market metadata."""
    return set(meta["yes_token_id"]) | set(meta["no_token_id"])


# ---------------------------------------------------------------- raw layer
def check_raw_fills(meta: pd.DataFrame, data_dir: Path = DATA_DIR) -> dict:
    """Raw-layer integrity on order_filled_raw.parquet.

    Returns diagnostics written into download_manifest.json. Amounts are required
    to be strictly > 0.
    """
    fills = pd.read_parquet(data_dir / "raw/goldsky/order_filled_raw.parquet")
    token_ids = token_id_set(meta)

    assert fills["id"].is_unique, "OrderFilledEvent id must be unique"
    maker_is_cash = fills["makerAssetId"] == "0"
    taker_is_cash = fills["takerAssetId"] == "0"
    assert (maker_is_cash ^ taker_is_cash).all(), "unexpected asset pair"
    token_col = fills["takerAssetId"].where(maker_is_cash, fills["makerAssetId"])
    assert token_col.isin(token_ids).all(), "token not in this event's 12 ids"
    assert (fills["makerAmountFilled"].astype("int64") > 0).all()
    assert (fills["takerAmountFilled"].astype("int64") > 0).all()
    assert (fills["fee"].astype("int64") >= 0).all()

    active = fills[fills["taker"].str.lower() == EXCHANGE_ADDRESS]
    return {
        "order_filled_rows": int(len(fills)),
        "active_order_rows_taker_eq_exchange": int(len(active)),
        "unique_makers": int(fills["maker"].str.lower().nunique()),
        "fee_nonzero_rows": int((fills["fee"].astype("int64") > 0).sum()),
    }


# ---------------------------------------------------------------- parsed layer
def assert_trades_event_level(trades: pd.DataFrame, token_ids: set[str]) -> None:
    """Parsed-layer assertions on the trades_event_level table."""
    assert list(trades.columns) == TRADES_EVENT_LEVEL_COLUMNS, "column drift vs schema"
    assert trades["transaction_hash"].is_unique
    assert trades["token_id"].isin(token_ids).all()
    assert trades["gross_shares"].gt(0).all()
    assert trades["gross_cash"].ge(0).all()
    assert trades["gross_price"].between(0, 1).all()
    assert trades["yes_price"].between(0, 1).all()
    assert len(trades) == EXPECTED_TRADE_ROWS, (
        f"expected {EXPECTED_TRADE_ROWS:,} rows, got {len(trades):,}"
    )


def validate_processed(data_dir: Path = DATA_DIR, write_summary: bool = True) -> dict:
    meta = pd.read_parquet(data_dir / "interim/market_metadata.parquet")
    assert len(meta) == N_MARKETS and meta["condition_id"].is_unique
    token_ids = token_id_set(meta)
    assert len(token_ids) == N_TOKENS

    trades = pd.read_parquet(data_dir / "processed/trades_event_level.parquet")
    assert_trades_event_level(trades, token_ids)

    summary = {
        "rows": int(len(trades)),
        "expected_rows": EXPECTED_TRADE_ROWS,
        "unique_active_wallets": int(trades["active_wallet"].nunique()),
        "n_markets": int(meta["condition_id"].nunique()),
        "n_tokens": len(token_ids),
        "rows_per_market": {
            q: int(n)
            for q, n in trades.groupby("question").size().sort_values(ascending=False).items()
        },
        "all_assertions_passed": True,
    }
    if write_summary:
        stats = data_dir / "stats"
        stats.mkdir(parents=True, exist_ok=True)
        (stats / "validation_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2)
        )
    return summary
