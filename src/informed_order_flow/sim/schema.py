# -*- coding: utf-8 -*-
"""Schema layer for the simulator
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.schemas import (
    DATA_DIR,
    MARKET_METADATA_COLUMNS,
    TRADES_EVENT_LEVEL_COLUMNS,
)

SIM_DIR = DATA_DIR / "sim"

CORE_COLUMNS = [
    "timestamp", "side", "gross_shares", "gross_cash",
    "active_wallet", "condition_id", "outcome", "token_id", "transaction_hash",
]

_STR_COLS = [
    "side", "active_wallet", "condition_id", "question",
    "outcome", "resolved_outcome", "token_id", "transaction_hash",
]


# ---------------------------------------------------------------- ID synthesis
def rand_hex(rng: np.random.Generator, n_bytes: int) -> str:
    """Random ``0x``-prefixed lowercase hex string of ``n_bytes`` bytes."""
    return "0x" + rng.integers(0, 256, size=n_bytes, dtype=np.uint8).tobytes().hex()


def rand_token_id(rng: np.random.Generator) -> str:
    """Random ERC-1155-style token id as a decimal string (256-bit integer)."""
    return str(int.from_bytes(rng.integers(0, 256, size=32, dtype=np.uint8).tobytes(), "big"))


def make_wallet_pool(rng: np.random.Generator, n: int) -> np.ndarray:
    """``n`` distinct synthetic wallet addresses (``0x`` + 40 hex)."""
    pool: list[str] = []
    seen: set[str] = set()
    while len(pool) < n:
        w = rand_hex(rng, 20)
        if w not in seen:
            seen.add(w)
            pool.append(w)
    return np.array(pool, dtype=object)


# ---------------------------------------------------------------- market metadata
def build_sim_market_metadata(
    rng: np.random.Generator, markets: list[dict]
) -> pd.DataFrame:

    rows = []
    for m in markets:
        rows.append(
            {
                "market_id": str(rng.integers(10_000, 99_999)),
                "condition_id": rand_hex(rng, 32),
                "question": m["question"],
                "start_date": m.get("start_date", ""),
                "scheduled_end_date": m["scheduled_end_date"],
                "closed_time": m.get("closed_time", ""),
                "yes_token_id": rand_token_id(rng),
                "no_token_id": rand_token_id(rng),
                "resolved_outcome": m["resolved_outcome"],
                "gamma_volume": float(m.get("gamma_volume", 0.0)),
                "neg_risk": False,
            }
        )
    meta = pd.DataFrame(rows)[MARKET_METADATA_COLUMNS]
    assert meta["condition_id"].is_unique
    return meta


def token_id_set(meta: pd.DataFrame) -> set[str]:
    """The outcome token ids of the synthetic markets."""
    return set(meta["yes_token_id"]) | set(meta["no_token_id"])


# ---------------------------------------------------------------- 14-column assembler
def to_sim_trades_event_level(core: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """Turn generator core columns into the frozen 14-column main-table schema.
    """
    df = core.copy()
    df["gross_price"] = df["gross_cash"] / df["gross_shares"]

    ctx = meta[["condition_id", "question", "resolved_outcome"]]
    df = df.merge(ctx, on="condition_id", how="left", validate="m:1")
    assert df["question"].notna().all(), "condition_id not found in sim metadata"

    is_yes = df["outcome"] == "Yes"
    df["yes_price"] = df["gross_price"].where(is_yes, 1.0 - df["gross_price"])
    is_long = ((df["side"] == "BUY") & is_yes) | ((df["side"] == "SELL") & ~is_yes)
    df["signed_yes_size"] = df["gross_shares"] * np.where(is_long, 1.0, -1.0)

    df = df.sort_values(["timestamp", "transaction_hash"]).reset_index(drop=True)
    df = df[TRADES_EVENT_LEVEL_COLUMNS]


    df["timestamp"] = df["timestamp"].astype("int64")
    for c in ["gross_shares", "gross_price", "yes_price", "signed_yes_size", "gross_cash"]:
        df[c] = df[c].astype("float64")
    for c in _STR_COLS:
        df[c] = df[c].astype(object)
    return df


# ---------------------------------------------------------------- sim validator
def assert_sim_trades_event_level(
    df: pd.DataFrame, token_ids: set[str], expected_rows: int | None = None
) -> None:

    assert list(df.columns) == TRADES_EVENT_LEVEL_COLUMNS, "column drift vs schema"
    assert df["transaction_hash"].is_unique, "transaction_hash must be unique"
    assert df["token_id"].isin(token_ids).all(), "token_id not in sim token set"
    assert df["side"].isin(["BUY", "SELL"]).all()
    assert df["outcome"].isin(["Yes", "No"]).all()
    assert df["gross_shares"].gt(0).all(), "gross_shares must be > 0"
    assert df["gross_cash"].ge(0).all(), "gross_cash must be >= 0"
    assert df["gross_price"].between(0, 1).all(), "gross_price out of [0,1]"
    assert df["yes_price"].between(0, 1).all(), "yes_price out of [0,1]"
    # |signed_yes_size| == gross_shares (sign only encodes direction).
    assert np.allclose(df["signed_yes_size"].abs(), df["gross_shares"]), "signed size magnitude"
    if expected_rows is not None:
        assert len(df) == expected_rows, f"expected {expected_rows} rows, got {len(df)}"
