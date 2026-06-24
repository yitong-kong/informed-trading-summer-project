# -*- coding: utf-8 -*-
"""
We define trade unit T_i = (t_i, a_i, s_i, p_i, w_i):
    t_i  arrival time         -> timestamp (unix seconds, UTC)
    a_i  side                  -> side (BUY/SELL, active-order direction)
    s_i  size                  -> gross_shares (token shares)
    p_i  implied probability   -> gross_price (raw) / yes_price (unified YES space)
    w_i  wallet address        -> active_wallet (active-order owner)

OrdersMatched is the trade stream; the active wallet is recovered from the same-tx taker==Exchange OrderFilled; 
amounts and side are parsed with /1e6; prices are unified to YES space.

Reads only data/raw.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from . import validate
from .schemas import (
    DATA_DIR,
    EXCHANGE_ADDRESS,
    MARKET_METADATA_COLUMNS,
    N_MARKETS,
    N_TOKENS,
    REPO_ROOT,
    TRADES_EVENT_LEVEL_COLUMNS,
)


# ---------------------------------------------------------------- interim: market metadata
def build_market_metadata(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Reconstruct market_metadata from the frozen raw/gamma JSON.
    """
    events = json.loads((data_dir / "raw/gamma/maduro_event.json").read_text())
    assert len(events) == 1, f"expected 1 event, got {len(events)}"
    event = events[0]

    rows = []
    for market in event["markets"]:
        tokens = json.loads(market["clobTokenIds"])
        outcomes = json.loads(market["outcomes"])
        prices = list(map(float, json.loads(market["outcomePrices"])))

        assert market["closed"], f"market still open: {market['question']}"
        assert market["umaResolutionStatus"] == "resolved", market["question"]
        assert sorted(prices) == [0.0, 1.0], (market["question"], prices)
        resolved = outcomes[prices.index(max(prices))]

        rows.append(
            {
                "market_id": market["id"],
                "condition_id": market["conditionId"],
                "question": market["question"],
                "start_date": market["startDate"],
                "scheduled_end_date": market["endDate"],
                "closed_time": market["closedTime"],
                "yes_token_id": tokens[outcomes.index("Yes")],
                "no_token_id": tokens[outcomes.index("No")],
                "resolved_outcome": resolved,
                "gamma_volume": float(market["volume"]),
                "neg_risk": bool(market.get("negRisk", False)),
            }
        )

    meta = pd.DataFrame(rows)[MARKET_METADATA_COLUMNS]
    assert len(meta) == N_MARKETS and meta["condition_id"].is_unique
    assert not meta["neg_risk"].any(), "expected plain V1 markets only"
    token_ids = set(meta["yes_token_id"]) | set(meta["no_token_id"])
    assert len(token_ids) == N_TOKENS

    interim = data_dir / "interim"
    interim.mkdir(parents=True, exist_ok=True)
    meta.to_parquet(interim / "market_metadata.parquet", index=False)
    return meta


def _token_map(meta: pd.DataFrame) -> tuple[pd.DataFrame, set[str]]:
    """Long token table (token_id -> condition/question/outcome/resolved)."""
    rows = []
    for _, m in meta.iterrows():
        for outcome, col in (("Yes", "yes_token_id"), ("No", "no_token_id")):
            rows.append(
                {
                    "token_id": m[col],
                    "condition_id": m["condition_id"],
                    "question": m["question"],
                    "outcome": outcome,
                    "resolved_outcome": m["resolved_outcome"],
                }
            )
    tok = pd.DataFrame(rows)
    assert tok["token_id"].is_unique and len(tok) == N_TOKENS
    return tok, set(tok["token_id"])


# ---------------------------------------------------------------- raw -> active wallet
def _active_wallet_map(data_dir: Path) -> pd.DataFrame:
    """maker of the same-tx taker==Exchange OrderFilledEvent = active wallet."""
    fills = pd.read_parquet(
        data_dir / "raw/goldsky/order_filled_raw.parquet",
        columns=["transactionHash", "maker", "taker"],
    )
    active = fills[fills["taker"].str.lower() == EXCHANGE_ADDRESS]
    dup = active["transactionHash"].duplicated()
    assert not dup.any(), (
        f"{dup.sum()} transactions carry multiple active fills — "
        "investigate before proceeding (data-sources spec: coverage issue)"
    )
    return active.rename(
        columns={"transactionHash": "transaction_hash", "maker": "active_wallet"}
    )[["transaction_hash", "active_wallet"]]


# ---------------------------------------------------------------- raw -> matched trades
def _parse_matches(data_dir: Path, token_ids: set[str]) -> pd.DataFrame:
    m = pd.read_parquet(data_dir / "raw/goldsky/orders_matched_raw.parquet")
    assert m["id"].is_unique

    maker_amt = m["makerAmountFilled"].astype("int64") / 1e6
    taker_amt = m["takerAmountFilled"].astype("int64") / 1e6
    maker_is_cash = m["makerAssetID"] == "0"
    taker_is_cash = m["takerAssetID"] == "0"
    # Exactly one leg is USDC, else it is an unexpected pair.
    assert (maker_is_cash ^ taker_is_cash).all(), "unexpected OrdersMatched asset pair"

    out = pd.DataFrame(
        {
            "transaction_hash": m["id"],
            "timestamp": m["timestamp"].astype("int64"),
            "side": maker_is_cash.map({True: "BUY", False: "SELL"}),
            "token_id": m["takerAssetID"].where(maker_is_cash, m["makerAssetID"]),
            "gross_cash": maker_amt.where(maker_is_cash, taker_amt),
            "gross_shares": taker_amt.where(maker_is_cash, maker_amt),
        }
    )
    assert out["token_id"].isin(token_ids).all()
    out["gross_price"] = out["gross_cash"] / out["gross_shares"]
    return out


# ---------------------------------------------------------------- main build
def build_trades_event_level(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Build processed/trades_event_level.parquet from raw; write its manifest."""
    meta = build_market_metadata(data_dir)
    tok, token_ids = _token_map(meta)

    trades = _parse_matches(data_dir, token_ids)
    wallets = _active_wallet_map(data_dir)
    trades = trades.merge(wallets, on="transaction_hash", how="left", validate="1:1")
    assert trades["active_wallet"].notna().all(), "active wallet missing for some matches"

    trades = trades.merge(tok, on="token_id", how="left", validate="m:1")

    # Unify to the YES probability space.
    is_yes = trades["outcome"] == "Yes"
    trades["yes_price"] = trades["gross_price"].where(is_yes, 1 - trades["gross_price"])
    buy_sign = trades["side"].map({"BUY": 1.0, "SELL": -1.0})
    trades["signed_yes_size"] = trades["gross_shares"] * buy_sign.where(is_yes, -buy_sign)

    trades = trades.sort_values(["timestamp", "transaction_hash"]).reset_index(drop=True)
    trades = trades[TRADES_EVENT_LEVEL_COLUMNS]

    # Parsed-layer assertions.
    validate.assert_trades_event_level(trades, token_ids)

    processed = data_dir / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    out_path = processed / "trades_event_level.parquet"
    trades.to_parquet(out_path, index=False)

    _write_manifest(data_dir, trades, out_path)
    return trades


def _write_manifest(data_dir: Path, trades: pd.DataFrame, out_path: Path) -> None:
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "output": str(out_path.relative_to(REPO_ROOT)),
        "rows": int(len(trades)),
        "unique_active_wallets": int(trades["active_wallet"].nunique()),
        "time_range_utc": [
            datetime.fromtimestamp(int(trades["timestamp"].min()), tz=timezone.utc).isoformat(),
            datetime.fromtimestamp(int(trades["timestamp"].max()), tz=timezone.utc).isoformat(),
        ],
        "rows_per_market": {
            q: int(n)
            for q, n in trades.groupby("question").size().sort_values(ascending=False).items()
        },
        "proposal_mapping": {
            "t_i": "timestamp", "a_i": "side", "s_i": "gross_shares",
            "p_i": "gross_price (raw) / yes_price (YES space)", "w_i": "active_wallet",
        },
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    manifests = data_dir / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    (manifests / "processed_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2)
    )
