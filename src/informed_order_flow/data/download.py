# -*- coding: utf-8 -*-
"""Maduro six-contract data ingestion.

Strategy: Goldsky subgraph is the single primary source, frozen to local parquet
after download.

Steps:
  1. Gamma API metadata -> data/raw/gamma/ (+ interim/market_metadata via build)
  2. Goldsky OrderFilledEvent / OrdersMatchedEvent (full) -> data/raw/goldsky/*.parquet
  3. Data API recent ~4,000/contract (cross-check only) -> data/raw/data_api/*.parquet
  4. raw-layer integrity checks (delegated to validate) + download_manifest.json

Run via scripts/01_download_data.py.
"""
from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from . import build, validate
from .schemas import DATA_DIR, N_TOKENS

GAMMA_URL = "https://gamma-api.polymarket.com/events"
EVENT_SLUG = "maduro-out-in-2025"
GOLDSKY_URL = (
    "https://api.goldsky.com/api/public/project_cl6mb8i9h0003e201j6li0diw"
    "/subgraphs/orderbook-subgraph/0.0.1/gn"
)
DATA_API_URL = "https://data-api.polymarket.com/trades"
PAGE_SIZE = 1000
DATA_API_OFFSETS = (0, 1000, 2000, 3000)  # offset > 3000 is rejected by the API

SESSION = requests.Session()
SESSION.headers["User-Agent"] = "ucl-msc-informed-trading/0.1"


# ---------------------------------------------------------------- Goldsky helpers
def goldsky_query(query: str, variables: dict, retries: int = 5) -> dict:
    for attempt in range(retries):
        try:
            resp = SESSION.post(
                GOLDSKY_URL,
                json={"query": query, "variables": variables},
                timeout=60,
            )
            resp.raise_for_status()
            payload = resp.json()
            if "errors" in payload:
                raise RuntimeError(payload["errors"])
            return payload["data"]
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2**attempt)
    raise RuntimeError("unreachable")


def fetch_goldsky_meta() -> dict:
    return goldsky_query("{ _meta { block { number hash } hasIndexingErrors } }", {})


# ---------------------------------------------------------------- 1. Gamma metadata
def download_metadata(data_dir: Path) -> pd.DataFrame:
    """Fetch the event JSON, freeze it to raw/, then materialise interim via build."""
    resp = SESSION.get(GAMMA_URL, params={"slug": EVENT_SLUG}, timeout=60)
    resp.raise_for_status()
    events = resp.json()
    assert len(events) == 1, f"expected 1 event, got {len(events)}"

    out = data_dir / "raw/gamma"
    out.mkdir(parents=True, exist_ok=True)
    (out / "maduro_event.json").write_text(json.dumps(events, indent=2))

    # Single parser owns metadata reconstruction (build's rebuildable-interim contract).
    return build.build_market_metadata(data_dir)


# ---------------------------------------------------------------- 2. Goldsky tables
FILLS_QUERY_TEMPLATE = """
query Fills($last: ID!, $tokens: [String!]) {{
  orderFilledEvents(
    first: {page_size}
    orderBy: id
    orderDirection: asc
    where: {{ id_gt: $last, {asset_field}_in: $tokens }}
  ) {{
    id transactionHash timestamp orderHash maker taker
    makerAssetId takerAssetId makerAmountFilled takerAmountFilled fee
  }}
}}
"""

MATCHED_QUERY_TEMPLATE = """
query Matches($last: ID!, $tokens: [String!]) {{
  ordersMatchedEvents(
    first: {page_size}
    orderBy: id
    orderDirection: asc
    where: {{ id_gt: $last, {asset_field}_in: $tokens }}
  ) {{
    id timestamp makerAssetID takerAssetID makerAmountFilled takerAmountFilled
  }}
}}
"""


def paginate(query_template: str, entity: str, asset_field: str,
             tokens: list, max_pages: int | None) -> list:
    """Keyset pagination (id_gt), not skip."""
    query = query_template.format(page_size=PAGE_SIZE, asset_field=asset_field)
    rows, last_id, pages = [], "", 0
    while True:
        page = goldsky_query(query, {"last": last_id, "tokens": tokens})[entity]
        if not page:
            break
        rows.extend(page)
        last_id = page[-1]["id"]
        pages += 1
        print(f"    {entity}/{asset_field}: page {pages}, total {len(rows)}")
        if len(page) < PAGE_SIZE or (max_pages and pages >= max_pages):
            break
    return rows


def download_goldsky(data_dir: Path, tokens: list, max_pages: int | None) -> dict:
    out = data_dir / "raw/goldsky"
    out.mkdir(parents=True, exist_ok=True)
    counts = {}

    # A token may be on the maker or taker side; query each, dedup by id
    # (id = txHash_orderHash).
    fills = (
        paginate(FILLS_QUERY_TEMPLATE, "orderFilledEvents", "takerAssetId", tokens, max_pages)
        + paginate(FILLS_QUERY_TEMPLATE, "orderFilledEvents", "makerAssetId", tokens, max_pages)
    )
    fills_df = pd.DataFrame(fills).drop_duplicates("id")
    fills_df.to_parquet(out / "order_filled_raw.parquet", index=False)
    counts["order_filled"] = len(fills_df)

    # OrdersMatchedEvent: fields are makerAssetID/takerAssetID; id = bare txHash
    matches = (
        paginate(MATCHED_QUERY_TEMPLATE, "ordersMatchedEvents", "takerAssetID", tokens, max_pages)
        + paginate(MATCHED_QUERY_TEMPLATE, "ordersMatchedEvents", "makerAssetID", tokens, max_pages)
    )
    matches_df = pd.DataFrame(matches).drop_duplicates("id")
    matches_df.to_parquet(out / "orders_matched_raw.parquet", index=False)
    counts["orders_matched"] = len(matches_df)
    return counts


# ---------------------------------------------------------------- 3. Data API cross-check
def download_data_api(data_dir: Path, meta: pd.DataFrame) -> int:
    out = data_dir / "raw/data_api"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for condition_id in meta["condition_id"]:
        for offset in DATA_API_OFFSETS:
            resp = SESSION.get(
                DATA_API_URL,
                params={"market": condition_id, "limit": PAGE_SIZE,
                        "offset": offset, "takerOnly": "true"},
                timeout=60,
            )
            resp.raise_for_status()
            page = resp.json()
            if not isinstance(page, list) or not page:
                break
            rows.extend(page)
        print(f"    data_api {condition_id[:10]}...: cumulative {len(rows)}")

    api_df = pd.DataFrame(rows)
    if not api_df.empty:
        api_df = api_df.drop_duplicates(
            ["transactionHash", "proxyWallet", "asset", "side", "size", "price"]
        ).sort_values(["timestamp", "transactionHash"])
    api_df.to_parquet(out / "recent_trades.parquet", index=False)
    return len(api_df)


# ---------------------------------------------------------------- 4. orchestration
def download_all(data_dir: Path = DATA_DIR, smoke: bool = False) -> dict:
    """Run the full ingestion and write data/manifests/*.json; return the manifest."""
    max_pages = 2 if smoke else None
    started = datetime.now(timezone.utc).isoformat()
    meta_block = fetch_goldsky_meta()

    print("[1/4] Gamma metadata")
    meta = download_metadata(data_dir)
    tokens = sorted(set(meta["yes_token_id"]) | set(meta["no_token_id"]))
    assert len(tokens) == N_TOKENS

    print("[2/4] Goldsky main tables")
    goldsky_counts = download_goldsky(data_dir, tokens, max_pages)

    print("[3/4] Data API cross-check sample")
    api_rows = download_data_api(data_dir, meta)

    print("[4/4] Integrity checks + manifest")
    checks = validate.check_raw_fills(meta, data_dir) if not smoke else {"smoke_run": True}

    manifests = data_dir / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    (manifests / "goldsky_meta.json").write_text(json.dumps(meta_block, indent=2))
    manifest = {
        "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "smoke": smoke,
        "event_slug": EVENT_SLUG,
        "goldsky_endpoint": GOLDSKY_URL,
        "goldsky_meta_block": meta_block["_meta"]["block"]["number"],
        "goldsky_counts": goldsky_counts,
        "data_api_rows": api_rows,
        "integrity": checks,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (manifests / "download_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest
