# -*- coding: utf-8 -*-
"""Main-table column definitions and shared constants (single source of truth).

All import from here to avoid schema drift.
"""
from __future__ import annotations

from pathlib import Path

# Repo root: src/informed_order_flow/data/schemas.py -> parents[3] = repo root.
REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data"

# V1 CTF Exchange address: the taker of the active OrderFilledEvent
EXCHANGE_ADDRESS = "0x4bfb41d5b3570defd03c39a9a4d8de6bd8b8982e"

# Event-level constants
N_MARKETS = 6
N_TOKENS = 12
EXPECTED_TRADE_ROWS = 80_049

# The 14 columns of the parsed main table trades_event_level.
# proposal trade unit T_i=(t_i,a_i,s_i,p_i,w_i) + unified YES space + contract context
# + retained raw fields.
TRADES_EVENT_LEVEL_COLUMNS = [
    # proposal T_i = (t, a, s, p, w)
    "timestamp", "side", "gross_shares", "gross_price", "active_wallet",
    # unified YES space (comparable across contracts)
    "yes_price", "signed_yes_size",
    # contract context and retained raw fields
    "condition_id", "question", "outcome", "resolved_outcome",
    "token_id", "gross_cash", "transaction_hash",
]

# Columns of market_metadata (interim layer).
MARKET_METADATA_COLUMNS = [
    "market_id", "condition_id", "question", "start_date", "scheduled_end_date",
    "closed_time", "yes_token_id", "no_token_id", "resolved_outcome",
    "gamma_volume", "neg_risk",
]
