# -*- coding: utf-8 -*-
"""Entry point: build the main table from frozen raw data.

raw -> processed/trades_event_level.parquet. 

Usage:
    python scripts/02_build_trades_table.py
"""
import json

from informed_order_flow.data import build


def main() -> None:
    trades = build.build_trades_event_level()
    print(json.dumps({
        "rows": int(len(trades)),
        "unique_active_wallets": int(trades["active_wallet"].nunique()),
    }, indent=2))


if __name__ == "__main__":
    main()
