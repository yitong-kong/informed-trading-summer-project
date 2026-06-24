# -*- coding: utf-8 -*-
"""Entry point: re-run parsed-layer integrity checks on the frozen main table.

Writes data/stats/validation_summary.json

Usage:
    python scripts/03_validate_data.py
"""
import json

from informed_order_flow.data import validate


def main() -> None:
    summary = validate.validate_processed()
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
