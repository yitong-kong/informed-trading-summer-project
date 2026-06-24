# -*- coding: utf-8 -*-
"""Entry point: download + freeze raw data and write manifests.

Usage:
    python scripts/01_download_data.py            # full download + freeze
    python scripts/01_download_data.py --smoke    # 2 pages/query, pipeline check only
"""
import argparse
import json

from informed_order_flow.data import download


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true",
                        help="2 pages per query (pipeline validation, not real data)")
    args = parser.parse_args()

    manifest = download.download_all(smoke=args.smoke)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
