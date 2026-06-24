# -*- coding: utf-8 -*-
"""Entry point: generate the data-quality figures from the processed main table.

Reads data/processed/trades_event_level.parquet directly. 
Writes four figures to results/figures/data_quality/:
    price_history, critical_window, wallet_concentration, crosscheck_dataapi.

Usage:
    python scripts/04_plot_data_quality.py
"""
from informed_order_flow.viz import data_quality


def main() -> None:
    data_quality.run_visualizations()


if __name__ == "__main__":
    main()
