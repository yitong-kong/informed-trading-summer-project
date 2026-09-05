# -*- coding: utf-8 -*-
"""Entry point: visualise the simulated dataset built under data/sim/.

Reads each scenario's produced trades_event_level.parquet + sim_manifest.json.
Writes to results/figures/simulated_data/:
    imbalance_signature_s<seed>[_no] (one figure per seed and resolved
    outcome) and wallet_concentration.

Run scripts/05_build_simulated_data.py first.

Usage:
    python scripts/06_plot_simulated_data.py
"""
from informed_order_flow.viz import simulated_data


def main() -> None:
    simulated_data.run_visualizations()


if __name__ == "__main__":
    main()
