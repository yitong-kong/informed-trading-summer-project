# -*- coding: utf-8 -*-
"""Entry point: visualise the Q1 GLR-CUSUM detector outputs (data/detect/).

Reads cusum_calibration.json + cusum_sim_eval.parquet and recomputes detector
paths where needed. Writes the CUSUM figure suite to results/figures/cusum/:

    Aggregates / real contracts (main-text tier):
        calibration_curve, real_signature_<question> (one per real contract),
        sim_onset_error, sim_window_outcomes
    Curated simulated example (main-text tier):
        sim_example_paths,
        sim_signatures/sim_signature_s<seed>[_no]
        (raw signatures with per-method alarm tracks: estimated
        alarm/evidence window [onset, alarm], onset tick, alarm glyph)
    Per-scenario diagnostics (appendix tier):
        scenario_paths/seed_<s>/L<lv>_<mode>[_no].png  (108 three-panel figures)
        seed_overviews/seed_<s>_overview[_no].png      (per-(seed, outcome)
                                                        contact sheets)

Every simulated detector pass recomputed for the diagnostics is asserted
against the frozen verdicts in cusum_sim_eval.parquet, so this script also
acts as a consistency audit against scripts/07_run_cusum.py.

Run scripts/07_run_cusum.py first.

Usage:
    python scripts/08_plot_cusum.py
"""
from informed_order_flow.viz import cusum


def main() -> None:
    cusum.run_visualizations()


if __name__ == "__main__":
    main()
