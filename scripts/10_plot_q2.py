# -*- coding: utf-8 -*-
"""Render the Q2 / Q3 wallet-attribution figures into results/figures/q2/.

Read-only: every number comes from the frozen products under data/attrib/ and
results/q2/q2_calibration.json, and nothing under data/attrib/ or results/q2/
is written, so the Q2/Q3 provenance hashes are untouched.

    python scripts/10_plot_q2.py
"""
from __future__ import annotations

from informed_order_flow.viz.attribution import build_all

if __name__ == "__main__":
    build_all()
