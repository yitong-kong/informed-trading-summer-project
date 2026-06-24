# -*- coding: utf-8 -*-
"""Simulated-dataset generator for online change-point detection.

The output schema of a simulated dataset is identical to
``data/processed/trades_event_level.parquet``.
See ``src/informed_order_flow/sim/README.md`` for the design and usage.

Public surface:
    estimate_baseline   estimate null parameters from a real control window
    generate_null       Level 0 / 1 / 2 H0 streams
    inject_h1           inject a parametric informed trader (known tau_info)
    build_scenario      one config + seed -> parquet + metadata + manifest
"""
from __future__ import annotations

from .estimate_baseline import estimate_baseline
from .generate_null import (
    generate_null,
    generate_null_bootstrap,
)
from .inject_h1 import inject_h1
from .run import build_scenario
from .schema import (
    SIM_DIR,
    assert_sim_trades_event_level,
    build_sim_market_metadata,
    to_sim_trades_event_level,
)

__all__ = [
    "estimate_baseline",
    "generate_null",
    "generate_null_bootstrap",
    "inject_h1",
    "build_scenario",
    "SIM_DIR",
    "assert_sim_trades_event_level",
    "build_sim_market_metadata",
    "to_sim_trades_event_level",
]
