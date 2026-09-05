# -*- coding: utf-8 -*-
"""Q1 online change-point detector: two-sided GLR-CUSUM on order-flow features.

Pipeline (see ``README.md`` in this package for the full design):

    build_features      trade table -> fixed-count event-time bucket features
    standardize         (x - mu0) / sigma0 with a robust/burn-in baseline
    run_gaussian_cusum  two-sided GLR-CUSUM (W form) over the standardized series
    calibrate_threshold finite-horizon false-alarm threshold from an H0 null
    evaluate_sim/real   power & delay on sim (known tau); alarm windows on real

The concentration channel uses HHI **only** (no top-k).
"""
from __future__ import annotations

from .calibrate import (
    CalibrationResult,
    DegenerateScale,
    calibrate_threshold,
    calibrate_threshold_pooled,
)
from .cusum import CusumConfig, CusumResult, run_gaussian_cusum
from .evaluate import (
    CUSUM,
    WINDOWED_GLR,
    DetectorRun,
    channel_path,
    channel_z,
    evaluate_real,
    evaluate_sim,
    monitoring_start,
    run_detector,
    to_utc_iso,
)
from .features import (
    CHANNELS,
    FEATURE_COLUMNS,
    SCALE_FLOOR,
    Baseline,
    FeatureConfig,
    StandardizedPath,
    WindowConfig,
    assign_buckets,
    build_features,
    iter_contracts,
    local_standardize,
    local_standardize_path,
    make_baseline,
    standardize,
    standardize_path,
)

__all__ = [
    "FeatureConfig",
    "FEATURE_COLUMNS",
    "CHANNELS",
    "SCALE_FLOOR",
    "Baseline",
    "StandardizedPath",
    "WindowConfig",
    "assign_buckets",
    "build_features",
    "iter_contracts",
    "make_baseline",
    "standardize",
    "standardize_path",
    "local_standardize",
    "local_standardize_path",
    "channel_path",
    "CusumConfig",
    "CusumResult",
    "run_gaussian_cusum",
    "CalibrationResult",
    "DegenerateScale",
    "calibrate_threshold",
    "calibrate_threshold_pooled",
    "DetectorRun",
    "channel_z",
    "run_detector",
    "monitoring_start",
    "evaluate_sim",
    "evaluate_real",
    "to_utc_iso",
    "CUSUM",
    "WINDOWED_GLR",
]
