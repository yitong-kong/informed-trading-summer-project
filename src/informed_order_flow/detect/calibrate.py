# -*- coding: utf-8 -*-
"""Empirical finite-horizon threshold calibration.

The real and simulated streams are finite-horizon, so the calibration targets
the empirical finite-window exceedance rate under the **designated H0 null**'s
bootstrap distribution, rather than an asymptotic ARL:

    alpha_hat(h) = mean over null replicates of 1{ max_b G_b >= h },
    h* = min{ h : alpha_hat(h) <= alpha }.

``alpha_hat`` is a property of that designated null and its resampling scheme,
so ``h*`` inherits whatever the null does and does not represent. Which streams
supply the null is recorded per calibration cell by the caller.

``h*`` is found by *searching* the observed max statistics for the smallest one
satisfying that inequality, not by taking a ``1 - alpha`` quantile: the alarm
rule is ``stat >= h``, and with ties a quantile does not guarantee
``alpha_hat(h*) <= alpha``. Normal-theory thresholds do not hold on
heavy-tailed, non-stationary, low-liquidity prediction-market data, so the
threshold is set this way, not from a textbook ARL.

Replicates are drawn by **circular block bootstrap of the standardized null
series** ``z``: the null stream is standardized once, with the same baseline the
production detector uses, and blocks of that ``z`` are resampled to preserve
short-range serial dependence. The detector then runs on each replicate under the
same ``CusumConfig`` -- including the same ``start_index`` -- as the deployed
detector, so the monitoring ban is present in the null distribution too.

A single null seed cannot certify ``alpha`` (its ``h*`` does not transfer across
seeds), so ``calibrate_threshold_pooled`` pools the per-replicate max statistics
across *several* independent null streams and reads one threshold off the pooled
sample. Note that pooled replicates buy Monte-Carlo precision, not independent
information: the statistical precision of ``h*`` is governed by the number of
independent source streams, which is why ``CalibrationResult`` reports the
replicate counts separately instead of one ``n_replicates`` number.

The conservative martingale bound ``h >= log(N / alpha)`` is reported alongside
as a sanity check.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .cusum import CusumConfig, run_gaussian_cusum

# A null whose standardized series is constant cannot be calibrated on: every
# replicate would be identical and the max statistic degenerate.
_MIN_SPREAD = 1e-8


class DegenerateScale(RuntimeError):
    """Raised when a null's standardized series is constant or non-finite."""


@dataclass
class CalibrationResult:
    threshold: float            # h*, the calibrated finite-horizon threshold
    alpha_target: float
    alpha_achieved: float       # alpha_hat(h*) on the replicates
    horizon: int                # N, buckets per replicate
    block_len: int
    theoretical_bound: float    # log(N / alpha), conservative martingale bound
    # Replicate accounting. ``n_replicates_total`` is a Monte-Carlo count; the
    # statistical precision of h* is governed by ``n_sources`` (independent null
    # streams) and, within a stream, by ``n_effective_blocks`` -- the number of
    # distinct blocks the resampler can draw from. Reporting only a per-source
    # replicate count invites a reader to over-count the evidence.
    n_sources: int
    n_replicates_per_source: int
    n_replicates_total: int
    n_effective_blocks: int
    # ``horizon`` divided by the shortest source length. Bootstrapping a short
    # null out to a much longer horizon extrapolates, so a large value here means
    # h* is read from nulls much shorter than the stream it will threshold.
    horizon_ratio: float
    grid: list[tuple[float, float]] = field(default_factory=list)  # (h, alpha_hat)

    def to_dict(self) -> dict:
        return {
            "threshold": self.threshold,
            "alpha_target": self.alpha_target,
            "alpha_achieved": self.alpha_achieved,
            "horizon": self.horizon,
            "block_len": self.block_len,
            "theoretical_bound_log_N_over_alpha": self.theoretical_bound,
            "n_sources": self.n_sources,
            "n_replicates_per_source": self.n_replicates_per_source,
            "n_replicates_total": self.n_replicates_total,
            "n_effective_blocks": self.n_effective_blocks,
            "horizon_ratio": self.horizon_ratio,
            "grid": [{"h": h, "alpha_hat": a} for h, a in self.grid],
        }


def circular_block_bootstrap(x: np.ndarray, block_len: int, horizon: int,
                             rng: np.random.Generator) -> np.ndarray:
    """Resample ``horizon`` points from ``x`` as wrapped blocks of ``block_len``.

    Circular wrapping avoids end effects; concatenated blocks are truncated to
    exactly ``horizon`` so every replicate shares the same finite horizon.
    """
    n = x.shape[0]
    if n == 0:
        raise ValueError("cannot bootstrap an empty series")
    block_len = max(1, min(block_len, n))
    n_blocks = math.ceil(horizon / block_len)
    starts = rng.integers(0, n, size=n_blocks)
    offsets = np.arange(block_len)
    idx = (starts[:, None] + offsets[None, :]).ravel() % n
    return x[idx[:horizon]]


def bootstrap_max_stats(z_null: np.ndarray, config: CusumConfig, *,
                        n_replicates: int, block_len: int, horizon: int,
                        seed: int = 0) -> np.ndarray:
    """Per-replicate max statistic from circular block bootstrap of ``z_null``.

    ``z_null`` is the already-standardized null series. Each replicate is run
    under the caller's ``config``, so the deployed detector's ``start_index``
    monitoring ban applies to the null distribution as well.
    """
    z_null = np.asarray(z_null, dtype="float64")
    if not np.all(np.isfinite(z_null)) or float(np.ptp(z_null)) < _MIN_SPREAD:
        raise DegenerateScale(
            "the null's standardized series is constant or non-finite; it is too "
            "close to degenerate to calibrate on")
    rng = np.random.default_rng(seed)
    maxes = [run_gaussian_cusum(
                circular_block_bootstrap(z_null, block_len, horizon, rng),
                config).max_stat
             for _ in range(n_replicates)]
    return np.asarray(maxes, dtype="float64")


def bootstrap_max_stats_raw(x_null: np.ndarray, config: CusumConfig, *,
                            standardizer, n_replicates: int, block_len: int,
                            horizon: int, seed: int = 0) -> np.ndarray:
    """Replicate max statistics, resampling the **raw** feature path.

    Not wired into any pipeline; provided so the alternative calibration regime
    is available without rewriting the calibrator when it is wanted.

    The difference from ``bootstrap_max_stats`` is where standardization sits.
    There, the null is standardized once and blocks of the resulting ``z`` are
    resampled, so every replicate inherits one baseline that was estimated from
    the whole stream -- the null therefore contains no baseline-estimation
    error, while the deployed detector estimates its baseline from 20 buckets.
    Here each replicate is resampled first and then standardizes *itself*, so
    the variability of that estimate is inside the null distribution:

        raw bucket path -> block bootstrap -> re-estimate baseline within the
        replicate -> standardize within the replicate -> run the detector

    ``standardizer`` is a callable ``raw -> z`` carrying the method's own
    baseline rule, so the replicate uses the same rule the deployment does.

    The two regimes do not produce the same threshold and their results must
    never be pooled; anything derived from either carries a
    ``calibration_variant`` tag for that reason.
    """
    x_null = np.asarray(x_null, dtype="float64")
    if not np.all(np.isfinite(x_null)) or float(np.ptp(x_null)) < _MIN_SPREAD:
        raise DegenerateScale(
            "the null's raw feature series is constant or non-finite; it is too "
            "close to degenerate to calibrate on")
    rng = np.random.default_rng(seed)
    maxes = []
    for _ in range(n_replicates):
        raw = circular_block_bootstrap(x_null, block_len, horizon, rng)
        maxes.append(run_gaussian_cusum(standardizer(raw), config).max_stat)
    return np.asarray(maxes, dtype="float64")


def threshold_from_max_stats(max_stats: np.ndarray, alpha: float) -> float:
    """Smallest observed ``h`` with ``mean(max_stats >= h) <= alpha``.

    The alarm rule is ``stat >= h``, so the threshold must be read off that same
    inequality. A ``1 - alpha`` quantile is not enough: when the max statistics
    have ties at the quantile, ``mean(max_stats >= quantile)`` can exceed
    ``alpha``. Searching the observed values for the smallest conservative one
    is exact by construction.
    """
    max_stats = np.asarray(max_stats, dtype="float64")
    if max_stats.size == 0:
        raise ValueError("cannot calibrate on an empty max-statistic sample")
    for h in np.unique(max_stats):          # np.unique returns sorted values
        if float(np.mean(max_stats >= h)) <= alpha:
            return float(h)
    # Even the largest observed value fires too often (alpha < 1/n): no observed
    # threshold is conservative, so step just above the sample.
    return float(np.nextafter(max_stats.max(), np.inf))


def empirical_alpha(max_stats: np.ndarray, h: float) -> float:
    return float(np.mean(np.asarray(max_stats) >= h))


def _result_from_max_stats(max_stats: np.ndarray, *, alpha: float, horizon: int,
                           block_len: int, n_sources: int,
                           n_replicates_per_source: int, n_effective_blocks: int,
                           horizon_ratio: float,
                           grid: np.ndarray | None) -> CalibrationResult:
    """Threshold + achieved alpha + alpha_hat(h) grid from pooled max statistics."""
    h_star = threshold_from_max_stats(max_stats, alpha)
    if grid is None:
        grid = np.linspace(0.0, float(max_stats.max()) + 1.0, 41)
    grid_pairs = [(float(h), empirical_alpha(max_stats, h)) for h in grid]
    return CalibrationResult(
        threshold=h_star,
        alpha_target=alpha,
        alpha_achieved=empirical_alpha(max_stats, h_star),
        horizon=horizon,
        block_len=block_len,
        theoretical_bound=math.log(horizon / alpha),
        n_sources=n_sources,
        n_replicates_per_source=n_replicates_per_source,
        n_replicates_total=int(max_stats.size),
        n_effective_blocks=n_effective_blocks,
        horizon_ratio=horizon_ratio,
        grid=grid_pairs,
    )


def calibrate_threshold(z_null: np.ndarray, config: CusumConfig, *,
                        alpha: float = 0.05, horizon: int | None = None,
                        n_replicates: int = 500, block_len: int = 10,
                        seed: int = 0, grid: np.ndarray | None = None
                        ) -> CalibrationResult:
    """Calibrate the finite-horizon threshold ``h*`` from one standardized null.

    ``z_null`` is the standardized feature of the designated H0 stream (sim
    L0/L1/L2 or a real control window). ``horizon`` defaults to the length of
    ``z_null``.
    """
    return calibrate_threshold_pooled([z_null], config, alpha=alpha,
                                      horizon=int(horizon or np.asarray(z_null).shape[0]),
                                      n_replicates=n_replicates, block_len=block_len,
                                      seed=seed, grid=grid)


def calibrate_threshold_pooled(z_nulls: list[np.ndarray], config: CusumConfig, *,
                               alpha: float = 0.05, horizon: int,
                               n_replicates: int = 500, block_len: int = 10,
                               seed: int = 0, grid: np.ndarray | None = None
                               ) -> CalibrationResult:
    """Calibrate ``h*`` by pooling bootstrap max-stats across several nulls.

    A threshold read off a single null seed reflects that seed's idiosyncrasies
    and under-covers other draws of the same null; pooling ``n_replicates``
    replicates from *each* independent null stream (different seeds of the same
    generator level) and reading one conservative threshold off the combined
    sample targets the false-alarm rate across the null ensemble.
    """
    if not z_nulls:
        raise ValueError("need at least one null series to calibrate on")
    arrays = [np.asarray(z, dtype="float64") for z in z_nulls]
    pieces = [bootstrap_max_stats(z, config, n_replicates=n_replicates,
                                  block_len=block_len, horizon=horizon, seed=seed + i)
              for i, z in enumerate(arrays)]
    max_stats = np.concatenate(pieces)
    shortest = min(z.shape[0] for z in arrays)
    return _result_from_max_stats(
        max_stats, alpha=alpha, horizon=horizon, block_len=block_len,
        n_sources=len(arrays), n_replicates_per_source=n_replicates,
        n_effective_blocks=sum(z.shape[0] // max(1, block_len) for z in arrays),
        horizon_ratio=float(horizon) / float(shortest), grid=grid)
