# -*- coding: utf-8 -*-
"""Bucket-level features for the change-point detector.

The raw per-trade ``signed_yes_size`` is heavy-tailed and must not be fed to a
textbook CUSUM directly. We first aggregate the active trade stream of each
contract into **fixed-count event-time buckets** (every ``K`` trades), then
build the order-flow features the detector runs on.

Per bucket we emit:

- ``imbalance``        : signed YES flow / total absolute flow, in [-1, 1] -- the
                         main detector feature.
- ``imbalance_winsor`` : the same ratio after capping each trade's shares at a
                         within-bucket quantile -- a robustness channel that
                         tells whether an alarm is driven by a few huge trades.
- ``imbalance_cash``   : a ``gross_cash``-weighted variant.
- ``max_trade_share``  : largest single trade's share of the bucket's shares --
                         a diagnostic, not a feature to threshold on.
- ``wallet_hhi``       : Herfindahl-Hirschman concentration of ``gross_cash``
                         across active wallets. This project uses HHI **only**
                         for the concentration channel (no top-k).
- bucket calendar boundaries ``start_ts``/``end_ts`` and ``bucket_duration``,
  needed to translate an event-time alarm back to lead time (and to flag that
  fixed-count buckets can span very uneven calendar durations).

Both the real main table and any simulated dataset share the 14-column schema,
so the same function runs on either with no branching.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Bucket-feature columns emitted by ``build_features`` (HHI only; no top-k).
FEATURE_COLUMNS = [
    "condition_id", "question", "bucket_index",
    "start_ts", "end_ts", "bucket_duration",
    "n_trades", "gross_cash", "abs_signed_yes_size",
    "imbalance", "imbalance_winsor", "imbalance_cash",
    "max_trade_share", "wallet_hhi",
]

# The standardized feature each detector channel monitors, and whether the
# change is one-sided. Concentration can only rise under H1, so HHI is
# one-sided positive on log(HHI); imbalance channels are two-sided.
CHANNELS: dict[str, dict] = {
    "imbalance":        {"column": "imbalance",        "two_sided": True,  "log": False},
    "imbalance_winsor": {"column": "imbalance_winsor", "two_sided": True,  "log": False},
    "imbalance_cash":   {"column": "imbalance_cash",   "two_sided": True,  "log": False},
    "hhi":              {"column": "wallet_hhi",       "two_sided": False, "log": True},
}


@dataclass(frozen=True)
class FeatureConfig:
    """Bucketing knobs.

    bucket_size     : ``K``, trades per fixed-count event-time bucket.
    winsor_quantile : within-bucket shares cap for ``imbalance_winsor``.
    min_buckets     : soft gate -- contracts with fewer buckets are flagged as
                      too shallow to headline (e.g. the shallowest contracts at
                      K=100).
    """

    bucket_size: int = 100
    winsor_quantile: float = 0.95
    min_buckets: int = 20

    def __post_init__(self) -> None:
        if self.bucket_size <= 0:
            raise ValueError("bucket_size must be > 0")
        if not 0.0 < self.winsor_quantile <= 1.0:
            raise ValueError("winsor_quantile must be in (0, 1]")


def build_features(trades: pd.DataFrame, config: FeatureConfig = FeatureConfig()) -> pd.DataFrame:
    """Aggregate a trade table into per-contract, per-bucket features.

    ``trades`` must carry the 14-column main-table schema (real or simulated).
    Each ``condition_id`` is bucketed independently after sorting by
    ``(timestamp, transaction_hash)``. Returns one row per bucket with
    ``FEATURE_COLUMNS``.
    """
    df = assign_buckets(trades, config.bucket_size)
    key = ["condition_id", "bucket_index"]

    shares = df["gross_shares"].to_numpy(dtype="float64")
    sign = np.sign(df["signed_yes_size"].to_numpy(dtype="float64"))
    cap = df.groupby(key)["gross_shares"].transform(lambda s: s.quantile(config.winsor_quantile))
    shares_w = np.minimum(shares, cap.to_numpy(dtype="float64"))
    df["_signed_w"] = sign * shares_w
    df["_abs_w"] = shares_w
    df["_signed_cash"] = sign * df["gross_cash"].to_numpy(dtype="float64")

    g = df.groupby(key, sort=True)
    feat = g.agg(
        question=("question", "first"),
        start_ts=("timestamp", "min"),
        end_ts=("timestamp", "max"),
        n_trades=("timestamp", "size"),
        gross_cash=("gross_cash", "sum"),
        abs_signed_yes_size=("signed_yes_size", lambda s: s.abs().sum()),
        _signed=("signed_yes_size", "sum"),
        _signed_w=("_signed_w", "sum"),
        _abs_w=("_abs_w", "sum"),
        _signed_cash=("_signed_cash", "sum"),
        _max_shares=("gross_shares", "max"),
        _sum_shares=("gross_shares", "sum"),
    )

    feat["bucket_duration"] = feat["end_ts"] - feat["start_ts"]
    # All three ratios are mathematically in [-1, 1]; clip only floating-point dust.
    feat["imbalance"] = (feat["_signed"] / feat["abs_signed_yes_size"]).clip(-1.0, 1.0)
    feat["imbalance_winsor"] = (feat["_signed_w"] / feat["_abs_w"]).clip(-1.0, 1.0)
    feat["imbalance_cash"] = (
        feat["_signed_cash"] / feat["gross_cash"].replace(0.0, np.nan)).clip(-1.0, 1.0)
    feat["max_trade_share"] = feat["_max_shares"] / feat["_sum_shares"]
    feat["wallet_hhi"] = _wallet_hhi(df, key)

    feat = feat.reset_index()
    return feat[FEATURE_COLUMNS].sort_values(key).reset_index(drop=True)


def assign_buckets(trades: pd.DataFrame, bucket_size: int) -> pd.DataFrame:
    """Sort a trade table into detector order and stamp its ``bucket_index``.

    The single source of truth for which trade belongs to which bucket. Trades
    are ordered by ``(condition_id, timestamp, transaction_hash)``: several
    trades can share a one-second timestamp, so the hash is the tie-break that
    makes the order total. Any downstream consumer that needs a bucket's members
    must call this rather than re-deriving membership from the bucket's UTC
    endpoints -- filtering the raw table by ``[start_ts, end_ts]`` can pick up a
    different set whenever a bucket boundary falls inside a shared second.
    """
    df = (trades.sort_values(["condition_id", "timestamp", "transaction_hash"])
                .reset_index(drop=True))
    df["bucket_index"] = df.groupby("condition_id").cumcount() // int(bucket_size)
    return df


def _wallet_hhi(df: pd.DataFrame, key: list[str]) -> pd.Series:
    """Per-bucket HHI of gross_cash across active wallets, indexed by ``key``."""
    by_wallet = df.groupby(key + ["active_wallet"], sort=False)["gross_cash"].sum()
    sq = by_wallet.pow(2).groupby(level=key).sum()
    tot = by_wallet.groupby(level=key).sum().pow(2)
    return (sq / tot.replace(0.0, np.nan)).rename("wallet_hhi")


def iter_contracts(feat: pd.DataFrame):
    """Yield ``(condition_id, question, sub_frame)`` per contract, bucket-ordered."""
    for cid, sub in feat.groupby("condition_id", sort=False):
        sub = sub.sort_values("bucket_index").reset_index(drop=True)
        yield cid, str(sub["question"].iloc[0]), sub


# ---------------------------------------------------------------- baseline
@dataclass(frozen=True)
class Baseline:
    """Center/scale used to standardize a feature into ``z``."""

    mu: float
    sigma: float
    method: str


_MAD_TO_SIGMA = 1.4826  # MAD -> sigma for a Gaussian
_EPS = 1e-9


def robust_baseline(x: np.ndarray) -> Baseline:
    """Pooled robust center/scale: median and 1.4826 * MAD (the default).

    Robust to the heavy-tailed, occasionally-extreme buckets of real prediction
    markets; the matching threshold is calibrated by bootstrap, not normal ARL.
    """
    x = np.asarray(x, dtype="float64")
    x = x[np.isfinite(x)]
    mu = float(np.median(x))
    mad = float(np.median(np.abs(x - mu)))
    return Baseline(mu=mu, sigma=_MAD_TO_SIGMA * mad, method="robust")


def burn_in_baseline(x: np.ndarray, n_burn: int) -> Baseline:
    """Per-market burn-in: robust median / MAD over the first ``n_burn`` buckets.

    Assumes the contract's early window is a clean pre-change period, so the
    center/scale it yields are uncontaminated by any later informed episode.
    Robust (median/MAD) rather than mean/std because even the burn-in window can
    hold a heavy-tailed bucket.
    """
    x = np.asarray(x, dtype="float64")
    head = x[:max(n_burn, 2)]
    head = head[np.isfinite(head)]
    mu = float(np.median(head))
    mad = float(np.median(np.abs(head - mu)))
    return Baseline(mu=mu, sigma=_MAD_TO_SIGMA * mad, method="burn_in")


def make_baseline(x: np.ndarray, method: str = "robust", n_burn: int = 20) -> Baseline:
    if method == "robust":
        return robust_baseline(x)
    if method == "burn_in":
        return burn_in_baseline(x, n_burn)
    raise ValueError(f"unknown baseline method {method!r}")


def standardize(x: np.ndarray, baseline: Baseline) -> np.ndarray:
    """Z = (x - mu0) / (sigma0 + eps), NaNs (empty buckets) mapped to 0."""
    return standardize_path(x, baseline).z


@dataclass(frozen=True)
class StandardizedPath:
    """``z`` together with the per-bucket baseline that produced it.

    The detector itself only needs ``z``, but attribution needs to know, bucket
    by bucket, *which* centre and scale were used -- the same numbers have to be
    reproduced exactly when the alarm statistic is decomposed into per-trade
    contributions. Keeping them here (rather than re-deriving them later) is what
    makes that decomposition auditable.

    imputed          : ``z`` was set to 0 by fiat rather than computed -- an
                       empty/non-finite bucket, or a warm-up bucket with no
                       baseline yet. ``mu``/``sigma`` are NaN there, so a
                       consumer can tell an imputed 0 from a genuine ``z == 0``.
    scale_degenerate : the estimated scale is at or below ``scale_floor`` while
                       the bucket itself is not, i.e. ``z`` is being produced by
                       the ``+ eps`` guard rather than by a real scale. Reported,
                       not raised on: the floor defaults to disabled so this is a
                       diagnostic until a floor is explicitly frozen.
    """

    z: np.ndarray
    mu: np.ndarray
    sigma: np.ndarray
    imputed: np.ndarray
    scale_degenerate: np.ndarray

    def __len__(self) -> int:
        return int(self.z.shape[0])


# Minimum credible scale for a standardized bucket. A `sigma` at or below this
# means `z` is being produced by the `+ eps` guard rather than by a real spread,
# so the bucket cannot be attributed and the run must be marked untestable.
#
# The floor has to be strictly positive to mean anything: at 0.0 the check can
# never fire, and a clean `scale_degenerate` column then proves only that a
# disabled test stayed quiet. It is set to 0.01, two orders of magnitude below
# the smallest scale observed anywhere in the current data (min sigma over every
# bucket of every stream = 0.126, min within any alarm excursion = 0.197), so
# enabling it flags nothing here and changes no result -- but the assertion
# becomes real, and a genuinely degenerate bucket in future data will trip it.
#
# Raising it into the range where it *would* bind is a different decision: that
# changes which buckets are usable and so requires a new feature version and a
# full re-calibration.
SCALE_FLOOR = 0.01


def _degenerate(sigma: np.ndarray, x: np.ndarray, floor: float) -> np.ndarray:
    """Buckets whose scale is not usable at ``floor``.

    A non-finite `sigma` on a bucket that does have an observed value counts as
    degenerate too: the scale is missing, not small.
    """
    if floor <= 0.0:
        return np.zeros(sigma.shape, dtype=bool)
    return np.isfinite(x) & (~np.isfinite(sigma) | (sigma <= floor))


def standardize_path(x: np.ndarray, baseline: Baseline, *,
                     scale_floor: float = SCALE_FLOOR) -> StandardizedPath:
    """``standardize`` with the (constant) baseline it used exposed per bucket."""
    x = np.asarray(x, dtype="float64")
    mu = np.full(x.shape, baseline.mu, dtype="float64")
    sigma = np.full(x.shape, baseline.sigma, dtype="float64")
    z = (x - baseline.mu) / (baseline.sigma + _EPS)
    imputed = ~np.isfinite(z)
    z = np.nan_to_num(z, nan=0.0)
    mu = np.where(imputed, np.nan, mu)
    sigma = np.where(imputed, np.nan, sigma)
    return StandardizedPath(z=z, mu=mu, sigma=sigma, imputed=imputed,
                            scale_degenerate=_degenerate(sigma, x, scale_floor))


@dataclass(frozen=True)
class WindowConfig:
    """Trailing-window local-baseline parameters for the windowed GLR-CUSUM.

    ref_window : number of preceding buckets used to estimate the local center
                 / scale (the longer it is, the slower it adapts, so the more
                 sensitive it is to a slow shift but the less it absorbs drift).
    gap        : buckets between the current bucket and the reference window's
                 end, so an emerging shift does not immediately pollute its own
                 baseline.
    min_ref    : minimum preceding buckets required before any detection (the
                 warm-up period; earlier buckets get z = 0).
    """

    ref_window: int = 30
    gap: int = 0
    min_ref: int = 10

    def __post_init__(self) -> None:
        if self.ref_window < 2:
            raise ValueError("ref_window must be >= 2")
        if self.gap < 0:
            raise ValueError("gap must be >= 0")


def local_standardize(x: np.ndarray, window: WindowConfig = WindowConfig()) -> np.ndarray:
    """Causal local-baseline standardization for the windowed GLR-CUSUM.

    For each bucket ``b`` the center/scale come from a *trailing* robust window
    ``x[b - gap - ref_window : b - gap]`` (strictly past, so no look-ahead),
    giving ``z_b = (x_b - median) / (1.4826 * MAD)``. A slow non-stationary
    drift is tracked by the moving baseline and produces no sustained z, while a
    shift abrupt relative to the recent window spikes z until the window catches
    up -- which is exactly the window-limited detection behaviour. Buckets with
    fewer than ``min_ref`` past observations are warm-up (z = 0).
    """
    return local_standardize_path(x, window).z


def local_standardize_path(x: np.ndarray, window: WindowConfig = WindowConfig(), *,
                           scale_floor: float = SCALE_FLOOR) -> StandardizedPath:
    """``local_standardize`` with the per-bucket trailing baseline exposed.

    Unlike the plain method the centre and scale move with every bucket, so
    attribution cannot reconstruct them from a single pair -- they have to be
    carried alongside ``z``.
    """
    x = np.asarray(x, dtype="float64")
    n = x.shape[0]
    z = np.zeros(n)
    mu_path = np.full(n, np.nan)
    sigma_path = np.full(n, np.nan)
    imputed = np.ones(n, dtype=bool)
    for b in range(n):
        hi = b - window.gap
        lo = max(0, hi - window.ref_window)
        ref = x[lo:hi] if hi > lo else x[:0]
        ref = ref[np.isfinite(ref)]
        if ref.shape[0] < window.min_ref or not np.isfinite(x[b]):
            continue
        mu = float(np.median(ref))
        sigma = _MAD_TO_SIGMA * float(np.median(np.abs(ref - mu)))
        z[b] = (x[b] - mu) / (sigma + _EPS)
        mu_path[b] = mu
        sigma_path[b] = sigma
        imputed[b] = False
    return StandardizedPath(z=z, mu=mu_path, sigma=sigma_path, imputed=imputed,
                            scale_degenerate=_degenerate(sigma_path, x, scale_floor))


def channel_baseline(feat_contract: pd.DataFrame, channel: str,
                     method: str = "robust", n_burn: int = 20) -> Baseline:
    """Estimate a channel's standardization baseline from a (null/control) frame."""
    return make_baseline(channel_series(feat_contract, channel),
                         method=method, n_burn=n_burn)


def channel_series(feat_contract: pd.DataFrame, channel: str) -> np.ndarray:
    """Raw (un-standardized) feature series a channel monitors.

    For the HHI channel the monitored quantity is ``log(HHI)``: concentration
    is multiplicative and only ever rises under H1, so the channel runs a
    one-sided positive CUSUM on the log.
    """
    spec = CHANNELS[channel]
    x = feat_contract[spec["column"]].to_numpy(dtype="float64")
    if spec["log"]:
        x = np.log(np.clip(x, _EPS, None))
    return x
