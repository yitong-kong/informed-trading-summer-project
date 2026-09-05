# -*- coding: utf-8 -*-
"""Run the detector on a contract and evaluate it.

Two evaluation paths, sharing one detector pass:

A fixed-count bucket is only complete when its K-th trade arrives, so an alarm
on bucket ``a`` is not actionable until ``end_ts[a]``. Every timing metric
(simulated ``delay_seconds``, real ``lead_time_to_close_s``) is measured from
that instant, exposed as ``alarm_available_utc``; ``alarm_start_utc`` is kept
purely as the alarm window's boundary.

- **Simulated** streams carry a known ``tau_info``, so we report ground-truth
  metrics: the bucket-level change point ``b_tau``, whether the alarm is genuine
  (fires at or after ``b_tau``), detection delay in buckets and seconds, any
  pre-``tau`` false alarm, and alarm-window quality (``onset_error_buckets`` and
  ``window_covers_tau``) -- these quantify the last-zero onset, which under a
  weak shift can land *after* the true change point; no margin is applied, the
  metrics exist to measure that gap rather than paper over it.
- **Real** streams have no ground truth, so we report the **alarm window**
  ``[onset, alarm]``: the last-zero onset (see ``cusum.onset_from_last_zero``),
  the alarm bucket, both as buckets and UTC timestamps, plus lead time to the
  contract's ``closed_time``, the alarm bucket's ``bucket_duration`` (so the
  lead time is not read as more precise than it is), direction, and the
  statistic vs threshold. The window is the Q2 wallet-audit identification
  scope: wallets first seen *after* the alarm mix with reactive/public-news
  flow and are not screenable by presence alone, so the window ends at the
  alarm bucket; flagged wallets are then examined over their full history.
  ``onset_bucket_mle`` (the winning delta's tighter start) is a triage
  diagnostic, and ``onset_at_stream_start`` / ``onset_in_burn_in`` warn that
  the excursion may predate a clean baseline (contamination risk).

The detector works in event time (bucket indices), which are not interpretable
on their own, so every alarm bucket is also written out as an explicit UTC
wall-clock timestamp: ``alarm_start_iso``/``alarm_end_iso`` on real streams and
``alarm_time_iso`` (with ``tau_info_iso``) on simulated ones. ``to_utc_iso``
does the bucket-time -> calendar-time conversion.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .cusum import CusumConfig, CusumResult, run_gaussian_cusum
from .features import (
    CHANNELS,
    Baseline,
    StandardizedPath,
    WindowConfig,
    channel_series,
    local_standardize_path,
    make_baseline,
    standardize_path,
)

# Detector methods that share the GLR-CUSUM core but differ in how they baseline.
CUSUM = "cusum"                 # plain: fixed per-contract burn-in / robust baseline
WINDOWED_GLR = "windowed_glr"   # trailing-window local baseline (drift-robust)


def to_utc_iso(unix_seconds: int | float | None) -> str | None:
    """Render Unix seconds as an explicit UTC wall-clock string (None passes through).

    The detector runs in event time (bucket indices), so every alarm is also
    written out as a human-readable calendar timestamp -- the bucket index alone
    is not interpretable without it.
    """
    if unix_seconds is None or (isinstance(unix_seconds, float) and np.isnan(unix_seconds)):
        return None
    return pd.Timestamp(int(unix_seconds), unit="s", tz="UTC").strftime("%Y-%m-%d %H:%M:%S UTC")


@dataclass
class DetectorRun:
    """One detector pass on one contract/channel."""

    channel: str
    method: str
    z: np.ndarray
    result: CusumResult
    baseline_label: str
    threshold: float
    path: StandardizedPath | None = None

    @property
    def alarm_bucket(self) -> int | None:
        return self.result.alarm_index


def channel_z(feat_contract: pd.DataFrame, channel: str, *, method: str = CUSUM,
              baseline: Baseline | None = None, baseline_method: str = "burn_in",
              n_burn: int = 20, window: WindowConfig | None = None
              ) -> tuple[np.ndarray, str]:
    """Standardized series the GLR-CUSUM runs on, plus a baseline label.

    ``cusum``        : one fixed (mu0, sigma0) for the whole stream -- a clean
                       control/burn-in window (anti-contamination: mu0 must not
                       be estimated from data that already contains the change).
    ``windowed_glr`` : a trailing local baseline per bucket, which tracks slow
                       non-stationarity and only reacts to recent abrupt shifts.
    """
    path, label = channel_path(feat_contract, channel, method=method, baseline=baseline,
                               baseline_method=baseline_method, n_burn=n_burn,
                               window=window)
    return path.z, label


def channel_path(feat_contract: pd.DataFrame, channel: str, *, method: str = CUSUM,
                 baseline: Baseline | None = None, baseline_method: str = "burn_in",
                 n_burn: int = 20, window: WindowConfig | None = None
                 ) -> tuple[StandardizedPath, str]:
    """``channel_z`` plus the per-bucket ``(mu, sigma)`` that produced ``z``.

    Attribution has to reproduce the detector's own arithmetic bucket by bucket,
    so it consumes this rather than ``z`` alone.
    """
    x = channel_series(feat_contract, channel)
    if method == WINDOWED_GLR:
        window = window or WindowConfig()
        return (local_standardize_path(x, window),
                f"windowed_w{window.ref_window}g{window.gap}")
    if method == CUSUM:
        if baseline is None:
            baseline = make_baseline(x, method=baseline_method, n_burn=n_burn)
        return standardize_path(x, baseline), baseline.method
    raise ValueError(f"unknown detector method {method!r}")


def monitoring_start(method: str, *, n_burn: int = 20,
                     window: WindowConfig | None = None) -> int:
    """First bucket a method may monitor -- its baseline/warm-up requirement.

    Plain CUSUM estimates one baseline from the first ``n_burn`` buckets, so
    bucket ``b < n_burn`` is standardized with data that is not available at
    ``b``; monitoring may only begin once the baseline exists. The windowed
    method needs ``min_ref`` trailing buckets. Both are stated here so the
    calibration and the real/sim runs cannot drift apart.
    """
    if method == WINDOWED_GLR:
        return (window or WindowConfig()).min_ref
    if method == CUSUM:
        return int(n_burn)
    raise ValueError(f"unknown detector method {method!r}")


def run_detector(feat_contract: pd.DataFrame, channel: str, threshold: float, *,
                 deltas: tuple[float, ...] = (0.5, 0.75, 1.0), method: str = CUSUM,
                 baseline: Baseline | None = None, baseline_method: str = "burn_in",
                 n_burn: int = 20, window: WindowConfig | None = None) -> DetectorRun:
    """Standardize a channel's feature (per ``method``) and run the GLR-CUSUM."""
    spec = CHANNELS[channel]
    path, label = channel_path(feat_contract, channel, method=method, baseline=baseline,
                               baseline_method=baseline_method, n_burn=n_burn,
                               window=window)
    cfg = CusumConfig(deltas=tuple(deltas), threshold=threshold,
                      two_sided=spec["two_sided"],
                      start_index=monitoring_start(method, n_burn=n_burn, window=window))
    return DetectorRun(channel=channel, method=method, z=path.z,
                       result=run_gaussian_cusum(path.z, cfg),
                       baseline_label=label, threshold=threshold, path=path)


def tau_bucket(feat_contract: pd.DataFrame, tau_info: int) -> int | None:
    """First bucket whose ``end_ts >= tau_info`` -- the bucket-level change point."""
    end = feat_contract["end_ts"].to_numpy()
    hit = np.flatnonzero(end >= tau_info)
    return int(hit[0]) if hit.size else None


def evaluate_sim(feat_contract: pd.DataFrame, run: DetectorRun, *,
                 tau_info: int | None, injection_mode: str | None,
                 scenario_id: str, level: str) -> dict:
    """Ground-truth metrics for a simulated contract with known ``tau_info``."""
    res = run.result
    b_tau = None if tau_info is None else tau_bucket(feat_contract, tau_info)
    a = res.alarm_index
    is_null = tau_info is None

    detected = (a is not None) and (b_tau is not None) and (a >= b_tau)
    pre_tau = (a is not None) and (b_tau is not None) and (a < b_tau)
    # On a null stream (no tau) any alarm is a false alarm.
    false_alarm = a is not None if is_null else pre_tau

    # A fixed-count bucket is only complete once its K-th trade arrives, so the
    # alarm cannot be acted on before the bucket's end_ts. Timing metrics use
    # that availability instant; start_ts is a window boundary, not a decision
    # time. (Using start_ts also made delay_seconds negative whenever the alarm
    # bucket straddled tau; measured from end_ts it is provably >= 0, because
    # tau_bucket is the first bucket with end_ts >= tau_info.)
    alarm_start = int(feat_contract["start_ts"].iloc[a]) if a is not None else None
    alarm_available = int(feat_contract["end_ts"].iloc[a]) if a is not None else None
    o = res.onset_index
    row = {
        "scenario_id": scenario_id,
        "level": level,
        "method": run.method,
        "channel": run.channel,
        "injection_mode": injection_mode,
        "n_buckets": int(len(feat_contract)),
        "tau_info_utc": tau_info,
        "tau_info_iso": to_utc_iso(tau_info),
        "tau_bucket": b_tau,
        "alarm_bucket": a,
        "alarm_start_utc": alarm_start,          # window boundary
        "alarm_available_utc": alarm_available,  # decision time (bucket end)
        "alarm_time_iso": to_utc_iso(alarm_available),
        "direction": res.direction,
        "detected": bool(detected),
        "false_alarm": bool(false_alarm),
        "delay_buckets": (a - b_tau) if detected else None,
        "delay_seconds": None,
        "onset_bucket": o,
        "onset_bucket_mle": res.onset_index_mle,
        "winning_delta": res.winning_delta,
        # Window quality vs ground truth: signed onset error (positive = the
        # window starts late and misses the earliest informed flow) and whether
        # the window [onset, alarm] contains the true change point at all.
        "onset_error_buckets": ((o - b_tau) if (o is not None and b_tau is not None)
                                else None),
        "window_covers_tau": (bool(o <= b_tau <= a)
                              if (o is not None and b_tau is not None) else None),
        "statistic": (float(res.stat[a]) if a is not None else res.max_stat),
        "threshold": run.threshold,
    }
    if detected:
        row["delay_seconds"] = int(alarm_available - tau_info)
    return row


def evaluate_real(feat_contract: pd.DataFrame, run: DetectorRun, *,
                  condition_id: str, question: str, closed_time: int | None,
                  min_buckets: int, bucket_size: int,
                  n_burn: int | None = None) -> dict:
    """Alarm-window report for a real contract (no ground-truth tau).

    The alarm window is ``[onset_bucket, alarm_bucket]`` (last-zero onset to
    first threshold crossing) -- the Q2 wallet-audit identification scope, in
    both bucket indices and UTC (``window_start_*`` to ``alarm_end_*``).
    ``n_burn`` (plain-CUSUM only) flags an onset inside the burn-in window,
    where the baseline itself may be contaminated by the flagged episode.
    """
    res = run.result
    a = res.alarm_index
    n = int(len(feat_contract))
    row = {
        "condition_id": condition_id,
        "question": question,
        "method": run.method,
        "channel": run.channel,
        "bucket_size": bucket_size,
        "baseline": run.baseline_label,
        "n_buckets": n,
        "shallow": n < min_buckets,
        "alarmed": a is not None,
        "alarm_bucket": a,
        "alarm_start_utc": None,
        "alarm_end_utc": None,
        "alarm_available_utc": None,
        "alarm_start_iso": None,
        "alarm_end_iso": None,
        "bucket_duration_s": None,
        "onset_bucket": res.onset_index,
        "onset_bucket_mle": res.onset_index_mle,
        "winning_delta": res.winning_delta,
        "window_start_utc": None,
        "window_start_iso": None,
        "window_n_buckets": None,
        "window_duration_s": None,
        "onset_at_stream_start": None,
        "onset_in_burn_in": None,
        "direction": res.direction,
        "lead_time_to_close_s": None,
        "statistic": (float(res.stat[a]) if a is not None else res.max_stat),
        "threshold": run.threshold,
        "closed_time_utc": closed_time,
        "closed_time_iso": to_utc_iso(closed_time),
    }
    if a is not None:
        # ``start`` bounds the alarm window; ``end`` is when the fixed-count
        # bucket completes and the alarm can actually be acted on. Lead time is
        # an operational claim, so it must be measured from ``end``.
        start = int(feat_contract["start_ts"].iloc[a])
        end = int(feat_contract["end_ts"].iloc[a])
        row["alarm_start_utc"] = start
        row["alarm_end_utc"] = end
        row["alarm_available_utc"] = end
        row["alarm_start_iso"] = to_utc_iso(start)
        row["alarm_end_iso"] = to_utc_iso(end)
        row["bucket_duration_s"] = int(feat_contract["bucket_duration"].iloc[a])
        o = res.onset_index
        window_start = int(feat_contract["start_ts"].iloc[o])
        row["window_start_utc"] = window_start
        row["window_start_iso"] = to_utc_iso(window_start)
        row["window_n_buckets"] = int(a - o + 1)
        row["window_duration_s"] = int(end - window_start)
        row["onset_at_stream_start"] = bool(o == 0)
        if n_burn is not None and run.method == CUSUM:
            row["onset_in_burn_in"] = bool(o < n_burn)
        if closed_time is not None:
            row["lead_time_to_close_s"] = int(closed_time - end)
    return row
