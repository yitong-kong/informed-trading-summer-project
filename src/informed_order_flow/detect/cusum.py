# -*- coding: utf-8 -*-
"""Two-sided GLR-CUSUM in the W (log-likelihood-ratio) form.

The detector runs on a standardized bucket-level feature ``z`` (see
``features.standardize``). For a Gaussian mean-shift of size ``delta`` the
single-step LLR is ``delta * z - delta**2 / 2``; the CUSUM recursion is the
reflected random walk

    W_b(delta) = max(0, W_{b-1}(delta) + delta * z_b - delta**2 / 2).

Because the post-change shift size is unknown, we run a small grid of
``deltas`` and take the running maximum *in the W form* (LLR / nats units --
NOT the ``C = W / delta`` form, which would select a different delta across the
grid). The negative side mirrors the positive side with ``-delta * z_b``. The
statistic is

    G_b = max over delta of max(W_b^+(delta), W_b^-(delta)),

and the detector alarms at the first bucket where ``G_b >= threshold``.

On an alarm the detector also estimates where the flagged episode *started*:
because every ``W`` is a reflected walk that returns to exactly 0 between
excursions, the natural change-point estimate is the start of the final
excursion -- one past the last bucket where the statistic was 0 before the
alarm (Page's estimator). Two variants are reported:

- ``onset_index``     : last zero of the alarm-direction max-over-deltas series.
                        The max is 0 only when *every* delta's W is 0, so this is
                        the earliest excursion start across the grid -- the
                        conservative (widest) window start, used as the official
                        alarm-window start ``[onset, alarm]`` for the wallet audit.
- ``onset_index_mle`` : last zero of the winning delta's own W path -- the
                        classical change-point MLE, never earlier than
                        ``onset_index``; a tighter diagnostic for triage.

This module is pure NumPy and has no I/O; calibration and evaluation live in
sibling modules.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Default GLR mean-shift grid (standardized sigma units).
DEFAULT_DELTAS: tuple[float, ...] = (0.5, 0.75, 1.0)


@dataclass(frozen=True)
class CusumConfig:
    """Detector configuration.

    deltas      : Gaussian mean-shift grid the GLR maximizes over.
    threshold   : alarm threshold ``h`` on the running statistic ``G_b``.
    two_sided   : run both the positive and negative side. The HHI channel
                  (concentration only ever *rises* under H1) uses
                  ``two_sided=False`` -- a one-sided positive CUSUM.
    start_index : first bucket the detector is allowed to monitor. Buckets
                  before it are **not** accumulated and cannot alarm, and the
                  ``W`` state is 0 on entry at ``start_index``. This is what
                  makes the detector online-implementable: the standardization
                  of bucket ``b`` must only use data available at ``b``, so a
                  method whose baseline is estimated from the first ``n_burn``
                  buckets cannot monitor any bucket before ``n_burn``. Plain
                  CUSUM passes ``n_burn``; the windowed method passes its
                  ``min_ref`` warm-up (which it previously enforced only
                  implicitly, via ``z = 0``).
    """

    deltas: tuple[float, ...] = DEFAULT_DELTAS
    threshold: float = 5.0
    two_sided: bool = True
    start_index: int = 0

    def __post_init__(self) -> None:
        if not self.deltas:
            raise ValueError("deltas must be non-empty")
        if any(d <= 0 for d in self.deltas):
            raise ValueError("every delta must be > 0")
        if self.start_index < 0:
            raise ValueError("start_index must be >= 0")


@dataclass
class CusumResult:
    """Output of one CUSUM pass over a standardized series.

    stat            : running statistic ``G_b`` per bucket (length == len(z)).
    pos             : per-bucket max over deltas of the positive-side W.
    neg             : per-bucket max over deltas of the negative-side W (zeros
                      if one-sided).
    alarm_index     : index of the first bucket with ``G_b >= threshold`` (or
                      ``None`` if the series never crosses).
    direction       : +1 if the positive side drove the alarm, -1 if the
                      negative side did (``None`` when there is no alarm).
    onset_index     : alarm-window start -- one past the last zero of the
                      alarm-direction max-over-deltas series before the alarm
                      (0 if it never touched zero, a baseline-contamination
                      warning sign). ``None`` when there is no alarm.
    onset_index_mle : same last-zero rule on the winning delta's own W path
                      (the classical change-point MLE); always
                      ``>= onset_index``. ``None`` when there is no alarm.
    winning_delta   : the grid delta whose W was largest at the alarm bucket
                      (``None`` when there is no alarm).
    pos_paths       : the individual ``W^+(delta)`` paths, shape
                      ``(n_buckets, n_deltas)``; ``pos`` is their row-wise max.
                      Attribution decomposes the *winning delta's own* path, so
                      it needs the un-maxed paths, not just their envelope.
    neg_paths       : same for the negative side (all zeros if one-sided).
    deltas          : the grid, so a path column can be mapped back to its delta.
    """

    stat: np.ndarray
    pos: np.ndarray
    neg: np.ndarray
    alarm_index: int | None
    direction: int | None
    onset_index: int | None = None
    onset_index_mle: int | None = None
    winning_delta: float | None = None
    pos_paths: np.ndarray | None = None
    neg_paths: np.ndarray | None = None
    deltas: tuple[float, ...] = ()

    def winning_path(self) -> np.ndarray | None:
        """The alarm side's ``W(delta*)`` path -- the series attribution splits.

        ``W`` is a reflected walk that is exactly 0 between excursions, so on
        ``[onset_index_mle, alarm_index]`` this path is a plain running sum of
        the single-step LLRs and ``W[alarm_index]`` is exactly their total.
        """
        if self.alarm_index is None or self.winning_delta is None:
            return None
        paths = self.pos_paths if self.direction == 1 else self.neg_paths
        if paths is None:
            return None
        return paths[:, self.deltas.index(self.winning_delta)]

    @property
    def alarmed(self) -> bool:
        return self.alarm_index is not None

    @property
    def max_stat(self) -> float:
        return float(self.stat.max()) if self.stat.size else 0.0


def run_gaussian_cusum(z: np.ndarray, config: CusumConfig) -> CusumResult:
    """Run the two-sided GLR-CUSUM recursion over standardized series ``z``.

    Returns the running statistic and the first threshold crossing. The pass is
    always run to the end of the series so callers can re-threshold the same
    ``stat`` array (used by the calibrator) without re-running the recursion.
    """
    z = np.asarray(z, dtype="float64")
    n = z.shape[0]
    deltas = np.asarray(config.deltas, dtype="float64")

    # Per-delta W paths, shape (n, n_deltas): needed to trace the winning
    # delta's excursion back to its last zero (the onset MLE).
    pos_paths = np.zeros((n, deltas.shape[0]))
    neg_paths = np.zeros((n, deltas.shape[0]))
    w_pos = np.zeros(deltas.shape[0])  # W^+(delta) carried across buckets
    w_neg = np.zeros(deltas.shape[0])
    half = 0.5 * deltas * deltas  # the -delta^2/2 drift term, per delta

    # Buckets before start_index are the baseline/warm-up period: they are not
    # accumulated (their W paths stay 0), so the state entering start_index is
    # exactly 0 and no pre-start_index bucket can trigger an alarm.
    start = min(config.start_index, n)
    for b in range(start, n):
        w_pos = np.maximum(0.0, w_pos + deltas * z[b] - half)
        pos_paths[b] = w_pos
        if config.two_sided:
            w_neg = np.maximum(0.0, w_neg - deltas * z[b] - half)
            neg_paths[b] = w_neg

    pos = pos_paths.max(axis=1) if n else np.zeros(0)
    neg = neg_paths.max(axis=1) if n else np.zeros(0)
    stat = np.maximum(pos, neg)
    alarm_index = first_crossing(stat, config.threshold, start=start)

    direction: int | None = None
    onset: int | None = None
    onset_mle: int | None = None
    winning_delta: float | None = None
    if alarm_index is not None:
        direction = 1 if pos[alarm_index] >= neg[alarm_index] else -1
        side = pos if direction == 1 else neg
        side_paths = pos_paths if direction == 1 else neg_paths
        onset = onset_from_last_zero(side, alarm_index)
        j = int(np.argmax(side_paths[alarm_index]))
        winning_delta = float(deltas[j])
        onset_mle = onset_from_last_zero(side_paths[:, j], alarm_index)
    return CusumResult(stat=stat, pos=pos, neg=neg,
                       alarm_index=alarm_index, direction=direction,
                       onset_index=onset, onset_index_mle=onset_mle,
                       winning_delta=winning_delta,
                       pos_paths=pos_paths, neg_paths=neg_paths,
                       deltas=tuple(float(d) for d in deltas))


def onset_from_last_zero(series: np.ndarray, alarm_index: int) -> int:
    """Excursion start: one past the last zero strictly before the alarm.

    The reflected W hits *exactly* 0 between excursions (``max(0, .)``), so no
    tolerance is needed. If the series never touched zero the excursion runs
    from the very first bucket (index 0) -- with a burn-in baseline that means
    the deviation started inside the burn-in window, which callers should flag.
    """
    zeros = np.flatnonzero(np.asarray(series[:alarm_index]) == 0.0)
    onset = int(zeros[-1]) + 1 if zeros.size else 0
    return min(onset, alarm_index)


def first_crossing(stat: np.ndarray, threshold: float, start: int = 0) -> int | None:
    """First index ``>= start`` where ``stat >= threshold``; ``None`` if never.

    ``start`` makes the monitoring ban explicit rather than relying on the
    pre-``start`` statistic being 0 (which a non-positive threshold would
    otherwise cross immediately).
    """
    hit = np.flatnonzero(np.asarray(stat)[start:] >= threshold)
    return int(hit[0]) + start if hit.size else None
