# -*- coding: utf-8 -*-
"""Tests for the Q1 GLR-CUSUM detector: recursion, alarms, features, calibration."""
import json

import numpy as np
import pandas as pd
import pytest

from informed_order_flow.data.schemas import DATA_DIR, TRADES_EVENT_LEVEL_COLUMNS
from informed_order_flow.detect import (
    FEATURE_COLUMNS,
    CusumConfig,
    DegenerateScale,
    FeatureConfig,
    build_features,
    calibrate_threshold,
    calibrate_threshold_pooled,
    iter_contracts,
    run_detector,
    run_gaussian_cusum,
)
from informed_order_flow.detect.calibrate import empirical_alpha, threshold_from_max_stats
from informed_order_flow.detect.evaluate import tau_bucket
from informed_order_flow.detect.features import standardize

PROCESSED = DATA_DIR / "processed" / "trades_event_level.parquet"
SIM_DIR = DATA_DIR / "sim"

pytestmark = pytest.mark.skipif(
    not PROCESSED.exists(), reason="frozen main table required for detector tests"
)


# ---------------------------------------------------------------- recursion
def test_cusum_recursion_known_sequence():
    """A hand-checked sequence reproduces the W-form recursion exactly."""
    cfg = CusumConfig(deltas=(1.0,), threshold=100.0)  # delta=1 -> W += z - 0.5
    res = run_gaussian_cusum(np.array([2.0, 2.0, 2.0]), cfg)
    np.testing.assert_allclose(res.pos, [1.5, 3.0, 4.5])
    assert not res.alarmed


def test_no_alarm_on_zero_sequence():
    cfg = CusumConfig(threshold=4.0)
    res = run_gaussian_cusum(np.zeros(200), cfg)
    assert not res.alarmed
    assert res.max_stat == 0.0


def test_positive_shift_alarms():
    """Zeros then a sustained +1 shift triggers a positive-side alarm."""
    z = np.concatenate([np.zeros(20), np.ones(40)])
    res = run_gaussian_cusum(z, CusumConfig(threshold=4.0))
    assert res.alarmed
    assert res.alarm_index >= 20      # only after the shift
    assert res.direction == +1


def test_two_sided_negative_shift_alarms():
    z = np.concatenate([np.zeros(20), -np.ones(40)])
    res = run_gaussian_cusum(z, CusumConfig(threshold=4.0, two_sided=True))
    assert res.alarmed
    assert res.direction == -1


def test_windowed_local_standardize_basics():
    """A constant series has zero local drift; the warm-up buckets are z = 0."""
    from informed_order_flow.detect import WindowConfig, local_standardize
    z = local_standardize(np.full(40, 0.3), WindowConfig(ref_window=10, min_ref=5))
    assert np.allclose(z, 0.0)                 # no drift -> no residual anywhere
    z2 = local_standardize(np.arange(40.0), WindowConfig(ref_window=10, min_ref=5))
    assert np.allclose(z2[:5], 0.0)            # first min_ref buckets are warm-up


def test_windowed_glr_suppresses_drift_catches_abrupt():
    """Windowed GLR is quiet on a slow drift but loud on a genuine abrupt shift."""
    from informed_order_flow.detect import WindowConfig, local_standardize
    from informed_order_flow.detect.features import robust_baseline

    def max_stat(z):
        return run_gaussian_cusum(z, CusumConfig(threshold=1e9)).max_stat

    rng = np.random.default_rng(1)
    decay = 1.2 * np.exp(-np.arange(120) / 25) + rng.normal(0, 0.4, 120)   # no change point
    abrupt = np.concatenate([np.zeros(60), np.full(60, 1.5)]) + rng.normal(0, 0.4, 120)
    w = WindowConfig(ref_window=20, min_ref=8)

    plain_decay = max_stat(standardize(decay, robust_baseline(decay)))
    win_decay = max_stat(local_standardize(decay, w))
    assert win_decay < plain_decay              # the launch-hype drift is suppressed
    assert max_stat(local_standardize(abrupt, w)) > 8.0   # the real shift still fires


def test_one_sided_ignores_negative_shift():
    """A one-sided positive CUSUM (the HHI channel) never alarms on a drop."""
    z = np.concatenate([np.zeros(20), -np.ones(80)])
    res = run_gaussian_cusum(z, CusumConfig(threshold=4.0, two_sided=False))
    assert not res.alarmed


# ---------------------------------------------------------------- alarm window
def test_onset_last_zero_known_shift():
    """The window start is the first bucket of the final excursion.

    z is exactly zero through bucket 19 (every W stays at 0), then a sustained
    +1 shift: the last zero is bucket 19, so the onset is bucket 20 -- the
    first bucket accumulating evidence.
    """
    z = np.concatenate([np.zeros(20), np.ones(60)])
    res = run_gaussian_cusum(z, CusumConfig(threshold=4.0))
    assert res.alarmed
    assert res.onset_index == 20
    assert res.onset_index <= res.onset_index_mle <= res.alarm_index
    assert res.winning_delta == 1.0     # steepest LLR slope on a unit shift


def test_onset_excludes_earlier_excursion_that_reset():
    """A noise blip that returns to zero is excluded from the alarm window."""
    z = np.concatenate([np.zeros(5), np.full(3, 2.0),     # blip at 5..7
                        np.full(6, -1.0),                 # pulls every W+ back to 0
                        np.ones(60)])                     # the real shift at 14
    res = run_gaussian_cusum(z, CusumConfig(threshold=6.0))
    assert res.alarmed and res.direction == +1
    assert res.onset_index == 14        # the blip stays outside the window


def test_onset_at_stream_start_when_never_zero():
    """A W that never touches zero yields onset 0 (contamination warning case)."""
    res = run_gaussian_cusum(np.ones(40), CusumConfig(threshold=4.0))
    assert res.alarmed
    assert res.onset_index == 0


def test_onset_none_without_alarm():
    res = run_gaussian_cusum(np.zeros(50), CusumConfig(threshold=4.0))
    assert res.onset_index is None
    assert res.onset_index_mle is None
    assert res.winning_delta is None


# ---------------------------------------------------------------- features
def test_feature_schema_matches_real_and_sim():
    """The same feature builder runs on the real table and a sim dataset."""
    fcfg = FeatureConfig(bucket_size=100)
    real = build_features(pd.read_parquet(PROCESSED), fcfg)
    assert list(real.columns) == FEATURE_COLUMNS
    assert real["imbalance"].between(-1, 1).all()
    assert real["wallet_hhi"].between(0, 1).all()
    assert real["max_trade_share"].between(0, 1).all()

    sim_dirs = [d for d in SIM_DIR.iterdir() if (d / "trades_event_level.parquet").exists()]
    if sim_dirs:
        sim = build_features(pd.read_parquet(sim_dirs[0] / "trades_event_level.parquet"), fcfg)
        assert list(sim.columns) == FEATURE_COLUMNS


def test_bucket_boundaries_are_consistent():
    fcfg = FeatureConfig(bucket_size=100)
    feat = build_features(pd.read_parquet(PROCESSED), fcfg)
    assert (feat["end_ts"] >= feat["start_ts"]).all()
    assert (feat["bucket_duration"] == feat["end_ts"] - feat["start_ts"]).all()
    assert (feat["n_trades"] > 0).all()


# ---------------------------------------------------------------- burn-in gating
def test_start_index_bans_monitoring_and_resets_state():
    """No accumulation or alarm before ``start_index``; state is 0 on entry.

    The burn-in buckets are standardized with a baseline that does not exist yet
    at those buckets, so an alarm there is not online-implementable. A huge
    pre-start excursion must therefore leave no trace in the statistic.
    """
    z = np.concatenate([np.full(20, 10.0), np.zeros(30)])
    gated = run_gaussian_cusum(z, CusumConfig(threshold=5.0, start_index=20))
    assert gated.alarm_index is None, "pre-start_index buckets must not alarm"
    assert np.all(gated.stat[:20] == 0.0), "pre-start_index state must stay 0"
    assert gated.max_stat == 0.0, "the excursion before start_index is discarded"
    # Without the gate the very same series alarms immediately -- i.e. the gate
    # is what removes the look-ahead, not a property of the data.
    assert run_gaussian_cusum(z, CusumConfig(threshold=5.0)).alarm_index == 0


def test_start_index_onset_never_precedes_it():
    """An excursion running from start_index reports onset == start_index."""
    z = np.concatenate([np.zeros(10), np.full(20, 3.0)])
    res = run_gaussian_cusum(z, CusumConfig(threshold=5.0, start_index=10))
    assert res.alarm_index is not None
    assert res.onset_index >= 10


# ---------------------------------------------------------------- calibration
def test_calibration_controls_false_alarm():
    """h* calibrated on an i.i.d. N(0,1) null keeps empirical alpha at/below target."""
    rng = np.random.default_rng(0)
    z_null = rng.standard_normal(400)
    cal = calibrate_threshold(z_null, CusumConfig(), alpha=0.05, horizon=200,
                              n_replicates=300, block_len=10, seed=1)
    assert cal.threshold > 0
    assert cal.alpha_achieved <= 0.05 + 1e-9


def test_calibration_uses_the_deployed_monitoring_ban():
    """The null distribution must be generated under the detector's ``start_index``.

    Banning the first ``start_index`` buckets removes their crossings from every
    replicate, so a calibration that ignored the ban would not describe the
    deployed detector. Same null, same seed, ban on vs off must differ.
    """
    rng = np.random.default_rng(3)
    z_null = rng.standard_normal(300)
    free = calibrate_threshold(z_null, CusumConfig(start_index=0), alpha=0.05,
                               horizon=200, n_replicates=300, seed=1)
    banned = calibrate_threshold(z_null, CusumConfig(start_index=20), alpha=0.05,
                                 horizon=200, n_replicates=300, seed=1)
    assert banned.threshold <= free.threshold


def test_threshold_is_conservative_under_heavy_ties():
    """With ties at the quantile, h* must still satisfy mean(stat >= h) <= alpha.

    A ``1 - alpha`` quantile fails here: it lands *on* the tied value, so the
    achieved alpha jumps to the whole tie mass. The threshold search steps past
    the tie instead.
    """
    max_stats = np.concatenate([np.zeros(80), np.full(20, 3.0)])  # 20% tied at 3.0
    h = threshold_from_max_stats(max_stats, alpha=0.05)
    assert empirical_alpha(max_stats, h) <= 0.05
    assert empirical_alpha(max_stats, float(np.quantile(max_stats, 0.95,
                                                        method="higher"))) > 0.05


def test_calibration_reports_replicate_accounting():
    """Pooled calibration must not report a per-source count as the sample size."""
    rng = np.random.default_rng(1)
    nulls = [rng.standard_normal(120) for _ in range(3)]
    cal = calibrate_threshold_pooled(nulls, CusumConfig(), alpha=0.05, horizon=100,
                                     n_replicates=40, block_len=10, seed=0)
    assert cal.n_sources == 3
    assert cal.n_replicates_per_source == 40
    assert cal.n_replicates_total == 120
    assert cal.n_effective_blocks == 3 * (120 // 10)
    assert cal.horizon_ratio == pytest.approx(100 / 120)


def test_degenerate_null_fails_closed():
    """A constant null carries no null distribution; calibration raises instead of returning 0."""
    with pytest.raises(DegenerateScale):
        calibrate_threshold(np.zeros(200), CusumConfig(), alpha=0.05,
                            horizon=100, n_replicates=50, seed=0)


# ---------------------------------------------------------------- sim H1
def test_sim_h1_alarm_after_tau_not_before(tmp_path):
    """Fixed-seed strong H1 alarms at/after tau, not before.

    Runs the real pipeline end to end: a clean L0 null supplies both the
    standardization baseline and the bootstrap-calibrated threshold; the same
    baseline+threshold are applied to a strong direction-tilt H1. Because the
    baseline comes from the null (not the H1 stream), the post-change mass cannot
    push the pre-change buckets across (the anti-contamination path).
    """
    from informed_order_flow.sim import build_scenario, estimate_baseline
    from informed_order_flow.detect import CusumConfig, calibrate_threshold
    from informed_order_flow.detect.evaluate import evaluate_sim
    from informed_order_flow.detect.features import channel_baseline, channel_series, standardize

    estimate_baseline(out_dir=tmp_path)
    market = {"question": "SIM: test?", "scheduled_end_date": "2026-03-31T12:00:00Z",
              "resolved_outcome": "Yes"}

    null = build_scenario({"scenario_id": "null", "level": "0", "seed": 1,
                           "n_trades": 4000, "p_long_yes": 0.5, "market": market,
                           "injection": None}, out_dir=tmp_path)
    _, _, sub_n = next(iter_contracts(build_features(
        pd.read_parquet(tmp_path / "null" / "trades_event_level.parquet"),
        FeatureConfig(bucket_size=100))))
    baseline = channel_baseline(sub_n, "imbalance", "robust")
    # The detector uses this fixed null-derived baseline, so the null is
    # standardized with the same one before its blocks are resampled.
    cal = calibrate_threshold(standardize(channel_series(sub_n, "imbalance"), baseline),
                              CusumConfig(), alpha=0.05, horizon=len(sub_n),
                              n_replicates=200, block_len=8, seed=0)

    h1 = build_scenario({"scenario_id": "h1", "level": "0", "seed": 3,
                         "n_trades": 4000, "p_long_yes": 0.5, "market": market,
                         "injection": {"mode": "direction_tilt_same_count",
                                       "tau_frac": 0.6, "tilt_frac": 0.9,
                                       "n_wallets": 3}}, out_dir=tmp_path)
    _, _, sub_h = next(iter_contracts(build_features(
        pd.read_parquet(tmp_path / "h1" / "trades_event_level.parquet"),
        FeatureConfig(bucket_size=100))))
    run = run_detector(sub_h, "imbalance", cal.threshold, baseline=baseline)
    row = evaluate_sim(sub_h, run, tau_info=h1["tau_info_utc"],
                       injection_mode="direction_tilt_same_count",
                       scenario_id="h1", level="0")

    assert row["detected"], "strong H1 should be detected post-tau"
    assert not row["false_alarm"], "must not raise a pre-tau false alarm"
    assert run.alarm_bucket >= tau_bucket(sub_h, h1["tau_info_utc"])
    # Alarm-window quality columns: onset precedes the alarm, and the window
    # metrics vs the known tau are populated (coverage is measured, not assumed).
    assert 0 <= row["onset_bucket"] <= row["alarm_bucket"]
    assert row["onset_bucket"] <= row["onset_bucket_mle"] <= row["alarm_bucket"]
    assert row["onset_error_buckets"] is not None
    assert row["window_covers_tau"] is not None
