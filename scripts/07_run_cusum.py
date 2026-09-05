# -*- coding: utf-8 -*-
"""Entry point: run the Q1 GLR-CUSUM change-point detector end to end.

Pipeline (see src/informed_order_flow/detect/README.md for the full design):

    1. build fixed-count event-time bucket features on the real main table
    2. calibrate a finite-horizon false-alarm threshold per (method, channel,
       null level, bucket size K, horizon N) by POOLING bootstrap replicates
       across every H0 null seed of that level (a single-seed threshold does
       not transfer across seeds)
    3. scan the real contracts -> alarm windows [onset, alarm]: the onset is
       the last zero of the alarm-side CUSUM before the crossing (the start of
       the final excursion), so the window is the Q2 wallet-audit scope; real
       contracts are thresholded at the most realistic (Level-2) calibration
    4. evaluate every simulated scenario at its own level's threshold ->
       power / detection delay / window quality (onset error, coverage of
       tau_info) on H1 and false alarm on the H0 nulls, aggregated per
       (method, channel, level, injection mode) into a summary table

The simulated grid (scripts/05_build_simulated_data.py --grid) is 4 seeds x
3 levels x (null + 4 single-mode injections x resolved outcome Yes/No); within
one (level, seed) the null and all H1 scenarios share the same base stream,
tau_info and informed wallets, so mode and outcome effects are paired. The
summary keeps resolved_outcome as its own dimension: a No market's insider
shorts YES, so its imbalance drifts down and must be caught by the two-sided
channel with direction -1 (HHI is direction-free).

Two detectors run side by side, tagged by a ``method`` column, for comparison:
``cusum`` (plain: fixed per-contract burn-in baseline) and ``windowed_glr``
(trailing local baseline, drift-robust). Channels: the main two-sided imbalance
CUSUM plus a one-sided HHI wallet-concentration channel (HHI only -- no top-k).
Writes under data/detect/.

Real contracts standardize against a per-contract burn-in baseline (their early,
pre-change window). Shallow contracts are handled by detect/realrun.py: the
December contract is bucketed finer (K=50) so its burn-in has enough depth, and
the February contract is too shallow to baseline and is excluded for now.

Usage:
    python scripts/07_run_cusum.py
    python scripts/07_run_cusum.py --methods cusum windowed_glr --ref-window 30
    python scripts/07_run_cusum.py --methods cusum --alpha 0.05 --n-burn 20
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from informed_order_flow.data.schemas import DATA_DIR
from informed_order_flow.detect import (
    FeatureConfig,
    build_features,
    calibrate_threshold_pooled,
    iter_contracts,
    run_detector,
)
from informed_order_flow.detect import CUSUM, WINDOWED_GLR, WindowConfig
from informed_order_flow.detect.cusum import CusumConfig
from informed_order_flow.detect.evaluate import (
    channel_z, evaluate_real, evaluate_sim, monitoring_start)
from informed_order_flow.detect.features import CHANNELS
from informed_order_flow.detect.realrun import bucket_size_for, is_excluded

METHODS = (CUSUM, WINDOWED_GLR)

PROCESSED = DATA_DIR / "processed" / "trades_event_level.parquet"
METADATA = DATA_DIR / "interim" / "market_metadata.parquet"
SIM_DIR = DATA_DIR / "sim"
DETECT_DIR = DATA_DIR / "detect"

DEFAULT_CHANNELS = ("imbalance", "hhi")


# ---------------------------------------------------------------- helpers
def _to_unix(value) -> int | None:
    """Parse an ISO / timestamp string into Unix seconds (None if blank)."""
    if value is None or (isinstance(value, float) and np.isnan(value)) or value == "":
        return None
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return int(ts.timestamp())


def _closed_times() -> dict[str, int | None]:
    """condition_id -> closed_time (Unix s) from the real market metadata."""
    meta = pd.read_parquet(METADATA)
    return {row["condition_id"]: _to_unix(row["closed_time"]) for _, row in meta.iterrows()}


def _discover_scenarios() -> list[dict]:
    """Every built simulated scenario under data/sim/ with its manifest."""
    scenarios = []
    for d in sorted(p for p in SIM_DIR.iterdir() if p.is_dir()):
        man = d / "sim_manifest.json"
        trades = d / "trades_event_level.parquet"
        if man.exists() and trades.exists():
            scenarios.append({"dir": d, "manifest": json.loads(man.read_text())})
    return scenarios


def _nulls_by_level(scenarios: list[dict]) -> dict[str, list[dict]]:
    """level -> its H0 null scenarios (the pooled calibration set per level)."""
    out: dict[str, list[dict]] = {}
    for s in scenarios:
        if s["manifest"].get("injection_mode") is None:
            out.setdefault(str(s["manifest"]["level"]), []).append(s)
    if not out:
        raise SystemExit("no H0 null scenario found to calibrate on; build the grid "
                         "with scripts/05_build_simulated_data.py --grid")
    return out


# ---------------------------------------------------------------- stages
def _scenario_contract(s: dict, fcfg: FeatureConfig) -> pd.DataFrame:
    """The (single) synthetic contract's bucket features for a scenario."""
    feat = build_features(pd.read_parquet(s["dir"] / "trades_event_level.parquet"), fcfg)
    _, _, sub = next(iter_contracts(feat))
    return sub


def real_contracts(winsor: float, min_buckets: int):
    """Yield ``(condition_id, question, bucket_size, features)`` per real contract.

    Each contract is bucketed at its own size (shallow contracts use a finer K)
    and excluded contracts are skipped -- see ``detect/realrun.py``.
    """
    trades = pd.read_parquet(PROCESSED)
    pairs = trades[["condition_id", "question"]].drop_duplicates()
    for _, pr in pairs.iterrows():
        question = pr["question"]
        if is_excluded(question):
            continue
        k = bucket_size_for(question)
        sub_trades = trades[trades["condition_id"] == pr["condition_id"]]
        feat = build_features(sub_trades, FeatureConfig(k, winsor, min_buckets))
        _, _, sub = next(iter_contracts(feat))
        yield pr["condition_id"], question, k, sub


def calibrate_for_k(cal_scns: list[dict], bucket_size: int, channels, deltas, *,
                    method: str, window: WindowConfig, alpha: float, horizon: int,
                    baseline_method: str, n_burn: int, n_replicates: int,
                    block_len: int, winsor: float, min_buckets: int) -> dict[str, dict]:
    """Calibrate a finite-horizon threshold per channel at this K, pooling nulls.

    ``cal_scns`` are all H0 null scenarios of one level (different seeds);
    replicates from every seed are pooled into one threshold, so it targets
    alpha across the null ensemble rather than one draw. The windowed method
    has a different null statistic distribution and so earns its own threshold.

    Each null is standardized with the same baseline the production detector
    applies for this ``method``, and blocks of that standardized series are
    resampled. The replicates run under the deployed ``start_index``, so the
    monitoring ban is present in the null distribution too.
    """
    fcfg = FeatureConfig(bucket_size, winsor, min_buckets)
    subs = [_scenario_contract(s, fcfg) for s in cal_scns]
    start = monitoring_start(method, n_burn=n_burn, window=window)
    out = {}
    for ch in channels:
        z_nulls = [channel_z(sub, ch, method=method, baseline_method=baseline_method,
                             n_burn=n_burn, window=window)[0] for sub in subs]
        cfg = CusumConfig(deltas=deltas, two_sided=CHANNELS[ch]["two_sided"],
                          start_index=start)
        cal = calibrate_threshold_pooled(z_nulls, cfg, alpha=alpha, horizon=horizon,
                                         n_replicates=n_replicates,
                                         block_len=block_len, seed=0)
        out[ch] = cal.to_dict()
    return out


def scan_real(subs, methods, channels, calib_for, deltas, *, baseline_method: str,
              n_burn: int, window: WindowConfig, min_buckets: int,
              closed: dict, real_level: str) -> pd.DataFrame:
    """Run every method x channel on every (included) real contract.

    Both detectors share the GLR-CUSUM core; ``cusum`` uses a fixed per-contract
    burn-in baseline, ``windowed_glr`` a trailing local baseline. Each is
    thresholded at its own (method, K, N), calibrated on the most realistic
    null level available (``real_level``).
    """
    rows = []
    for cid, question, k, sub in subs:
        for method in methods:
            cal = calib_for(method, real_level, k, len(sub))
            for ch in channels:
                run = run_detector(sub, ch, cal[ch]["threshold"], deltas=deltas,
                                   method=method, baseline_method=baseline_method,
                                   n_burn=n_burn, window=window)
                rows.append(evaluate_real(sub, run, condition_id=cid, question=question,
                                          closed_time=closed.get(cid),
                                          min_buckets=min_buckets, bucket_size=k,
                                          n_burn=n_burn))
    return pd.DataFrame(rows)


def evaluate_sims(scenarios, methods, channels, calib_for, deltas, *,
                  baseline_method: str, n_burn: int, window: WindowConfig,
                  fcfg: FeatureConfig, default_k: int) -> pd.DataFrame:
    """Run every method x channel on every simulated scenario -> eval table.

    Each scenario is thresholded at its OWN level's pooled calibration, so an
    L0 stream is not judged against an L2 threshold.
    """
    rows = []
    for s in scenarios:
        man = s["manifest"]
        sub = _scenario_contract(s, fcfg)
        # the side the insider bets; None on nulls (their flow is outcome-free)
        outcome = ((man.get("config") or {}).get("market") or {}).get(
            "resolved_outcome") if man.get("injection_mode") else None
        for method in methods:
            cal = calib_for(method, str(man["level"]), default_k, len(sub))
            for ch in channels:
                run = run_detector(sub, ch, cal[ch]["threshold"], deltas=deltas,
                                   method=method, baseline_method=baseline_method,
                                   n_burn=n_burn, window=window)
                row = evaluate_sim(
                    sub, run, tau_info=man.get("tau_info_utc"),
                    injection_mode=man.get("injection_mode"),
                    scenario_id=man["scenario_id"], level=man["level"])
                row["resolved_outcome"] = outcome
                rows.append(row)
    return pd.DataFrame(rows)


def summarize_sim(sim: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the grid verdicts per (method, channel, level, mode, outcome).

    H1 rows report power (detected at/after tau), pre-tau false alarm, median
    detection delay, and the alarm-window quality metrics (median signed onset
    error; share of detections whose window covers the true tau), separately
    per resolved outcome (Yes = drift up, No = drift down) so the two-sided
    channel's direction symmetry is visible. Null rows (outcome-free) report
    the cross-seed false-alarm rate at the pooled threshold -- with 4 seeds
    per level this is a coarse check, not an alpha certification.
    """
    rows = []
    grouped = sim.groupby(
        ["method", "channel", "level", "injection_mode", "resolved_outcome"],
        dropna=False)
    for (method, ch, level, mode, outcome), g in grouped:
        row = {"method": method, "channel": ch, "level": level,
               "mode": "null" if pd.isna(mode) else mode,
               "outcome": None if pd.isna(mode) else outcome,
               "n_scenarios": int(len(g))}
        if pd.isna(mode):
            row.update({"power": None, "false_alarm_rate": float(g["false_alarm"].mean()),
                        "median_delay_buckets": None, "median_onset_error_buckets": None,
                        "window_coverage_rate": None})
        else:
            det = g["detected"]
            delays = g.loc[det, "delay_buckets"].dropna().astype(float)
            onset_err = g.loc[det, "onset_error_buckets"].dropna().astype(float)
            covers = g.loc[det, "window_covers_tau"].dropna()
            row.update({
                "power": float(det.mean()),
                "false_alarm_rate": float(g["false_alarm"].mean()),   # pre-tau
                "median_delay_buckets": (float(delays.median()) if len(delays) else None),
                "median_onset_error_buckets": (float(onset_err.median())
                                               if len(onset_err) else None),
                "window_coverage_rate": (float(covers.mean()) if len(covers) else None),
            })
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["method", "channel", "level", "mode", "outcome"],
        na_position="first").reset_index(drop=True)


# ---------------------------------------------------------------- CLI
def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bucket-size", type=int, default=100,
                    help="K, trades per fixed-count event-time bucket (default: %(default)s)")
    ap.add_argument("--winsor-quantile", type=float, default=0.95,
                    help="within-bucket shares cap for imbalance_winsor (default: %(default)s)")
    ap.add_argument("--min-buckets", type=int, default=20,
                    help="soft gate flagging shallow contracts (default: %(default)s)")
    ap.add_argument("--alpha", type=float, default=0.05,
                    help="target finite-horizon false-alarm rate (default: %(default)s)")
    ap.add_argument("--deltas", type=float, nargs="+", default=[0.5, 0.75, 1.0],
                    help="GLR shift grid (default: 0.5 0.75 1.0)")
    ap.add_argument("--baseline", choices=("burn_in", "robust"), default="burn_in",
                    help="standardization baseline: per-contract burn-in (first "
                         "n_burn buckets) or full robust (default: %(default)s)")
    ap.add_argument("--n-burn", type=int, default=20,
                    help="burn-in window in buckets for the burn_in baseline "
                         "(default: %(default)s)")
    ap.add_argument("--channels", nargs="+", default=list(DEFAULT_CHANNELS),
                    choices=list(CHANNELS), help="detector channels (default: imbalance hhi)")
    ap.add_argument("--methods", nargs="+", default=list(METHODS), choices=list(METHODS),
                    help="detectors to run side by side (default: cusum windowed_glr)")
    ap.add_argument("--ref-window", type=int, default=30,
                    help="windowed_glr: trailing local-baseline window in buckets "
                         "(default: %(default)s)")
    ap.add_argument("--gap", type=int, default=0,
                    help="windowed_glr: buckets between the current bucket and its "
                         "reference window (default: %(default)s)")
    ap.add_argument("--n-replicates", type=int, default=500,
                    help="bootstrap replicates per null seed for calibration "
                         "(default: %(default)s)")
    ap.add_argument("--block-len", type=int, default=10,
                    help="circular block-bootstrap block length in buckets (default: %(default)s)")
    return ap


def main() -> None:
    args = _build_parser().parse_args()
    deltas = tuple(args.deltas)
    channels = tuple(args.channels)
    DETECT_DIR.mkdir(parents=True, exist_ok=True)

    methods = tuple(args.methods)
    window = WindowConfig(ref_window=args.ref_window, gap=args.gap)
    scenarios = _discover_scenarios()
    nulls_by_level = _nulls_by_level(scenarios)
    real_level = max(nulls_by_level)   # most realistic level with nulls, e.g. "2"
    subs = list(real_contracts(args.winsor_quantile, args.min_buckets))

    # Finite-horizon thresholds are calibrated per (method, null level, bucket
    # size K, horizon N): the windowed detector has a different null statistic,
    # each level has its own null distribution, a finer K is noisier, and a
    # longer stream has more chances to cross -- so each earns its own h*.
    # Every fit pools bootstrap replicates across the level's null seeds.
    cache: dict[tuple[str, str, int, int], dict] = {}

    def calib_for(method: str, level: str, k: int, n: int) -> dict:
        key = (method, level, k, n)
        if key not in cache:
            cache[key] = calibrate_for_k(
                nulls_by_level[level], k, channels, deltas, method=method,
                window=window, alpha=args.alpha, horizon=n,
                baseline_method=args.baseline, n_burn=args.n_burn,
                n_replicates=args.n_replicates, block_len=args.block_len,
                winsor=args.winsor_quantile, min_buckets=args.min_buckets)
        return cache[key]

    pooled = {lv: len(ss) for lv, ss in sorted(nulls_by_level.items())}
    print(f"[calibrate] pooled nulls per level={pooled} alpha={args.alpha} "
          f"methods={','.join(methods)} (real contracts use L{real_level})")
    for _cid, question, k, sub in subs:
        for method in methods:
            cal = calib_for(method, real_level, k, len(sub))
            hs = "  ".join(f"{ch}={cal[ch]['threshold']:.2f}" for ch in channels)
            print(f"    {question[:30]:30s} {method:13s} K={k:<3d} N={len(sub):<3d} h*: {hs}")

    print("[real] scanning real contracts for alarm windows")
    real = scan_real(subs, methods, channels, calib_for, deltas,
                     baseline_method=args.baseline, n_burn=args.n_burn, window=window,
                     min_buckets=args.min_buckets, closed=_closed_times(),
                     real_level=real_level)
    real.to_parquet(DETECT_DIR / "cusum_real_alarms.parquet", index=False)
    _print_real(real, methods)

    print(f"[sim] evaluating {len(scenarios)} scenarios (power / delay / false alarm)")
    sim = evaluate_sims(scenarios, methods, channels, calib_for, deltas,
                        baseline_method=args.baseline, n_burn=args.n_burn, window=window,
                        fcfg=FeatureConfig(args.bucket_size, args.winsor_quantile,
                                           args.min_buckets), default_k=args.bucket_size)
    sim.to_parquet(DETECT_DIR / "cusum_sim_eval.parquet", index=False)
    summary = summarize_sim(sim)
    summary.to_csv(DETECT_DIR / "cusum_sim_summary.csv", index=False)
    _print_summary(summary, methods)

    calib_meta = {
        "alpha": args.alpha, "deltas": list(deltas), "baseline": args.baseline,
        "n_burn": args.n_burn, "default_bucket_size": args.bucket_size,
        "window": {"ref_window": args.ref_window, "gap": args.gap},
        "methods": list(methods),
        "real_calibration_level": real_level,
        "nulls_pooled_per_level": {lv: [s["manifest"]["scenario_id"] for s in ss]
                                   for lv, ss in sorted(nulls_by_level.items())},
        "calibrations": {f"{m}_L{lv}_K{k}_N{n}": {"method": m, "level": lv,
                                                  "bucket_size": k, "horizon": n,
                                                  "channels": v}
                         for (m, lv, k, n), v in sorted(cache.items())},
    }
    (DETECT_DIR / "cusum_calibration.json").write_text(
        json.dumps(calib_meta, indent=2, ensure_ascii=False))

    print(f"\nwrote {DETECT_DIR}/cusum_calibration.json, cusum_real_alarms.parquet, "
          "cusum_sim_eval.parquet, cusum_sim_summary.csv")


def _print_real(real: pd.DataFrame, methods) -> None:
    for method in methods:
        print(f"  -- method={method}")
        sel = real[(real["channel"] == "imbalance") & (real["method"] == method)]
        for _, r in sel.iterrows():
            flag = " [shallow]" if r["shallow"] else ""
            if r["alarmed"]:
                lead = r["lead_time_to_close_s"]
                lead_h = f"{lead / 3600:.1f}h" if pd.notna(lead) else "n/a"
                print(f"    {r['question'][:30]:30s} ALARM dir={int(r['direction']):+d} "
                      f"at {r['alarm_start_iso']} (bucket {int(r['alarm_bucket'])}, "
                      f"span {r['bucket_duration_s'] / 3600:.1f}h) "
                      f"lead_to_close={lead_h} stat={r['statistic']:.2f}{flag}")
                onset_flag = (" [onset at stream start]" if r["onset_at_stream_start"]
                              else " [onset in burn-in]" if r["onset_in_burn_in"] else "")
                print(f"    {'':30s} window buckets {int(r['onset_bucket'])}-"
                      f"{int(r['alarm_bucket'])} ({int(r['window_n_buckets'])} buckets, "
                      f"{r['window_duration_s'] / 3600:.1f}h) "
                      f"[{r['window_start_iso']} -> {r['alarm_end_iso']}] "
                      f"mle_onset={int(r['onset_bucket_mle'])} "
                      f"delta*={r['winning_delta']:.2f}{onset_flag}")
            else:
                print(f"    {r['question'][:30]:30s} no alarm (max stat={r['statistic']:.2f})"
                      f"{flag}")


def _print_summary(summary: pd.DataFrame, methods) -> None:
    """Aggregated grid verdicts: one line per (channel, level, mode) per method."""
    def fmt(v, spec="5.2f"):
        return "  n/a" if v is None or pd.isna(v) else f"{v:{spec}}"

    for method in methods:
        print(f"  -- method={method}")
        print(f"    {'channel':10s} {'L':>2s} {'mode':26s} {'out':>3s} {'power':>5s} "
              f"{'FA':>5s} {'delay':>6s} {'onset_err':>9s} {'covers_tau':>10s}")
        sel = summary[summary["method"] == method]
        for _, r in sel.iterrows():
            out = "-" if r["outcome"] is None or pd.isna(r["outcome"]) else r["outcome"]
            print(f"    {r['channel']:10s} {r['level']:>2s} {r['mode']:26s} {out:>3s} "
                  f"{fmt(r['power'])} {fmt(r['false_alarm_rate'])} "
                  f"{fmt(r['median_delay_buckets'], '5.1f'):>6s} "
                  f"{fmt(r['median_onset_error_buckets'], '+5.1f'):>9s} "
                  f"{fmt(r['window_coverage_rate']):>10s}")


if __name__ == "__main__":
    main()
