# -*- coding: utf-8 -*-
"""Step 6 of Q2: count the control worlds and turn the counts into p-values.

Every wallet-window pair gets

    p = (1 + #{T_r >= T_obs}) / (B + 1),   B = 1,999,999

built from B relabelled worlds in which nothing but the wallet names moved.
Three details in that formula are load-bearing. The plus one on both sides means
a p-value is never reported as zero: 2,000,000 finite draws cannot establish an
impossible event, and 1 / 2,000,000 is as small as this design can speak. Ties
count as exceedances (``>=``), which is the conservative direction -- a wallet
that merely equals its own observed score in a control world is not evidence
against the null. And a wallet with no movable slot scores identically in every
world, so its count is B and its p is exactly 1, which is the honest answer
rather than a missing value.

Magnitude and direction are the two headline legs. DNC and DFA remain
sensitivities. All four statistics are read off the same relabelled worlds, so
each extra statistic costs one weighted sum rather than another permutation.

Monte Carlo error is not ignorable at one seed. At the confirmatory first
threshold the expected exceedance count is a few hundred, so the relative error
of p there is several per cent. Every eligible pair therefore reports

    mc_sigma_leg = (p_raw_leg - holm_threshold_leg) / sqrt(p (1 - p) / B)

against the first Holm threshold ``alpha_leg / m_confirmatory`` -- the most
demanding one in the sequence, and within a fraction of a per cent of every
later one. A pair inside three standard errors of it is flagged for a
second-seed review of its window. The rule is written down before any p-value
is looked at, which is the only thing that makes it a rule.

Windows are independent: each cell draws from its own stream, addressed by
``(window_id, cell_id)``, so running windows in parallel, in any order, on any
number of workers, gives bit-identical counts to a single sequential pass.
"""
from __future__ import annotations

import hashlib
import time
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd

from . import permute
from .plan import CONFIG, persist_report, sha256_file

B = int(CONFIG["permutation"]["B"])
DENOMINATOR = int(CONFIG["permutation"]["p_denominator"])
SIGMA_TRIGGER = float(CONFIG["permutation"]["mc_review"]["second_seed_trigger_sigma"])
NUMPY_VERSION = str(CONFIG["numerics"]["numpy_version"])
WEIGHT_SUFFIX = {"dnc": "dnc", "dfa": "dfa",
                 "score_vdw": "mag", "score_sign": "dir"}
WEIGHTS = tuple(WEIGHT_SUFFIX)
if WEIGHTS != permute.WEIGHT_COLUMNS:
    raise AssertionError("p-value weights and permutation-window weights drifted")
ALPHA_LEG = (Fraction(str(CONFIG["multiplicity"]["alpha"]))
             * Fraction(str(CONFIG["statistics"]["alpha_split"])))

P_COLUMNS = [column for suffix in WEIGHT_SUFFIX.values()
             for column in (f"n_exceed_{suffix}", f"p_raw_{suffix}")]
P_COLUMNS += ["permutation_draws", "p_dir_floor", "p_dir_floor_log10",
              "dir_reachable", "mc_threshold_mag", "mc_sigma_mag",
              "mc_threshold_dir", "mc_sigma_dir", "mc_review_required"]
DROP_P_COLUMNS = P_COLUMNS + ["mc_threshold", "mc_sigma_to_threshold"]


def window_counts(window: permute.Window, draws: int = B,
                  batch: int = permute.BATCH_SIZE) -> pd.DataFrame:
    """Exceedance counts for one window, one row per wallet."""
    weights = {name: window.weights[name].to_numpy() for name in WEIGHTS}
    counts = permute.exceedance(window, weights, draws, batch)
    floors = permute.direction_floors(window)
    data = {"window_id": window.window_id, "active_wallet": window.wallets,
            **{f"n_exceed_{WEIGHT_SUFFIX[name]}": counts[name] for name in WEIGHTS}}
    for column in floors:
        data[column] = floors[column].to_numpy()
    return pd.DataFrame(data).reset_index(drop=True)


def all_counts(windows: dict[str, permute.Window], draws: int = B,
               batch: int = permute.BATCH_SIZE, workers: int = 1) -> pd.DataFrame:
    """Every window's counts, in one process or several.

    Parallelism is by window and changes nothing: a window's draws come from its
    own per-cell streams, so the result does not depend on which worker ran it
    or in what order the windows finished.
    """
    order = sorted(windows)
    if workers <= 1:
        frames = [window_counts(windows[name], draws, batch) for name in order]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            frames = list(pool.map(window_counts, [windows[name] for name in order],
                                   [draws] * len(order), [batch] * len(order)))
    return pd.concat(frames, ignore_index=True)


def p_values(counts: pd.DataFrame, draws: int = B) -> pd.DataFrame:
    """Plus-one p-values; never zero, never below ``1 / (draws + 1)``."""
    out = counts.copy()
    for suffix in WEIGHT_SUFFIX.values():
        out[f"p_raw_{suffix}"] = (1 + out[f"n_exceed_{suffix}"]) / (draws + 1)
    out["permutation_draws"] = draws
    return out


def mc_sigma(p_raw: np.ndarray, threshold: np.ndarray, draws: int) -> np.ndarray:
    """Distance from the Holm threshold in Monte Carlo standard errors.

    A pair at ``p = 1`` has no Monte Carlo error left -- every draw tied or beat
    it -- so it sits infinitely far from any threshold and never asks for a
    second seed.
    """
    error = np.sqrt(p_raw * (1.0 - p_raw) / draws)
    return np.divide(p_raw - threshold, error, out=np.full(np.shape(p_raw), np.inf),
                     where=error > 0)


def attach(rows: pd.DataFrame, counts: pd.DataFrame, draws: int = B) -> pd.DataFrame:
    """Merge the p-values onto the wallet-window rows and flag borderline pairs."""
    out = rows.drop(columns=DROP_P_COLUMNS, errors="ignore").merge(
        p_values(counts, draws), on=["window_id", "active_wallet"], how="left")
    roster = out["in_mle_roster"].to_numpy()
    for suffix in WEIGHT_SUFFIX.values():
        if out.loc[roster, f"p_raw_{suffix}"].isna().any():
            raise AssertionError(f"a roster wallet came back without p_raw_{suffix}")

    # Before Holm has an ordering, each leg's first threshold is the only
    # threshold that exists. Step 7 later replaces it with the threshold that
    # the corresponding leg's decision actually used.
    eligible = out["confirmatory_eligible"].fillna(False).to_numpy(dtype=bool)
    threshold = np.where(eligible,
                         out["holm_first_threshold_confirmatory"].to_numpy(), np.nan)
    exact_floor = out.pop("_p_dir_floor_exact").to_numpy()
    reachable = np.zeros(len(out), dtype=bool)
    reachable[eligible] = [floor <= ALPHA_LEG / int(m) for floor, m in zip(
        exact_floor[eligible], out.loc[eligible, "m_confirmatory"])]
    out["dir_reachable"] = pd.array(
        np.where(eligible, reachable, None), dtype="boolean")
    return review(out, {"mag": threshold, "dir": threshold})


def review(rows: pd.DataFrame, thresholds: dict[str, np.ndarray]) -> pd.DataFrame:
    """Store each headline leg's threshold, MC distance and joint trigger."""
    out = rows.copy()
    draws = out["permutation_draws"].to_numpy()
    trigger = np.zeros(len(out), dtype=bool)
    measurable_any = np.zeros(len(out), dtype=bool)
    for suffix in ("mag", "dir"):
        threshold = np.asarray(thresholds[suffix], dtype="float64")
        measurable = np.isfinite(threshold)
        sigma = np.full(len(out), np.nan)
        sigma[measurable] = mc_sigma(out.loc[measurable, f"p_raw_{suffix}"].to_numpy(),
                                     threshold[measurable], draws[measurable])
        out[f"mc_threshold_{suffix}"] = threshold
        out[f"mc_sigma_{suffix}"] = sigma
        trigger |= measurable & (np.abs(sigma) < SIGMA_TRIGGER)
        measurable_any |= measurable
    out["mc_review_required"] = pd.array(
        np.where(measurable_any, trigger, None), dtype="boolean")
    return out


# --------------------------------------------------------------------- gate 6
def check_p_values(rows: pd.DataFrame, draws: int = B) -> list[str]:
    """A p-value is a probability, never zero, and never finer than the grid."""
    roster = rows[rows["in_mle_roster"]]
    failures = []
    for suffix in WEIGHT_SUFFIX.values():
        p, counts = roster[f"p_raw_{suffix}"], roster[f"n_exceed_{suffix}"]
        if (p <= 0).any():
            failures.append(f"p_raw_{suffix} reached zero")
        if (p > 1.0).any() or (counts < 0).any() or (counts > draws).any():
            failures.append(f"p_raw_{suffix} or its count left the valid range")
        if (p < 1.0 / (draws + 1) - 1e-15).any():
            failures.append(f"p_raw_{suffix} below the Monte Carlo grid 1 / (B + 1)")
        rebuilt = (1 + counts) / (draws + 1)
        if (rebuilt - p).abs().max() > 0:
            failures.append(f"p_raw_{suffix} is not (1 + exceedances) / (B + 1)")
    stuck = roster[roster["no_movable_slots"]]
    for suffix in WEIGHT_SUFFIX.values():
        if len(stuck) and not (stuck[f"p_raw_{suffix}"] == 1.0).all():
            failures.append(f"a wallet with no movable slot did not get p_raw_{suffix} = 1")
    if (not np.isfinite(roster["p_dir_floor_log10"]).all()
            or (roster["p_dir_floor_log10"] > 0).any()
            or (roster["p_dir_floor"] < 0).any()
            or (roster["p_dir_floor"] > 1).any()):
        failures.append("a readable direction floor is outside [0, 1]")
    if (roster["p_dir_floor_log10"] + 1e-12
            < roster["p_orbit_floor_log10"]).any():
        failures.append("a direction floor fell below the unrestricted orbit floor")
    eligible = roster[roster["confirmatory_eligible"].astype(bool)]
    log_reachable = (eligible["p_dir_floor_log10"].to_numpy()
                     <= np.log10(eligible["holm_first_threshold_confirmatory"].to_numpy()))
    if not np.array_equal(log_reachable, eligible["dir_reachable"].to_numpy(dtype=bool)):
        failures.append("dir_reachable disagrees with the stored direction floor")
    if roster.loc[~roster["confirmatory_eligible"].astype(bool),
                  "dir_reachable"].notna().any():
        failures.append("a screening-only pair was given confirmatory dir_reachable")
    if rows.loc[~rows["in_mle_roster"], P_COLUMNS].notna().any().any():
        failures.append("a context-only wallet was given a p-value")
    return failures


# --------------------------------------------------------------------- gate 7
def check_reproducible(window: permute.Window, draws: int = 2048) -> list[str]:
    """Gate 7 on one window: batching, worker count and seed all change nothing."""
    reference = window_counts(window, draws, batch=permute.BATCH_SIZE)
    failures = []
    for batch in (1, 128, 512, 1024):        # row by row, 4 x 128, the frozen 512
        again = window_counts(window, draws, batch=batch)
        if not reference.equals(again):
            failures.append(f"{window.window_id}: batch {batch} changed the counts")
    if not reference.equals(window_counts(window, draws, batch=permute.BATCH_SIZE)):
        failures.append(f"{window.window_id}: the same seed did not reproduce")
    return failures


def check_parallel_matches_sequential(windows: dict[str, permute.Window],
                                      draws: int = 2048,
                                      workers: int = 4) -> list[str]:
    """Running windows across processes must be bit-identical to one pass."""
    names = sorted(windows)[:min(4, len(windows))]
    sample = {name: windows[name] for name in names}
    sequential = all_counts(sample, draws, workers=1)
    parallel = all_counts(sample, draws, workers=workers)
    return ([] if sequential.equals(parallel)
            else [f"parallel counts differ from sequential over {len(sample)} windows"])


def dnc_vector_sha256(rows: pd.DataFrame) -> tuple[str, int]:
    """Digest of every roster pair's frozen DNC count and p-value, in key order."""
    key = ["window_id", "active_wallet"]
    current = (rows.loc[rows["in_mle_roster"], key + ["n_exceed_dnc", "p_raw_dnc"]]
               .sort_values(key).reset_index(drop=True))
    counts = current["n_exceed_dnc"].to_numpy(dtype="int64")
    payload = "\n".join(
        f"{window}|{wallet}|{count}|{p:.17g}"
        for window, wallet, count, p in current.assign(n_exceed_dnc=counts)
        .itertuples(index=False, name=None)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest(), len(current)


def frozen_dnc_regression(track: str, rows: pd.DataFrame, draws: int) -> dict:
    """Prove that adding weights did not move one frozen DNC count or p-value.

    The baseline is the digest frozen in the configuration under
    ``expected_counts[track]["dnc_v1_2_0"]``, not a file. It used to be read from
    ``wallet_window_tests.parquet``, which step 9 rewrites: once that step has
    run, such a baseline is the current run's own output and the regression
    compares a run against itself. A digest in the hashed pre-registration
    cannot be overwritten by the pipeline that it audits.
    """
    if draws != B:
        return {"status": "not_applicable", "reason": "the run did not use frozen B"}
    expected = CONFIG["expected_counts"][track]["dnc_v1_2_0"]
    digest, rows_seen = dnc_vector_sha256(rows)
    if rows_seen != int(expected["rows"]):
        raise AssertionError(f"G2 failed: the roster holds {rows_seen} pairs, the "
                             f"frozen baseline {expected['rows']}")
    if digest != str(expected["vector_sha256"]):
        raise AssertionError(
            "G2 failed: the frozen DNC counts or p-values moved. Same seeds, same "
            f"cells, same draws must give {expected['vector_sha256']}, got {digest}")
    return {"status": "bit_equal", "rows": rows_seen, "vector_sha256": digest,
            "baseline": "q2_config.json expected_counts.dnc_v1_2_0"}


# ----------------------------------------------------------------------- entry
def run(repo_root: Path, track: str, workers: int = 1, draws: int = B) -> dict:
    """Draw the control worlds for one track, write the p-values, close gates 6-7."""
    if np.__version__ != NUMPY_VERSION:
        raise AssertionError(f"the permutation null is pinned to numpy {NUMPY_VERSION}, "
                             f"this is {np.__version__}")
    windows, rows, _ = permute.load_windows(repo_root, track)
    smallest = min(windows.values(), key=lambda w: len(w.labels))
    blocked = (check_reproducible(smallest)
               + check_parallel_matches_sequential(windows, workers=max(2, workers)))
    if blocked:
        raise AssertionError(f"gate 7 failed, the run is not reproducible: {blocked}")

    started = time.time()
    counts = all_counts(windows, draws, workers=workers)
    elapsed = time.time() - started

    rows = attach(rows, counts, draws)
    failures = check_p_values(rows, draws)
    if failures:
        raise AssertionError(f"gate 6 failed, the p-values are not valid: {failures}")

    out_dir = repo_root / "data" / "attrib" / track
    dnc_regression = frozen_dnc_regression(track, rows, draws)
    out_path = out_dir / "wallet_windows.parquet"
    rows.to_parquet(out_path, index=False)

    roster = rows[rows["in_mle_roster"]]
    eligible = roster[roster["confirmatory_eligible"].astype(bool)]
    report = {
        "track": track,
        "freeze_build_id": str(rows["freeze_build_id"].iloc[0]),
        "draws": draws, "p_denominator": draws + 1, "seed_base": permute.SEED_BASE,
        "n_seeds": CONFIG["permutation"]["n_seeds"],
        "batch_size": permute.BATCH_SIZE, "workers": workers,
        "checkpointing": CONFIG["permutation"]["checkpointing"],
        "restart_policy": CONFIG["permutation"]["restart_policy"],
        "numpy_version": np.__version__,
        "tie_rule": CONFIG["permutation"]["tie_rule"],
        "windows": len(windows), "pairs": int(len(roster)),
        "elapsed_seconds": round(elapsed, 1),
        "gate_6_failures": [], "gate_7_failures": [],
        "dnc_frozen_regression": dnc_regression,
        **{f"p_raw_{suffix}": {
            "min": float(roster[f"p_raw_{suffix}"].min()),
            "median": float(roster[f"p_raw_{suffix}"].median()),
            "at_grid_floor": int((roster[f"n_exceed_{suffix}"] == 0).sum()),
            "exactly_one": int((roster[f"p_raw_{suffix}"] == 1.0).sum())}
           for suffix in WEIGHT_SUFFIX.values()},
        "p_raw_below_orbit_floor": int(
            (roster["p_raw_dnc"] < roster["p_orbit_floor"]).sum()),
        "p_raw_below_orbit_floor_note": "expected and never corrected: the orbit "
                                        "floor bounds the exact orbit p-value, "
                                        "while a sampled p can undershoot it when "
                                        "B draws happen to miss the ties that the "
                                        "complete orbit contains",
        "confirmatory": {
            "pairs": int(len(eligible)),
            "magnitude": {
                "min_p_raw": float(eligible["p_raw_mag"].min()),
                "below_first_threshold": int((eligible["p_raw_mag"]
                    <= eligible["holm_first_threshold_confirmatory"]).sum())},
            "direction": {
                "min_p_raw": float(eligible["p_raw_dir"].min()),
                "below_first_threshold": int((eligible["p_raw_dir"]
                    <= eligible["holm_first_threshold_confirmatory"]).sum()),
                "structurally_reachable": int(eligible["dir_reachable"].sum()),
                "floor_log10_min": float(eligible["p_dir_floor_log10"].min())},
            "mc_review_required": int(eligible["mc_review_required"].sum()),
            "second_seed_trigger_sigma": SIGMA_TRIGGER},
        "outputs": {"wallet_windows.parquet": sha256_file(out_path)},
    }
    persist_report(out_dir / "pvalue_report.json", report)
    return report
