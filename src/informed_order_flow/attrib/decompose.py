# -*- coding: utf-8 -*-
"""Step 2 of Q2: split the alarm statistic into per-trade contributions.

Only three rules are at work. A trade counts positive when it runs with the
alarm direction and negative when it runs against it; inside a bucket a larger
trade contributes proportionally more; and the more stable Q1 judged that
bucket to be, the more the same displacement counts.

For trade ``i`` in bucket ``b`` of the MLE excursion, with ``q_i`` its
``signed_yes_size``, ``A_b = sum_i |q_i|``, ``d`` the alarm direction,
``delta*`` the winning shift, ``mu_b`` / ``sigma_b`` the baseline Q1 actually
used and ``eps = 1e-9``::

    x_b   = sum_i q_i / A_b
    l_b   = delta* d (x_b - mu_b) / (sigma_b + eps) - delta*^2 / 2
    DNC_i = delta* d q_i / (A_b (sigma_b + eps))
    kappa_b = -delta* d mu_b / (sigma_b + eps) - delta*^2 / 2
    AGC_i = kappa_b |q_i| / A_b
    DFA_i = DNC_i + AGC_i

Two things follow, and both are checked here. Within a bucket ``DNC_i = C_b
q_i``, so DNC is not a pure directional tendency but a joint contribution of
direction agreement and signed size -- a with-the-alarm block trade scores high
and an against-the-alarm block trade scores very negative. And ``DFA`` is an
exactly conserved ledger: it sums to ``l_b`` per bucket and to ``W_alarm`` per
run. That conservation is why DFA is reported as a sensitivity only -- the
``kappa_b`` share can be spread by any weights summing to one without breaking
it, so wallet DFA rankings are not a second headline.

A trade in the wide audit window but outside the MLE excursion gets no
contribution: it is context, not evidence, and its DNC is left null rather than
zero so the two cannot be confused.

The two headline legs use the same frozen MLE slots but not the ledger weights:
``score_vdw`` is the van der Waerden score of ``d q_i`` ranked within
``(detector_run_id, bucket_index)``, and ``score_sign`` is ``sign(d q_i)``.
Neither score enters the conservation identities above.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from ..detect.features import SCALE_FLOOR
from . import freeze
from .plan import CONFIG, dumps, sha256_file

EPS = float(CONFIG["numerics"]["epsilon"])
TOLERANCE = CONFIG["numerics"]["conservation_tolerances"]

CONTRIBUTION_COLUMNS = [
    "abs_flow_bucket", "dnc", "agc", "dfa", "score_vdw", "score_sign",
]


def load_tables(repo_root: Path, track: str) -> dict[str, pd.DataFrame]:
    """Read a track's freeze tables and refuse to mix build ids."""
    out_dir = repo_root / "data" / "attrib" / track
    tables = {name: pd.read_parquet(out_dir / f"{name}.parquet")
              for name in freeze.TABLES}
    stamps = {digest for table in tables.values()
              for digest in table["freeze_build_id"].unique()}
    if len(stamps) != 1:
        raise AssertionError(f"{track} freeze tables carry {len(stamps)} build ids: "
                             f"{sorted(stamps)}; re-run the freeze")
    return tables


def decompose(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Per-trade ledger and headline-leg scores over every MLE excursion."""
    payload = tables["trade_attribution"].drop(columns=CONTRIBUTION_COLUMNS,
                                               errors="ignore").copy()
    bucket_key = ["detector_run_id", "bucket_index"]

    path = tables["detector_path"].set_index(bucket_key)[["mu", "sigma", "z"]]
    window = tables["canonical_windows"].set_index("detector_run_id")
    run = payload["detector_run_id"]

    q = payload["signed_yes_size"].to_numpy(dtype="float64")
    mle = payload["in_mle"].to_numpy()
    # A_b sums the whole bucket: the MLE window is a bucket range, so every
    # bucket it contains is present in full.
    abs_flow = (payload.assign(_abs=np.abs(q)).groupby(bucket_key)["_abs"].transform("sum")
                .to_numpy(dtype="float64"))

    joined = payload.set_index(bucket_key).join(path)
    if len(joined) != len(payload):
        raise AssertionError("detector_path is not unique per (run, bucket); the join "
                             "duplicated trades")
    scale = joined["sigma"].to_numpy(dtype="float64") + EPS
    mu = joined["mu"].to_numpy(dtype="float64")
    delta = run.map(window["winning_delta"]).to_numpy(dtype="float64")
    direction = run.map(window["direction"]).to_numpy(dtype="float64")

    signed = delta * direction
    dnc = signed * q / (abs_flow * scale)
    kappa = -signed * mu / scale - delta ** 2 / 2
    agc = kappa * np.abs(q) / abs_flow

    payload["abs_flow_bucket"] = np.where(mle, abs_flow, np.nan)
    payload["dnc"] = np.where(mle, dnc, np.nan)
    payload["agc"] = np.where(mle, agc, np.nan)
    payload["dfa"] = payload["dnc"] + payload["agc"]

    signed_q = direction * q
    payload["score_sign"] = np.where(mle, np.sign(signed_q), np.nan)

    # Rank d*q inside the full bucket, using MLE slots only.  The cell/profile
    # partition is deliberately irrelevant here and ties receive average rank.
    ranked = pd.Series(np.where(mle, signed_q, np.nan), index=payload.index)
    groups = [payload[name] for name in bucket_key]
    rank = ranked.groupby(groups, sort=False).rank(method="average")
    size = ranked.notna().groupby(groups, sort=False).transform("sum")
    payload["score_vdw"] = norm.ppf(rank / (size + 1.0))
    return payload


# ------------------------------------------------------------------- gate 3
def check_conservation(payload: pd.DataFrame,
                       tables: dict[str, pd.DataFrame]) -> tuple[list[str], dict]:
    """The ledger must close at three levels, each with its own tolerance."""
    mle = payload[payload["in_mle"]]
    path = tables["detector_path"]
    windows = tables["canonical_windows"].set_index("detector_run_id")

    per_trade = (mle["dfa"] - (mle["dnc"] + mle["agc"])).abs().max()

    # per bucket: sum(DFA) must equal that bucket's single-step LLR
    increments = path[path["in_mle"]].set_index(["detector_run_id", "bucket_index"])
    delta = increments.index.get_level_values(0).map(windows["winning_delta"])
    direction = increments.index.get_level_values(0).map(windows["direction"])
    llr = delta * direction * increments["z"] - delta ** 2 / 2
    by_bucket = mle.groupby(["detector_run_id", "bucket_index"])["dfa"].sum()
    per_bucket = (by_bucket - llr.reindex(by_bucket.index)).abs().max()

    # per run: sum(DFA) must equal W_alarm
    by_run = mle.groupby("detector_run_id")["dfa"].sum()
    per_run = (by_run - windows["w_alarm"].reindex(by_run.index)).abs().max()

    residuals = {"per_trade_dfa": float(per_trade), "per_bucket_llr": float(per_bucket),
                 "per_run_w_alarm": float(per_run)}
    failures = [f"{name}: residual {value:.3e} exceeds {TOLERANCE[name]:.0e}"
                for name, value in residuals.items() if value > TOLERANCE[name]]
    return failures, residuals


def check_reconstruction(payload: pd.DataFrame, tables: dict[str, pd.DataFrame],
                         tolerance: float = 1e-12) -> tuple[list[str], float]:
    """The frozen slots must rebuild the detector's own ``x_b``.

    ``x_b = sum q_i / A_b`` recomputed from the membership has to reproduce the
    monitored series the detector standardized. It is the one check that ties
    the trade-level tables and the bucket-level path together; if it drifts, the
    two describe different sets of trades.
    """
    mle = payload[payload["in_mle"]]
    rebuilt = (mle.groupby(["detector_run_id", "bucket_index"])["signed_yes_size"].sum()
               / mle.groupby(["detector_run_id", "bucket_index"])["abs_flow_bucket"].first())
    path = tables["detector_path"]
    observed = (path[path["in_mle"]].set_index(["detector_run_id", "bucket_index"])["x"]
                .reindex(rebuilt.index))
    worst = float((rebuilt - observed).abs().max())
    failures = ([] if worst <= tolerance
                else [f"x_b rebuilt from slots differs by {worst:.3e}"])
    return failures, worst


# ------------------------------------------------------------------- gate 4
def check_fail_closed(payload: pd.DataFrame,
                      tables: dict[str, pd.DataFrame]) -> list[str]:
    """Refuse to attribute a run whose MLE baseline is not usable.

    Step 1 already rejects imputed buckets and a degenerate scale; this repeats
    the test on the numbers the decomposition actually divided by, and adds the
    one denominator that only exists here -- a bucket with no absolute flow at
    all, where ``x_b`` is undefined.
    """
    failures = []
    windows = tables["canonical_windows"]
    if windows["untestable_reason"].notna().any():
        blocked = windows.loc[windows["untestable_reason"].notna(), "window_id"].tolist()
        failures.append(f"untestable runs reached the decomposition: {blocked}")

    path = tables["detector_path"]
    mle_path = path[path["in_mle"]]
    if mle_path["imputed"].any() or mle_path["scale_degenerate"].any():
        failures.append("MLE buckets with an imputed or degenerate baseline")
    if not np.isfinite(mle_path[["x", "mu", "sigma", "z"]].to_numpy()).all():
        failures.append("non-finite baseline inside an MLE window")
    if (mle_path["sigma"] <= SCALE_FLOOR).any():
        failures.append(f"sigma at or below the scale floor {SCALE_FLOOR}")

    mle = payload[payload["in_mle"]]
    if (mle["abs_flow_bucket"] <= 0).any():
        failures.append("MLE bucket with zero absolute flow: x_b is undefined")
    if not np.isfinite(mle[CONTRIBUTION_COLUMNS].to_numpy()).all():
        failures.append("non-finite contribution")
    if payload.loc[~payload["in_mle"], CONTRIBUTION_COLUMNS].notna().any().any():
        failures.append("a trade outside the MLE excursion was given a contribution")
    return failures


def check_proportional(payload: pd.DataFrame, tolerance: float = 1e-12) -> list[str]:
    """Within a bucket ``DNC_i = C_b q_i``: one constant, no per-trade freedom."""
    mle = payload[payload["in_mle"] & (payload["signed_yes_size"] != 0)]
    ratio = mle["dnc"] / mle["signed_yes_size"]
    spread = ratio.groupby([mle["detector_run_id"], mle["bucket_index"]]).agg(
        lambda values: float(values.max() - values.min())).max()
    return ([] if spread <= tolerance
            else [f"DNC is not proportional to signed size within a bucket: {spread:.3e}"])


def check_scores(payload: pd.DataFrame, tables: dict[str, pd.DataFrame],
                 tolerance: float = 1e-12) -> tuple[list[str], dict]:
    """Check G3 and describe van der Waerden tie-induced bucket imbalance."""
    bucket_key = ["detector_run_id", "bucket_index"]
    mle = payload[payload["in_mle"]].copy()
    direction = tables["canonical_windows"].set_index("detector_run_id")["direction"]
    mle["_signed_q"] = (mle["detector_run_id"].map(direction)
                         * mle["signed_yes_size"])

    ordered = mle.sort_values([*bucket_key, "_signed_q"], kind="mergesort")
    differences = ordered.groupby(bucket_key, sort=False)["score_vdw"].diff()
    failures = []
    if (differences < -tolerance).any():
        failures.append("score_vdw is not monotone in d*q inside a (run, bucket)")

    expected_sign = np.sign(mle["_signed_q"].to_numpy(dtype="float64"))
    observed_sign = mle["score_sign"].to_numpy(dtype="float64")
    if not np.array_equal(observed_sign, expected_sign):
        failures.append("score_sign differs from sign(d*q), including the zero rule")

    grouped = mle.groupby(bucket_key, sort=False)
    bucket_shape = grouped["_signed_q"].agg(["size", "nunique"])
    bucket_sums = grouped["score_vdw"].sum()
    tied = bucket_shape["nunique"] < bucket_shape["size"]
    untied_deviation = bucket_sums.loc[~tied].abs()
    max_untied = float(untied_deviation.max()) if len(untied_deviation) else 0.0
    if max_untied > tolerance:
        failures.append(f"an untied bucket has non-zero score_vdw sum: {max_untied:.3e}")

    tied_deviation = bucket_sums.loc[tied].abs()
    diagnostics = {
        "buckets": int(len(bucket_sums)),
        "buckets_with_ties": int(tied.sum()),
        "max_abs_bucket_sum": float(bucket_sums.abs().max()),
        "max_abs_untied_bucket_sum": max_untied,
        "max_abs_tied_bucket_sum": (float(tied_deviation.max())
                                     if len(tied_deviation) else 0.0),
        "sum_abs_tied_bucket_deviation": float(tied_deviation.sum()),
    }
    return failures, diagnostics


# ----------------------------------------------------------------------- entry
def run(repo_root: Path, track: str) -> dict:
    """Decompose one track, run gates 3 and 4, and write the contributions back."""
    tables = load_tables(repo_root, track)
    payload = decompose(tables)

    failures, residuals = check_conservation(payload, tables)
    rebuilt, worst = check_reconstruction(payload, tables)
    failures += rebuilt
    score_failures, score_diagnostics = check_scores(payload, tables)
    failures += score_failures
    if failures:
        raise AssertionError(f"gate 3 failed: {failures}")
    blocked = check_fail_closed(payload, tables) + check_proportional(payload)
    if blocked:
        raise AssertionError(f"gate 4 failed: {blocked}")

    out_path = repo_root / "data" / "attrib" / track / "trade_attribution.parquet"
    payload.to_parquet(out_path, index=False)

    mle = payload[payload["in_mle"]]
    report = {
        "track": track,
        "freeze_build_id": str(payload["freeze_build_id"].iloc[0]),
        "mle_slots": int(len(mle)),
        "epsilon": EPS,
        "conservation_residuals": residuals,
        "conservation_tolerances": TOLERANCE,
        "x_b_reconstruction_residual": worst,
        "gate_3_failures": [], "gate_4_failures": [],
        "score_g3": score_diagnostics,
        "score_sign": {"negative": int((mle["score_sign"] == -1).sum()),
                       "zero": int((mle["score_sign"] == 0).sum()),
                       "positive": int((mle["score_sign"] == 1).sum())},
        "dnc": {"positive": int((mle["dnc"] > 0).sum()),
                "negative": int((mle["dnc"] < 0).sum()),
                "total": float(mle["dnc"].sum()),
                "abs_total": float(mle["dnc"].abs().sum())},
        "dfa_total": float(mle["dfa"].sum()),
        "w_alarm_total": float(tables["canonical_windows"]["w_alarm"].sum()),
        "outputs": {"trade_attribution.parquet": sha256_file(out_path)},
    }
    (repo_root / "data" / "attrib" / track / "decompose_report.json").write_bytes(
        dumps(report))
    return report
