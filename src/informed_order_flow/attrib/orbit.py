# -*- coding: utf-8 -*-
"""Step 4 of Q2: who can be tested confirmatorily, and who could ever be rejected.

Two separate limits are drawn here, and conflating them is the mistake this
module exists to prevent.

*Eligibility* is a decision about the estimand. A wallet with one or two trades
in the test window can be screened, but it cannot carry confirmatory inference,
so the confirmatory family is ``n_trades_mle >= 3`` and everything else is
recorded with an explicit ``outside_confirmatory_reason``. Cell-wise shuffling
preserves each wallet's per-cell multiplicity, so ``n_trades_mle`` is constant
over the whole permutation orbit: the filter is fixed before randomisation and
does not disturb FWER control.

*Resolution* is a fact about arithmetic, not a decision. If wallet ``w`` holds
``k_wc`` of the ``m_c`` slots of cell ``c``, its label vector has

    Omega_w        = prod_c C(m_c, k_wc)        reachable arrangements
    p_orbit_floor  = 1 / Omega_w

and no permutation p-value for that wallet can ever fall below the floor. A
pair whose floor already exceeds the multiplicity threshold cannot be rejected
however extreme its data -- it is a power ceiling, and reporting it is what
makes an empty result interpretable rather than merely empty.

Two floors that must never be written as one: ``p_orbit_floor`` is structural
(with ties the true minimum tail probability is higher still), while
``p_mc_min = 1 / (B + 1)`` is the finite Monte Carlo grid. A sampled p-value can
sit below the orbit floor through Monte Carlo noise alone.

``1 / Omega`` is never zero, but a busy wallet's orbit runs past ``10^400`` and
the float rendering of its floor saturates at zero. Writing that zero down would
assert exactly what this column exists to deny -- that the p-value can be
arbitrarily small -- so ``p_orbit_floor_log10`` carries the authoritative value
(always finite, taken from the exact integer) and ``p_orbit_floor`` is kept as
the readable rendering, saturating below about ``1e-308`` and never used in a
decision.

A family is the frozen ``family_id`` written by aggregation. Under the Q2 rule
``family_unit = window``, each alarm window is a separate family on both tracks.
Reachability is reported against the first Holm threshold ``alpha_leg / m`` of
both families -- the screening one because the ceiling describes what would
have happened had the whole family carried confirmatory inference, the
confirmatory one because that is the threshold the analysis actually uses.

Wide-only rows carry ``family_id`` as provenance but do not enter either family.
Every inferential field written here is null for them, which is not the same as
being ineligible.
"""
from __future__ import annotations

import math
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd

from . import aggregate, decompose
from .plan import CONFIG, persist_report, sha256_file

# the exact decimal, not the binary float: an orbit floor can land on the Holm
# threshold exactly, and 0.05 as a float sits a hair above 1/20
ALPHA = Fraction(str(CONFIG["multiplicity"]["alpha"]))
ALPHA_LEG = ALPHA * Fraction(str(CONFIG["statistics"]["alpha_split"]))
P_MC_MIN = 1.0 / (int(CONFIG["permutation"]["B"]) + 1)

# derived from the frozen rule rather than restated, so the two cannot drift
CONFIRMATORY_RULE = str(CONFIG["eligibility"]["confirmatory_rule"])
MIN_MLE_SLOTS = int(CONFIRMATORY_RULE.rsplit(">=", 1)[1])
OUTSIDE_REASON = str(CONFIG["eligibility"]["outside_reason_code"])

CELL_KEY = ["detector_run_id", "bucket_index", "profile"]
ORBIT_COLUMNS = ["confirmatory_eligible", "outside_confirmatory_reason",
                 "log_orbit_size", "p_orbit_floor", "p_orbit_floor_log10",
                 "m_screening", "m_confirmatory",
                 "holm_first_threshold_screening", "holm_first_threshold_confirmatory",
                 "orbit_reachable_screening_family",
                 "orbit_reachable_confirmatory_family",
                 "orbit_floor_at_screening_threshold",
                 "orbit_floor_at_confirmatory_threshold"]


# ----------------------------------------------------------------- orbit size
def orbit_sizes(labelled: pd.DataFrame) -> pd.Series:
    """``Omega_w = prod_c C(m_c, k_wc)`` per wallet-window, as an exact integer.

    ``math.comb`` and Python integers are exact, so no factorial cancellation
    can creep into a bound that decides whether a pair is testable at all. The
    products get astronomically large for busy wallets, which costs nothing here
    and is what lets every comparison below be made without a tolerance.
    """
    m_c = labelled.groupby(CELL_KEY).size().rename("m_c")
    k_wc = labelled.groupby(CELL_KEY + ["active_wallet"]).size().rename("k_wc")
    counts = k_wc.reset_index().join(m_c, on=CELL_KEY)
    counts["choose"] = [math.comb(int(m), int(k))
                        for m, k in zip(counts["m_c"], counts["k_wc"])]
    return (counts.groupby(["detector_run_id", "active_wallet"])["choose"]
            .agg(math.prod).rename("orbit_size"))


def reaches(omega: np.ndarray, family_size: np.ndarray,
            alpha: Fraction = ALPHA_LEG) -> np.ndarray:
    """``1 / Omega <= alpha / m``, decided in exact rational arithmetic."""
    return np.array([alpha / int(m) * int(size) >= 1
                     for size, m in zip(omega, family_size)], dtype=bool)


def at_threshold(omega: np.ndarray, family_size: np.ndarray,
                 alpha: Fraction = ALPHA_LEG) -> np.ndarray:
    """``1 / Omega == alpha / m`` exactly: the orbit floor sits on the line.

    Holm rejects at ``p <= alpha / (m - j + 1)``, so such a pair counts as
    reachable here. It is flagged separately because the pre-registered counts
    were computed with a floating log-gamma orbit size, which puts the tie a few
    ulps on the other side; the strict count is reported next to it so both
    numbers can be checked.
    """
    return np.array([alpha / int(m) * int(size) == 1
                     for size, m in zip(omega, family_size)], dtype=bool)


def max_holm_step(omega: np.ndarray, family_size: int,
                  alpha: Fraction = ALPHA_LEG) -> int:
    """How far a Holm sequence could walk before the orbit alone stops it.

    Holm tests the ``j``-th smallest p-value against
    ``alpha_leg / (m - j + 1)`` and stops at the first failure. Walking the
    sorted orbit floors against those
    thresholds bounds the number of rejections the family can produce whatever
    the data say.
    """
    ordered = sorted((int(size) for size in omega), reverse=True)
    for step, size in enumerate(ordered):
        if not alpha / (family_size - step) * size >= 1:
            return step
    return family_size


# ------------------------------------------------------------------- resolve
def keyed_sizes(rows: pd.DataFrame, sizes: pd.Series) -> np.ndarray:
    """The exact orbit of every roster row, in row order."""
    keyed = pd.MultiIndex.from_arrays([rows["detector_run_id"], rows["active_wallet"]])
    omega = sizes.reindex(keyed).to_numpy()[rows["in_mle_roster"].to_numpy()]
    if pd.isna(omega).any():
        raise AssertionError("a roster wallet has no orbit; its slots are not in a cell")
    return omega


def resolve(rows: pd.DataFrame, sizes: pd.Series) -> pd.DataFrame:
    """Eligibility, orbit floor and family sizes for every wallet-window row."""
    out = rows.drop(columns=ORBIT_COLUMNS, errors="ignore").copy()
    roster = out["in_mle_roster"].to_numpy()
    omega = keyed_sizes(out, sizes)

    eligible = roster & (out["n_trades_mle"].to_numpy() >= MIN_MLE_SLOTS)
    out["confirmatory_eligible"] = pd.array(np.where(roster, eligible, None),
                                            dtype="boolean")
    out["outside_confirmatory_reason"] = np.where(roster & ~eligible, OUTSIDE_REASON,
                                                  None)

    log_orbit = np.full(len(out), np.nan)
    log_orbit[roster] = [math.log(int(size)) for size in omega]
    out["log_orbit_size"] = log_orbit

    # the authoritative floor: finite for every orbit, straight from the integer
    log10 = np.full(len(out), np.nan)
    log10[roster] = [-math.log10(int(size)) for size in omega]
    out["p_orbit_floor_log10"] = log10
    # the readable rendering of the same number; it saturates to 0 below about
    # 1e-308, which is why nothing here decides anything on it
    out["p_orbit_floor"] = np.exp(-log_orbit)

    family = out.loc[roster, "family_id"]
    m_screening = family.map(family.value_counts()).to_numpy()
    m_confirmatory = family.map(family[eligible[roster]].value_counts()).to_numpy()
    if pd.isna(m_confirmatory).any():
        raise AssertionError("a family has no confirmatory pair, so no Holm threshold "
                             "exists for it")

    for name, size in (("screening", m_screening), ("confirmatory", m_confirmatory)):
        out[f"m_{name}"] = pd.Series(pd.NA, index=out.index, dtype="Int64")
        out.loc[roster, f"m_{name}"] = size.astype("int64")
        threshold = np.full(len(out), np.nan)
        threshold[roster] = [float(ALPHA_LEG / int(m)) for m in size]
        out[f"holm_first_threshold_{name}"] = threshold
        for field, rule in ((f"orbit_reachable_{name}_family", reaches),
                            (f"orbit_floor_at_{name}_threshold", at_threshold)):
            flag = np.zeros(len(out), dtype=bool)
            flag[roster] = rule(omega, size)
            out[field] = pd.array(np.where(roster, flag, None), dtype="boolean")

    # family sizes and thresholds stay on every roster
    # row; whether a pair reaches the confirmatory threshold is a pair-level
    # question that simply does not apply outside that family
    outside = roster & ~eligible
    for field in ("orbit_reachable_confirmatory_family",
                  "orbit_floor_at_confirmatory_threshold"):
        out.loc[outside, field] = None
    return out


def family_report(rows: pd.DataFrame, sizes: pd.Series) -> list[dict]:
    """Per-family sizes, thresholds and reachable counts."""
    roster = rows[rows["in_mle_roster"]].copy()
    roster["orbit_size"] = keyed_sizes(rows, sizes)
    report = []
    for family_id, group in roster.groupby("family_id"):
        eligible = group[group["confirmatory_eligible"].astype(bool)]
        report.append({
            "family_id": str(family_id),
            "m_screening": int(group["m_screening"].iloc[0]),
            "m_confirmatory": int(group["m_confirmatory"].iloc[0]),
            "holm_first_threshold_screening":
                float(group["holm_first_threshold_screening"].iloc[0]),
            "holm_first_threshold_confirmatory":
                float(group["holm_first_threshold_confirmatory"].iloc[0]),
            "orbit_reachable_screening_family":
                int(group["orbit_reachable_screening_family"].sum()),
            "orbit_reachable_confirmatory_family":
                int(eligible["orbit_reachable_confirmatory_family"].sum()),
            "orbit_floor_at_threshold_screening":
                int(group["orbit_floor_at_screening_threshold"].sum()),
            "orbit_floor_at_threshold_confirmatory":
                int(eligible["orbit_floor_at_confirmatory_threshold"].sum()),
            "max_holm_step_screening_family":
                max_holm_step(group["orbit_size"].to_numpy(), len(group)),
        })
    return report


def resolution_by_n_trades(rows: pd.DataFrame) -> list[dict]:
    """Each family's resolution ceiling by number of wallet slots."""
    roster = rows[rows["in_mle_roster"]].copy()
    n = roster["n_trades_mle"].clip(upper=6)
    label = np.where(n >= 6, "6+", n.astype(str))
    roster["_n_trades_label"] = label
    return [{"family_id": str(family_id),
             "n_trades_mle": str(key),
             "pairs": int(len(group)),
             "confirmatory_eligible": int(group["confirmatory_eligible"].sum()),
             "orbit_reachable_screening_family":
                 int(group["orbit_reachable_screening_family"].sum()),
             "orbit_reachable_confirmatory_family":
                 int(group["orbit_reachable_confirmatory_family"].sum()),
             "median_log_orbit_size": float(group["log_orbit_size"].median())}
            for (family_id, key), group in
            roster.groupby(["family_id", "_n_trades_label"], sort=True)]


# --------------------------------------------------------------------- gate 8
def check_gate_8(track: str, rows: pd.DataFrame, families: list[dict]) -> list[str]:
    """Recompute S3 family sizes and resolution from the persisted columns."""
    want = CONFIG["expected_counts"][track]
    roster = rows[rows["in_mle_roster"]]
    eligible = roster[roster["confirmatory_eligible"].astype(bool)]
    failures = []

    def compare(name, got, expected):
        if got != expected:
            failures.append(f"{name}: got {got}, expected {expected}")

    def total(field):
        return sum(f[field] for f in families)

    compare("confirmatory_family", int(len(eligible)),
            want.get("confirmatory_family", want["pairs_by_n_trades"]["ge3"]))
    compare("families", len(families), int(want["canonical_runs"]))
    compare("family_report_confirmatory_total",
            sum(f["m_confirmatory"] for f in families), len(eligible))
    if "confirmatory_pairs_by_window" in want:
        compare("confirmatory_pairs_by_window",
                sorted(f["m_confirmatory"] for f in families),
                sorted(want["confirmatory_pairs_by_window"]))
    if "confirmatory_family_median" in want:
        sizes = sorted(f["m_confirmatory"] for f in families)
        compare("confirmatory_family_median", float(np.median(sizes)),
                float(want["confirmatory_family_median"]))
        compare("confirmatory_family_range", [sizes[0], sizes[-1]],
                want["confirmatory_family_range"])

    # Independent resolution audit: use only the persisted log floor and Holm
    # threshold, not the exact-integer helper that produced the reachability flag.
    log_reaches = (eligible["p_orbit_floor_log10"].to_numpy()
                   <= np.log10(eligible["holm_first_threshold_confirmatory"]
                               .to_numpy()))
    flags = eligible["orbit_reachable_confirmatory_family"].to_numpy(dtype=bool)
    compare("log10_recomputed_reachability", bool(np.array_equal(log_reaches, flags)),
            True)
    reachable = int(flags.sum())
    compare("family_report_reachable_total",
            total("orbit_reachable_confirmatory_family"), reachable)

    # The pre-registered resolution counts live in the frozen config, never as a
    # literal here: a track-specific number in a shared module would break the
    # single-fork-point rule and would make the count unfalsifiable, since the
    # engine would be asserting the very figure it is supposed to produce.
    compare("orbit_reachable_confirmatory_family", reachable,
            int(want["orbit_reachable_confirmatory_family"]))
    ties = int(eligible["orbit_floor_at_confirmatory_threshold"].to_numpy(dtype=bool).sum())
    compare("orbit_boundary_ties_confirmatory", ties,
            int(want["orbit_boundary_ties_confirmatory"]))
    compare("orbit_reachable_confirmatory_family_legacy_float", reachable - ties,
            int(want["orbit_reachable_confirmatory_family_legacy_float"]))

    screening_reachable = int(
        roster["orbit_reachable_screening_family"].to_numpy(dtype=bool).sum())
    compare("full_family_reachable_at_first_threshold", screening_reachable,
            int(want["full_family_reachable_at_first_threshold"]))
    compare("orbit_boundary_ties_screening",
            int(roster["orbit_floor_at_screening_threshold"].to_numpy(dtype=bool).sum()),
            int(want["orbit_boundary_ties_screening"]))
    return failures


def check_orbit_is_a_bound(rows: pd.DataFrame) -> list[str]:
    """The floor must behave like a bound: never above one, never below the grid."""
    roster = rows[rows["in_mle_roster"]]
    failures = []
    if (roster["log_orbit_size"] < 0).any():
        failures.append("an orbit is smaller than one arrangement")
    floor, log10 = roster["p_orbit_floor"], roster["p_orbit_floor_log10"]
    if not np.isfinite(log10).all():
        failures.append("p_orbit_floor_log10 is not finite: the floor is unreadable")
    if (log10 > 0).any() or (floor > 1.0).any():
        failures.append("an orbit floor above one")
    # a zero is only ever the float rendering giving up, never the true value
    saturated = floor == 0.0
    if not (log10[saturated] < -307).all():
        failures.append("p_orbit_floor is zero where the true floor is representable")
    if (floor[~saturated] <= 0).any():
        failures.append("a non-saturated p_orbit_floor is not positive")
    # a wallet holding every slot of every cell it touches cannot move at all
    frozen = roster[roster["log_orbit_size"] == 0.0]
    if not (frozen["p_orbit_floor"] == 1.0).all():
        failures.append("a wallet with no movable arrangement has a floor below one")
    if roster["confirmatory_eligible"].isna().any():
        failures.append("a roster wallet has no eligibility decision")
    if not (rows.loc[~rows["in_mle_roster"], ORBIT_COLUMNS].isna().all().all()):
        failures.append("a context-only wallet was given a family or a floor")
    return failures


# ----------------------------------------------------------------------- entry
def run(repo_root: Path, track: str) -> dict:
    """Draw both limits for one track and close gate 8."""
    tables = decompose.load_tables(repo_root, track)
    out_dir = repo_root / "data" / "attrib" / track
    rows = pd.read_parquet(out_dir / "wallet_windows.parquet")
    if "profile" not in rows.columns:
        raise AssertionError("wallet_windows carries no profiles; run the aggregate "
                             "step before the orbit audit")
    if set(rows["freeze_build_id"]) != set(tables["canonical_windows"]["freeze_build_id"]):
        raise AssertionError("wallet_windows and the freeze tables disagree on the "
                             "build id; re-run the aggregate step")

    # the exact orbit is an unbounded integer, which no parquet column can hold:
    # log_orbit_size is stored and orbit_sizes() rebuilds the exact one on demand
    sizes = orbit_sizes(aggregate.labelled_slots(tables, rows))
    rows = resolve(rows, sizes)
    families = family_report(rows, sizes)
    blocked = check_orbit_is_a_bound(rows)
    if blocked:
        raise AssertionError(f"the orbit audit is not self-consistent: {blocked}")
    failures = check_gate_8(track, rows, families)
    if failures:
        raise AssertionError(f"gate 8 failed, resolution counts differ from the "
                             f"frozen plan: {failures}")

    out_path = out_dir / "wallet_windows.parquet"
    rows.to_parquet(out_path, index=False)

    roster = rows[rows["in_mle_roster"]]
    eligible = roster[roster["confirmatory_eligible"].astype(bool)]
    report = {
        "track": track,
        "freeze_build_id": str(rows["freeze_build_id"].iloc[0]),
        "confirmatory_rule": CONFIRMATORY_RULE,
        "outside_confirmatory_reason": OUTSIDE_REASON,
        "alpha": float(ALPHA),
        "alpha_leg": float(ALPHA_LEG),
        "p_mc_min": P_MC_MIN,
        "p_mc_min_note": "the Monte Carlo grid floor: a sampled p-value can fall "
                         "below p_orbit_floor through noise, so the two are never "
                         "written as one bound",
        "family_unit": CONFIG["multiplicity"]["family_unit"],
        "families_count": len(families),
        "pairs": int(len(roster)),
        "confirmatory_family": int(len(eligible)),
        "outside_confirmatory": int((~roster["confirmatory_eligible"]
                                     .astype(bool)).sum()),
        "orbit_reachable_screening_family":
            sum(f["orbit_reachable_screening_family"] for f in families),
        "orbit_reachable_confirmatory_family":
            sum(f["orbit_reachable_confirmatory_family"] for f in families),
        "orbit_floor_at_threshold_screening":
            sum(f["orbit_floor_at_threshold_screening"] for f in families),
        "orbit_floor_at_threshold_confirmatory":
            sum(f["orbit_floor_at_threshold_confirmatory"] for f in families),
        "orbit_reachable_confirmatory_family_legacy_float":
            sum(f["orbit_reachable_confirmatory_family"] for f in families)
            - sum(f["orbit_floor_at_threshold_confirmatory"] for f in families),
        "boundary_tie_note": "a pair whose orbit floor equals the first Holm "
                             "threshold exactly. Holm rejects at p <= threshold, "
                             "so it is reachable; the legacy_float count is the "
                             "same number with such ties removed, which is what a "
                             "floating log-gamma orbit size produced before this "
                             "was corrected",
        "max_holm_step_screening_family":
            max(f["max_holm_step_screening_family"] for f in families),
        "gate_8_failures": [],
        "log_orbit_size": {"min": float(roster["log_orbit_size"].min()),
                           "median": float(roster["log_orbit_size"].median()),
                           "max": float(roster["log_orbit_size"].max()),
                           "zero_orbit_pairs": int((roster["log_orbit_size"] == 0).sum())},
        "p_orbit_floor_log10_min": float(roster["p_orbit_floor_log10"].min()),
        "p_orbit_floor_float_underflow_pairs": int((roster["p_orbit_floor"] == 0).sum()),
        "p_orbit_floor_note": "1 / Omega is never zero; the float column saturates "
                              "to 0 below about 1e-308, so p_orbit_floor_log10 is "
                              "the field to read and to publish",
        "boundary_tie_pairs": [
            {"family_id": str(row.family_id), "active_wallet": str(row.active_wallet),
             "n_trades_mle": int(row.n_trades_mle),
             "m_confirmatory": int(row.m_confirmatory),
             "log_orbit_size": float(row.log_orbit_size)}
            for row in rows[rows["orbit_floor_at_confirmatory_threshold"]
                            .fillna(False).astype(bool)].itertuples()],
        "by_n_trades": resolution_by_n_trades(rows),
        "families": families if len(families) <= 5 else families[:5],
        "families_truncated": len(families) > 5,
        "outputs": {"wallet_windows.parquet": sha256_file(out_path)},
    }
    persist_report(out_dir / "orbit_report.json", report)
    return report
