# -*- coding: utf-8 -*-
"""Step 3 of Q2: add a wallet's trades up inside one window, and classify it.

Two things happen here, and they are deliberately kept apart.

*Aggregation* sums the per-trade contributions of step 2 over the trades one
wallet holds in one window::

    T_DNC_w    = sum_{i: label_i = w} DNC_i
    T_mag_w    = sum_{i: label_i = w} score_vdw_i
    T_dir_w    = sum_{i: label_i = w} score_sign_i
    dnc_scaled = T_DNC_w / sum_i |DNC_i|          (all MLE trades of the window)
    e_MLE_w    = d * sum_{i: label_i = w} q_i     (detector-weight-free exposure)

``dnc_scaled`` is a comparable magnitude, **not** a share of the alarm score:
signed sums cancel, so it does not add to 100% across wallets and must never be
described that way. ``e_MLE`` is reported next to it because it uses no
detector weight at all -- if the two disagree, the disagreement is the finding.

*Classification* gives each wallet a history profile, and reads nothing but the
same stream's trades from before the official onset -- ``onset_bucket``, the
wide window's first bucket, so that not one trade of either window can move a
label::

    NEW        no active fill in this stream before the onset bucket
    OLD_SMALL  median pre-onset per-trade gross_shares <= cutoff
    OLD_LARGE  median pre-onset per-trade gross_shares >  cutoff

The cutoff is the median of those medians over the old wallets of this window's
MLE roster. The values are pre-onset only; the set they are taken over is the
observed roster, so the honest statement is "the numbers never come from after
the onset", not "no post-onset information at all". Roster, cutoff and profile
are all frozen here, before any label is permuted.

Profiles are the second half of a permutation cell (``cell_id = window_id |
bucket | profile``), which is why they are computed from history and never from
side, outcome, truth or any known address: a cell that knew the answer would
permute labels inside a group already sorted by the answer.

Wide-only wallets are kept as audit context with counts of zero; their MLE
fields are null rather than zero, so context can never be read as a measured
contribution of nothing.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..detect.features import assign_buckets
from . import decompose, sources
from .plan import CONFIG, FAMILY_UNIT, persist_report, sha256_file

PROFILES = tuple(CONFIG["profile"]["labels"])          # NEW, OLD_SMALL, OLD_LARGE

# The only trade columns the history rule may touch. side, outcome,
# resolved_outcome, prices and token ids are not in it and never will be.
HISTORY_COLUMNS = ("condition_id", "timestamp", "transaction_hash",
                   "active_wallet", "gross_shares")

RANK_COLUMNS = ("rank_dnc", "rank_mag", "rank_dir")
# measured contributions a context-only wallet must leave null. The ranks are
# checked separately through RANK_COLUMNS, which keeps "was never measured" and
# "was never ranked" as two distinct failures rather than one blurred one.
MLE_FIELDS = ("gross_mle", "dnc", "dnc_scaled", "agc", "dfa",
              "score_vdw", "score_sign", "e_mle")

COLUMNS = ["freeze_build_id", "stream_id", "condition_id", "window_id", "family_id",
           "detector_run_id", "representative_method", "active_wallet",
           "in_mle_roster", "n_trades_mle", "n_buckets_active", "gross_mle",
           "n_trades_wide", "gross_wide", "pre_onset_n_trades",
           "pre_onset_median_gross", "pre_onset_first_trade_utc",
           "first_contract_trade_asof_alarm_utc", "profile", "profile_cutoff",
           "dnc", "dnc_scaled", "agc", "dfa", "score_vdw", "score_sign",
           "e_mle", "rank_dnc", "rank_mag", "rank_dir"]


# ------------------------------------------------------------------- history
def bucketed_history(stream: sources.Stream) -> pd.DataFrame:
    """The stream's trades in Q1 bucket order, narrowed to the history columns."""
    trades = stream.trades.loc[:, list(HISTORY_COLUMNS)]
    return assign_buckets(trades, stream.bucket_size)


def pre_onset_history(bucketed: pd.DataFrame, onset_bucket: int) -> pd.DataFrame:
    """Per-wallet history strictly before the onset bucket, one row per wallet."""
    before = bucketed[bucketed["bucket_index"] < int(onset_bucket)]
    by_wallet = before.groupby("active_wallet")
    return pd.DataFrame({
        "pre_onset_n_trades": by_wallet.size(),
        "pre_onset_median_gross": by_wallet["gross_shares"].median(),
        "pre_onset_first_trade_utc": by_wallet["timestamp"].min(),
    })


def window_profiles(history: pd.DataFrame,
                    roster: pd.Index) -> tuple[pd.DataFrame, float]:
    """Profiles for the whole wide roster, cut at the MLE roster's own median.

    ``roster`` carries the MLE membership flag: only wallets that trade inside
    the test window may set the cutoff, but every wallet of the wide window is
    labelled so that the audit context is classified on the same scale.
    """
    aligned = history.reindex(roster.get_level_values("active_wallet"))
    aligned.index = roster
    n = aligned["pre_onset_n_trades"].fillna(0).to_numpy()
    median = aligned["pre_onset_median_gross"].to_numpy(dtype="float64")

    in_mle = roster.get_level_values("in_mle_roster").to_numpy()
    old_in_mle = median[(n > 0) & in_mle]
    cutoff = float(np.median(old_in_mle)) if old_in_mle.size else float("nan")

    aligned["pre_onset_n_trades"] = n.astype("int64")
    aligned["pre_onset_first_trade_utc"] = (aligned["pre_onset_first_trade_utc"]
                                            .astype("Int64"))
    aligned["profile"] = np.where(n == 0, PROFILES[0],
                                  np.where(median <= cutoff, PROFILES[1], PROFILES[2]))
    aligned["profile_cutoff"] = cutoff
    return aligned, cutoff


# ----------------------------------------------------------------- aggregate
def aggregate_window(window: pd.Series, slots: pd.DataFrame,
                     bucketed: pd.DataFrame) -> pd.DataFrame:
    """One canonical window: sum contributions per wallet, then label the wallet."""
    mle = slots[slots["in_mle"]]
    wide_by = slots.groupby("active_wallet")
    mle_by = mle.groupby("active_wallet")
    wallets = pd.Index(sorted(slots["active_wallet"].unique()), name="active_wallet")

    out = pd.DataFrame(index=wallets)
    out["n_trades_wide"] = wide_by.size().reindex(wallets).astype("int64")
    out["gross_wide"] = wide_by["gross_shares"].sum().reindex(wallets)
    out["n_trades_mle"] = mle_by.size().reindex(wallets).fillna(0).astype("int64")
    out["n_buckets_active"] = (mle_by["bucket_index"].nunique()
                               .reindex(wallets).fillna(0).astype("int64"))
    out["in_mle_roster"] = out["n_trades_mle"] > 0
    # sums over an empty selection stay null: a context wallet has no measured
    # contribution, which is not the same as a contribution measured at zero
    for name, column in (("gross_mle", "gross_shares"), ("dnc", "dnc"),
                         ("agc", "agc"), ("dfa", "dfa"),
                         ("score_vdw", "score_vdw"),
                         ("score_sign", "score_sign")):
        out[name] = mle_by[column].sum().reindex(wallets)
    out["e_mle"] = (int(window["direction"])
                    * mle_by["signed_yes_size"].sum().reindex(wallets))
    out["dnc_scaled"] = out["dnc"] / mle["dnc"].abs().sum()

    # a wallet's rank must not depend on its address: break ties on the frozen
    # order of its first MLE trade, which a rename carries along with the wallet
    first = (mle.sort_values(["timestamp", "transaction_hash"], kind="mergesort")
                .drop_duplicates("active_wallet").set_index("active_wallet"))
    arrival = pd.Series(np.arange(len(first)), index=first.index)
    roster_rows = out[out["in_mle_roster"]]
    for score, rank in (("dnc", "rank_dnc"), ("score_vdw", "rank_mag"),
                        ("score_sign", "rank_dir")):
        ordered = (roster_rows.assign(_arrival=arrival.reindex(roster_rows.index))
                   .sort_values([score, "_arrival"], ascending=[False, True],
                                kind="mergesort"))
        out[rank] = pd.Series(np.arange(1, len(ordered) + 1),
                              index=ordered.index, dtype="Int64")

    # the wallet's first appearance on this contract, cut at the alarm bucket:
    # a description that could have been read the moment the alarm existed
    asof = bucketed[bucketed["bucket_index"] <= int(window["alarm_bucket"])]
    out["first_contract_trade_asof_alarm_utc"] = (
        asof.groupby("active_wallet")["timestamp"].min().reindex(wallets)
        .astype("Int64"))

    roster = pd.MultiIndex.from_arrays([wallets, out["in_mle_roster"].to_numpy()],
                                       names=["active_wallet", "in_mle_roster"])
    history = pre_onset_history(bucketed, window["onset_bucket"])
    labelled, _ = window_profiles(history, roster)
    out = out.join(labelled.reset_index("in_mle_roster", drop=True))

    for field in ("window_id", "detector_run_id", "stream_id", "condition_id",
                  "representative_method", "freeze_build_id"):
        out[field] = window[field]
    out["family_id"] = out["window_id"] if FAMILY_UNIT == "window" else out["stream_id"]
    return out.reset_index()[COLUMNS]


def aggregate(tables: dict[str, pd.DataFrame],
              streams: dict[tuple[str, str], sources.Stream]) -> pd.DataFrame:
    """Wallet x window rows for every canonical window of one track."""
    payload = tables["trade_attribution"]
    required = {"dnc", "score_vdw", "score_sign"}
    missing = required - set(payload.columns)
    if missing:
        raise AssertionError(f"trade_attribution is missing {sorted(missing)}; run the "
                             "decompose step before aggregating")
    by_run = dict(tuple(payload.groupby("detector_run_id")))
    frames = []
    for _, window in tables["canonical_windows"].iterrows():
        stream = streams[(window["stream_id"], window["condition_id"])]
        frames.append(aggregate_window(window, by_run[window["detector_run_id"]],
                                       bucketed_history(stream)))
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------- gate 5
def scramble_after_onset(bucketed: pd.DataFrame, onset_bucket: int,
                         seed: int = 0) -> pd.DataFrame:
    """Rewrite every value from the onset bucket on, leaving the row order alone."""
    out = bucketed.copy()
    after = out["bucket_index"] >= int(onset_bucket)
    rng = np.random.default_rng(seed)
    rows = int(after.sum())
    labels = out.loc[after, "active_wallet"].to_numpy()
    out.loc[after, "active_wallet"] = rng.permutation(labels)
    out.loc[after, "gross_shares"] = (out.loc[after, "gross_shares"].to_numpy()
                                      * (1.0 + rng.random(rows) * 1e3))
    return out


def check_profile_reads_only_history(window: pd.Series, frozen: pd.DataFrame,
                                     bucketed: pd.DataFrame) -> list[str]:
    """Gate 5: mutating everything from the onset bucket on must change nothing.

    The wallet labels and trade sizes of both windows are overwritten with
    arbitrary values and the profile is rebuilt on the frozen roster. Bucket
    boundaries are left in place because they are a Q1 output that step 1 already
    froze; what is under test is that no *value* from the onset bucket onwards
    reaches a label. The roster is held fixed on purpose -- the design conditions
    the comparison set on the observed roster and says so.
    """
    roster = pd.MultiIndex.from_frame(
        frozen[["active_wallet", "in_mle_roster"]].sort_values("active_wallet"))
    mutated = scramble_after_onset(bucketed, window["onset_bucket"])
    rebuilt, cutoff = window_profiles(pre_onset_history(mutated, window["onset_bucket"]),
                                      roster)
    rebuilt = rebuilt.reset_index(drop=True)
    reference = frozen.sort_values("active_wallet").reset_index(drop=True)

    failures = []
    for field in ("pre_onset_n_trades", "pre_onset_median_gross",
                  "pre_onset_first_trade_utc", "profile"):
        # Series.equals compares dtype and treats two missing values as equal
        if not rebuilt[field].equals(reference[field]):
            failures.append(f"{window['window_id']}: {field} moved when post-onset "
                            f"values were scrambled")
    was = float(reference["profile_cutoff"].iloc[0])
    if not (cutoff == was or (np.isnan(cutoff) and np.isnan(was))):
        failures.append(f"{window['window_id']}: cutoff moved from {was!r} to {cutoff!r}")
    return failures


def check_profile_rule(rows: pd.DataFrame) -> list[str]:
    """The three labels must be exactly what the frozen definition says."""
    failures = []
    if not rows["profile"].isin(PROFILES).all():
        failures.append(f"profiles outside {PROFILES}")
    new = rows["profile"] == PROFILES[0]
    if not (new == (rows["pre_onset_n_trades"] == 0)).all():
        failures.append("NEW is not exactly 'no pre-onset active fill'")
    old = rows[~new]
    small = old["profile"] == PROFILES[1]
    if not (small == (old["pre_onset_median_gross"] <= old["profile_cutoff"])).all():
        failures.append("OLD_SMALL is not exactly 'median at or below the cutoff'")
    if old["pre_onset_median_gross"].isna().any():
        failures.append("an old wallet has no pre-onset median")

    for window_id, group in rows.groupby("window_id"):
        mle = group[group["in_mle_roster"]]
        medians = mle.loc[mle["pre_onset_n_trades"] > 0, "pre_onset_median_gross"]
        want = float(np.median(medians)) if len(medians) else float("nan")
        got = float(group["profile_cutoff"].iloc[0])
        if group["profile_cutoff"].nunique(dropna=False) != 1 or not (
                got == want or (np.isnan(got) and np.isnan(want))):
            failures.append(f"{window_id}: cutoff {got!r} is not the MLE roster's "
                            f"median of medians {want!r}")
    return failures


def labelled_slots(tables: dict[str, pd.DataFrame], rows: pd.DataFrame) -> pd.DataFrame:
    """Every MLE slot carrying its wallet's frozen profile: the cell coordinates.

    ``(detector_run_id, bucket_index, profile)`` is the cell a slot belongs to.
    The cells themselves are built in the permutation step; this is the one
    place their coordinates are derived, so later steps share it rather than
    re-deriving a profile.
    """
    payload = tables["trade_attribution"]
    mle = payload[payload["in_mle"]]
    profile = rows.set_index(["detector_run_id", "active_wallet"])["profile"]
    keyed = pd.MultiIndex.from_arrays([mle["detector_run_id"], mle["active_wallet"]])
    labelled = mle.assign(profile=profile.reindex(keyed).to_numpy())
    if labelled["profile"].isna().any():
        raise AssertionError("an MLE slot belongs to a wallet with no profile")
    return labelled


def cell_report(tables: dict[str, pd.DataFrame], rows: pd.DataFrame) -> dict:
    """Count the permutation cells the frozen profiles imply.

    Cells are built in the permutation step, not here, but their count is the
    sharpest available check on the profile rule: it was pre-registered per
    track in step 0, and it moves as soon as a single wallet changes label.
    """
    labelled = labelled_slots(tables, rows)
    per_cell = labelled.groupby(["detector_run_id", "bucket_index", "profile"])
    wallets = per_cell["active_wallet"].nunique()
    by_window = wallets.groupby(level="detector_run_id").size()
    return {"total": int(len(wallets)),
            "by_window": sorted(by_window.tolist(), reverse=True),
            "single_wallet_cells": int((wallets == 1).sum())}


def check_context_rows_are_null(rows: pd.DataFrame) -> list[str]:
    """A wide-only wallet has counts of zero and no measured contribution."""
    context = rows[~rows["in_mle_roster"]]
    failures = []
    if context[list(MLE_FIELDS)].notna().any().any():
        failures.append("a context-only wallet was given an MLE contribution")
    if context[list(RANK_COLUMNS)].notna().any().any():
        failures.append("a context-only wallet was ranked")
    if not ((context["n_trades_mle"] == 0) & (context["n_buckets_active"] == 0)).all():
        failures.append("a context-only wallet has MLE trades")
    required = ["dnc", "agc", "dfa", "score_vdw", "score_sign", "e_mle",
                *RANK_COLUMNS]
    if not rows.loc[rows["in_mle_roster"], required].notna().all().all():
        failures.append("a roster wallet is missing a contribution")
    return failures


def check_rank_permutations(rows: pd.DataFrame) -> list[str]:
    """Every score ranks the MLE roster exactly once inside each window."""
    failures = []
    for window_id, group in rows[rows["in_mle_roster"]].groupby("window_id"):
        expected = list(range(1, len(group) + 1))
        for column in RANK_COLUMNS:
            values = group[column]
            if values.isna().any() or sorted(values.dropna().astype(int).tolist()) != expected:
                failures.append(f"{window_id}: {column} is not a 1..m permutation")
    return failures


def check_counts(track: str, rows: pd.DataFrame, cells: dict) -> list[str]:
    """Pairs, wallets and cells against the counts frozen in step 0."""
    want = CONFIG["expected_counts"][track]
    mle = rows[rows["in_mle_roster"]]
    per_pair = mle["n_trades_mle"]
    windows_per_wallet = mle.groupby(["stream_id", "active_wallet"])["window_id"].nunique()
    got = {
        "pairs": int(len(mle)),
        "pairs_by_n_trades": {"1": int((per_pair == 1).sum()),
                              "2": int((per_pair == 2).sum()),
                              "ge3": int((per_pair >= 3).sum())},
        "distinct_wallets": int(mle.groupby(["stream_id", "active_wallet"]).ngroups),
        "wallets_in_1_2_3_windows": [int((windows_per_wallet == k).sum())
                                     for k in (1, 2, 3)],
        "cells": {key: cells[key] for key in want["cells"]},
    }
    return [f"{name}: got {value}, expected {want[name]}"
            for name, value in got.items() if name in want and value != want[name]]


# ----------------------------------------------------------------------- entry
def run(repo_root: Path, track: str) -> dict:
    """Aggregate one track to wallet x window rows and close gate 5's profile line."""
    tables = decompose.load_tables(repo_root, track)
    windows = tables["canonical_windows"]
    untestable = windows.loc[windows["untestable_reason"].notna(), "window_id"]
    if len(untestable):
        raise AssertionError(f"fail-closed runs must not be aggregated: "
                             f"{untestable.tolist()}")

    streams, _, _, _ = sources.load_track(repo_root, track)
    by_key = {(stream.stream_id, stream.condition_id): stream for stream in streams}
    rows = aggregate(tables, by_key)

    blocked = (check_profile_rule(rows) + check_context_rows_are_null(rows)
               + check_rank_permutations(rows))
    for _, window in windows.iterrows():
        stream = by_key[(window["stream_id"], window["condition_id"])]
        frozen = rows[rows["window_id"] == window["window_id"]]
        blocked += check_profile_reads_only_history(window, frozen,
                                                    bucketed_history(stream))
    if blocked:
        raise AssertionError(f"gate 5 failed, the profile rule is not history-only: "
                             f"{blocked[:5]}")

    cells = cell_report(tables, rows)
    failures = check_counts(track, rows, cells)
    if failures:
        raise AssertionError(f"aggregated counts differ from the frozen plan: {failures}")

    out_dir = repo_root / "data" / "attrib" / track
    out_path = out_dir / "wallet_windows.parquet"
    rows.to_parquet(out_path, index=False)

    mle = rows[rows["in_mle_roster"]]
    report = {
        "track": track,
        "freeze_build_id": str(rows["freeze_build_id"].iloc[0]),
        "rows": int(len(rows)),
        "pairs": int(len(mle)),
        "context_only_rows": int((~rows["in_mle_roster"]).sum()),
        "family_unit": FAMILY_UNIT,
        "families": int(rows["family_id"].nunique()),
        "official_onset": "onset_bucket, the first bucket of the wide window",
        "history_columns": list(HISTORY_COLUMNS),
        "gate_5_profile_failures": [], "rank_failures": [], "count_failures": [],
        "profiles": {label: int((mle["profile"] == label).sum()) for label in PROFILES},
        "cells": cells,
        "windows": [{"window_id": str(window_id),
                     "profile_cutoff": float(group["profile_cutoff"].iloc[0]),
                     "pairs": int(group["in_mle_roster"].sum()),
                     "context_only": int((~group["in_mle_roster"]).sum()),
                     "abs_wallet_dnc_total": float(group["dnc"].abs().sum()),
                     "dnc_scaled_abs_sum": float(group["dnc_scaled"].abs().sum()),
                     "profiles": {label: int((group.loc[group["in_mle_roster"],
                                                        "profile"] == label).sum())
                                  for label in PROFILES}}
                    for window_id, group in rows.groupby("window_id")],
        "dnc_scaled_note": "a comparable magnitude, not a share of the alarm score: "
                           "signed sums cancel, so it does not add to one",
        "outputs": {"wallet_windows.parquet": sha256_file(out_path)},
    }
    persist_report(out_dir / "aggregate_report.json", report)
    return report
