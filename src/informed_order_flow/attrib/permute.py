# -*- coding: utf-8 -*-
"""Step 5 of Q2: the engine that swaps wallet names inside a cell.

The control world keeps everything except who owns a trade. Times, sides,
sizes, prices, the bucketing and the whole Q1 path stay exactly as they are;
only the wallet labels move, and only between records that were already judged
comparable. A cell is

    cell_id = window_id | b<bucket_index> | <profile>

so a label can only travel to a slot in the same bucket of the same window held
by a wallet with the same kind of history. Side is deliberately not a stratum:
direction is the signal under test, and stratifying on it would permute the
answer away.

Each draw shuffles the complete repeated label vector inside every cell, which
preserves exactly what the null conditions on -- each wallet's number of slots
in each cell, and therefore its bucket counts, its profile, its orbit and its
``n_trades_mle`` eligibility. A single-wallet cell is left as the identity and
is never merged into a neighbour: merging would change the null to make a
number come out.

Randomness is addressed, not sequential. Every ``(window, cell)`` gets its own
``Philox`` stream seeded from ``SHA256(seed_base | window_id | cell_id)``, and
cells are drawn in one frozen order, so a run is reproducible from the ids
alone and batching cannot change a single draw: ``permuted(axis=1)`` over 512
rows consumes a stream exactly as 512 single-row calls do.

Two structural facts are recorded here rather than discovered later.
``window_fixed_slot_share`` is the share of a window's slots sitting in
single-wallet cells, and a window above 20% is flagged ``low_power_window``
before any result exists. ``wallet_fixed_slot_share`` is the same share for one
wallet; at 1 the wallet has no movable slot at all, its statistic is identical
in every permutation, and its p-value is exactly 1 -- it still stays in its
family rather than being quietly dropped.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from itertools import permutations, product
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.random import Generator, Philox

from . import decompose, ids
from .plan import CONFIG, persist_report, sha256_file

SEED_BASE = int(CONFIG["permutation"]["seed_base"])
BATCH_SIZE = int(CONFIG["permutation"]["batch_size"])
LOW_POWER_THRESHOLD = float(
    CONFIG["permutation"]["low_power_window"]["window_fixed_slot_share_threshold"])
PROFILE_ORDER = {label: index for index, label in enumerate(CONFIG["profile"]["labels"])}
WEIGHT_COLUMNS = ("dnc", "dfa", "score_vdw", "score_sign")

CELL_TABLE_COLUMNS = ["freeze_build_id", "window_id", "detector_run_id", "cell_id",
                      "bucket_index", "profile", "n_slots", "n_wallets", "movable",
                      "cell_seed"]
SHARE_COLUMNS = ["wallet_fixed_slot_share", "window_fixed_slot_share",
                 "low_power_window", "no_movable_slots"]


@dataclass(frozen=True)
class Window:
    """One window's frozen slots, grouped into cells and ready to permute.

    ``labels`` holds a wallet code per slot and ``bounds`` the cell boundaries,
    with slots ordered by ``(bucket_index, profile, slot_index)`` -- the frozen
    cell order every stream and every draw follows.
    """

    window_id: str
    wallets: pd.Index               # code -> active_wallet
    labels: np.ndarray              # wallet code per slot
    bounds: np.ndarray              # cell start offsets, length n_cells + 1
    cells: pd.DataFrame             # one row per cell, in draw order
    weights: pd.DataFrame           # fixed per-slot statistics, in slot order

    @property
    def n_cells(self) -> int:
        return len(self.cells)

    def cell_of_slot(self) -> np.ndarray:
        return np.repeat(np.arange(self.n_cells), np.diff(self.bounds))


def build_window(slots: pd.DataFrame, window_id: str) -> Window:
    """Group one window's MLE slots into cells in the frozen draw order."""
    ordered = slots.assign(_profile=slots["profile"].map(PROFILE_ORDER)).sort_values(
        ["bucket_index", "_profile", "slot_index"], kind="mergesort")
    if ordered["_profile"].isna().any():
        raise AssertionError(f"{window_id}: a slot carries an unknown profile")

    keys = list(zip(ordered["bucket_index"], ordered["profile"]))
    cell_ids = [ids.cell_id(window_id, bucket, profile) for bucket, profile in keys]
    ordered = ordered.assign(cell_id=cell_ids)
    grouped = ordered.groupby("cell_id", sort=False)
    cells = pd.DataFrame({
        "cell_id": ordered["cell_id"].drop_duplicates().tolist(),
        "bucket_index": grouped["bucket_index"].first().to_numpy(),
        "profile": grouped["profile"].first().to_numpy(),
        "n_slots": grouped.size().to_numpy(),
        "n_wallets": grouped["active_wallet"].nunique().to_numpy(),
    })
    cells["movable"] = cells["n_wallets"] > 1
    cells["cell_seed"] = [np.uint64(ids.cell_seed(SEED_BASE, window_id, cell))
                          for cell in cells["cell_id"]]

    wallets = pd.Index(sorted(ordered["active_wallet"].unique()), name="active_wallet")
    bounds = np.concatenate([[0], np.cumsum(cells["n_slots"].to_numpy())])
    return Window(window_id=window_id, wallets=wallets,
                  labels=wallets.get_indexer(ordered["active_wallet"]),
                  bounds=bounds, cells=cells,
                  weights=ordered[list(WEIGHT_COLUMNS)].reset_index(drop=True))


# ------------------------------------------------------------------- engine
def cell_streams(window: Window) -> list[Generator]:
    """One Philox stream per cell, addressed by ``(window_id, cell_id)``."""
    return [Generator(Philox(int(seed))) for seed in window.cells["cell_seed"]]


def permute(window: Window, streams: list[Generator], size: int) -> np.ndarray:
    """``size`` independent draws: shuffle the label vector inside every cell.

    Cells are visited in the frozen order and each consumes only its own
    stream, so the result depends on the ids and the draw count alone -- never
    on how the draws were batched.
    """
    out = np.empty((size, len(window.labels)), dtype=window.labels.dtype)
    for cell, stream in enumerate(streams):
        start, end = window.bounds[cell], window.bounds[cell + 1]
        block = np.tile(window.labels[start:end], (size, 1))
        out[:, start:end] = stream.permuted(block, axis=1)
    return out


def wallet_totals(window: Window, labels: np.ndarray,
                  weight: np.ndarray) -> np.ndarray:
    """``T_w`` for every wallet and every row of ``labels``."""
    rows, slots = labels.shape
    n = len(window.wallets)
    flat = labels + n * np.arange(rows, dtype=labels.dtype)[:, None]
    totals = np.bincount(flat.ravel(), weights=np.tile(weight, rows),
                         minlength=rows * n)
    return totals.reshape(rows, n)


def exceedance(window: Window, weights: dict[str, np.ndarray], draws: int,
               batch: int = BATCH_SIZE) -> dict[str, np.ndarray]:
    """Count draws whose wallet total is at least the observed one (ties count).

    Every statistic is read off the same draw -- DFA costs a second weighted sum,
    not a second permutation -- and only the running counts are kept, never a
    ``draws x wallets`` matrix.
    """
    observed = {name: wallet_totals(window, window.labels[None, :], weight)[0]
                for name, weight in weights.items()}
    counts = {name: np.zeros(len(window.wallets), dtype="int64") for name in weights}
    streams = cell_streams(window)
    remaining = draws
    while remaining > 0:
        size = min(batch, remaining)
        drawn = permute(window, streams, size)
        for name, weight in weights.items():
            totals = wallet_totals(window, drawn, weight)
            counts[name] += (totals >= observed[name]).sum(axis=0)
        remaining -= size
    return counts


# ------------------------------------------------------------- fixed slots
def fixed_slot_shares(window: Window) -> tuple[float, pd.Series]:
    """Share of slots that cannot move, for the window and for each wallet."""
    cell_of = window.cell_of_slot()
    fixed = ~window.cells["movable"].to_numpy()[cell_of]
    per_wallet = pd.DataFrame({"wallet": window.wallets[window.labels], "fixed": fixed})
    share = per_wallet.groupby("wallet")["fixed"].mean().reindex(window.wallets)
    return float(fixed.mean()), share


def direction_floors(window: Window) -> pd.DataFrame:
    """The smallest exact upper-tail probability attainable by ``score_sign``.

    In one cell a wallet keeps ``k`` labels and receives a uniformly selected
    ``k``-subset of the fixed {-1, 0, +1} scores. Its largest attainable sum
    takes positive slots first, then zeros, then negatives. The probability of
    that maximum is the number of subsets attaining it divided by ``C(m, k)``;
    cells are independent, so the window floor is the product of those ratios.
    """
    signs = window.weights["score_sign"].to_numpy()
    if not np.isin(signs, (-1.0, 0.0, 1.0)).all():
        raise AssertionError("score_sign left {-1, 0, +1}; no direction floor exists")

    floors = [Fraction(1, 1) for _ in window.wallets]
    for cell in range(window.n_cells):
        start, end = window.bounds[cell], window.bounds[cell + 1]
        block = signs[start:end]
        sign_counts = {value: int((block == value).sum()) for value in (1.0, 0.0, -1.0)}
        wallet_counts = np.bincount(window.labels[start:end], minlength=len(window.wallets))
        for wallet in np.flatnonzero(wallet_counts):
            k = int(wallet_counts[wallet])
            remaining = k
            favourable = 1
            for value in (1.0, 0.0, -1.0):
                take = min(remaining, sign_counts[value])
                favourable *= math.comb(sign_counts[value], take)
                remaining -= take
            if remaining:
                raise AssertionError("a wallet holds more labels than its cell has slots")
            floors[wallet] *= Fraction(favourable, math.comb(len(block), k))

    log10 = np.array([math.log10(value.numerator) - math.log10(value.denominator)
                      for value in floors])
    return pd.DataFrame({"p_dir_floor": [float(value) for value in floors],
                         "p_dir_floor_log10": log10,
                         "_p_dir_floor_exact": floors}, index=window.wallets)


# --------------------------------------------------------------- closure 1
def check_multiplicity(window: Window, draws: int = 8) -> list[str]:
    """Every draw must leave each wallet's per-cell slot count untouched."""
    drawn = permute(window, cell_streams(window), draws)
    failures = []
    for cell in range(window.n_cells):
        start, end = window.bounds[cell], window.bounds[cell + 1]
        before = np.sort(window.labels[start:end])
        after = np.sort(drawn[:, start:end], axis=1)
        if not (after == before).all():
            failures.append(f"{window.window_id}: cell {cell} changed its labels")
    return failures


def enumerate_labels(window: Window) -> np.ndarray:
    """Every arrangement the null allows, for a window small enough to enumerate."""
    per_cell = []
    for cell in range(window.n_cells):
        start, end = window.bounds[cell], window.bounds[cell + 1]
        per_cell.append(sorted(set(permutations(window.labels[start:end]))))
    return np.array([np.concatenate(combo) for combo in product(*per_cell)])


def check_tiny_cells_match_enumeration(window: Window, weight: np.ndarray,
                                       draws: int = 20000,
                                       sigma: float = 4.0) -> list[str]:
    """A tiny window's sampler must hit the complete orbit, and only the orbit.

    Two things are compared against the full enumeration: the set of label
    arrangements the sampler can produce, which must be exactly the orbit, and
    the exact orbit p-value, which the sampled p must match within Monte Carlo
    error.
    """
    orbit = enumerate_labels(window)
    drawn = permute(window, cell_streams(window), draws)
    seen = {row.tobytes() for row in drawn}
    expected = {row.astype(drawn.dtype).tobytes() for row in orbit}
    failures = []
    if not seen <= expected:
        failures.append(f"{window.window_id}: the sampler left the orbit")
    if seen != expected:
        failures.append(f"{window.window_id}: {len(expected) - len(seen)} of "
                        f"{len(expected)} arrangements were never drawn")

    observed = wallet_totals(window, window.labels[None, :], weight)[0]
    exact = (wallet_totals(window, orbit, weight) >= observed).mean(axis=0)
    sampled = (1 + exceedance(window, {"dnc": weight}, draws)["dnc"]) / (draws + 1)
    error = np.sqrt(np.maximum(exact * (1 - exact), 1e-12) / draws)
    drift = np.abs(sampled - exact) / error
    if drift.max() > sigma:
        failures.append(f"{window.window_id}: sampled p differs from the exact orbit "
                        f"p by {drift.max():.1f} standard errors")
    return failures


# --------------------------------------------------------------- closure 2
def check_frozen_wallet_gets_p_one(window: Window, weight: np.ndarray,
                                   draws: int = 512) -> list[str]:
    """A wallet with no movable slot must score identically in every draw."""
    _, share = fixed_slot_shares(window)
    frozen = share[share == 1.0].index
    if not len(frozen):
        return []
    p = (1 + exceedance(window, {"dnc": weight}, draws)["dnc"]) / (draws + 1)
    stuck = window.wallets.get_indexer(frozen)
    failures = []
    if not (p[stuck] == 1.0).all():
        failures.append(f"{window.window_id}: a wallet with no movable slot has "
                        f"p = {p[stuck].min()}, not 1")
    return failures


# ------------------------------------------------------------ engine fixture
SELF_TEST_STREAM = "selftest"
SELF_TEST_CONDITION = "0x" + "0" * 64


def self_test_window(cells: list[list[str]], alarm_bucket: int) -> Window:
    """A hand-built window, small enough to enumerate completely.

    Real windows hold hundreds of slots and no real wallet turns out to be
    completely stuck, so the two closure properties -- agreement with the full
    enumeration, and p exactly 1 for a wallet that cannot move -- are checked on
    fixtures built through the same code path, ids and seeds as the real ones.
    """
    rows = [{"bucket_index": bucket, "profile": CONFIG["profile"]["labels"][0],
             "slot_index": slot, "active_wallet": wallet,
             "dnc": 1.0 + slot + 10 * bucket, "dfa": 0.5 - slot,
             "score_vdw": 0.25 + slot - bucket,
             "score_sign": float(1 if slot % 2 == 0 else -1)}
            for bucket, labels in enumerate(cells)
            for slot, wallet in enumerate(labels)]
    window_id = ids.window_id(SELF_TEST_STREAM, SELF_TEST_CONDITION, "imbalance",
                              CONFIG["windows"]["bucket_size"], alarm_bucket)
    return build_window(pd.DataFrame(rows), window_id)


def engine_self_test() -> tuple[list[str], dict]:
    """Close both engine properties on fixtures, in exact terms where possible."""
    tiny = self_test_window([["A", "A", "B"], ["C", "C", "D", "E"]], alarm_bucket=1)
    stuck = self_test_window([["A", "A"], ["B", "C"]], alarm_bucket=2)
    weight_tiny = tiny.weights["dnc"].to_numpy()
    weight_stuck = stuck.weights["dnc"].to_numpy()

    failures = (check_multiplicity(tiny) + check_multiplicity(stuck)
                + check_tiny_cells_match_enumeration(tiny, weight_tiny)
                + check_frozen_wallet_gets_p_one(stuck, weight_stuck))
    return failures, {
        "enumerated_window": {"slots": int(len(tiny.labels)),
                              "cells": tiny.n_cells,
                              "arrangements": int(len(enumerate_labels(tiny)))},
        "frozen_wallet_window": {
            "slots": int(len(stuck.labels)), "cells": stuck.n_cells,
            "wallets_with_no_movable_slot":
                int((fixed_slot_shares(stuck)[1] == 1.0).sum())},
    }


def load_windows(repo_root: Path,
                 track: str) -> tuple[dict[str, Window], pd.DataFrame, dict]:
    """Every canonical window of a track, built once and shared by later steps."""
    tables = decompose.load_tables(repo_root, track)
    out_dir = repo_root / "data" / "attrib" / track
    rows = pd.read_parquet(out_dir / "wallet_windows.parquet")
    if "p_orbit_floor" not in rows.columns:
        raise AssertionError("wallet_windows carries no orbit audit; run the orbit "
                             "step before building cells")

    payload = tables["trade_attribution"]
    members = tables["window_membership"][["detector_run_id", "transaction_hash",
                                           "slot_index"]]
    slots = payload[payload["in_mle"]].merge(members, on=["detector_run_id",
                                                          "transaction_hash"])
    profile = rows.set_index(["detector_run_id", "active_wallet"])["profile"]
    keyed = pd.MultiIndex.from_arrays([slots["detector_run_id"], slots["active_wallet"]])
    slots = slots.assign(profile=profile.reindex(keyed).to_numpy())

    by_run = dict(tuple(slots.groupby("detector_run_id")))
    windows = {row.window_id: build_window(by_run[row.detector_run_id], row.window_id)
               for row in tables["canonical_windows"].itertuples()}
    return windows, rows, tables


def run(repo_root: Path, track: str) -> dict:
    """Build the cells and the seeds for one track and close both engine checks."""
    windows, rows, tables = load_windows(repo_root, track)
    out_dir = repo_root / "data" / "attrib" / track

    failures, fixtures = engine_self_test()
    if failures:
        raise AssertionError(f"the permutation engine does not close: {failures}")

    cells, shares = [], []
    for row in tables["canonical_windows"].itertuples():
        window = windows[row.window_id]
        failures += check_multiplicity(window)

        window_share, wallet_share = fixed_slot_shares(window)
        table = window.cells.assign(window_id=row.window_id,
                                    detector_run_id=row.detector_run_id,
                                    freeze_build_id=row.freeze_build_id)
        cells.append(table[CELL_TABLE_COLUMNS])
        shares.append(pd.DataFrame({
            "detector_run_id": row.detector_run_id,
            "active_wallet": wallet_share.index,
            "wallet_fixed_slot_share": wallet_share.to_numpy(),
            "window_fixed_slot_share": window_share}))

    if failures:
        raise AssertionError(f"the shuffle does not preserve the null's conditioning: "
                             f"{failures[:5]}")
    cell_table = pd.concat(cells, ignore_index=True)
    share_table = pd.concat(shares, ignore_index=True)

    rows = rows.drop(columns=SHARE_COLUMNS, errors="ignore").merge(
        share_table, on=["detector_run_id", "active_wallet"], how="left")
    rows["low_power_window"] = rows["window_fixed_slot_share"] > LOW_POWER_THRESHOLD
    rows["no_movable_slots"] = rows["wallet_fixed_slot_share"] == 1.0
    mismatched = rows["in_mle_roster"] & (rows["no_movable_slots"]
                                          != (rows["log_orbit_size"] == 0.0))
    if mismatched.any():
        raise AssertionError("no_movable_slots disagrees with a unit orbit for "
                             f"{int(mismatched.sum())} pairs")

    rows.to_parquet(out_dir / "wallet_windows.parquet", index=False)
    cell_table.to_parquet(out_dir / "permutation_cells.parquet", index=False)

    roster = rows[rows["in_mle_roster"]]
    report = {
        "track": track,
        "freeze_build_id": str(rows["freeze_build_id"].iloc[0]),
        "seed_base": SEED_BASE,
        "batch_size": BATCH_SIZE,
        "cell_definition": CONFIG["permutation"]["cell_definition"],
        "stratify_by_side": CONFIG["permutation"]["stratify_by_side"],
        "cells": {"total": int(len(cell_table)),
                  "movable": int(cell_table["movable"].sum()),
                  "single_wallet": int((~cell_table["movable"]).sum()),
                  "distinct_seeds": int(cell_table["cell_seed"].nunique()),
                  "largest": int(cell_table["n_slots"].max())},
        "multiplicity_failures": [],
        "engine_self_test": fixtures,
        "fixed_slots": {
            "window_share_max": float(roster["window_fixed_slot_share"].max()),
            "low_power_windows": int(roster.groupby("window_id")["low_power_window"]
                                     .first().sum()),
            "pairs_with_no_movable_slots": int(roster["no_movable_slots"].sum())},
        "outputs": {name: sha256_file(out_dir / name)
                    for name in ("wallet_windows.parquet",
                                 "permutation_cells.parquet")},
    }
    persist_report(out_dir / "permutation_report.json", report)
    return report
