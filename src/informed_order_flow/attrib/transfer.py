# -*- coding: utf-8 -*-
"""Q3: carry an online alarm's time interval across to the other contracts of the
event cluster, and test who was trading there.

Q1 monitors each contract on its own. A contract can therefore stay silent while
a wallet accumulates a large one-sided position in it, simply because that
contract's own reflected walk keeps being pushed back to zero. Q3 asks a
different question, conditioned on a different event:

    given that *some* contract of the cluster raised a genuine online imbalance
    alarm, which wallets on the *other* contracts of the cluster took an
    unusually concentrated position, in the alarm's own direction, during the
    very same wall-clock interval?

Nothing in the detector changes. The mapping happens strictly after an alarm has
already fired, so the stopping time and the online property of Q1 are untouched;
what is new is only the window a wallet-level permutation test is run on.

Every rule below is frozen in ``CONFIG`` before a single window is mapped. This
version runs on the real event cluster only: the simulated grid is not part of
it, and no module here imports the evaluator or names a manifest, so the study
is structurally incapable of reading ground truth.

**Source.** A source is an ``imbalance`` alarm with ``alarmed`` true, either
method, any bucket size -- including the K=50 contract that the Q2 population
excluded. The HHI channel may never be a source: HHI is a function of the wallet
labels, so selecting a window with it would let the selection event and the
permutation act on the same labels and the null distribution would collapse.

**Mapping.** A target bucket qualifies when ``[start_ts, end_ts]`` intersects the
source interval ``[window_start_utc, alarm_end_utc]``; the window is the
contiguous span from the first to the last qualifying bucket. The frozen Q1
bucketing of each contract is reused unchanged -- re-cutting the target by UTC
endpoints would pick a different set of trades whenever a boundary falls inside
a shared second. A mapped window identical to a Q2 primary window (same contract,
same bucket span) is dropped, because Q2 already tested it.

**Statistic.** The primary statistic is the direction-weighted exposure

    e_w = d * sum_{i: label_i = w} q_i

with ``d`` the *source* alarm's direction and ``q_i`` the signed YES size of a
trade. It needs no baseline, no delta and no detector path, it is exactly
conserved (``sum_w e_w = d * sum_i q_i``), and it says one thing only: how much
net exposure in the alarm's direction this wallet took inside the window. ``d``
is taken from the source alarm and never from the target contract's resolved
outcome, which would feed the answer into the statistic.

A transferred window contains no excursion of its own, so the Q2 DNC weight
loses the interpretation it had there. DNC is therefore kept as a secondary
sensitivity, computed only where the target contract has a clean burn-in, and
over a window statistic defined as the **plain** sum of single-step LLRs

    W_window = sum_{b=start}^{end} llr_b,   llr_b = delta* d z_b - delta*^2 / 2

with no reflection: reflection is a requirement of the sequential test, but once
a window is given, the plain sum is the natural object for a per-trade ledger,
and ``sum DFA = W_window`` still holds exactly.

**Inference.** Everything else is Q2's frozen machinery, reused unmodified:
cells are ``(bucket_index, profile)``, labels are shuffled inside a cell so each
wallet's per-cell multiplicity is preserved exactly, each cell draws from its own
Philox stream, ``p = (1 + #{T_r >= T_obs}) / (B + 1)`` with B = 1,999,999, and
the confirmatory family is ``n_trades >= 3``. The seed base and the id namespace
are new (``q3|`` prefix, ``SEED_BASE`` below), so no Q3 stream can collide with a
Q2 stream, and the Q3 Holm family is its own -- it is never merged with Q2's.

**Why the null still holds.** The window on target contract A is fixed by the
order flow of source contract B. B's order flow is a function of B's trades, not
of A's wallet labels, so conditioning on the window and then permuting A's labels
is legitimate -- the same argument Q2 makes for selecting on a label-invariant
imbalance alarm, one step further out. The one gap is a wallet that traded in
*both* the source window and the target window: for it the selection event is no
longer label-independent. Every pair therefore carries
``crosses_source_window`` and the whole Holm family is run twice, once as it
stands and once with those wallets removed.

**Direction transfer is a domain assumption.** Carrying ``d`` from one contract
to another assumes the cluster's contracts share an informed direction in that
interval. That is an assumption about these questions, not a statistical result,
so the mirror statistic ``-e`` is reported next to ``e``: a window where both
tails are extreme shows concentration without direction and must be read
descriptively.

**Placebo windows.** A transferred window is only interesting if the test would
*not* have fired on a window nobody selected. Every transferred window therefore
gets a placebo: the nearest span of the same length on the same contract that
touches no transferred window and no window Q2 already tested. It is chosen by
arithmetic on the transferred window's own length and by no alarm at all. Its
pairs form a family of their own, adjudicated separately and never merged with
the study, and the contrast between the two is reported beside the results.

Read the contrast for what it is. These are real contracts with no ground truth,
so a rejection inside a placebo window is not provably a false positive -- the
cluster may hold informed trading outside the alarm windows, which is the whole
premise of the project. What the contrast can say is whether the transferred
windows are distinguishable from unselected ones at all. If they are not, the
selection is doing no work and the rejections should be read as a property of
the statistic on this order flow rather than of the window it was run in.

**What is not established.** The permutation null conditions on each wallet's
per-cell slot count and treats the wallets inside a cell as exchangeable. The
cell stratifies on the bucket and on a wallet's *pre-window* trade sizes, so a
wallet with no pre-window history is pooled with every other such wallet however
large it trades inside the window. Where trade size is persistent at the wallet
level -- and in real order flow it is -- that assumption is approximate, and the
family-wise error rate is correspondingly approximate. This version ships no
measurement of how approximate. That is the main reason the study is reported as
exploratory rather than confirmatory.

This is an exploratory third question. It is reported as such, its family-wise
error claim stands only for its own family, and it never feeds back into Q1's
thresholds or channels or Q2's population, rules or products.
"""
from __future__ import annotations

import json
import hashlib
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from ..detect.features import assign_buckets
from ..detect.realrun import bucket_size_for, is_excluded
from . import aggregate, decompose, ids, multiplicity, orbit, permute, sources
from .plan import CONFIG as Q2_CONFIG
from .plan import dumps, sha256_file

VERSION = "q3-config-2.1.0"
FROZEN_ON = "2026-08-23"

# A namespace of its own. 2026081601 seeds Q2 and 2026081602 its second-seed
# review, so neither can be reproduced by accident from a Q3 id.
SEED_BASE = 2026081901
B = 1999999
BATCH_SIZE = 512
ALPHA = 0.05
ALPHA_LEG = ALPHA * float(Q2_CONFIG["statistics"]["alpha_split"])
BH_Q = 0.10
MIN_SLOTS = orbit.MIN_MLE_SLOTS            # the Q2 rule, imported rather than restated
EPS = decompose.EPS
# the bucket size Q2's primary selection runs at, read from Q2's frozen config so
# that the dedup rule cannot drift away from the population it is meant to skip
Q2_PRIMARY_BUCKET_SIZE = int(Q2_CONFIG["windows"]["bucket_size"])

CHANNEL = sources.CHANNEL                  # imbalance; HHI can never be a source
PREFIX = "q3"
TARGET_TAG = ">"
SEP = ids.SEP

OUT_DIR = ("data", "attrib", "q3")
# Which Q2 state this study was frozen against. A provenance record, not a gate:
# Q2 may legitimately be re-run afterwards, and the claim Q3 has to defend is
# that *it* never writes into Q2, which is checked per run against a snapshot
# taken at the start of that run (see q2_products_moved).
Q2_BASELINE_FILE = "q2_products_at_q3_freeze.json"

# What a window is for. A transfer carries a source alarm's interval onto another
# contract; a placebo is the control span no alarm selected. The role is part of
# the window id, so the two can never be confused for one another or share a
# permutation stream.
TRANSFER = "transfer"
PLACEBO = "placebo"
ROLES = (TRANSFER, PLACEBO)

DOJ_WALLET = "0x31a56e9e690c621ed21de08cb559e9524cdb8ed9"
DOJ_SOURCE = "Maduro out by December 31, 2026?"
DOJ_TARGET = "Maduro out by January 31, 2026?"
FROZEN_E_ROWS = 9176
FROZEN_E_VECTOR_SHA256 = "e74811df3515dd23832f556be2a3f46f5cba724d5205a85e2bb865eccd5f508d"


# --------------------------------------------------------------------- ids
def _token(value: object, field: str) -> str:
    text = str(value)
    if not text or SEP in text or any(char.isspace() for char in text):
        raise ValueError(f"{field} is not a valid id token: {value!r}")
    return text


def _run_token(value: object, field: str) -> str:
    """A composite detector run id, carried whole into a Q3 window id.

    The one component that legitimately contains separators of its own: it is
    ``<stream>|<contract>|<channel>|K<K>|<method>`` and all five parts are kept,
    because the source of a transferred window is exactly one detector run.
    """
    text = str(value)
    if not text or any(char.isspace() for char in text):
        raise ValueError(f"{field} is not a valid id token: {value!r}")
    if text.count(SEP) != 4:
        raise ValueError(f"{field} is not a detector run id: {value!r}")
    return text


def _index(value: object, field: str) -> int:
    """A non-negative integer component, asserted integral.

    The Q1 alarm tables store ``alarm_bucket`` and ``bucket_size`` as floats;
    formatting 54.0 without this cast would yield "54.0" and seed a different
    Philox stream from the one 54 seeds.
    """
    number = int(value)
    if number != value:
        raise ValueError(f"{field} is not integral: {value!r}")
    if number < 0:
        raise ValueError(f"{field} is negative: {value!r}")
    return number


def window_id(role: str, source_detector_run_id: str, target_stream_id: str,
              target_condition_id: object, bucket_size: object,
              bucket_start: object, bucket_end: object) -> str:
    """``q3|<role>|<source run>|><target stream>|<contract>|K<K>|b<start>-<end>``.

    Both ends of the transfer are in the key, because the same source alarm maps
    onto several targets and the same target receives several sources, and the
    ``q3`` prefix keeps the whole space disjoint from Q2's window ids.

    The target carries its *stream* as well as its contract for the reason Q2's
    ids do: a stream is one parallel universe, and on the simulated grid the
    scenarios share only four contract ids, so a target named by contract alone
    would collapse different scenarios onto one key -- and therefore onto one
    Philox stream. On the real track the stream is ``real`` throughout and the
    field is constant.

    The role is carried too, because a placebo window sits on the same contract
    as the transfer it controls and must be a different window in every respect
    that matters: a different key, a different cell namespace, a different set
    of Philox streams and a different Holm family.
    """
    start = _index(bucket_start, "bucket_start")
    end = _index(bucket_end, "bucket_end")
    if end < start:
        raise ValueError(f"bucket span runs backwards: {start}-{end}")
    if role not in ROLES:
        raise ValueError(f"role must be one of {ROLES}, got {role!r}")
    return SEP.join([
        PREFIX,
        role,
        _run_token(source_detector_run_id, "source_detector_run_id"),
        TARGET_TAG + _token(target_stream_id, "target_stream_id"),
        ids.condition(target_condition_id),
        f"K{_index(bucket_size, 'bucket_size')}",
        f"b{start}-{end}",
    ])


def cell_id(window: str, bucket_index: object, profile: str) -> str:
    """``<q3_window_id>|b<bucket_index>|<profile>`` -- Q2's cell layout, Q3's window."""
    if not window.startswith(PREFIX + SEP):
        raise ValueError(f"not a Q3 window id: {window!r}")
    if profile not in ids.PROFILES:
        raise ValueError(f"profile must be one of {ids.PROFILES}, got {profile!r}")
    return SEP.join([window, f"b{_index(bucket_index, 'bucket_index')}", profile])


def cell_seed(seed_base: int, window: str, cell: str) -> int:
    """Per-cell Philox seed; the Q2 rule with the Q3 seed base and Q3 ids."""
    return ids.cell_seed(seed_base, window, cell)


# ------------------------------------------------------------------ config
def _vectors() -> dict:
    """Golden id vectors, computed from the frozen templates and pinned by tests."""
    condition = "0x" + "0" * 63 + "1"
    target = "0x" + "0" * 63 + "2"
    # the track names are sources.py's to own, so they are read from it rather
    # than written here: this package's one fork point stays in one module
    stream = ids.stream_id()
    run = ids.detector_run_id(stream, condition, CHANNEL, 50, "windowed_glr")
    window = window_id(TRANSFER, run, stream, target, 100, 44, 53)
    cell = cell_id(window, 47, "OLD_LARGE")
    return {
        "source_detector_run_id": run,
        "q3_window_id": window,
        "q3_cell_id": cell,
        "cell_seed": cell_seed(SEED_BASE, window, cell),
        "vector_inputs": {
            "source_stream_id": stream, "source_condition_id": condition,
            "source_channel": CHANNEL, "source_bucket_size": 50,
            "source_method": "windowed_glr", "role": TRANSFER,
            "target_stream_id": stream, "target_condition_id": target,
            "target_bucket_size": 100.0, "bucket_start": 44, "bucket_end": 53.0,
            "bucket_index": 47, "profile": "OLD_LARGE", "seed_base": SEED_BASE,
        },
    }


# The mapping arithmetic is deterministic given the frozen alarm table and the
# frozen bucketing, so the window table it must produce is pre-registered here
# and the run refuses to differ. These are counts of trades and wallets, not
# results: no statistic, no p-value and no wallet identity enters this block.
EXPECTED_WINDOWS = [
    {"source_question": "Maduro out by November 30, 2025?", "source_method": "cusum",
     "target_question": "Maduro out in 2025?", "bucket_start": 176, "bucket_end": 196,
     "n_trades": 2100, "n_wallets": 1254, "n_pairs_ge3": 151},
    {"source_question": "Maduro out by November 30, 2025?", "source_method": "cusum",
     "target_question": "Maduro out by March 31, 2026?", "bucket_start": 15,
     "bucket_end": 20, "n_trades": 600, "n_wallets": 390, "n_pairs_ge3": 30},
    {"source_question": "Maduro out by March 31, 2026?", "source_method": "windowed_glr",
     "target_question": "Maduro out by December 31, 2026?", "bucket_start": 53,
     "bucket_end": 61, "n_trades": 450, "n_wallets": 247, "n_pairs_ge3": 43},
    {"source_question": "Maduro out by March 31, 2026?", "source_method": "windowed_glr",
     "target_question": "Maduro out by January 31, 2026?", "bucket_start": 46,
     "bucket_end": 85, "n_trades": 4000, "n_wallets": 1396, "n_pairs_ge3": 347},
    {"source_question": "Maduro out by March 31, 2026?", "source_method": "windowed_glr",
     "target_question": "Maduro out by February 28, 2026?", "bucket_start": 7,
     "bucket_end": 13, "n_trades": 700, "n_wallets": 329, "n_pairs_ge3": 42},
    {"source_question": "Maduro out by December 31, 2026?",
     "source_method": "windowed_glr", "target_question": "Maduro out by March 31, 2026?",
     "bucket_start": 76, "bucket_end": 79, "n_trades": 400, "n_wallets": 174,
     "n_pairs_ge3": 24},
    {"source_question": "Maduro out by December 31, 2026?",
     "source_method": "windowed_glr",
     "target_question": "Maduro out by January 31, 2026?", "bucket_start": 44,
     "bucket_end": 53, "n_trades": 1000, "n_wallets": 408, "n_pairs_ge3": 88},
    {"source_question": "Maduro out by December 31, 2026?",
     "source_method": "windowed_glr",
     "target_question": "Maduro out by February 28, 2026?", "bucket_start": 6,
     "bucket_end": 8, "n_trades": 300, "n_wallets": 181, "n_pairs_ge3": 21},
    {"source_question": "Maduro out by December 31, 2026?",
     "source_method": "windowed_glr",
     "target_question": "Maduro out by December 31, 2026?", "bucket_start": 48,
     "bucket_end": 54, "n_trades": 350, "n_wallets": 177, "n_pairs_ge3": 39},
]

# The placebo spans follow from the transferred spans and the contract bucketing
# by the rule above, so they are arithmetic too and are pre-registered the same
# way. One transferred window gets none: its contract is fifteen buckets long and
# both sides are blocked, which the run records rather than papers over.
EXPECTED_PLACEBOS = [
    {"source_question": "Maduro out by December 31, 2026?",
     "source_method": "windowed_glr",
     "target_question": "Maduro out by December 31, 2026?", "bucket_start": 41,
     "bucket_end": 47, "n_trades": 350, "n_wallets": 179, "n_pairs_ge3": 29},
    {"source_question": "Maduro out by December 31, 2026?",
     "source_method": "windowed_glr",
     "target_question": "Maduro out by February 28, 2026?", "bucket_start": 3,
     "bucket_end": 5, "n_trades": 300, "n_wallets": 212, "n_pairs_ge3": 12},
    {"source_question": "Maduro out by December 31, 2026?",
     "source_method": "windowed_glr",
     "target_question": "Maduro out by January 31, 2026?", "bucket_start": 34,
     "bucket_end": 43, "n_trades": 1000, "n_wallets": 535, "n_pairs_ge3": 49},
    {"source_question": "Maduro out by December 31, 2026?",
     "source_method": "windowed_glr",
     "target_question": "Maduro out by March 31, 2026?", "bucket_start": 72,
     "bucket_end": 75, "n_trades": 400, "n_wallets": 148, "n_pairs_ge3": 11},
    {"source_question": "Maduro out by March 31, 2026?", "source_method": "windowed_glr",
     "target_question": "Maduro out by December 31, 2026?", "bucket_start": 39,
     "bucket_end": 47, "n_trades": 450, "n_wallets": 225, "n_pairs_ge3": 29},
    {"source_question": "Maduro out by March 31, 2026?", "source_method": "windowed_glr",
     "target_question": "Maduro out by January 31, 2026?", "bucket_start": 4,
     "bucket_end": 43, "n_trades": 4000, "n_wallets": 1977, "n_pairs_ge3": 225},
    {"source_question": "Maduro out by November 30, 2025?", "source_method": "cusum",
     "target_question": "Maduro out by March 31, 2026?", "bucket_start": 9,
     "bucket_end": 14, "n_trades": 600, "n_wallets": 364, "n_pairs_ge3": 35},
    {"source_question": "Maduro out by November 30, 2025?", "source_method": "cusum",
     "target_question": "Maduro out in 2025?", "bucket_start": 155, "bucket_end": 175,
     "n_trades": 2100, "n_wallets": 980, "n_pairs_ge3": 117},
]

CONFIG = {
    "config_version": VERSION,
    "frozen_on": FROZEN_ON,
    "study": "Q3 -- cross-contract transfer of an online alarm window",
    "standing": ("exploratory. Its family-wise error claim covers its own family "
                 "only, it is reported as a third question beside Q1 and Q2, and "
                 "no Q3 result may be used to change a Q1 threshold, channel or "
                 "bucket size, or a Q2 rule, population, seed or statistic"),

    "source": {
        "channel": CHANNEL,
        "require_alarmed": True,
        "methods": list(ids.METHODS),
        "bucket_sizes": "all, including K=50",
        "interval": "[window_start_utc, alarm_end_utc] of the source alarm",
        "hhi_excluded_reason": ("HHI is a function of the wallet labels, so it is "
                                "not invariant to the permutation; selecting a "
                                "window with it would make the selection event and "
                                "the permutation act on the same labels and the "
                                "null distribution would no longer hold"),
    },

    "targets": {
        "population": "every contract of the event cluster, including the one Q1 "
                      "excluded from detection for lack of a clean burn-in: the "
                      "primary statistic needs no baseline",
        "bucket_size": "each contract keeps the bucket size Q1 froze for it",
    },

    "mapping": {
        "rule": ("a target bucket qualifies iff [start_ts, end_ts] intersects the "
                 "source interval [window_start_utc, alarm_end_utc]; the window is "
                 "the contiguous span from the first to the last qualifying bucket"),
        "span_not_set": ("the contiguous span is taken rather than the set of "
                         "qualifying buckets, so that the cell structure is the "
                         "same object Q2 permutes"),
        "rebucketing": ("forbidden; the frozen per-contract bucketing is reused "
                        "unchanged, because re-cutting by UTC endpoints picks up a "
                        "different set of trades whenever a bucket boundary falls "
                        "inside a shared second"),
    },

    "dedup": {
        "rule": ("drop a mapped window whose (target contract, bucket span) equals "
                 "a Q2 primary window's; Q2 has already tested it"),
        "q2_primary": (f"the imbalance alarms of the frozen Q1 table at "
                       f"K={Q2_PRIMARY_BUCKET_SIZE}: the wide span "
                       f"[onset_bucket, alarm_bucket] and the test span "
                       f"[onset_bucket_mle, alarm_bucket] both count as a match"),
    },

    "statistic": {
        "primary": "e = d * sum(signed_yes_size), d the source alarm's direction",
        "primary_conservation": "sum_w e_w = d * sum_i q_i, exactly",
        "mirror": ("-e is reported beside e: a window extreme in both tails shows "
                   "concentration without direction and is descriptive only"),
        "secondary": ("DNC / AGC / DFA over the window, with W defined as the plain "
                      "sum of single-step LLRs and no reflection, so that "
                      "sum DFA = W_window still holds exactly"),
        "secondary_status": ("a sensitivity with a family of its own, because it "
                             "covers a different set of pairs; it may disagree "
                             "with the primary and the disagreement is reported, "
                             "but it can never promote a pair"),
        "secondary_availability": ("only where the target contract has a clean "
                                   "burn-in: it is not excluded from Q1 detection, "
                                   "no window bucket is imputed or scale-degenerate, "
                                   "and the window starts at or after the baseline "
                                   "warm-up of the source alarm's method"),
        "secondary_baseline": ("the target contract's own mu_b / sigma_b under the "
                               "source alarm's method; delta* is the source alarm's "
                               "winning delta and d its direction"),
        "direction": ("taken from the source alarm, never from the target "
                      "contract's resolved outcome"),
        "headline_legs": {
            "magnitude": ("score_vdw: van der Waerden score of d*q, ranked with "
                          "average ties inside each target bucket"),
            "direction": "score_sign: sign(d*q), with zero size scoring zero",
        },
    },

    "direction_assumption": ("transferring d assumes the cluster's contracts share "
                             "an informed direction over the interval. That is a "
                             "domain assumption about these questions, not a "
                             "statistical result, and the mirror statistic is "
                             "reported so a reader can see both tails"),

    "permutation": {
        "cell_definition": ["bucket_index", "profile"],
        "stratify_by_side": False,
        "B": B,
        "p_denominator": B + 1,
        "tie_rule": ">=",
        "batch_size": BATCH_SIZE,
        "seed_base": SEED_BASE,
        "seed_base_note": ("new constant; Q2 uses 2026081601 and its second-seed "
                           "review 2026081602, so no Q3 stream can be reproduced "
                           "from a Q2 id or the other way round"),
        "numpy_version": "2.4.6",
    },

    "profile": {
        "labels": list(ids.PROFILES),
        "rule": ("Q2's rule with the window's first bucket as the onset: NEW has no "
                 "fill on this contract before the window starts, OLD_SMALL and "
                 "OLD_LARGE split the rest at the median of the window roster's "
                 "pre-window median trade sizes"),
    },

    "eligibility": {"confirmatory_rule": f"n_trades >= {MIN_SLOTS}"},

    "multiplicity": {
        "alpha": ALPHA,
        "alpha_leg": ALPHA_LEG,
        "bh_q": BH_Q,
        "family_unit": "window",
        "family": ("each declared family is adjudicated separately inside each "
                   "q3_window_id; no family is pooled across windows"),
        "parallel_evidence": ("magnitude and direction each run at alpha/2 over "
                              "the same eligible transferred pairs as primary; "
                              "primary e remains the only headline family"),
        "isolation": ("never merged with the Q2 family; the two family-wise error "
                      "claims are separate"),
        "leakage_sensitivity": ("the whole Holm family is run a second time with "
                                "every wallet that also traded inside the source "
                                "window removed, and both results are reported; a "
                                "pair significant only in the first is a lead, not "
                                "a finding"),
    },

    "placebo": {
        "purpose": ("a transferred window is only interesting if the same test "
                    "would not have fired on a window nobody selected"),
        "rule": ("for each transferred window [s, e] of length L on contract C, "
                 "the placebo is the nearest same-length span on C that lies "
                 "inside C's bucket range and intersects no transferred window "
                 "and no Q2 primary span on C. Candidates are walked outward from "
                 "the window one bucket at a time on both sides and the first "
                 "that qualifies is taken, the earlier side winning an exact tie; "
                 "if the contract has no room at all the window has no placebo "
                 "and that is recorded. Two placebos may overlap each other, "
                 "because two transferred windows on one contract already can"),
        "selection": ("arithmetic on the transferred window's own length, and no "
                      "alarm: nothing crossed a threshold inside a placebo span"),
        "direction": "the same d as the window it controls",
        "family": ("each placebo window is its own Holm family at the same alpha, "
                   "adjudicated separately and never merged with the study or "
                   "another placebo window"),
        "reading": ("a contrast, not a false-alarm rate. These are real contracts "
                    "with no ground truth, so a rejection inside a placebo window "
                    "is not provably a false positive -- the cluster may hold "
                    "informed trading outside the alarm windows, which is the "
                    "premise of the project. What the contrast can say is whether "
                    "the transferred windows are distinguishable from unselected "
                    "ones at all; if they are not, the selection is doing no work"),
    },

    "not_established": {
        "exchangeability": ("the permutation null treats the wallets inside a "
                            "cell as exchangeable. The cell stratifies on the "
                            "bucket and on a wallet's pre-window trade sizes, so "
                            "a wallet with no pre-window history is pooled with "
                            "every other such wallet however large it trades "
                            "inside the window. Where trade size is persistent at "
                            "the wallet level the assumption is approximate and "
                            "the family-wise error rate is approximate with it"),
        "calibration": ("this version ships no measurement of how approximate. "
                        "That is the main reason the study is exploratory rather "
                        "than confirmatory, and it is why the placebo contrast is "
                        "reported beside every result"),
        "scope": ("the real event cluster only; no simulated stream is read, and "
                  "no module of this study can open a scenario manifest"),
    },

    "order": ["freeze the configuration", "run the real cluster once",
              "read the results"],

    "id_scheme": {
        "separator": SEP,
        "python_hash_forbidden": True,
        "templates": {
            "q3_window_id": "q3|<role>|<source_detector_run_id>|><target_stream_id>"
                            "|<target_condition_id>|K<K>|b<bucket_start>-<bucket_end>",
            "q3_cell_id": "<q3_window_id>|b<bucket_index>|<profile>",
            "cell_seed": "int.from_bytes(sha256('<seed_base>|<q3_window_id>|"
                         "<q3_cell_id>')[:8], 'big')",
            "membership_sha256": ("sha256 over '<bucket_index>|<transaction_hash>\\n' "
                                  "for the slots of a window in frozen order"),
        },
        "vectors": _vectors(),
    },

    "expected_counts": {
        "source_alarms": 4,
        "intersecting_pairs": 12,
        "dropped_as_q2_primary": 3,
        "windows": len(EXPECTED_WINDOWS),
        "n_trades": sum(row["n_trades"] for row in EXPECTED_WINDOWS),
        "n_pairs_ge3": sum(row["n_pairs_ge3"] for row in EXPECTED_WINDOWS),
        "by_window": EXPECTED_WINDOWS,
        "placebo_windows": len(EXPECTED_PLACEBOS),
        "placebo_windows_impossible": len(EXPECTED_WINDOWS) - len(EXPECTED_PLACEBOS),
        "placebo_n_trades": sum(row["n_trades"] for row in EXPECTED_PLACEBOS),
        "placebo_n_pairs_ge3": sum(row["n_pairs_ge3"] for row in EXPECTED_PLACEBOS),
        "by_placebo": EXPECTED_PLACEBOS,
    },

    "scope": {
        "supported_claim": ("inside the interval another contract of the cluster "
                            "alarmed on, this wallet's directional exposure on this "
                            "contract is unusual against a label permutation, and "
                            "is worth review"),
        "unsupported_claim": ("informed, insider, illegal, or a probability of any "
                              "of those; and not that Q3 repairs Q1 -- Q3 "
                              "conditions on a different event and carries its own "
                              "false-alarm argument"),
        "attribution": ("active order owner only; the passive leg of every fill is "
                        "structurally invisible to this design"),
    },
}


# ------------------------------------------------------------------ streams
def cluster_streams(repo_root: Path,
                    calibration: sources.Calibration) -> list[sources.Stream]:
    """Every contract of the real cluster, each at the bucket size Q1 froze for it.

    ``sources.real_streams`` is deliberately narrower: it drops the contract Q1
    excluded from detection and the one that only runs at K=50, because the Q2
    population is the K=100 alarms. Q3 tests targets, not alarms, and its primary
    statistic needs no baseline, so both contracts are in scope here.
    """
    trades_path = repo_root / "data" / "processed" / "trades_event_level.parquet"
    pairs = (pd.read_parquet(trades_path, columns=["condition_id", "question"])
             .drop_duplicates().sort_values("condition_id"))
    return [sources.Stream(stream_id=ids.stream_id(),
                           condition_id=str(row["condition_id"]),
                           question=str(row["question"]),
                           bucket_size=bucket_size_for(str(row["question"])),
                           level=calibration.real_level, trades_path=trades_path)
            for _, row in pairs.iterrows()]


def source_alarms(repo_root: Path) -> pd.DataFrame:
    """The frozen Q1 alarms that may open a Q3 window."""
    alarms = pd.read_parquet(repo_root / "data" / "detect" / "cusum_real_alarms.parquet")
    picked = alarms[(alarms["channel"] == CHANNEL) & alarms["alarmed"]].copy()
    picked["detector_run_id"] = [
        ids.detector_run_id(ids.stream_id(), row.condition_id, CHANNEL,
                            row.bucket_size, row.method)
        for row in picked.itertuples()]
    return picked.sort_values("detector_run_id").reset_index(drop=True)


def q2_primary_spans(repo_root: Path) -> set[tuple[str, int, int]]:
    """The (contract, span) pairs Q2 already tested, rebuilt from the Q1 table.

    Both of Q2's spans count: the wide audit window ``[onset_bucket, alarm]`` and
    the narrower test window ``[onset_bucket_mle, alarm]``. Rebuilding them from
    the frozen alarm table rather than reading Q2's own outputs keeps Q3 from
    depending on a directory it must not touch.
    """
    alarms = pd.read_parquet(repo_root / "data" / "detect" / "cusum_real_alarms.parquet")
    primary = alarms[(alarms["channel"] == CHANNEL) & alarms["alarmed"]
                     & (alarms["bucket_size"] == Q2_PRIMARY_BUCKET_SIZE)]
    spans = set()
    for row in primary.itertuples():
        alarm = _index(row.alarm_bucket, "alarm_bucket")
        for start in (row.onset_bucket, row.onset_bucket_mle):
            spans.add((str(row.condition_id), _index(start, "onset"), alarm))
    return spans


# ------------------------------------------------------------------ mapping
def bucket_bounds(stream: sources.Stream) -> pd.DataFrame:
    """``bucket_index -> (start_ts, end_ts, n_trades)`` under the frozen bucketing."""
    bucketed = assign_buckets(stream.trades, stream.bucket_size)
    return bucketed.groupby("bucket_index").agg(start_ts=("timestamp", "min"),
                                                end_ts=("timestamp", "max"),
                                                n_trades=("timestamp", "size"))


def map_span(bounds: pd.DataFrame, low: float,
             high: float) -> tuple[int, int] | None:
    """The contiguous span of buckets whose calendar extent meets ``[low, high]``."""
    hit = bounds[(bounds["start_ts"] <= high) & (bounds["end_ts"] >= low)]
    if not len(hit):
        return None
    return int(hit.index.min()), int(hit.index.max())


def source_interval(alarm: pd.Series) -> tuple[float, float]:
    return float(alarm["window_start_utc"]), float(alarm["alarm_end_utc"])


def baseline_usable(stream: sources.Stream, method: str, bucket_start: int,
                    calibration: sources.Calibration) -> tuple[bool, str | None]:
    """Whether the secondary DNC arm may run on this target window.

    Three separate ways it may not. The contract may be one Q1 refused to detect
    on at all, for want of a clean burn-in. The window may start inside the very
    buckets the baseline is estimated from, which would standardize a window
    against itself. Or the standardization may simply not exist yet, because the
    method is still warming up.
    """
    if is_excluded(stream.question):
        return False, "contract has no clean burn-in and is outside Q1 detection"
    warm_up = (calibration.window.min_ref if method == "windowed_glr"
               else calibration.n_burn)
    if bucket_start < warm_up:
        return False, (f"window starts at bucket {bucket_start}, inside the "
                       f"{warm_up}-bucket baseline of {method}")
    return True, None


def placebo_span(span: tuple[int, int], n_buckets: int,
                 blocked: list[tuple[int, int]]) -> tuple[int, int] | None:
    """The control span for a transferred window: same length, nearest, unselected.

    Candidates are the same-length spans on either side of the window, walked
    outward one bucket at a time and taken as soon as one qualifies, with the
    earlier side winning an exact tie. A candidate is disqualified only if it
    runs off the contract or touches something an alarm did select -- any
    transferred window, or a window Q2 already tested -- because a control that
    contains part of a selected region is not a control.

    Two placebos may overlap each other, and are allowed to, because two
    transferred windows on one contract already can: the same-contract windows
    of this study are not disjoint, and the control set is built to mirror them
    rather than to be tidier than them.
    """
    start, end = span
    length = end - start + 1

    def free(low: int, high: int) -> bool:
        return (low >= 0 and high <= n_buckets - 1
                and not any(low <= taken_high and taken_low <= high
                            for taken_low, taken_high in blocked))

    for offset in range(n_buckets):
        for low, high in ((start - length - offset, start - 1 - offset),
                          (end + 1 + offset, end + length + offset)):
            if free(low, high):
                return low, high
    return None


def plan_placebos(windows: pd.DataFrame, bounds: dict[str, pd.DataFrame],
                  already: set[tuple[str, int, int]]) -> pd.DataFrame:
    """One control window per transferred window, where the contract has room."""
    rows = []
    for contract, group in windows.groupby("target_condition_id"):
        blocked = [(int(row.bucket_start), int(row.bucket_end))
                   for row in group.itertuples()]
        blocked += [(low, high) for cid, low, high in already if cid == contract]
        for row in group.sort_values("q3_window_id").itertuples():
            span = placebo_span((int(row.bucket_start), int(row.bucket_end)),
                                len(bounds[contract]), blocked)
            if span is None:
                continue
            start, end = span
            rows.append({**{field: getattr(row, field) for field in PLACEBO_CARRIED},
                         "role": PLACEBO,
                         "controls_window_id": row.q3_window_id,
                         "q3_window_id": window_id(
                             PLACEBO, row.source_detector_run_id, row.target_stream_id,
                             contract, row.target_bucket_size, start, end),
                         "bucket_start": start, "bucket_end": end,
                         "n_buckets": end - start + 1})
    return pd.DataFrame(rows)


# Fields a placebo inherits verbatim from the window it controls: it is the same
# source, the same contract and the same direction, moved to an unselected span.
PLACEBO_CARRIED = (
    "source_detector_run_id", "source_condition_id", "source_question",
    "source_method", "source_bucket_size", "source_alarm_bucket",
    "source_onset_bucket", "source_direction", "source_winning_delta",
    "source_window_start_utc", "source_alarm_end_utc", "source_alarm_available_utc",
    "target_stream_id", "target_condition_id", "target_question",
    "target_bucket_size", "is_self_window")


def plan_windows(repo_root: Path, calibration: sources.Calibration) -> pd.DataFrame:
    """Map every source alarm onto every contract of the cluster, dedup, control."""
    alarms = source_alarms(repo_root)
    streams = {stream.condition_id: stream
               for stream in cluster_streams(repo_root, calibration)}
    bounds = {cid: bucket_bounds(stream) for cid, stream in streams.items()}
    already = q2_primary_spans(repo_root)

    rows = []
    for _, alarm in alarms.iterrows():
        low, high = source_interval(alarm)
        for cid in sorted(streams):
            stream = streams[cid]
            span = map_span(bounds[cid], low, high)
            if span is None:
                continue
            start, end = span
            if (cid, start, end) in already:
                continue
            usable, reason = baseline_usable(stream, alarm["method"], start,
                                             calibration)
            rows.append({
                "role": TRANSFER,
                "controls_window_id": None,
                "q3_window_id": window_id(TRANSFER, alarm["detector_run_id"],
                                          stream.stream_id, cid,
                                          stream.bucket_size, start, end),
                "source_detector_run_id": alarm["detector_run_id"],
                "source_condition_id": str(alarm["condition_id"]),
                "source_question": str(alarm["question"]),
                "source_method": str(alarm["method"]),
                "source_bucket_size": _index(alarm["bucket_size"], "bucket_size"),
                "source_alarm_bucket": _index(alarm["alarm_bucket"], "alarm_bucket"),
                "source_onset_bucket": _index(alarm["onset_bucket"], "onset_bucket"),
                "source_direction": int(alarm["direction"]),
                "source_winning_delta": float(alarm["winning_delta"]),
                "source_window_start_utc": low,
                "source_alarm_end_utc": high,
                "source_alarm_available_utc": float(alarm["alarm_available_utc"]),
                "target_stream_id": stream.stream_id,
                "target_condition_id": cid,
                "target_question": stream.question,
                "target_bucket_size": stream.bucket_size,
                "bucket_start": start, "bucket_end": end,
                "n_buckets": end - start + 1,
                "is_self_window": cid == str(alarm["condition_id"]),
                "baseline_available": usable,
                "baseline_unavailable_reason": reason,
            })
    transfers = pd.DataFrame(rows).sort_values("q3_window_id").reset_index(drop=True)
    placebos = plan_placebos(transfers, bounds, already)
    if len(placebos):
        placebos = placebos.assign(
            baseline_available=False,
            baseline_unavailable_reason="a placebo window carries no secondary arm")
    out = pd.concat([transfers, placebos], ignore_index=True)
    return out.sort_values(["role", "q3_window_id"]).reset_index(drop=True)


# -------------------------------------------------------------------- slots
# Every slot of every window, with the weights it carried. DNC / AGC / DFA are
# null on a window whose target has no clean burn-in, which is not the same as
# a contribution measured at zero.
SLOT_COLUMNS = ["q3_window_id", "slot_index", "bucket_index", "timestamp",
                "transaction_hash", "active_wallet", "signed_yes_size",
                "gross_shares", "profile", "cell_id", "e", "e_mirror",
                "score_vdw", "score_sign", "dnc", "agc", "dfa"]


def window_slots(stream: sources.Stream, bucket_start: int,
                 bucket_end: int) -> pd.DataFrame:
    """The window's trades in the frozen slot order, one row per trade."""
    bucketed = assign_buckets(stream.trades, stream.bucket_size)
    slots = bucketed[bucketed["bucket_index"].between(bucket_start, bucket_end)]
    slots = slots.reset_index(drop=True)
    slots.insert(0, "slot_index", np.arange(len(slots), dtype="int64"))
    return slots


def window_profiles(stream: sources.Stream, slots: pd.DataFrame,
                    bucket_start: int) -> tuple[pd.DataFrame, float]:
    """Q2's history profile, cut at the window's first bucket.

    Nothing from the window itself can reach a label: the history is the same
    contract's fills strictly before ``bucket_start``, narrowed to the columns
    Q2 froze for this purpose, which carry no side, outcome or price.
    """
    wallets = pd.Index(sorted(slots["active_wallet"].unique()), name="active_wallet")
    roster = pd.MultiIndex.from_arrays([wallets, np.ones(len(wallets), dtype=bool)],
                                       names=["active_wallet", "in_mle_roster"])
    history = aggregate.pre_onset_history(aggregate.bucketed_history(stream),
                                          bucket_start)
    labelled, cutoff = aggregate.window_profiles(history, roster)
    return labelled.reset_index("in_mle_roster", drop=True), cutoff


# --------------------------------------------------------------- statistics
def exposure(slots: pd.DataFrame, direction: int) -> np.ndarray:
    """``e_i = d * q_i`` -- the per-slot share of the primary statistic."""
    return int(direction) * slots["signed_yes_size"].to_numpy(dtype="float64")


def headline_scores(slots: pd.DataFrame, direction: int) -> pd.DataFrame:
    """The two Q2 headline-leg weights over every slot of a Q3 window."""
    signed_q = int(direction) * slots["signed_yes_size"].to_numpy(dtype="float64")
    ranked = pd.Series(signed_q, index=slots.index)
    rank = ranked.groupby(slots["bucket_index"]).rank(method="average")
    size = ranked.groupby(slots["bucket_index"]).transform("size")
    return pd.DataFrame({"score_vdw": norm.ppf(rank / (size + 1.0)),
                         "score_sign": np.sign(signed_q)}, index=slots.index)


def target_path(stream: sources.Stream, method: str,
                calibration: sources.Calibration) -> pd.DataFrame:
    """The target contract's own standardized path, bucket by bucket.

    The source alarm's method decides how the target is standardized, so that
    delta* and z come from one detector definition rather than two.
    """
    from ..detect import build_features, channel_path
    from ..detect.features import FeatureConfig, channel_series, iter_contracts

    features = build_features(stream.trades,
                              FeatureConfig(bucket_size=stream.bucket_size))
    _, _, contract = next(iter_contracts(features))
    path, _ = channel_path(contract, CHANNEL, method=method,
                           baseline_method=calibration.baseline_method,
                           n_burn=calibration.n_burn, window=calibration.window)
    return pd.DataFrame({"bucket_index": contract["bucket_index"].to_numpy(),
                         "x": channel_series(contract, CHANNEL),
                         "mu": path.mu, "sigma": path.sigma, "z": path.z,
                         "imputed": path.imputed,
                         "scale_degenerate": path.scale_degenerate})


def contributions(slots: pd.DataFrame, path: pd.DataFrame, delta: float,
                  direction: int) -> pd.DataFrame:
    """Per-trade DNC / AGC / DFA and headline scores over a Q3 window.

    Identical arithmetic to Q2's decomposition -- the only difference is which
    buckets it runs over and that the window statistic it sums to is the plain
    sum of single-step LLRs rather than a reflected walk's value.
    """
    q = slots["signed_yes_size"].to_numpy(dtype="float64")
    abs_flow = (slots.assign(_abs=np.abs(q)).groupby("bucket_index")["_abs"]
                .transform("sum").to_numpy(dtype="float64"))
    joined = slots[["bucket_index"]].merge(path, on="bucket_index", how="left")
    if len(joined) != len(slots):
        raise AssertionError("the detector path is not unique per bucket")
    scale = joined["sigma"].to_numpy(dtype="float64") + EPS
    mu = joined["mu"].to_numpy(dtype="float64")

    signed = float(delta) * int(direction)
    dnc = signed * q / (abs_flow * scale)
    kappa = -signed * mu / scale - float(delta) ** 2 / 2
    agc = kappa * np.abs(q) / abs_flow
    ledger = pd.DataFrame({"abs_flow_bucket": abs_flow, "dnc": dnc, "agc": agc,
                           "dfa": dnc + agc}, index=slots.index)
    return pd.concat([ledger, headline_scores(slots, direction)], axis=1)


def window_llr(path: pd.DataFrame, bucket_start: int, bucket_end: int, delta: float,
               direction: int) -> float:
    """``W_window`` -- the plain sum of single-step LLRs, with no reflection."""
    inside = path[path["bucket_index"].between(bucket_start, bucket_end)]
    llr = (float(delta) * int(direction) * inside["z"].to_numpy(dtype="float64")
           - float(delta) ** 2 / 2)
    return float(llr.sum())


# ------------------------------------------------------- permutation engine
def build_window(slots: pd.DataFrame, q3_window_id: str,
                 weights: list[str]) -> permute.Window:
    """Group a Q3 window's slots into cells in the frozen draw order.

    ``permute.build_window`` cannot be called directly for one reason only: it
    stamps cells with ``ids.cell_id``, which validates the five-part Q2 window
    template and rejects a Q3 id. Everything else -- the ordering key, the cell
    boundaries, the label vector, the per-cell seed rule -- is Q2's, and a test
    pins the two against each other on a Q2-shaped id.
    """
    order = {label: index for index, label in enumerate(ids.PROFILES)}
    ordered = slots.assign(_profile=slots["profile"].map(order)).sort_values(
        ["bucket_index", "_profile", "slot_index"], kind="mergesort")
    if ordered["_profile"].isna().any():
        raise AssertionError(f"{q3_window_id}: a slot carries an unknown profile")

    cell_ids = [cell_id(q3_window_id, bucket, profile)
                for bucket, profile in zip(ordered["bucket_index"], ordered["profile"])]
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
    cells["cell_seed"] = [np.uint64(cell_seed(SEED_BASE, q3_window_id, cell))
                          for cell in cells["cell_id"]]

    wallets = pd.Index(sorted(ordered["active_wallet"].unique()), name="active_wallet")
    bounds = np.concatenate([[0], np.cumsum(cells["n_slots"].to_numpy())])
    return permute.Window(window_id=q3_window_id, wallets=wallets,
                          labels=wallets.get_indexer(ordered["active_wallet"]),
                          bounds=bounds, cells=cells,
                          weights=ordered[weights].reset_index(drop=True))


WEIGHT_SUFFIX = {"e": "e", "e_mirror": "e_mirror",
                 "score_vdw": "mag", "score_sign": "dir",
                 "dnc": "dnc", "dfa": "dfa"}


def window_counts(window: permute.Window, draws: int = B,
                  batch: int = BATCH_SIZE) -> pd.DataFrame:
    """Exceedance counts for one Q3 window, one row per wallet and statistic."""
    names = list(window.weights.columns)
    counts = permute.exceedance(
        window, {name: window.weights[name].to_numpy(dtype="float64")
                 for name in names}, draws, batch)
    out = pd.DataFrame({"q3_window_id": window.window_id,
                        "active_wallet": window.wallets}).reset_index(drop=True)
    for name in names:
        out[f"n_exceed_{WEIGHT_SUFFIX[name]}"] = counts[name]
    return out


def all_counts(windows: dict[str, permute.Window], draws: int = B,
               batch: int = BATCH_SIZE, workers: int = 1) -> pd.DataFrame:
    """Every window's counts, in one process or several -- the result is the same.

    A window's draws come from its own per-cell streams, so neither the worker
    count nor the order the windows finish in can move a single count.
    """
    order = sorted(windows)
    if workers <= 1:
        frames = [window_counts(windows[name], draws, batch) for name in order]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            frames = list(pool.map(window_counts, [windows[name] for name in order],
                                   [draws] * len(order), [batch] * len(order)))
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------- assembly
def prepare_window(row: pd.Series, stream: sources.Stream,
                   calibration: sources.Calibration,
                   with_secondary: bool = True) -> tuple[pd.DataFrame, dict]:
    """One window's slots, carrying their profile, cell and every weight."""
    start, end = int(row["bucket_start"]), int(row["bucket_end"])
    slots = window_slots(stream, start, end)
    profiles, cutoff = window_profiles(stream, slots, start)
    slots = slots.assign(
        q3_window_id=row["q3_window_id"],
        profile=slots["active_wallet"].map(profiles["profile"]).to_numpy(),
        e=exposure(slots, int(row["source_direction"])))
    slots["e_mirror"] = -slots["e"]
    if slots["profile"].isna().any():
        raise AssertionError(f"{row['q3_window_id']}: a slot has no profile")

    meta = {"profile_cutoff": cutoff, "w_window": float("nan"),
            "membership_sha256": ids.membership_sha256(
                zip(slots["bucket_index"], slots["transaction_hash"]))}
    if with_secondary and bool(row["baseline_available"]):
        path = target_path(stream, str(row["source_method"]), calibration)
        inside = path[path["bucket_index"].between(start, end)]
        if inside["imputed"].any() or inside["scale_degenerate"].any() \
                or not np.isfinite(inside[["mu", "sigma", "z"]].to_numpy()).all():
            # fail closed, and say so in the window table rather than only here:
            # a window that carries no DNC must not be described as one that does
            meta["baseline_rejected"] = ("an imputed, degenerate or non-finite "
                                         "bucket inside the window")
        else:
            parts = contributions(slots, path, float(row["source_winning_delta"]),
                                  int(row["source_direction"]))
            slots = pd.concat([slots, parts], axis=1)
            meta["w_window"] = window_llr(path, start, end,
                                          float(row["source_winning_delta"]),
                                          int(row["source_direction"]))
    if "score_vdw" not in slots.columns:
        slots = pd.concat([slots, headline_scores(
            slots, int(row["source_direction"]))], axis=1)
    slots = slots.assign(cell_id=[cell_id(row["q3_window_id"], bucket, profile)
                                  for bucket, profile
                                  in zip(slots["bucket_index"], slots["profile"])])
    return slots, meta


def weight_names(slots: pd.DataFrame) -> list[str]:
    weights = ["e", "e_mirror", "score_vdw", "score_sign"]
    return weights + (["dnc", "dfa"] if "dnc" in slots.columns else [])


def wallet_rows(row: pd.Series, slots: pd.DataFrame, window: permute.Window,
                meta: dict, source_wallets: set[str]) -> pd.DataFrame:
    """One row per (window, wallet): the observed statistics and the audit fields."""
    by_wallet = slots.groupby("active_wallet")
    wallets = pd.Index(sorted(slots["active_wallet"].unique()), name="active_wallet")
    out = pd.DataFrame(index=wallets)
    out["n_trades"] = by_wallet.size().reindex(wallets).astype("int64")
    out["n_buckets_active"] = (by_wallet["bucket_index"].nunique()
                               .reindex(wallets).astype("int64"))
    out["gross_shares"] = by_wallet["gross_shares"].sum().reindex(wallets)
    for name in weight_names(slots):
        out[name] = by_wallet[name].sum().reindex(wallets)
    out["profile"] = by_wallet["profile"].first().reindex(wallets)
    out["profile_cutoff"] = meta["profile_cutoff"]
    out["crosses_source_window"] = wallets.isin(source_wallets)

    window_share, wallet_share = permute.fixed_slot_shares(window)
    out["wallet_fixed_slot_share"] = wallet_share.reindex(wallets).to_numpy()
    out["window_fixed_slot_share"] = window_share
    out["no_movable_slots"] = out["wallet_fixed_slot_share"] == 1.0

    # a rank must not depend on an address: ties break on the frozen order of the
    # wallet's first slot, which a rename carries along with the wallet
    first = (slots.sort_values(["timestamp", "transaction_hash"], kind="mergesort")
             .drop_duplicates("active_wallet").set_index("active_wallet"))
    arrival = pd.Series(np.arange(len(first)), index=first.index)
    ordered = (out.assign(_arrival=arrival.reindex(out.index))
               .sort_values(["e", "_arrival"], ascending=[False, True],
                            kind="mergesort"))
    out["rank_e"] = pd.Series(np.arange(1, len(ordered) + 1), index=ordered.index,
                              dtype="Int64")
    out["eligible"] = out["n_trades"] >= MIN_SLOTS

    for field in ("q3_window_id", "role", "controls_window_id",
                  "source_detector_run_id", "source_question",
                  "source_method", "source_direction", "target_condition_id",
                  "target_question", "target_bucket_size", "bucket_start",
                  "bucket_end", "is_self_window", "baseline_available"):
        out[field] = row[field]
    out["w_window"] = meta["w_window"]
    return out.reset_index()


def orbit_sizes(labelled: pd.DataFrame) -> pd.Series:
    """``Omega_w = prod_c C(m_c, k_wc)`` per pair, as an exact Python integer.

    This is Q2's definition and a test pins the two against each other. It is
    re-implemented here for one mechanical reason: ``orbit.orbit_sizes`` finishes
    with ``groupby(...).agg(math.prod)`` over a column of Python integers, and
    pandas then tries to give that column a numeric dtype. Q2's largest orbit is
    about ``2**192`` and survives the conversion; a Q3 window can be forty
    buckets wide and produce an orbit past ``10**308``, where converting to a
    float overflows and the run dies. Accumulating the product in a dict keeps it
    an exact integer of whatever size it needs to be, which is the property the
    bound depends on.
    """
    m_c = labelled.groupby(orbit.CELL_KEY).size()
    k_wc = labelled.groupby(orbit.CELL_KEY + ["active_wallet"]).size()
    sizes: dict[tuple[object, object], int] = {}
    for key, held in k_wc.items():
        cell, wallet = key[:-1], key[-1]
        choose = math.comb(int(m_c.loc[cell]), int(held))
        sizes[(cell[0], wallet)] = sizes.get((cell[0], wallet), 1) * choose
    return pd.Series(list(sizes.values()), name="orbit_size", dtype=object,
                     index=pd.MultiIndex.from_tuples(
                         sizes, names=["detector_run_id", "active_wallet"]))


def orbit_fields(slots: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    """Q2's exact orbit bound, per pair: how fine a p-value can ever be."""
    labelled = slots.assign(detector_run_id=slots["q3_window_id"])
    sizes = orbit_sizes(labelled)
    keyed = pd.MultiIndex.from_arrays([rows["q3_window_id"], rows["active_wallet"]])
    omega = sizes.reindex(keyed).to_numpy()
    if pd.isna(omega).any():
        raise AssertionError("a wallet has no orbit; its slots are not in a cell")
    out = rows.copy()
    out["log_orbit_size"] = [math.log(int(size)) for size in omega]
    # the authoritative floor, finite for every orbit and taken from the exact
    # integer; the readable rendering below saturates to 0 below about 1e-308,
    # which is why nothing decides anything on it
    out["p_orbit_floor_log10"] = [-math.log10(int(size)) for size in omega]
    out["p_orbit_floor"] = np.exp(-out["log_orbit_size"].to_numpy())
    return out


def source_window_wallets(repo_root: Path, calibration: sources.Calibration,
                          windows: pd.DataFrame) -> dict[str, set[str]]:
    """Per source alarm, the wallets that traded inside its own alarm window."""
    streams = {stream.condition_id: stream
               for stream in cluster_streams(repo_root, calibration)}
    out: dict[str, set[str]] = {}
    for run_id, group in windows.groupby("source_detector_run_id"):
        row = group.iloc[0]
        stream = streams[str(row["source_condition_id"])]
        slots = window_slots(stream, int(row["source_onset_bucket"]),
                             int(row["source_alarm_bucket"]))
        out[str(run_id)] = set(slots["active_wallet"].unique())
    return out


# ---------------------------------------------------------------- inference
def p_values(rows: pd.DataFrame, counts: pd.DataFrame,
             draws: int) -> pd.DataFrame:
    """Plus-one p-values for every statistic a window carried."""
    out = rows.merge(counts, on=["q3_window_id", "active_wallet"], how="left")
    for column in [name for name in counts.columns if name.startswith("n_exceed_")]:
        statistic = column[len("n_exceed_"):]
        out[f"p_raw_{statistic}"] = (1 + out[column]) / (draws + 1)
    out["permutation_draws"] = draws
    if out["p_raw_e"].isna().any():
        raise AssertionError("a wallet came back without a p-value")
    return out


def _measured(rows: pd.DataFrame, column: str) -> np.ndarray:
    """Pairs whose statistic exists: a window without the arm has none."""
    if column not in rows.columns:
        return np.zeros(len(rows), dtype=bool)
    return rows[column].notna().to_numpy()


# Each family answers a different question and none of them is a subfamily of
# another that a reader may quietly swap in. ``primary`` is the study.
# ``no_leak`` repeats it without the wallets whose presence in the source window
# weakens the selection argument. ``mirror`` is the opposite tail, and exists so
# that a window which is merely busy cannot be read as directional.
# ``secondary_dnc`` is the detector-weighted sensitivity, over the windows whose
# target has a clean burn-in -- a different set of pairs, and therefore its own
# family. It is a sensitivity in the strict sense: it may disagree with the
# primary, and the disagreement is a result, but it can never promote a pair the
# primary did not reject.
def _transferred(rows: pd.DataFrame) -> np.ndarray:
    """Pairs of the study proper. A placebo pair is never one of them."""
    if "role" not in rows.columns:
        return np.ones(len(rows), dtype=bool)
    return (rows["role"] == TRANSFER).to_numpy()


HOLM_FAMILIES = {
    # suffix -> (p-value column, which pairs the family contains, alpha)
    "primary": ("p_raw_e", lambda rows: (rows["eligible"].to_numpy()
                                         & _transferred(rows)), ALPHA),
    "mag": ("p_raw_mag", lambda rows: (rows["eligible"].to_numpy()
                                         & _transferred(rows)
                                         & _measured(rows, "p_raw_mag")), ALPHA_LEG),
    "dir": ("p_raw_dir", lambda rows: (rows["eligible"].to_numpy()
                                         & _transferred(rows)
                                         & _measured(rows, "p_raw_dir")), ALPHA_LEG),
    "no_leak": ("p_raw_e", lambda rows: (rows["eligible"].to_numpy()
                                         & _transferred(rows)
                                         & ~rows["crosses_source_window"].to_numpy()),
                ALPHA),
    "mirror": ("p_raw_e_mirror", lambda rows: (rows["eligible"].to_numpy()
                                               & _transferred(rows)), ALPHA),
    "secondary_dnc": ("p_raw_dnc", lambda rows: (rows["eligible"].to_numpy()
                                                 & _transferred(rows)
                                                 & _measured(rows, "p_raw_dnc")),
                      ALPHA),
    "placebo": ("p_raw_e", lambda rows: (rows["eligible"].to_numpy()
                                         & ~_transferred(rows)), ALPHA),
}
HEADLINE_FAMILY = "primary"


def _window_groups(rows: pd.DataFrame, member: np.ndarray):
    """Indexes of one declared family, split at the frozen window boundary."""
    selected = rows.index[member]
    if not len(selected):
        return []
    if "q3_window_id" not in rows.columns:
        return [selected]
    return [group.index for _, group in rows.loc[selected].groupby(
        "q3_window_id", sort=False)]


def apply_multiplicity(rows: pd.DataFrame, alpha: float = ALPHA,
                       bh_q: float = BH_Q) -> pd.DataFrame:
    """Holm over each declared family, plus a BH screen over every Q3 pair.

    Every declared family is split again by ``q3_window_id``. Primary remains
    the only headline; magnitude and direction are parallel evidence at
    ``alpha / 2``. BH is a review screen and makes no FDR claim.
    """
    out = rows.copy()
    for suffix, (column, select, family_alpha) in HOLM_FAMILIES.items():
        for field, dtype in ((f"p_holm_{suffix}", "float64"),
                             (f"holm_threshold_{suffix}", "float64"),
                             (f"holm_rank_{suffix}", "Int64"),
                             (f"m_{suffix}", "Int64"),
                             (f"reject_{suffix}", "boolean")):
            out[field] = pd.Series(np.nan if dtype == "float64" else pd.NA,
                                   index=out.index, dtype=dtype)
        member = select(out)
        out[f"in_family_{suffix}"] = member
        for index in _window_groups(out, member):
            out.loc[index, f"m_{suffix}"] = len(index)
            adjusted, reject, threshold, rank = multiplicity.holm(
                out.loc[index, column].to_numpy(),
                family_alpha * alpha / ALPHA)
            out.loc[index, f"p_holm_{suffix}"] = adjusted
            out.loc[index, f"reject_{suffix}"] = reject
            out.loc[index, f"holm_threshold_{suffix}"] = threshold
            out.loc[index, f"holm_rank_{suffix}"] = rank

    # the screen covers the study's own pairs; a placebo pair is not under review
    study = _transferred(out)
    out["q_bh_e"] = pd.Series(np.nan, index=out.index, dtype="float64")
    out["bh_screen_e"] = pd.Series(pd.NA, index=out.index, dtype="boolean")
    out["m_screening"] = pd.Series(pd.NA, index=out.index, dtype="Int64")
    if study.any():
        out.loc[study, "m_screening"] = 0
    for index in _window_groups(out, study):
        q_values, screened = multiplicity.benjamini_hochberg(
            out.loc[index, "p_raw_e"].to_numpy(), bh_q)
        out.loc[index, "q_bh_e"] = q_values
        out.loc[index, "bh_screen_e"] = screened
        out.loc[index, "m_screening"] = len(index)
    out["both_tails_reject"] = (out["reject_primary"].fillna(False).astype(bool)
                                & out["reject_mirror"].fillna(False).astype(bool))
    headline = out[f"reject_{HEADLINE_FAMILY}"].fillna(False).astype(bool)
    secondary = out["reject_secondary_dnc"].fillna(False).astype(bool)
    out["secondary_only"] = secondary & ~headline
    return out


def check_secondary_cannot_promote(rows: pd.DataFrame) -> list[str]:
    """The headline set is the primary family's, whatever the sensitivity says."""
    headline = rows[f"reject_{HEADLINE_FAMILY}"].fillna(False).astype(bool)
    secondary = rows["reject_secondary_dnc"].fillna(False).astype(bool)
    failures = []
    if HEADLINE_FAMILY != "primary":
        failures.append(f"the headline family drifted to {HEADLINE_FAMILY}")
    if bool((headline & rows["secondary_only"]).any()):
        failures.append("a pair is both a headline and a secondary-only result")
    if int(rows["secondary_only"].sum()) != int((secondary & ~headline).sum()):
        failures.append("secondary_only is not the sensitivity's own rejections")
    return failures


def reachable(rows: pd.DataFrame) -> pd.DataFrame:
    """Whether a pair's orbit floor can reach its family's first Holm threshold."""
    out = rows.copy()
    for suffix, (_, _, family_alpha) in HOLM_FAMILIES.items():
        member = out[f"in_family_{suffix}"].to_numpy(dtype=bool)
        flag = np.zeros(len(out), dtype=bool)
        if member.any():
            thresholds = (float(family_alpha)
                          / out.loc[member, f"m_{suffix}"].to_numpy(dtype="int64"))
            # 1 / Omega <= alpha / m, compared on the log. The orbit itself runs
            # past 10^400 for a busy wallet and has no float representation at
            # all, so the log is the only form the comparison can be made in;
            # the log is monotone, and a pair that lands exactly on the line is
            # reachable because Holm rejects at p <= threshold
            flag[member] = (np.log(thresholds)
                            + out.loc[member, "log_orbit_size"].to_numpy()) >= 0.0
        out[f"orbit_reachable_{suffix}"] = pd.array(
            np.where(member, flag, None), dtype="boolean")
    return out


# -------------------------------------------------------------------- gates
def check_conservation(slots: pd.DataFrame, rows: pd.DataFrame, meta: dict,
                       direction: int, tolerance: float = 1e-8) -> list[str]:
    """The two ledgers must close: ``e`` exactly, ``DFA`` to ``tolerance``.

    ``e`` is checked where it is exact. Per slot ``e_i = d q_i`` is a sign
    change, which IEEE arithmetic performs without error, so that identity is
    asserted bit for bit. The totals are then compared with a relative
    tolerance, because summing the same numbers grouped by wallet and summing
    them in slot order are two different orders of addition and can differ in
    the last unit in the last place -- that is float addition not being
    associative, not a contribution going missing.

    ``sum DFA = W_window`` is the transferred window's version of Q2's ledger
    identity, and it is what makes the plain (unreflected) LLR sum the right
    object to decompose.
    """
    failures = []
    window = str(slots["q3_window_id"].iloc[0])
    exact = int(direction) * slots["signed_yes_size"].to_numpy(dtype="float64")
    if not np.array_equal(slots["e"].to_numpy(dtype="float64"), exact):
        failures.append(f"{window}: a slot's e is not d * q")
    if not np.array_equal(slots["score_sign"].to_numpy(dtype="float64"),
                          np.sign(exact)):
        failures.append(f"{window}: score_sign is not sign(d * q)")
    scored = slots.assign(_signed_q=exact).sort_values(
        ["bucket_index", "_signed_q"], kind="mergesort")
    monotone = scored.groupby("bucket_index", sort=False)["score_vdw"].diff().dropna()
    if (monotone < -1e-12).any():
        failures.append(f"{window}: score_vdw is not monotone inside a bucket")
    if not np.isfinite(slots["score_vdw"]).all():
        failures.append(f"{window}: score_vdw is not finite")
    total = int(direction) * float(slots["signed_yes_size"].sum())
    drift = abs(float(rows["e"].sum()) - total)
    if drift > 1e-9 * max(1.0, abs(total)):
        failures.append(f"{window}: sum(e) differs from d * sum(q) by {drift:.3e}")
    if "dfa" in slots.columns:
        residual = abs(float(slots["dfa"].sum()) - meta["w_window"])
        if residual > tolerance:
            failures.append(f"{window}: sum(DFA) differs from W_window by "
                            f"{residual:.3e}")
    return failures


def check_namespace(rows: pd.DataFrame) -> list[str]:
    """Every Q3 id must be a Q3 id, and no Q2 id may appear as one."""
    failures = []
    bad = [w for w in rows["q3_window_id"].unique() if not w.startswith(PREFIX + SEP)]
    if bad:
        failures.append(f"window ids outside the q3 namespace: {bad[:3]}")
    if SEED_BASE in (permute.SEED_BASE, multiplicity.REVIEW_SEED_BASE):
        failures.append("the Q3 seed base collides with a Q2 seed base")
    return failures


def check_p_values(rows: pd.DataFrame, draws: int) -> list[str]:
    """A p-value is a probability, never zero, never finer than the draw grid."""
    failures = []
    for column in [name for name in rows.columns if name.startswith("p_raw_")]:
        p = rows[column]
        if (p <= 0).any() or (p > 1).any():
            failures.append(f"{column} is not a probability")
        if (p < 1.0 / (draws + 1) - 1e-15).any():
            failures.append(f"{column} is below the Monte Carlo grid")
    stuck = rows[rows["no_movable_slots"]]
    for column in [name for name in rows.columns if name.startswith("p_raw_")]:
        measured = stuck[column].dropna()
        if len(measured) and not (measured == 1.0).all():
            failures.append(f"a wallet with no movable slot did not get {column} = 1")
    return failures


def frozen_e_regression(rows: pd.DataFrame, draws: int) -> dict:
    """Adding weights must not move one frozen primary count or p-value."""
    if draws != B:
        return {"status": "not_applicable", "reason": "the run did not use frozen B"}
    ordered = rows.sort_values(["q3_window_id", "active_wallet"])
    payload = "\n".join(
        f"{row.q3_window_id}|{row.active_wallet}|{int(row.n_exceed_e)}|"
        f"{float(row.p_raw_e).hex()}" for row in ordered.itertuples())
    observed = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    if len(ordered) != FROZEN_E_ROWS or observed != FROZEN_E_VECTOR_SHA256:
        raise AssertionError("G7 failed: the frozen Q3 primary e vector changed")
    return {"status": "bit_equal", "rows": len(ordered),
            "vector_sha256": observed}


def check_gate_7(rows: pd.DataFrame) -> tuple[list[str], dict]:
    """The pre-registered DOJ pair must expose all three S6 decisions."""
    picked = rows[(rows["role"] == TRANSFER)
                  & (rows["source_question"] == DOJ_SOURCE)
                  & (rows["target_question"] == DOJ_TARGET)
                  & (rows["active_wallet"] == DOJ_WALLET)]
    if len(picked) != 1:
        return ([f"DOJ gate selected {len(picked)} rows, not one"], {})
    row = picked.iloc[0]
    failures = []
    expected = {
        "eligible": True, "n_trades": 4, "rank_e": 1,
        "m_primary": 88, "m_mag": 88, "m_dir": 88,
        "holm_rank_primary": 1, "holm_rank_mag": 3,
        "reject_primary": True, "reject_mag": True, "reject_dir": False,
    }
    for field, value in expected.items():
        if row[field] != value:
            failures.append(f"DOJ {field} is {row[field]!r}, expected {value!r}")
    detail = {field: (bool(row[field]) if isinstance(value, bool) else int(row[field]))
              for field, value in expected.items()}
    detail.update({"active_wallet": DOJ_WALLET,
                   "q3_window_id": str(row["q3_window_id"]),
                   "p_raw_e": float(row["p_raw_e"]),
                   "p_raw_mag": float(row["p_raw_mag"]),
                   "p_raw_dir": float(row["p_raw_dir"]),
                   "p_holm_primary": float(row["p_holm_primary"]),
                   "p_holm_mag": float(row["p_holm_mag"]),
                   "p_holm_dir": float(row["p_holm_dir"])})
    return failures, detail


def check_reproducible(window: permute.Window, draws: int = 2048) -> list[str]:
    """Batching and re-running must not move a single count."""
    reference = window_counts(window, draws, batch=BATCH_SIZE)
    failures = []
    for batch in (1, 128, 512, 1024):
        if not reference.equals(window_counts(window, draws, batch=batch)):
            failures.append(f"{window.window_id}: batch {batch} changed the counts")
    if not reference.equals(window_counts(window, draws, batch=BATCH_SIZE)):
        failures.append(f"{window.window_id}: the same seed did not reproduce")
    return failures


def check_parallel_matches_sequential(windows: dict[str, permute.Window],
                                      draws: int = 2048,
                                      workers: int = 4) -> list[str]:
    sample = {name: windows[name] for name in sorted(windows)[:min(4, len(windows))]}
    sequential = all_counts(sample, draws, workers=1)
    parallel = all_counts(sample, draws, workers=workers)
    return ([] if sequential.equals(parallel)
            else ["parallel counts differ from sequential"])


COUNT_COLUMNS = ["source_question", "source_method", "target_question",
                 "bucket_start", "bucket_end", "n_trades", "n_wallets",
                 "n_pairs_ge3"]
SORT_COLUMNS = ["source_question", "target_question", "bucket_start"]


def check_expected_windows(windows: pd.DataFrame) -> list[str]:
    """Both window tables must land on exactly what the configuration frozen."""
    failures = []
    for role, expected in ((TRANSFER, EXPECTED_WINDOWS), (PLACEBO, EXPECTED_PLACEBOS)):
        got = (windows.loc[windows["role"] == role, COUNT_COLUMNS]
               .sort_values(SORT_COLUMNS).reset_index(drop=True))
        want = (pd.DataFrame(expected).sort_values(SORT_COLUMNS)
                .reset_index(drop=True))
        if len(got) != len(want):
            failures.append(f"mapped {len(got)} {role} windows, expected {len(want)}")
            continue
        for column in want.columns:
            differing = got.index[got[column].to_numpy() != want[column].to_numpy()]
            for index in differing:
                failures.append(f"{role} {want.loc[index, 'target_question']} from "
                                f"{want.loc[index, 'source_question']}: {column} is "
                                f"{got.loc[index, column]}, expected "
                                f"{want.loc[index, column]}")
    return failures


# ------------------------------------------------------------------ step 0
def config_path(repo_root: Path) -> Path:
    return repo_root.joinpath(*OUT_DIR, "q3_config.json")


def freeze(repo_root: Path) -> dict:
    """Write the frozen rules. Nothing else in Q3 may run before this exists.

    The ids are RNG inputs, so the id scheme has to be sealed before the first
    window is mapped; the file also carries the window table the mapping must
    reproduce, which is arithmetic over the frozen alarm table and contains no
    result.
    """
    out = config_path(repo_root)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(dumps(CONFIG))
    baseline = out.parent / Q2_BASELINE_FILE
    baseline.write_bytes(dumps(current_q2_products(repo_root)))
    return {"path": out.as_posix(), "sha256": sha256_file(out),
            "config_version": VERSION, "seed_base": SEED_BASE,
            "windows_expected": len(EXPECTED_WINDOWS),
            "q2_baseline_sha256": sha256_file(baseline)}


def require_frozen(repo_root: Path) -> None:
    """Refuse to run against anything but the configuration this code writes."""
    out = config_path(repo_root)
    if not out.is_file():
        raise AssertionError("the Q3 configuration is not frozen; run the freeze "
                             "step before mapping any window")
    if json.loads(out.read_text(encoding="utf-8")) != json.loads(dumps(CONFIG)):
        raise AssertionError("the frozen Q3 configuration differs from the one this "
                             "code would write; re-freeze deliberately or restore it")
    if not (out.parent / Q2_BASELINE_FILE).is_file():
        raise AssertionError("the Q2 product baseline was not frozen with Q3")


# ------------------------------------------------------- the transfer itself
def transfer_windows(windows: pd.DataFrame,
                     streams: dict[str, sources.Stream],
                     calibration: sources.Calibration,
                     with_secondary: bool = True
                     ) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    """Build every window's slots and cells, and settle the secondary arm.

    ``baseline_available`` is provisional until the path has actually been
    built: a contract can clear the burn-in rule and still hold an imputed or
    degenerate bucket inside the window. The flag is corrected here, so the
    window table can never say a window carries a DNC ledger that it does not.
    """
    windows = windows.copy()
    prepared: dict[str, permute.Window] = {}
    slot_frames, meta = [], {}
    for index, row in windows.iterrows():
        stream = streams[str(row["target_condition_id"])]
        slots, info = prepare_window(row, stream, calibration, with_secondary)
        if "baseline_rejected" in info:
            windows.loc[index, "baseline_available"] = False
            windows.loc[index, "baseline_unavailable_reason"] = info["baseline_rejected"]
        window = build_window(slots, str(row["q3_window_id"]), weight_names(slots))
        failures = permute.check_multiplicity(window)
        if failures:
            raise AssertionError(f"the shuffle does not preserve the null's "
                                 f"conditioning: {failures}")
        if str(row["q3_window_id"]) in prepared:
            raise AssertionError(f"two windows share an id: {row['q3_window_id']}")
        prepared[str(row["q3_window_id"])] = window
        slot_frames.append(slots)
        meta[str(row["q3_window_id"])] = info
    return (pd.concat(slot_frames, ignore_index=True), windows, prepared, meta)


def observed_rows(windows: pd.DataFrame, slots: pd.DataFrame,
                  prepared: dict[str, permute.Window], meta: dict,
                  crossing: dict[str, set[str]]) -> tuple[pd.DataFrame, list[str]]:
    """Per-pair observed statistics for every window, plus the conservation gate."""
    by_window = dict(tuple(slots.groupby("q3_window_id")))
    frames, failures = [], []
    for _, row in windows.iterrows():
        key = str(row["q3_window_id"])
        piece = by_window[key]
        rows = wallet_rows(row, piece, prepared[key], meta[key],
                           crossing.get(str(row["source_detector_run_id"]), set()))
        failures += check_conservation(piece, rows, meta[key],
                                       int(row["source_direction"]))
        frames.append(orbit_fields(piece, rows))
    return pd.concat(frames, ignore_index=True), failures


def describe_windows(windows: pd.DataFrame, slots: pd.DataFrame,
                     rows: pd.DataFrame, meta: dict) -> pd.DataFrame:
    """Attach the per-window counts and the membership digest."""
    counts = slots.groupby("q3_window_id").agg(
        n_trades=("slot_index", "size"), n_wallets=("active_wallet", "nunique"))
    eligible = (rows[rows["eligible"]].groupby("q3_window_id").size()
                .rename("n_pairs_ge3"))
    out = windows.merge(counts, left_on="q3_window_id", right_index=True)
    out = out.merge(eligible, left_on="q3_window_id", right_index=True, how="left")
    out["n_pairs_ge3"] = out["n_pairs_ge3"].fillna(0).astype("int64")
    out["membership_sha256"] = [meta[key]["membership_sha256"]
                                for key in out["q3_window_id"]]
    out["w_window"] = [meta[key]["w_window"] for key in out["q3_window_id"]]
    out["profile_cutoff"] = [meta[key]["profile_cutoff"] for key in out["q3_window_id"]]
    return out


# -------------------------------------------------------------- the real run
def run(repo_root: Path, workers: int = 1, draws: int = B) -> dict:
    """Map, test and adjudicate the real cluster -- once.

    The transferred windows and their placebo controls are built, permuted and
    adjudicated in the same pass and by the same code, so the contrast between
    them is a contrast between two window choices and nothing else: the same
    contracts, the same bucketing, the same statistic, the same draw count, the
    same alpha, and separate families.
    """
    require_frozen(repo_root)
    # Q2's products are snapshotted here, not compared against the freeze: Q2
    # being re-run between the Q3 freeze and this call is a provenance fact and
    # not a breach, and asserting on it made the guard fire on a wall-clock
    # timing field that no result depends on. What Q3 must not do is *write*
    # into Q2, and that is a claim about this run, checked against this snapshot
    # at the end of it.
    q2_at_start = current_q2_products(repo_root)
    if np.__version__ != CONFIG["permutation"]["numpy_version"]:
        raise AssertionError(f"the permutation null is pinned to numpy "
                             f"{CONFIG['permutation']['numpy_version']}, this is "
                             f"{np.__version__}")
    calibration = sources.load_calibration(repo_root)
    streams = {stream.condition_id: stream
               for stream in cluster_streams(repo_root, calibration)}
    windows = plan_windows(repo_root, calibration)
    crossing = source_window_wallets(repo_root, calibration, windows)

    slots, windows, prepared, meta = transfer_windows(windows, streams, calibration)
    rows, failures = observed_rows(windows, slots, prepared, meta, crossing)
    windows = describe_windows(windows, slots, rows, meta)

    blocked = check_expected_windows(windows)
    if blocked:
        raise AssertionError(f"the mapped windows differ from the frozen table: "
                             f"{blocked[:5]}")
    if failures:
        raise AssertionError(f"a window's statistics do not close: {failures[:5]}")

    smallest = min(prepared.values(), key=lambda w: len(w.labels))
    blocked = (check_reproducible(smallest)
               + check_parallel_matches_sequential(prepared,
                                                   workers=max(2, workers)))
    if blocked:
        raise AssertionError(f"the run is not reproducible: {blocked}")

    counts = all_counts(prepared, draws, workers=workers)
    rows = p_values(rows, counts, draws)
    e_regression = frozen_e_regression(rows, draws)
    rows = reachable(apply_multiplicity(rows))
    blocked = (check_p_values(rows, draws) + check_namespace(rows)
               + check_secondary_cannot_promote(rows))
    gate_7_failures, gate_7 = (([], {}) if draws != B else check_gate_7(rows))
    blocked += gate_7_failures
    if blocked:
        raise AssertionError(f"the p-values are not valid: {blocked}")

    out_dir = repo_root.joinpath(*OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    windows.to_parquet(out_dir / "q3_windows.parquet", index=False)
    slots.reindex(columns=SLOT_COLUMNS).to_parquet(
        out_dir / "q3_membership.parquet", index=False)
    rows.to_parquet(out_dir / "q3_wallet_windows.parquet", index=False)

    alarms = source_alarms(repo_root).assign(
        opened_a_q3_window=lambda frame: frame["detector_run_id"].isin(
            windows["source_detector_run_id"]))
    report = summary(alarms, windows, rows, draws, workers)
    q2_after = q2_products_moved(repo_root, q2_at_start)
    if any(q2_after.values()):
        raise AssertionError(f"Q3 changed a Q2 product: {q2_after}")
    report["gate_7_failures"] = []
    report["doj_gate"] = gate_7
    report["frozen_e_regression"] = e_regression
    report["q2_products_changed"] = q2_after
    (out_dir / "q3_summary.json").write_bytes(dumps(report))
    return report


def lead_lag(alarms: pd.DataFrame) -> dict:
    """Which contract of the cluster lit up first, and by how long.

    A by-product rather than a question: the source alarms are Q1 output, and
    once they are anchored to the earliest of them the ordering costs nothing to
    report. Every source is listed, including one whose only intersecting target
    was the window Q2 had already tested and which therefore opened no Q3 window
    of its own. It is descriptive only -- the contracts open at different times
    and carry very different depth, so an ordering is not a claim that one
    market led another in any causal sense.
    """
    anchor = float(alarms["alarm_available_utc"].min())
    ordered = alarms.sort_values(["alarm_available_utc", "detector_run_id"])
    return {
        "anchor_utc": anchor,
        "anchor_question": str(ordered["question"].iloc[0]),
        "alarms": [{"question": str(row.question),
                    "method": str(row.method),
                    "bucket_size": int(row.bucket_size),
                    "direction": int(row.direction),
                    "alarm_available_utc": float(row.alarm_available_utc),
                    "seconds_after_anchor":
                        float(row.alarm_available_utc) - anchor,
                    "opened_a_q3_window": bool(row.opened_a_q3_window)}
                   for row in ordered.itertuples()],
        "note": "descriptive only: the contracts open at different times and "
                "differ greatly in depth, so an ordering of alarm instants is "
                "not a claim that one market led another",
    }


def placebo_contrast(windows: pd.DataFrame, rows: pd.DataFrame) -> dict:
    """The study's windows beside the spans no alarm selected.

    Rates per eligible pair, not per window, because the two sets hold different
    numbers of pairs. Read it as a contrast and not as a false-alarm rate: these
    are real contracts with no ground truth, so a rejection inside a placebo
    window is not provably a false positive. What it can say is whether the
    transferred windows are distinguishable from unselected ones at all.
    """
    out = {}
    for role, family in ((TRANSFER, "primary"), (PLACEBO, "placebo")):
        piece = rows[rows["role"] == role]
        eligible = piece[piece["eligible"]]
        rejected = piece[f"reject_{family}"].fillna(False).astype(bool)
        out[role] = {
            "windows": int((windows["role"] == role).sum()),
            "pairs": int(len(piece)),
            "eligible_pairs": int(len(eligible)),
            "rejections": int(rejected.sum()),
            "rejection_rate_per_eligible_pair": (float(rejected.sum() / len(eligible))
                                                 if len(eligible) else None),
            "windows_with_a_rejection": int(
                piece.loc[rejected, "q3_window_id"].nunique()),
            "min_p_raw_e": float(piece["p_raw_e"].min()) if len(piece) else None}
    uncontrolled = windows[(windows["role"] == TRANSFER)
                           & ~windows["q3_window_id"].isin(
                               windows["controls_window_id"].dropna())]
    return {
        "by_role": out,
        "transferred_windows_without_a_placebo":
            sorted(str(name) for name in uncontrolled["q3_window_id"]),
        "rule": CONFIG["placebo"]["rule"],
        "reading": CONFIG["placebo"]["reading"],
        "families": "separate Holm families at the same alpha; never merged",
    }


def summary(alarms: pd.DataFrame, windows: pd.DataFrame, rows: pd.DataFrame,
            draws: int, workers: int) -> dict:
    """What the run produced, in the terms the study is allowed to make."""
    eligible = rows[rows["eligible"]]
    family_report = {}
    for suffix, (_, _, family_alpha) in HOLM_FAMILIES.items():
        members = rows[rows[f"in_family_{suffix}"]]
        sizes = members.groupby("q3_window_id").size()
        family_report[suffix] = {
            "alpha": family_alpha,
            "m": len(members),
            "windows": len(sizes),
            "family_size_range": ([int(sizes.min()), int(sizes.max())]
                                  if len(sizes) else None),
            "rejections": int(members[f"reject_{suffix}"].fillna(False).sum()),
            "windows_with_a_rejection": int(
                members.loc[members[f"reject_{suffix}"].fillna(False),
                            "q3_window_id"].nunique()),
            "min_adjusted_p": (float(members[f"p_holm_{suffix}"].min())
                               if len(members) else None),
            "orbit_reachable": int(members[f"orbit_reachable_{suffix}"]
                                   .fillna(False).sum()),
        }
    return {
        "study": CONFIG["study"],
        "scope_note": CONFIG["not_established"]["scope"],
        "standing": CONFIG["standing"],
        "config_version": VERSION,
        "seed_base": SEED_BASE,
        "draws": draws, "p_denominator": draws + 1, "workers": workers,
        "numpy_version": np.__version__,
        "windows": {
            "count": int(len(windows)),
            "transferred": int((windows["role"] == TRANSFER).sum()),
            "placebo": int((windows["role"] == PLACEBO).sum()),
            "trades": int(windows["n_trades"].sum()),
            "pairs": int(len(rows)),
            "eligible_pairs": int(len(eligible)),
            "self_windows": int(windows["is_self_window"].sum()),
            "with_secondary_arm": int(windows["baseline_available"].sum()),
            "detail": [{
                "q3_window_id": str(row.q3_window_id),
                "role": str(row.role),
                "controls_window_id": (None if pd.isna(row.controls_window_id)
                                       else str(row.controls_window_id)),
                "source_question": str(row.source_question),
                "source_method": str(row.source_method),
                "source_direction": int(row.source_direction),
                "target_question": str(row.target_question),
                "target_bucket_size": int(row.target_bucket_size),
                "bucket_span": [int(row.bucket_start), int(row.bucket_end)],
                "n_trades": int(row.n_trades), "n_wallets": int(row.n_wallets),
                "n_pairs_ge3": int(row.n_pairs_ge3),
                "is_self_window": bool(row.is_self_window),
                "baseline_available": bool(row.baseline_available),
                "baseline_unavailable_reason": (
                    None if pd.isna(row.baseline_unavailable_reason)
                    else str(row.baseline_unavailable_reason)),
                "w_window": (None if not np.isfinite(row.w_window)
                             else float(row.w_window)),
                "membership_sha256": str(row.membership_sha256)}
                for row in windows.itertuples()]},
        "families": family_report,
        "bh_screen": {"m": int(rows["q_bh_e"].notna().sum()),
                      "windows": int(rows.loc[rows["q_bh_e"].notna(),
                                                  "q3_window_id"].nunique()), "q": BH_Q,
                      "screened": int(rows["bh_screen_e"].sum()),
                      "caveat": "a review screen only: the pairs share slots, so "
                                "neither independence nor PRDS is established and "
                                "no false discovery rate is claimed"},
        "leakage": {
            "pairs_crossing_the_source_window": int(rows["crosses_source_window"].sum()),
            "eligible_crossing": int(eligible["crosses_source_window"].sum()),
            "rule": CONFIG["multiplicity"]["leakage_sensitivity"]},
        "both_tails": {
            "pairs": int(rows["both_tails_reject"].sum()),
            "note": "a pair extreme in both tails shows concentration without "
                    "direction and is descriptive only"},
        "secondary_sensitivity": {
            "headline_family": HEADLINE_FAMILY,
            "secondary_only_pairs": int(rows["secondary_only"].sum()),
            "note": "the DNC family runs over a different set of pairs -- only "
                    "the windows whose target has a clean burn-in -- so it is a "
                    "sensitivity and never a second headline; where the two "
                    "disagree, the disagreement is the result"},
        "p_raw_e": {"min": float(rows["p_raw_e"].min()),
                    "median": float(rows["p_raw_e"].median()),
                    "at_grid_floor": int((rows["n_exceed_e"] == 0).sum()),
                    "exactly_one": int((rows["p_raw_e"] == 1.0).sum())},
        "placebo_contrast": placebo_contrast(windows, rows),
        "lead_lag": lead_lag(alarms),
        "scope": CONFIG["scope"],
        "direction_assumption": CONFIG["direction_assumption"],
        "not_established": CONFIG["not_established"],
    }


# ------------------------------------------------------------------ export
ENGINE_MODULES = ("transfer.py",)


def engine_digests() -> dict[str, str]:
    package = Path(__file__).resolve().parent
    return {name: sha256_file(package / name) for name in ENGINE_MODULES}


def hashes(repo_root: Path) -> dict:
    """Everything needed to say which bytes produced the Q3 results."""
    out_dir = repo_root.joinpath(*OUT_DIR)
    reused = ("aggregate.py", "decompose.py", "ids.py", "multiplicity.py",
              "orbit.py", "permute.py", "sources.py")
    package = Path(__file__).resolve().parent
    inputs = ("data/detect/cusum_real_alarms.parquet",
              "data/detect/cusum_calibration.json",
              "data/processed/trades_event_level.parquet")
    return {
        "study": CONFIG["study"],
        "config_version": VERSION,
        "seed_base": SEED_BASE,
        "authoritative_inputs": {name: sha256_file(repo_root / name)
                                 for name in inputs},
        "engine": engine_digests(),
        "reused_q2_modules": {name: sha256_file(package / name) for name in reused},
        "outputs": {path.name: sha256_file(path) for path in sorted(out_dir.iterdir())
                    if path.is_file() and path.name != "q3_hashes.json"},
        "isolation": {
            "q2_products": "read-only; Q3 writes nothing under data/attrib/real, "
                           "data/attrib/sim or results/q2",
            "multiplicity": "the Q3 Holm family is never merged with the Q2 family",
            "ground_truth": "the study reads the real cluster only. No simulated "
                            "stream and no scenario manifest is opened anywhere in "
                            "this module, so no file it writes can carry a truth "
                            "label",
        },
        "self_reference": "q3_hashes.json is the only file of this directory it "
                          "does not hash, because it cannot hash itself",
    }


def current_q2_products(repo_root: Path) -> dict:
    """Every current Q2 product, hashed without trusting an older hash file."""
    products = {}
    for track in sources.TRACKS:
        out_dir = repo_root / "data" / "attrib" / track
        products[track] = {path.name: sha256_file(path)
                           for path in sorted(out_dir.iterdir()) if path.is_file()}
    return products


def q2_products_moved(repo_root: Path, baseline: dict) -> dict:
    """Which Q2 products differ from a snapshot taken earlier in this process.

    The claim being checked is containment -- *Q3 never writes into Q2* -- and
    containment is a property of one run, so the baseline is taken at the start
    of that run. Comparing against the Q3 freeze instead would fold in every
    legitimate Q2 re-run that happened in between, which is provenance and not
    a breach; the file Q3 froze is kept as that provenance record and is no
    longer an assertion.
    """
    current = current_q2_products(repo_root)
    moved = {}
    for track in sources.TRACKS:
        names = set(baseline[track]) | set(current[track])
        moved[track] = sorted(name for name in names
                              if baseline[track].get(name) != current[track].get(name))
    return moved


def export(repo_root: Path) -> dict:
    """Write the provenance file and assert this step wrote nothing into Q2."""
    q2_at_start = current_q2_products(repo_root)
    payload = hashes(repo_root)
    moved = q2_products_moved(repo_root, q2_at_start)
    if any(moved.values()):
        raise AssertionError(f"Q3 changed a Q2 product: {moved}")
    payload["q2_products_changed"] = moved
    repo_root.joinpath(*OUT_DIR, "q3_hashes.json").write_bytes(dumps(payload))
    return payload
