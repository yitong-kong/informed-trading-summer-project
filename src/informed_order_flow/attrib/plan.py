# -*- coding: utf-8 -*-
"""Step 0 of Q2: freeze the analysis plan, the run configuration and the code state.

Everything that can influence a wallet-level p-value is sealed here, before the
first run of either track. Three artefacts are written under ``data/attrib/``:

    q2_analysis_plan.json   what is being tested, on which population, and
                            which design questions are already adjudicated
    q2_config.json          the numeric rules, shared byte-for-byte by both
                            tracks (a copy lands in ``real/`` and ``sim/``)
    q2_code_state.json      per-file SHA-256 of the working tree

The two dictionaries below are the frozen text. They contain no timestamps and
no runtime values, so re-running step 0 on an unchanged tree reproduces the same
bytes and therefore the same hashes.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import ids

FROZEN_ON = "2026-08-23"
CONFIG_VERSION = "q2-config-2.0.0"
PLAN_VERSION = "q2-analysis-plan-2.0.0"

# Step S0 may change the test statistics and the multiplicity family, but must
# fail closed if an ID/RNG input has drifted. The check is a regression against
# the frozen cell table rather than against a declared digest of the previous
# config: a constant written down today cannot be falsified by anything in the
# repository, whereas every cell seed in ``permutation_cells.parquet`` was
# written before this revision and is recomputed here from the live id scheme.
CELL_TABLE = "permutation_cells.parquet"

# Baseline materials whose hashes go into the analysis plan. Paths are supplied
# on the command line so that no location is recorded in the source tree; the
# label set is fixed, and step 0 refuses to run with anything else.
REQUIRED_SPEC_LABELS = (
    "q2_design_spec",
    "q2_minimal_revision_spec",
    "q2_minimal_demo_readme",
    "q1_critique",
)

# Working-tree scope of the code-state record.
CODE_STATE_ROOTS = ("src", "scripts", "tests")
CODE_STATE_FILES = ("pyproject.toml", "environment.yml")
CODE_STATE_SKIP_PARTS = ("__pycache__", ".ipynb_checkpoints")

CONFIG = {
    "config_version": CONFIG_VERSION,
    "frozen_on": FROZEN_ON,
    "shared_by_tracks": ["real", "sim"],

    "id_scheme": {
        "separator": ids.SEP,
        "python_hash_forbidden": True,
        "templates": {
            "stream_id": "real | <scenario_id>",
            "window_id": "<stream_id>|<condition_id>|<channel>|K<K>|a<alarm_bucket>",
            "episode_id": "<stream_id>|<condition_id>|<channel>|K<K>",
            "detector_run_id": "<stream_id>|<condition_id>|<channel>|K<K>|<method>",
            "cell_id": "<window_id>|b<bucket_index>|<profile>",
            "cell_seed": "int.from_bytes(sha256('<seed_base>|<window_id>|<cell_id>')[:8], 'big')",
            "membership_sha256": ("sha256 over '<bucket_index>|<transaction_hash>\\n' "
                                  "for the slots of a run in frozen order"),
        },
        "serialisation": {
            "condition_id": "full length, lower case, 0x prefix retained, never truncated",
            "integers": ("no zero padding; bucket_size, alarm_bucket and bucket_index are "
                         "cast with int() and asserted integral, because the alarm tables "
                         "store alarm_bucket as a float and '103.0' would seed differently"),
            "tags": "a<alarm_bucket> for a window, b<bucket_index> for a cell",
            "channels": list(ids.CHANNELS),
            "methods": list(ids.METHODS),
            "profiles": list(ids.PROFILES),
        },
        "vectors": {
            "window_id": ("real|0xafc235557ace53ff0b0d2e93392314a7c3f3daab26a79050e985c1"
                          "1282f66df7|imbalance|K100|a103"),
            "episode_id": ("real|0xafc235557ace53ff0b0d2e93392314a7c3f3daab26a79050e985c"
                           "11282f66df7|imbalance|K100"),
            "detector_run_id": ("real|0xafc235557ace53ff0b0d2e93392314a7c3f3daab26a79050e"
                                "985c11282f66df7|imbalance|K100|cusum"),
            "cell_id": ("real|0xafc235557ace53ff0b0d2e93392314a7c3f3daab26a79050e985c1128"
                        "2f66df7|imbalance|K100|a103|b65|OLD_LARGE"),
            "cell_seed": 5806979916949404646,
            "membership_sha256": ("89a772de60482365358da717887da0653329bd7734b8431cd22f7b"
                                  "7f94e8ad27"),
            "vector_inputs": {
                "stream_id": "real",
                "condition_id": ("0xafc235557ace53ff0b0d2e93392314a7c3f3daab26a79050e985c"
                                 "11282f66df7"),
                "channel": "imbalance",
                "bucket_size": 100,
                "alarm_bucket": 103.0,
                "method": "cusum",
                "bucket_index": 65,
                "profile": "OLD_LARGE",
                "seed_base": 2026081601,
                "membership_slots": [[65, "0xabc"], [65, "0xdef"], [66, "0x001"]],
            },
        },
    },

    "inputs": {
        "verify": "SHA-256 checked before use; a mismatch aborts the freeze",
        "shared": {"data/detect/cusum_calibration.json":
                   "f01a6264f023bf6444135cf9bdcfd6be8642cffc89fb5622311650d24714e264"},
        "real": {"data/processed/trades_event_level.parquet":
                 "4834d015d855f2ca652b1865ebb2418986ae275e3315be3bb0fac280951288c6",
                 "data/detect/cusum_real_alarms.parquet":
                 "80688d44fd10ef3e82ae4759d05d18251d31171b15a778e4e74ef81299605f05"},
        "sim": {"note": ("data/detect/cusum_sim_eval.parquet and the 108 scenario trade "
                         "tables have no pre-registered digest; they are hashed and "
                         "recorded at freeze time")},
    },

    "windows": {
        "bucket_size": 100,
        "bucketing": ("sort_values(['condition_id', 'timestamp', 'transaction_hash']); "
                      "bucket_index = groupby('condition_id').cumcount() // K"),
        "trade_join_key": ["detector_run_id", "transaction_hash"],
        "primary_selection": ("imbalance channel, alarmed, bucket_size == 100; the HHI "
                              "channel is not invariant to wallet-label permutation and "
                              "never enters the formal path"),
        "episode": ("(stream_id, condition_id, channel, K); the method is deliberately "
                    "excluded so that a window found by both detectors is one episode. "
                    "The contract is kept because the real track is a single stream over "
                    "three contracts; on the simulated track a stream carries exactly one "
                    "condition id, so the field is constant and the grouping is unchanged"),
        "episode_representative": ("per episode keep the run with the "
                                   "largest statistic / threshold, ties broken by "
                                   "detector_run_id; method is demoted to a "
                                   "representative_method column, and every field of a "
                                   "canonical window must come from that one run"),
        "audit_window": "wide [onset_bucket, alarm_bucket]: roster and context only",
        "test_window": ("MLE [onset_bucket_mle, alarm_bucket]: contributions, permutation "
                        "and confirmatory eligibility"),
        "wide_only_wallets": ("kept in window_membership and trade_attribution as audit "
                              "context; they never enter a test and get no p-value"),
        "excluded": [
            {"key": "december_2026_k50",
             "reason": ("the December contract alarms only on a K=50 imbalance window; the "
                        "K=50 sensitivity analysis is dropped in this version and the "
                        "omission is stated explicitly in the paper")},
        ],
        "scan_coverage": ("a contract scanned at K=100 without an alarm is 'scanned, no "
                          "alarm'; the February contract was never scanned and must not be "
                          "described as scanned without alarm"),
        "real_primary": [
            {"condition_id": ("0xafc235557ace53ff0b0d2e93392314a7c3f3daab26a79050e985c112"
                              "82f66df7"),
             "question": "Maduro out in 2025?", "method": "cusum", "direction": -1,
             "wide_buckets": [48, 103], "mle_buckets": [65, 103],
             "statistic": 28.695404, "threshold": 28.499264,
             "wide_trades": 5600, "wide_wallets": 2979,
             "mle_trades": 3900, "mle_wallets": 2209},
            {"condition_id": ("0xa953bea944d7264285c0a2cc1f92809a7d9db78138b1c3de9cc23d89"
                              "17f14d6a"),
             "question": "Maduro out by November 30, 2025?", "method": "cusum",
             "direction": -1,
             "wide_buckets": [59, 88], "mle_buckets": [61, 88],
             "statistic": 15.551274, "threshold": 14.789786,
             "wide_trades": 3000, "wide_wallets": 1806,
             "mle_trades": 2800, "mle_wallets": 1711},
            {"condition_id": ("0x18d8c59309811ce5618ea941f9bde2a96afa5d876a69c42fba2da4bc"
                              "c56d3c5e"),
             "question": "Maduro out by March 31, 2026?", "method": "windowed_glr",
             "direction": 1,
             "wide_buckets": [78, 84], "mle_buckets": [78, 84],
             "statistic": 8.535656, "threshold": 7.982928,
             "wide_trades": 700, "wide_wallets": 318,
             "mle_trades": 700, "mle_wallets": 318},
        ],
    },

    "profile": {
        "labels": list(ids.PROFILES),
        "definition": {
            "NEW": "no active fill in the same stream before the official onset",
            "OLD_SMALL": "median pre-onset per-trade gross_shares at or below the cutoff",
            "OLD_LARGE": "median pre-onset per-trade gross_shares above the cutoff",
        },
        "cutoff": ("median of the pre-onset medians of the old wallets in this window's "
                   "MLE roster"),
        "history_scope": ("values come only from before the official onset of the same "
                          "stream; the comparison set is conditioned on the observed "
                          "roster, so this is not 'no post-onset information at all'"),
        "frozen_before_permutation": ["roster", "profile", "cutoff"],
        "fallback_branches": ("none; every simulated alarm run has onset_bucket >= n_burn "
                              "= 20, so no burn-in fallback is reachable, and adding one "
                              "would itself be a method difference between the tracks"),
    },

    "eligibility": {
        "confirmatory_rule": "n_trades_mle >= 3",
        "outside_reason_code": "fewer_than_3_mle_slots",
        "screening_family": "all wallet-window pairs, review screen only",
        "unit": "active_wallet x MLE window pair",
        "invariance": ("cell-wise shuffling preserves each wallet's per-cell multiplicity, "
                       "so n_trades_mle is constant over the whole permutation orbit; the "
                       "filter is therefore fixed before randomisation and does not "
                       "disturb FWER control"),
        "orbit": {
            "orbit_size": "prod_c C(m_c, k_wc) over the cells the wallet occupies",
            "p_orbit_floor": ("1 / orbit_size: a structural lower bound on the exact "
                              "orbit p-value, not a floor on the sampled Monte Carlo "
                              "p-value"),
            "p_mc_min": "1 / (B + 1): the finite Monte Carlo grid floor, reported separately",
            "arithmetic": ("orbit sizes are exact integers (math.comb) and reachability is "
                           "decided against alpha as the exact decimal 1/20, never a "
                           "floating log-gamma orbit size"),
            "boundary_rule": ("a pair whose orbit floor equals the threshold exactly is "
                              "reachable, because Holm rejects at p <= threshold; such "
                              "ties are counted in orbit_boundary_ties_* and the "
                              "superseded floating counts are kept as "
                              "*_legacy_float = exact - ties"),
            "float_range": ("1 / orbit_size is positive for every orbit but underflows "
                            "the double range past about 1e-308, so the floor is "
                            "published as p_orbit_floor_log10 = -log10(orbit_size), "
                            "always finite; p_orbit_floor is the readable rendering and "
                            "no decision is taken on it"),
            "required_fields": ["log_orbit_size", "p_orbit_floor", "p_orbit_floor_log10",
                                "m_screening",
                                "m_confirmatory", "orbit_reachable_screening_family",
                                "orbit_reachable_confirmatory_family"],
        },
    },

    "permutation": {
        "B": 1999999,
        "p_denominator": 2000000,
        "p_formula": "(1 + #{T_r >= T_obs}) / (B + 1)",
        "tie_rule": ">=",
        "seed_base": 2026081601,
        "n_seeds": 1,
        "rng": "numpy.random.Philox, one Generator per (window, cell)",
        "draw": ("successive permutation / permuted calls in frozen cell order; "
                 "random-key or argsort shuffles are forbidden"),
        "cell_definition": ["bucket_index", "profile"],
        "stratify_by_side": False,
        "stratify_by_side_reason": "direction is the signal under test",
        "singleton_cells": "kept as identity, never merged",
        "batch_size": 512,
        "checkpointing": False,
        "restart_policy": "a failed run restarts from the beginning",
        "low_power_window": {"window_fixed_slot_share_threshold": 0.20,
                             "flag": "low_power_window",
                             "note": "assigned before results, never re-stratified after"},
        "mc_review": {
            "statistic": ("mc_sigma_to_threshold = (p_raw - holm_threshold) / "
                          "sqrt(p_raw * (1 - p_raw) / B)"),
            "second_seed_trigger_sigma": 3.0,
            "scope": "only the window containing the borderline pair is re-run",
        },
    },

    "statistics": {
        "legs": {
            "magnitude": {
                "weight": "score_vdw",
                "rule": ("Phi^-1(R_b / (m_b + 1)) over d*q, ranked inside "
                         "(detector_run_id, bucket_index), MLE slots only, "
                         "ties averaged"),
            },
            "direction": {
                "weight": "score_sign",
                "rule": "sign(d * q), zero size scores 0",
            },
        },
        "headline": ["magnitude", "direction"],
        "alpha_split": 0.5,
        "best_of_forbidden": True,
        "dnc_role": "effect size, attribution ledger and sensitivity; never headline",
    },

    "multiplicity": {
        "study_unit": ("one window is one study; alpha is split equally over the two "
                       "pre-registered headline legs, and the family size m is data "
                       "determined within each window"),
        "family_unit": "window",
        "alpha": 0.05,
        "holm": {"family": ("confirmatory p-values within a study, run once per "
                            "headline leg"),
                 "legs": ["magnitude", "direction"],
                 "leg_alpha": 0.025,
                 "combination": ("union of the two leg rejections; each leg is judged "
                                 "against its own pre-registered alpha / 2 and never "
                                 "against the other leg's outcome"),
                 "status": "confirmed_repeat_active",
                 "note": ("the only tests that produce a headline status; selecting the "
                          "better-looking leg after the fact is forbidden")},
        "bh": {"q": 0.10, "family": "all p-values of a leg within a study",
               "legs": ["magnitude", "direction"],
               "status": "bh_review_screen",
               "caveat": ("wallets share slots, so independence / PRDS is unproven; this "
                          "is a review screen and carries no FDR claim")},
        "dnc_sensitivity": {"procedure": "separate Holm at alpha = 0.05 over confirmatory "
                                         "DNC p-values",
                            "note": ("sensitivity only; it can never promote a candidate. "
                                     "Its raw p-values double as the regression that "
                                     "proves the permutation engine was not disturbed, "
                                     "because seeds and draws are unchanged")},
        "dfa_sensitivity": {"procedure": "separate Holm at alpha = 0.05 over confirmatory "
                                         "DFA p-values",
                            "note": "sensitivity only; it can never promote a candidate"},
        "review_queue": {
            "status": "review_queue",
            "rule": ("a magnitude-leg Holm rejection whose raw p is above the "
                     "empirical threshold t_star measured on injection-free "
                     "replicas"),
            "note": ("not a finding and not discarded. The magnitude leg's "
                     "permutation null does not hold on real order flow, so its "
                     "nominal Holm rejection alone is a candidate for review; the "
                     "direction leg needs no such tier because its measured error "
                     "matches its nominal level")},
        "descriptive": {"status": "top10_descriptive",
                        "top_n": 10,
                        "rule": ("per window, top 10 on either headline leg with "
                                 "exposure > 0"),
                        "tie_rule": ("the cut is taken on the score, not the rank: "
                                     "every wallet sharing the 10th largest value "
                                     "comes in, so the tier can hold more than 10. "
                                     "Splitting an exact tie on arrival order would "
                                     "read as a difference in evidence where the "
                                     "statistic shows none"),
                        "tie_flag": "top10_tied_at_cut"},
        "status_priority": ["confirmed_repeat_active", "review_queue",
                            "bh_review_screen", "top10_descriptive"],
        "fwer_claim": ("the 5% study-wise FWER applies to the confirmed_repeat_active "
                       "decision of one window, split as alpha / 2 over the two headline "
                       "legs; error across studies is not controlled by Holm and must be "
                       "stated separately"),
    },

    "numerics": {
        "epsilon": 1e-9,
        "scale_floor": 0.01,
        "fail_closed": ("reject a run whose MLE window contains NaN, an imputed baseline, "
                        "or sigma <= scale_floor; report the exclusion, never patch it"),
        "conservation_tolerances": {"per_trade_dfa": 1e-10, "per_bucket_llr": 1e-8,
                                    "per_run_w_alarm": 1e-8},
        "descriptive_field_tolerance": ("descriptive timestamps and durations may differ "
                                        "from Q1 within a recorded tolerance; fields that "
                                        "enter a statistic must match bit for bit"),
        "numpy_version": "2.4.6",
    },

    "expected_counts": {
        "real": {
            "canonical_runs": 3, "mle_slots": 7400, "pairs": 4238, "distinct_wallets": 4096,
            "wallets_in_1_2_3_windows": [3964, 122, 10],
            "wide_trades": [5600, 3000, 700], "mle_trades": [3900, 2800, 700],
            "pairs_by_n_trades": {"1": 3001, "2": 810, "ge3": 427},
            "confirmatory_pairs_by_window": [235, 131, 61],
            "confirmatory_family": 427,
            "cells": {"total": 222, "by_window": [117, 84, 21], "single_wallet_cells": 1},
            # resolution diagnostics under one study per window at alpha / 2 per
            # leg; the superseded merged-family values are named in the amendment
            "holm_first_threshold_confirmatory_by_window": [1.0638297872340425e-4,
                                                            1.9083969465648855e-4,
                                                            4.0983606557377047e-4],
            "orbit_reachable_confirmatory_family": 381,
            "orbit_reachable_confirmatory_family_legacy_float": 381,
            "orbit_boundary_ties_confirmatory": 0,
            "orbit_boundary_ties_screening": 0,
            "full_family_reachable_at_first_threshold": 339,
            "full_family_max_holm_step": 192,
            # the family change is the only thing that moved DNC adjudication:
            # the v1.2.0 stream-pooled run is replayed from the same frozen
            # p-values and must reproduce bit for bit, and the window families
            # then reject one more pair
            "pooled_dnc_v1_2_0": {
                "rows": 427, "rejections": 6,
                "vector_sha256": ("c544fbdd8e82d77108251381f2fc9db4de33387028b68a"
                                  "4d3fd41800fdf4c979")},
            "window_family_dnc_rejections": 7,
            # G2's baseline: sha256 over "window_id|active_wallet|n_exceed_dnc|p"
            # for every roster pair in frozen order, measured under q2-config-1.2.0.
            # It lives here rather than in wallet_window_tests.parquet because the
            # pipeline rewrites that file, and a regression whose baseline its own
            # run overwrites proves nothing after step 9 has been executed once.
            "dnc_v1_2_0": {"rows": 4238,
                           "vector_sha256": ("31749fc630fcc7db9195ceab80c617d89e84577a3e856d"
                                             "ed377d5e7e1535b4ce")},
        },
        "sim": {
            "canonical_runs": 61, "distinct_episodes": 53, "studies": 61,
            "mle_slots": 87900, "pairs": 44723,
            "pairs_by_n_trades": {"1": 33026, "2": 7378, "ge3": 4319},
            "confirmatory_family_median": 39, "confirmatory_family_range": [4, 470],
            "cells": {"total": 2621, "single_wallet_cells": 0},
            "holm_first_threshold_confirmatory_median_approx": 6.41e-4,
            "orbit_reachable_confirmatory_family": 4207,
            "orbit_reachable_confirmatory_family_legacy_float": 4206,
            "orbit_boundary_ties_confirmatory": 1,
            "orbit_boundary_ties_screening": 0,
            "full_family_reachable_at_first_threshold": 2758,
            "full_family_max_holm_step": 243,
            # a simulated stream holds exactly one window, so pooling by stream
            # and by window are the same family here and the count cannot move
            "pooled_dnc_v1_2_0": {
                "rows": 4319, "rejections": 83,
                "vector_sha256": ("61f47cb863e6b85abf62c933361efa45f3b41ac59f1fc2"
                                  "066c7824a5e66f7471")},
            "window_family_dnc_rejections": 83,
            # G2's baseline: sha256 over "window_id|active_wallet|n_exceed_dnc|p"
            # for every roster pair in frozen order, measured under q2-config-1.2.0.
            # It lives here rather than in wallet_window_tests.parquet because the
            # pipeline rewrites that file, and a regression whose baseline its own
            # run overwrites proves nothing after step 9 has been executed once.
            "dnc_v1_2_0": {"rows": 44723,
                           "vector_sha256": ("54a52c257213feb26fe7667a390cf658094e6947ce3c89"
                                             "639f5afe296587de0b")},
            "canonical_method_split": {"windowed_glr": 33, "cusum": 28},
        },
    },
}

FAMILY_UNIT = CONFIG["multiplicity"]["family_unit"]
if FAMILY_UNIT not in {"window", "stream"}:
    raise ValueError(f"unsupported multiplicity family_unit: {FAMILY_UNIT!r}")

PLAN = {
    "plan_version": PLAN_VERSION,
    "frozen_on": FROZEN_ON,
    "frozen_at_step": 0,
    "rng_invariance_assertion": ("family_unit and statistics do not enter cell_seed; "
                                 "cell_seed remains a function only of seed_base, "
                                 "window_id and cell_id"),
    "question": ("within a frozen Q1 alarm window, which active wallets contribute more "
                 "alarm-direction signed volume than wallet labels alone can explain"),

    "prior_spec_available": False,
    "prior_spec_note": ("The earlier v4 / v5 specification files are no longer in the "
                        "working tree and no hashes are reconstructed for them. The "
                        "baseline materials that do still exist are hashed below."),
    "development_contact_disclosure": ("An early read-only prototype used B = 199,999 and "
                                       "temporary seeds to inspect anonymised aggregate "
                                       "rejection counts. No address or ranking was "
                                       "disclosed to the design owner and no contemporaneous "
                                       "config file exists to hash. This contact must be "
                                       "stated in the methods section of the paper."),

    "estimand": {
        "name": ("alarm-direction rank position (magnitude leg) and alarm-direction "
                 "trade count (direction leg)"),
        "statement": ("Two pre-registered estimands are tested side by side inside the "
                      "MLE window. The magnitude leg is a wallet's summed van der "
                      "Waerden score of d * q_i ranked within its bucket: where its "
                      "trades sit in the bucket's order, with bounded influence, so a "
                      "single block trade cannot set the width of the null. The "
                      "direction leg is its count of with-the-alarm trades minus "
                      "against-the-alarm trades, which carries no size at all. They "
                      "answer different questions and are expected to select different "
                      "wallets."),
        "not_claimed": ("Neither leg claims to separate direction from size in a single "
                        "number; that is why there are two of them and why the paper "
                        "reports them apart."),
        "alpha_allocation": ("alpha / 2 to each leg, Bonferroni over the two, fixed "
                             "before any p-value of either leg was seen; the headline "
                             "set is the union of the two rejections and never the "
                             "better-looking one"),
        "dnc_role": ("DNC -- the alarm-direction signed size contribution, DNC_i = C_b * "
                     "q_i within a bucket -- is retained as the effect size, the "
                     "attribution ledger and a sensitivity. It is no longer a headline "
                     "statistic: its permutation null is too wide on real order flow, "
                     "where the variance is set by the heavy tail of |q| inside a cell."),
        "amendment_status": ("The two headline legs are a prospective revision of the "
                             "analysis plan, recorded in amendments below and made "
                             "before any address-level run under them; they are not an "
                             "original pre-registration. The superseded revision, which "
                             "made DNC the headline, is likewise recorded and its "
                             "results remain reproducible as the sensitivity."),
        "dfa_role": ("DFA = DNC + AGC is an exactly conserved ledger summing to W_alarm, "
                     "reported as a sensitivity. Because the AGC part can be spread by any "
                     "weights summing to one without breaking conservation, wallet DFA "
                     "rankings are not a second headline."),
        "concentration_note": ("A single trade dominating a cell does not manufacture a "
                               "false positive: the permutation gives that wallet an "
                               "equally high value in the random worlds. Concentration "
                               "costs power, and the orbit audit reports that honestly."),
    },

    "confirmatory_population": {
        "rule": "n_trades_mle >= 3",
        "outside_reason_code": "fewer_than_3_mle_slots",
        "real_family_size": 427,
        "timing": ("fixed after inspecting membership structure and before any "
                   "address-level p-value; recorded as a prospective analysis-plan "
                   "amendment"),
        "scope_change": ("formal findings apply only to repeatedly active wallets; the "
                         "research object changes with the rule"),
        "justification": ("the resolution ceiling, not alpha, is the binding constraint: "
                          "at the 4,238-pair first Holm threshold only 323 pairs could "
                          "ever reach it, whereas 374 of the 427 confirmatory pairs reach "
                          "the narrowed threshold. The 374 was measured under the "
                          "superseded merged family; the count under one study per window "
                          "at alpha / 2 per leg is recomputed by the orbit step and "
                          "recorded there, and the argument is directional rather than "
                          "dependent on the exact figure"),
        "validity_argument": ("conditional on the frozen windows, cells and per-wallet "
                              "per-cell multiplicities: (1) eligibility is identical across "
                              "the whole orbit, (2) each true-null randomisation p-value is "
                              "still super-uniform, (3) Holm controls FWER over the "
                              "confirmatory items of a study under arbitrary dependence, "
                              "(4) no formal rejection is ever produced outside the "
                              "subfamily, so study-wise FWER stays at or below 0.05, "
                              "split as alpha / 2 over the two headline legs"),
        "prohibited_wording": [
            "the remaining 3,811 pairs are mathematically untestable",
            "the remaining 3,811 pairs show no anomaly",
            "an empty result confirms that no wallet traded on private information",
        ],
        "empty_result_wording": ("Among the 427 repeatedly active wallet-window pairs, no "
                                 "confirmed anomaly passed the 5% FWER threshold. The other "
                                 "3,811 pairs lie outside the confirmatory estimand and no "
                                 "claim is made about them."),
    },

    "result_tiers": {
        "confirmed_repeat_active": ("rejected by either headline leg's Holm within the "
                                    "confirmatory family of its window, each leg at "
                                    "alpha / 2"),
        "bh_review_screen": ("selected by a leg's BH screen over all pairs; not a formal "
                             "finding"),
        "top10_descriptive": "per-window top 10 on either headline leg; descriptive only",
    },

    "inherited_limitations": [
        ("active_wallet is a proxy for the owner of the aggressive order, not a natural "
         "person: one actor may hold several addresses and a wallet purely providing "
         "passive liquidity is structurally invisible rather than merely low-scoring"),
        ("the frozen in-2025 alarm exceeds its threshold by only 0.688% "
         "(28.695404 / 28.499264 = 1.00688) yet supplies 2,209 / 4,238 = 52.1% of screened "
         "pairs and 235 / 427 = 55.0% of the confirmatory family"),
        ("Q2 is wallet attribution conditional on a label-invariant audit window; it "
         "neither confirms nor rescues the Q1 change-point significance"),
        ("the '1 alarm in 5 seeds' figure and the maxT coefficients in the Q1 critique "
         "belong to a reverted raw-path bootstrap version and are not a robustness "
         "recomputation of the frozen detector"),
    ],

    "adjudicated": [
        {"issue": "restrict the formal family to MLE slots >= 3",
         "ruling": ("accepted; the formal question narrows to repeatedly active wallets")},
        {"issue": "claim that 3,915 pairs can never be rejected",
         "ruling": ("rejected; 1 / orbit size is the most optimistic bound on the exact "
                    "orbit p-value, not a hard floor on a finite Monte Carlo p-value")},
        {"issue": "reduce B to 199,999",
         "ruling": "rejected; B stays at 1,999,999, compute is not the bottleneck"},
        {"issue": "promote size-binned DNC to a co-primary analysis",
         "ruling": ("rejected; DNC still carries size variation inside a size bin and a "
                    "second primary analysis would add cross-analysis multiplicity")},
        {"issue": "direction-count sensitivity",
         "ruling": ("dropped in this version to protect the schedule; the cost is carried "
                    "by the estimand declaration")},
        {"issue": "audit seed / two-seed intersection",
         "ruling": ("dropped; replaced by a single seed plus conditional review via "
                    "mc_sigma_to_threshold")},
        {"issue": "K=50 sensitivity",
         "ruling": "dropped; the omission must be stated in the paper"},
        {"issue": "wallet_candidates.parquet",
         "ruling": "dropped; top-N lists are generated from the tests table when writing"},
        {"issue": "full provenance system",
         "ruling": "downgraded to freeze_build_id plus q2_hashes.json"},
        {"issue": "whether an episode includes the method",
         "ruling": ("it does not; an episode is (stream, channel, K) with one complete "
                    "representative run")},
        {"issue": "generate sim_truth.parquet",
         "ruling": ("not generated; the evaluator reads the 108 manifests directly, so no "
                    "file under data/attrib/ contains ground truth")},
        {"issue": "modify the pipeline to avoid the condition-id substring coincidence",
         "ruling": ("no modification; assertion A proves behaviourally that no rule reads "
                    "address text")},
        {"issue": "role of the simulated grid",
         "ruling": ("simulation is validation and a fallback, the real Maduro contracts are "
                    "the research target; no repositioning of simulation as primary "
                    "evidence")},
    ],

    "known_arithmetic_correction": ("With B = 199,999 and M = 4,238 the first Holm "
                                    "threshold admits floor(200000 * 0.05 / 4238) = 2 Monte "
                                    "Carlo grid points, not 4; M = 427 gives 23."),

    "amendments": [
        {"date": "2026-08-23", "step": 4,
         "change": ("the resolution diagnostics in expected_counts are restated under one "
                    "study per window at alpha / 2 per leg: real orbit-reachable "
                    "confirmatory 374 -> 381, screening 323 -> 339, max Holm step "
                    "325 -> 192; simulated confirmatory 4,276 -> 4,207 and its "
                    "legacy_float 4,275 -> 4,206"),
         "reason": ("these counts are a function of the family boundary and the alpha a "
                    "pair is measured against, both of which the same-day step-0 "
                    "amendment changed. Nothing about the orbits themselves moved: orbit "
                    "sizes, cells and cell seeds are bit-identical, and the counts are "
                    "recomputed from the persisted exact log floors against the new "
                    "thresholds. The one simulated boundary tie recorded on 2026-08-18 "
                    "survives the change, so the exact-integer treatment it argued for "
                    "still separates the exact count from the legacy floating one."),
         "effect": ("diagnostics only; no rule, family, seed, threshold or p-value is "
                    "defined by them. They are the regression gate 8 measures a rerun "
                    "against, so they are stated here rather than written into the "
                    "engine, where a track-specific literal would both break the "
                    "single-fork-point rule and make the count unfalsifiable")},
        {"date": "2026-08-23", "step": 0,
         "change": ("the headline statistic becomes two pre-registered orthogonal legs -- "
                    "magnitude (bucket van der Waerden score) and direction (bucket sign "
                    "count) -- each judged at alpha / 2; a study becomes one window rather "
                    "than one stream; and DNC is demoted from headline to effect size, "
                    "ledger and sensitivity"),
         "reason": ("DNC's permutation null is too wide on real order flow: its variance "
                    "is set by the heavy tail of |q| inside a cell, so a wallet that "
                    "merely holds one block trade looks extreme while a wallet that tilts "
                    "many ordinary trades does not. The simulated grid shows the failure "
                    "directly -- injected wallets rank at the top by DNC yet carry "
                    "p-values around 0.28 (direction tilt) and 0.066 (size tilt) -- so "
                    "the ordering carries the information and the null width destroys it. "
                    "Merging a stream's windows into one family also judged the real "
                    "track far more strictly than the simulated track on which recall was "
                    "calibrated, and pushed the smallest achievable p of some pairs below "
                    "their own orbit floor, making them unrejectable in principle."),
         "effect": ("the test statistics and the family boundary change; the permutation "
                    "engine does not. Cells, cell seeds, the orbit sizes, B, the seed "
                    "base, the Philox addressing and every step 0-1 freeze table are "
                    "untouched, and the DNC p-values are recomputed from the same draws "
                    "as a bit-identical regression. Both legs were fixed, with their "
                    "alpha split, before any p-value of either was seen on the real "
                    "track; taking the better-looking leg after the fact is prohibited."),
         "supersedes": ("the revision that made DNC the headline statistic; that analysis "
                        "remains reproducible and is reported as the DNC sensitivity")},
        {"date": "2026-08-18", "step": 4,
         "change": ("the simulated orbit-reachable confirmatory count is 4,276 of 4,319, "
                    "superseding 4,275"),
         "reason": ("study L2_size_tilt_s1000_no holds a pair with orbit size 1560 in a "
                    "family of 78, so 1 / 1560 = alpha / 78 exactly. Holm rejects at "
                    "p <= threshold, so the pair is reachable. The superseded value came "
                    "from a floating log-gamma orbit size, which places the tie a few ulps "
                    "outside the threshold; this version computes orbit sizes as exact "
                    "integers and compares against alpha as the exact decimal 1/20."),
         "effect": ("a resolution diagnostic only. No rule, family, seed, threshold or "
                    "p-value changes, and the real track is untouched because it has no "
                    "boundary tie. Both numbers stay in the configuration and the gate "
                    "checks the identity exact - ties == legacy, so the superseded count "
                    "remains reproducible.")},
        {"date": "2026-08-18", "step": 4,
         "change": ("the orbit floor is published as p_orbit_floor_log10 alongside "
                    "p_orbit_floor"),
         "reason": ("1 / orbit_size is positive for every orbit, but the largest orbits "
                    "run past 10^400 and the double rendering of their floor saturates at "
                    "zero -- nine simulated pairs, the smallest true floor near 1e-447. A "
                    "stored zero asserts that the p-value can be arbitrarily small, which "
                    "is the opposite of what the field means."),
         "effect": ("a reporting field only. No count, threshold or decision changes; "
                    "every comparison was already made in exact or logarithmic form, and "
                    "the checks now assert that a zero in p_orbit_floor can only be the "
                    "float rendering giving up.")},
    ],
}


def dumps(payload: dict) -> bytes:
    """Deterministic JSON bytes."""
    return (json.dumps(payload, indent=2, ensure_ascii=True) + "\n").encode("utf-8")


# A step report is provenance, and provenance that cannot be reproduced is worse
# than none: it makes every guard that hashes it fire on something no result
# depends on. Two fields are like that and both are dropped before the report is
# written, while still being returned to the caller, which prints them.
#
#   elapsed_seconds        wall-clock time.
#   wallet_windows.parquet the mutable accumulator. Every step reads it, adds its
#                          columns and writes it back, and no step strips the
#                          columns a later step wrote, so its bytes at any given
#                          step depend on what this machine ran before rather
#                          than on the data. Its final state is hashed once, in
#                          q2_hashes.json, together with every other file.
VOLATILE_REPORT_FIELDS = ("elapsed_seconds",)
VOLATILE_OUTPUTS = ("wallet_windows.parquet",)


def persist_report(path: Path, report: dict) -> dict:
    """Write a step report with the fields no result depends on removed."""
    payload = {key: value for key, value in report.items()
               if key not in VOLATILE_REPORT_FIELDS}
    if "outputs" in payload:
        payload["outputs"] = {name: digest
                              for name, digest in payload["outputs"].items()
                              if name not in VOLATILE_OUTPUTS}
    path.write_bytes(dumps(payload))
    return report


def subtree_sha256(payload: dict) -> str:
    """Canonical digest for a JSON subtree, independent of dictionary key order."""
    serialised = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                            ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(serialised).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def code_state(repo_root: Path) -> dict:
    """Per-file SHA-256 of the working tree, plus a single digest over that listing.

    Version control is deliberately not consulted: this project stays local until
    it is complete, so the working tree itself is the provenance record.
    """
    files: dict[str, str] = {}
    for name in CODE_STATE_ROOTS:
        root = repo_root / name
        if not root.is_dir():
            raise FileNotFoundError(f"code-state root is missing: {name}")
        for path in sorted(root.rglob("*")):
            if path.is_file() and not any(part in CODE_STATE_SKIP_PARTS
                                          for part in path.parts):
                files[path.relative_to(repo_root).as_posix()] = sha256_file(path)
    for name in CODE_STATE_FILES:
        path = repo_root / name
        if path.is_file():
            files[name] = sha256_file(path)

    listing = "".join(f"{name} {digest}\n" for name, digest in sorted(files.items()))
    return {
        "scope": {"roots": list(CODE_STATE_ROOTS), "files": list(CODE_STATE_FILES),
                  "skipped": list(CODE_STATE_SKIP_PARTS)},
        "version_control": ("not consulted; the per-file working-tree digests below are "
                            "the provenance record for this version"),
        "python": ".".join(str(part) for part in sys.version_info[:3]),
        "numpy": CONFIG["numerics"]["numpy_version"],
        "file_count": len(files),
        "tree_sha256": hashlib.sha256(listing.encode("utf-8")).hexdigest(),
        "files": dict(sorted(files.items())),
    }


def spec_document_hashes(spec_docs: dict[str, Path]) -> list[dict]:
    """Hash the baseline materials; only the label is recorded, never the path."""
    missing = set(REQUIRED_SPEC_LABELS) - set(spec_docs)
    unknown = set(spec_docs) - set(REQUIRED_SPEC_LABELS)
    if missing or unknown:
        raise ValueError(f"spec documents must be exactly {REQUIRED_SPEC_LABELS}; "
                         f"missing={sorted(missing)} unknown={sorted(unknown)}")
    return [{"label": label,
             "bytes": spec_docs[label].stat().st_size,
             "sha256": sha256_file(spec_docs[label])}
            for label in REQUIRED_SPEC_LABELS]


def cell_seed_regression(repo_root: Path) -> dict:
    """Recompute every frozen cell seed from the live id scheme.

    ``cell_seed`` is ``SHA-256(seed_base | window_id | cell_id)`` truncated to
    eight bytes, so this reproduces the whole RNG address space of both tracks
    from the ids module and the seed base alone. A revision that leaves it intact
    has demonstrably not moved an RNG input; one that does not is refused here,
    before anything is written.

    A track with no frozen cell table yet is recorded as unavailable rather than
    silently passing, so the report always says what was actually checked.
    """
    seed_base = int(CONFIG["permutation"]["seed_base"])
    checked = {}
    for track in CONFIG["shared_by_tracks"]:
        path = repo_root / "data" / "attrib" / track / CELL_TABLE
        if not path.exists():
            checked[track] = {"status": "unavailable", "cells": 0}
            continue
        cells = pd.read_parquet(path, columns=["window_id", "cell_id", "cell_seed"])
        recomputed = np.array([ids.cell_seed(seed_base, window, cell)
                               for window, cell in zip(cells["window_id"],
                                                       cells["cell_id"])],
                              dtype="uint64")
        drifted = int((recomputed != cells["cell_seed"].to_numpy(dtype="uint64")).sum())
        if drifted:
            raise AssertionError(
                f"{drifted} of {len(cells)} frozen cell seeds no longer reproduce on the "
                f"{track} track; an ID or RNG input has moved and step 0 must not be "
                "re-frozen over a live permutation address space")
        checked[track] = {"status": "reproduced", "cells": int(len(cells))}
    return checked


def write_step0(repo_root: Path, spec_docs: dict[str, Path]) -> dict:
    """Write the three step-0 artefacts and return the closure report."""
    id_scheme_sha256 = subtree_sha256(CONFIG["id_scheme"])
    seed_regression = cell_seed_regression(repo_root)

    out_dir = repo_root / "data" / "attrib"
    track_dirs = [out_dir / track for track in CONFIG["shared_by_tracks"]]
    for directory in [out_dir, *track_dirs]:
        directory.mkdir(parents=True, exist_ok=True)

    # One serialisation, copied verbatim, so the two tracks cannot drift apart.
    config_bytes = dumps(CONFIG)
    config_paths = [out_dir / "q2_config.json"] + [d / "q2_config.json" for d in track_dirs]
    for path in config_paths:
        path.write_bytes(config_bytes)
    config_hashes = {path.relative_to(repo_root).as_posix(): sha256_file(path)
                     for path in config_paths}
    if len(set(config_hashes.values())) != 1:
        raise AssertionError(f"q2_config.json differs across tracks: {config_hashes}")

    plan = dict(PLAN)
    plan["specification_documents"] = spec_document_hashes(spec_docs)
    plan["id_scheme_regression"] = {
        "id_scheme_sha256": id_scheme_sha256,
        "evidence": ("every cell seed of every frozen cell table recomputed from the live "
                     "id scheme and the seed base; the digest above names the subtree "
                     "that was exercised, it is not itself the check"),
        "frozen_cell_seeds": seed_regression,
    }
    plan["q2_config"] = {"config_version": CONFIG_VERSION,
                         "sha256": next(iter(config_hashes.values()))}
    plan_path = out_dir / "q2_analysis_plan.json"
    plan_path.write_bytes(dumps(plan))

    state = code_state(repo_root)
    state_path = out_dir / "q2_code_state.json"
    state_path.write_bytes(dumps(state))

    return {
        "config_sha256": next(iter(config_hashes.values())),
        "config_paths": config_hashes,
        "id_scheme_sha256": id_scheme_sha256,
        "frozen_cell_seeds": seed_regression,
        "plan_sha256": sha256_file(plan_path),
        "code_state_sha256": sha256_file(state_path),
        "code_state_files": state["file_count"],
        "code_tree_sha256": state["tree_sha256"],
        "spec_documents": plan["specification_documents"],
    }
