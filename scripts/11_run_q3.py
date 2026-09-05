# -*- coding: utf-8 -*-
"""Entry point: Q3 -- cross-contract transfer of an online alarm window.

Q1 watches one contract at a time, so a contract can stay silent while a wallet
builds a large one-sided position in it. Q3 conditions on a different event: an
online imbalance alarm on *some* contract of the event cluster, whose wall-clock
interval is then mapped onto the frozen buckets of the other contracts, and the
wallets trading there are tested with Q2's permutation machinery. The detector is
untouched and the mapping happens after the alarm, so Q1 stays online.

This study runs on the real event cluster only. No simulated stream is read and
no scenario manifest can be opened from here, so nothing it writes can carry a
ground-truth label.

Every transferred window is built together with a placebo: the nearest span of
the same length on the same contract that no alarm selected. The two are
permuted by the same code at the same alpha in separate Holm families, and the
contrast between them is reported beside the results. Read it as a contrast and
not as a false-alarm rate -- these are real contracts with no ground truth, so a
rejection inside a placebo window is not provably a false positive.

Three subcommands, in this order:

    freeze  write data/attrib/q3/q3_config.json -- the source set, the mapping
            rule, the placebo rule, the statistics, the seed base, alpha, B and
            the id scheme. The ids seed the permutation streams, so nothing may
            run before this.
    run     map the real cluster, test it and adjudicate it -- once.
    export  write data/attrib/q3/q3_hashes.json and assert no Q2 product moved.

    python scripts/11_run_q3.py freeze
    python scripts/11_run_q3.py run --workers 12
    python scripts/11_run_q3.py export

The run is one shot. Nothing about the rules may be adjusted after its output has
been looked at, which is what freezing them into a file first is for.

Q3 is exploratory. Its family-wise error claim covers its own family only, it is
never merged with Q2's, and no Q3 result may change a Q1 threshold or a Q2 rule.
"""
import argparse
from pathlib import Path

from informed_order_flow.attrib import transfer as q3

REPO_ROOT = Path(__file__).resolve().parents[1]


def _cmd_freeze(args: argparse.Namespace) -> None:
    del args
    report = q3.freeze(REPO_ROOT)
    print("step 0 -- Q3 rules frozen")
    print(f"  config  {report['sha256']}  {report['path']}")
    print(f"  q2      current products frozen read-only "
          f"({report['q2_baseline_sha256']})")
    print(f"  seeds   seed_base={report['seed_base']} (Q2 uses "
          f"{q3.permute.SEED_BASE} and {q3.multiplicity.REVIEW_SEED_BASE})")
    print(f"  windows {report['windows_expected']} pre-registered by the mapping rule")


def _cmd_run(args: argparse.Namespace) -> None:
    report = q3.run(REPO_ROOT, workers=args.workers, draws=args.draws)
    windows, families = report["windows"], report["families"]
    print("run -- the real cluster")
    print(f"  windows {windows['transferred']} transferred + {windows['placebo']} "
          f"placebo, {windows['trades']} trades, {windows['pairs']} pairs, "
          f"{windows['eligible_pairs']} eligible "
          f"({windows['with_secondary_arm']} windows carry the DNC arm)")
    for suffix, values in families.items():
        print(f"  {suffix:13s} Holm at alpha={values['alpha']} over "
              f"{values['m']} pairs / {values['windows']} windows "
              f"(m range {values['family_size_range']}): "
              f"{values['rejections']} rejections, "
              f"{values['orbit_reachable']} could reach the first threshold")
    contrast = report["placebo_contrast"]["by_role"]
    for role, values in contrast.items():
        rate = values["rejection_rate_per_eligible_pair"]
        print(f"  {role:9s} {values['rejections']} rejections over "
              f"{values['eligible_pairs']} eligible pairs in {values['windows']} "
              f"windows" + (f" ({rate:.4f} per pair)" if rate is not None else ""))
    missing = report["placebo_contrast"]["transferred_windows_without_a_placebo"]
    if missing:
        print(f"          {len(missing)} transferred window(s) had no room for a "
              f"placebo on their contract")
    print(f"  screen  {report['bh_screen']['screened']} of "
          f"{report['bh_screen']['m']} pairs at q={report['bh_screen']['q']} "
          f"(review only)")
    print(f"  leakage {report['leakage']['eligible_crossing']} eligible pairs also "
          f"traded in the source window; both families are reported")
    print(f"  tails   {report['both_tails']['pairs']} pair(s) extreme in both tails; "
          f"{report['secondary_sensitivity']['secondary_only_pairs']} rejected by "
          f"the DNC sensitivity alone")
    if report["doj_gate"]:
        doj = report["doj_gate"]
        print(f"  G7      DOJ m={doj['m_primary']}, e/mag/dir="
              f"{doj['reject_primary']}/{doj['reject_mag']}/{doj['reject_dir']}, "
              f"ranks e/mag={doj['rank_e']}/{doj['holm_rank_mag']}")
    lag = report["lead_lag"]
    print(f"  lag     alarms after {lag['anchor_question']}: "
          + ", ".join(f"{entry['question']} +{entry['seconds_after_anchor'] / 3600:.1f}h"
                      for entry in lag["alarms"][1:]))


def _cmd_export(args: argparse.Namespace) -> None:
    del args
    payload = q3.export(REPO_ROOT)
    print("export -- provenance written")
    print(f"  engine  {payload['engine']['transfer.py']}  transfer.py")
    print(f"  outputs {len(payload['outputs'])} file(s) under data/attrib/q3/")
    print(f"  q2      unchanged: {payload['q2_products_changed']}")


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    freeze_ap = sub.add_parser("freeze", help="step 0: freeze the Q3 rules")
    freeze_ap.set_defaults(func=_cmd_freeze)

    run_ap = sub.add_parser("run", help="the real cluster, once")
    run_ap.add_argument("--workers", type=int, default=1,
                        help="windows to permute in parallel (default: 1)")
    run_ap.add_argument("--draws", type=int, default=q3.B,
                        help=f"permutations per window (default: {q3.B})")
    run_ap.set_defaults(func=_cmd_run)

    export_ap = sub.add_parser("export", help="q3_hashes.json and the Q2 isolation check")
    export_ap.set_defaults(func=_cmd_export)
    return ap


def main() -> None:
    args = _build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
