# -*- coding: utf-8 -*-
"""Entry point: the Q2 wallet-attribution study (real Maduro contracts + simulated grid).

Subcommands follow the ten pipeline steps.

    plan    step 0 -- freeze the analysis plan, the run configuration and the
            code state under data/attrib/
    freeze  step 1 -- replay Q1 on a track and write its five freeze tables
    decompose  step 2 -- split the alarm statistic into per-trade DNC/AGC/DFA
    aggregate  step 3 -- sum contributions per wallet and freeze its history
               profile, the roster and the cutoff
    orbit      step 4 -- confirmatory eligibility, orbit floors and the
               resolution ceiling of each family
    permute    step 5 -- permutation cells, per-cell seeds and the shuffle engine
    pvalues    step 6 -- draw B control worlds and turn the counts into p-values
    multiplicity  step 7 -- Holm on the confirmatory family, BH as a review
                  screen, and a separate Holm on DFA as a sensitivity
    evaluate   step 8 -- simulated ground truth: stratified recall, engine
               correctness and the two truth-isolation assertions
    run     steps 2-9 for one track in order, ending in the delivered table
    export  step 9 -- the provenance file of both tracks

Step 0 seals every input that can move a wallet-level p-value before either
track runs for the first time. In particular the composite-key id scheme is an
RNG input (each cell's Philox stream is seeded from window_id and cell_id), so
it must be frozen first; changing it later forces both tracks to be re-run.

The baseline specification documents are hashed into the analysis plan, but
their locations are passed on the command line so that no path is recorded in
the repository. The three labels are fixed and all three are required.

Both tracks run the same replay code; the only fork is which files are read
and which columns have to be filled in (see attrib/sources.py).

The whole study is six commands. The simulated track runs first and is the
end-to-end acceptance; the real track is run once, after that acceptance, and
is not re-run to chase a result:

    python scripts/09_run_q2.py freeze --track sim  --config data/attrib/q2_config.json
    python scripts/09_run_q2.py freeze --track real --config data/attrib/q2_config.json
    python scripts/09_run_q2.py run      --track sim
    python scripts/09_run_q2.py evaluate --track sim
    python scripts/09_run_q2.py run      --track real
    python scripts/09_run_q2.py export

``run`` chains steps 2 to 9 for one track; each step is also available on its
own, with the same code and the same outputs, for working through the pipeline:

    python scripts/09_run_q2.py plan \
        --spec-doc q2_design_spec=<path> \
        --spec-doc q2_minimal_demo_readme=<path> \
        --spec-doc q1_critique=<path>
    python scripts/09_run_q2.py decompose --track both
    python scripts/09_run_q2.py aggregate --track both
    python scripts/09_run_q2.py orbit --track both
    python scripts/09_run_q2.py permute --track both
    python scripts/09_run_q2.py pvalues --track both --workers 8
    python scripts/09_run_q2.py multiplicity --track both
"""
import argparse
from pathlib import Path

from informed_order_flow.attrib import aggregate as q2_aggregate
from informed_order_flow.attrib import decompose as q2_decompose
from informed_order_flow.attrib import evaluate as q2_evaluate
from informed_order_flow.attrib import export as q2_export
from informed_order_flow.attrib import freeze as q2_freeze
from informed_order_flow.attrib import multiplicity as q2_multiplicity
from informed_order_flow.attrib import orbit as q2_orbit
from informed_order_flow.attrib import permute as q2_permute
from informed_order_flow.attrib import plan as q2_plan
from informed_order_flow.attrib import pvalues as q2_pvalues
from informed_order_flow.attrib import sources as q2_sources

REPO_ROOT = Path(__file__).resolve().parents[1]


def _spec_doc(argument: str) -> tuple[str, Path]:
    label, _, raw_path = argument.partition("=")
    path = Path(raw_path).expanduser()
    if not label or not raw_path:
        raise argparse.ArgumentTypeError(f"expected label=path, got {argument!r}")
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"no such file for label {label!r}")
    return label, path


def _cmd_plan(args: argparse.Namespace) -> None:
    report = q2_plan.write_step0(REPO_ROOT, dict(args.spec_doc))
    print("step 0 -- rules frozen")
    for doc in report["spec_documents"]:
        print(f"  spec  {doc['label']:24s} {doc['sha256']}")
    print(f"  plan  q2_analysis_plan.json    {report['plan_sha256']}")
    print(f"  code  working tree ({report['code_state_files']:d} files) "
          f"{report['code_tree_sha256']}")
    print("  config copies (must be identical):")
    for path, digest in report["config_paths"].items():
        print(f"    {digest}  {path}")


def _tracks(args: argparse.Namespace) -> tuple[str, ...]:
    return tuple(q2_sources.TRACKS) if args.track == "both" else (args.track,)


def _cmd_freeze(args: argparse.Namespace) -> None:
    if args.config is not None:
        frozen = REPO_ROOT / "data" / "attrib" / "q2_config.json"
        if q2_plan.sha256_file(args.config) != q2_plan.sha256_file(frozen):
            raise SystemExit(f"{args.config} is not the frozen configuration; "
                             f"re-run the plan step before freezing")
    for track in _tracks(args):
        report = q2_freeze.run(REPO_ROOT, track)
        counts = report["counts"]
        print(f"step 1 -- {track} frozen  build_id={report['freeze_build_id']}")
        print(f"  gate 1  {report['replayed_runs']} runs replayed, all fields bit-equal")
        print(f"  gate 2  canonical={counts['canonical_runs']} "
              f"episodes={counts['distinct_episodes']} "
              f"mle_slots={counts['mle_slots']} pairs={counts['pairs']} "
              f"n>=3={counts['pairs_by_n_trades']['ge3']}")
        for name, digest in report["outputs"].items():
            print(f"    {digest}  data/attrib/{track}/{name}")


def _cmd_decompose(args: argparse.Namespace) -> None:
    for track in _tracks(args):
        report = q2_decompose.run(REPO_ROOT, track)
        residuals = report["conservation_residuals"]
        print(f"step 2 -- {track} decomposed  build_id={report['freeze_build_id']}")
        print(f"  gate 3  {report['mle_slots']} slots; worst residual "
              + ", ".join(f"{name}={value:.1e}" for name, value in residuals.items()))
        print(f"          x_b rebuilt from slots: {report['x_b_reconstruction_residual']:.1e}")
        print(f"  gate 4  no fail-closed run; DNC {report['dnc']['positive']}+ / "
              f"{report['dnc']['negative']}-")
        print(f"    sum(DFA)={report['dfa_total']:.12f} "
              f"W_alarm={report['w_alarm_total']:.12f}")


def _cmd_aggregate(args: argparse.Namespace) -> None:
    for track in _tracks(args):
        report = q2_aggregate.run(REPO_ROOT, track)
        cells = report["cells"]
        profiles = report["profiles"]
        print(f"step 3 -- {track} aggregated  build_id={report['freeze_build_id']}")
        print(f"  gate 5  profiles read pre-onset history only "
              f"({', '.join(f'{k}={v}' for k, v in profiles.items())})")
        print(f"  counts  pairs={report['pairs']} context_rows="
              f"{report['context_only_rows']} cells={cells['total']} "
              f"single_wallet_cells={cells['single_wallet_cells']}")
        for name, digest in report["outputs"].items():
            print(f"    {digest}  data/attrib/{track}/{name}")


def _cmd_orbit(args: argparse.Namespace) -> None:
    for track in _tracks(args):
        report = q2_orbit.run(REPO_ROOT, track)
        print(f"step 4 -- {track} resolved  build_id={report['freeze_build_id']}")
        print(f"  family  {report['families_count']} families; screening={report['pairs']} "
              f"confirmatory={report['confirmatory_family']} "
              f"outside={report['outside_confirmatory']}")
        print(f"  gate 8  reachable {report['orbit_reachable_confirmatory_family']}"
              f"/{report['confirmatory_family']} confirmatory, "
              f"{report['orbit_reachable_screening_family']}/{report['pairs']} screening; "
              f"max Holm step {report['max_holm_step_screening_family']}")
        for name, digest in report["outputs"].items():
            print(f"    {digest}  data/attrib/{track}/{name}")


def _cmd_permute(args: argparse.Namespace) -> None:
    for track in _tracks(args):
        report = q2_permute.run(REPO_ROOT, track)
        cells, fixed = report["cells"], report["fixed_slots"]
        print(f"step 5 -- {track} cells built  build_id={report['freeze_build_id']}")
        print(f"  cells   {cells['total']} ({cells['movable']} movable, "
              f"{cells['single_wallet']} single-wallet), "
              f"{cells['distinct_seeds']} distinct seeds, largest {cells['largest']}")
        print(f"  engine  multiplicity preserved; tiny window matches its full "
              f"enumeration ({report['engine_self_test']['enumerated_window']['arrangements']}"
              f" arrangements); a stuck wallet gets p = 1")
        print(f"  fixed   window share max {fixed['window_share_max']:.4f}; "
              f"{fixed['low_power_windows']} low-power windows; "
              f"{fixed['pairs_with_no_movable_slots']} pairs with no movable slot")
        for name, digest in report["outputs"].items():
            print(f"    {digest}  data/attrib/{track}/{name}")


def _cmd_pvalues(args: argparse.Namespace) -> None:
    for track in _tracks(args):
        report = q2_pvalues.run(REPO_ROOT, track, workers=args.workers,
                                draws=args.draws)
        mag, direction = report["p_raw_mag"], report["p_raw_dir"]
        confirmatory = report["confirmatory"]
        print(f"step 6 -- {track} p-values  build_id={report['freeze_build_id']}")
        print("  gate 7  batching, worker count and seed all reproduce")
        print(f"  gate 6  {report['pairs']} pairs, B={report['draws']}, "
              f"min mag/dir p={mag['min']:.3g}/{direction['min']:.3g}, "
              f"grid-floor mag/dir={mag['at_grid_floor']}/{direction['at_grid_floor']}")
        print(f"  review  mag/dir below first threshold "
              f"{confirmatory['magnitude']['below_first_threshold']}/"
              f"{confirmatory['direction']['below_first_threshold']} of "
              f"{confirmatory['pairs']} confirmatory pairs; "
              f"{confirmatory['mc_review_required']} within "
              f"{confirmatory['second_seed_trigger_sigma']} sigma")
        print(f"  {report['elapsed_seconds']}s on {report['workers']} worker(s)")
        for name, digest in report["outputs"].items():
            print(f"    {digest}  data/attrib/{track}/{name}")


def _cmd_multiplicity(args: argparse.Namespace) -> None:
    for track in _tracks(args):
        report = q2_multiplicity.run(REPO_ROOT, track)
        holm, bh = report["holm"], report["bh_review"]
        review = report["second_seed_review"]
        print(f"step 7 -- {track} adjudicated  build_id={report['freeze_build_id']}")
        print(f"  gate 9  {report['families']} window families; mag/dir each saw "
              f"{holm['mag']['pairs']} confirmatory pairs at alpha={report['alpha_leg']}; "
              "headline is their fixed union")
        print(f"  holm    mag/dir={holm['mag']['rejections']}/"
              f"{holm['dir']['rejections']}; union={report['headline']['rejections']} "
              f"({report['headline']['by_leg']})")
        print(f"  bh      mag/dir={bh['mag']['screened']}/{bh['dir']['screened']} "
              f"at q={report['bh_q']} (review only)")
        print(f"  sens    dnc/dfa={holm['dnc']['rejections']}/"
              f"{holm['dfa']['rejections']}; cannot promote "
              f"{report['sensitivity_cannot_promote']}")
        regression = report["pooled_dnc_frozen_regression"]
        print(f"  regress pooled DNC bit-equal ({regression['rejections']} rejects); "
              f"window-family DNC={holm['dnc']['rejections']}")
        print(f"  seed 2  {review['pairs']} borderline pair(s) in "
              f"{review['windows']} window(s); "
              f"{review['unstable_leg_decisions']} unstable leg decision(s), "
              f"{review['unstable_headline_decisions']} unstable headline(s)")
        for name, digest in report["outputs"].items():
            print(f"    {digest}  data/attrib/{track}/{name}")


def _cmd_evaluate(args: argparse.Namespace) -> None:
    report = q2_evaluate.run(REPO_ROOT)
    counts, recall = report["instances"], report["recall"]
    print("step 8 -- simulated grid evaluated")
    print(f"  recall  {counts['directional_modes']} injected instances "
          f"({counts['negative_control']} negative control kept separate); "
          f"{recall['eligible']} eligible")
    print(f"          mag {recall['mag_rejected']}, dir {recall['dir_rejected']}, "
          f"union {recall['union_rejected']} of {recall['eligible']} confirmatory "
          f"({recall['instances']} instances in all); "
          f"{recall['magnitude_top10']} in a magnitude top-10")
    by_bin = ", ".join(f"{label}: {n}" for label, n in counts["by_bin"].items())
    print(f"          by slots {by_bin}")
    calibrated = report["calibration"]
    for leg, item in calibrated["legs"].items():
        verdict = "nominal holds" if item["nominal_holm_is_valid"] else "NOMINAL FAILS"
        print(f"  calib   {leg}: measured study-wise "
              f"{item['empirical_study_wise_error']} vs nominal "
              f"{item['nominal_study_wise_error']} ({verdict}); "
              f"t_star={item['t_star']:.3g}")
    applied = calibrated["applied"]
    if applied["censored"]:
        print(f"          WARNING t_star is censored at the grid floor "
              f"{calibrated['monte_carlo_grid_floor']:.3g}: the "
              f"{applied['leg']} leg has no measurable calibrated threshold at this B")
    for name, item in report["gate_checks"].items():
        mark = "pass" if item["passed"] else "FAIL"
        print(f"  gate    {mark}  {name} = {item['value']} ({item['threshold']})")
    agreement = report["leg_agreement"]
    print(f"  legs    union {agreement['union']} = mag-only "
          f"{agreement['mag_only']} + dir-only {agreement['dir_only']} + both "
          f"{agreement['both']}")
    print(f"  truth   assertion A on {len(report['assertion_a']['windows'])} windows, "
          f"assertion B over {len(report['assertion_b']['modules_scanned'])} modules "
          f"({report['assertion_b']['files_opened']} files opened, no manifest)")
    print(f"  episodes {report['episodes']['canonical_runs']} canonical -> "
          f"{report['episodes']['distinct']} distinct")
    for name, digest in report["outputs"].items():
        print(f"    {digest}  results/q2/{name}")


def _cmd_run(args: argparse.Namespace) -> None:
    """Steps 2 to 9 for one track, in order, on an already frozen track."""
    for track in _tracks(args):
        for command in (_cmd_decompose, _cmd_aggregate, _cmd_orbit, _cmd_permute,
                        _cmd_pvalues, _cmd_multiplicity):
            command(argparse.Namespace(track=track, workers=args.workers,
                                       draws=args.draws))
        _cmd_tests(argparse.Namespace(track=track))


def _cmd_tests(args: argparse.Namespace) -> None:
    """Step 9's table: statuses, summary and the build-id clause of gate 10."""
    for track in _tracks(args):
        report = q2_export.run(REPO_ROOT, track)
        counts, status = report["counts"], report["status_counts"]
        print(f"step 9 -- {track} exported  build_id={report['freeze_build_id']}")
        print("  gate 10 one build id across every table of the track")
        print(f"  table   {counts['pairs']} primary pairs, "
              f"{counts['confirmatory_family']} confirmatory, "
              f"{counts['distinct_wallets']} distinct wallets")
        print("  status  " + ", ".join(f"{name}={value}"
                                       for name, value in status.items()))
        print(f"  lead    median {report['lead_time_minutes']['median']:.1f} min "
              f"from a pair's first MLE trade to the alarm")
        for name, digest in report["outputs"].items():
            print(f"    {digest}  data/attrib/{track}/{name}")


def _cmd_export(args: argparse.Namespace) -> None:
    del args
    files = q2_export.export(REPO_ROOT)
    print("step 9 -- provenance written; gate 10 closed")
    for track, payload in files.items():
        print(f"  {track:4s} build_id={payload['freeze_build_id']} "
              f"engine={payload['engine']['sha256'][:16]} "
              f"config={payload['plan']['q2_config.json'][:16]}")
        print(f"         {len(payload['authoritative_inputs'])} authoritative input(s), "
              f"{len(payload['outputs'])} output file(s) hashed")


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    plan_ap = sub.add_parser("plan", help="step 0: freeze plan, config and code state")
    plan_ap.add_argument("--spec-doc", type=_spec_doc, action="append", required=True,
                         metavar="LABEL=PATH",
                         help=f"baseline document to hash; required labels: "
                              f"{', '.join(q2_plan.REQUIRED_SPEC_LABELS)}")
    plan_ap.set_defaults(func=_cmd_plan)

    freeze_ap = sub.add_parser("freeze", help="step 1: replay Q1 and write the freeze tables")
    freeze_ap.add_argument("--track", choices=(*q2_sources.TRACKS, "both"), default="both",
                           help="which track to freeze (default: both)")
    freeze_ap.add_argument("--config", type=Path, default=None,
                           help="configuration to freeze against; it must be the one "
                                "step 0 wrote, and is compared by SHA-256")
    freeze_ap.set_defaults(func=_cmd_freeze)

    run_ap = sub.add_parser("run", help="steps 2-9: one track end to end")
    run_ap.add_argument("--track", choices=(*q2_sources.TRACKS, "both"), default="both",
                        help="which track to run (default: both)")
    run_ap.add_argument("--workers", type=int, default=1,
                        help="windows to permute in parallel (default: 1)")
    run_ap.add_argument("--draws", type=int, default=q2_pvalues.B,
                        help=f"permutations per window (default: {q2_pvalues.B})")
    run_ap.set_defaults(func=_cmd_run)

    decompose_ap = sub.add_parser("decompose",
                                  help="step 2: per-trade DNC / AGC / DFA")
    decompose_ap.add_argument("--track", choices=(*q2_sources.TRACKS, "both"),
                              default="both", help="which track (default: both)")
    decompose_ap.set_defaults(func=_cmd_decompose)

    aggregate_ap = sub.add_parser("aggregate",
                                  help="step 3: wallet x window rows and profiles")
    aggregate_ap.add_argument("--track", choices=(*q2_sources.TRACKS, "both"),
                              default="both", help="which track (default: both)")
    aggregate_ap.set_defaults(func=_cmd_aggregate)

    orbit_ap = sub.add_parser("orbit",
                              help="step 4: eligibility, orbit floors, resolution")
    orbit_ap.add_argument("--track", choices=(*q2_sources.TRACKS, "both"),
                          default="both", help="which track (default: both)")
    orbit_ap.set_defaults(func=_cmd_orbit)

    permute_ap = sub.add_parser("permute",
                                help="step 5: cells, per-cell seeds, shuffle engine")
    permute_ap.add_argument("--track", choices=(*q2_sources.TRACKS, "both"),
                            default="both", help="which track (default: both)")
    permute_ap.set_defaults(func=_cmd_permute)

    pvalues_ap = sub.add_parser("pvalues", help="step 6: permutation p-values")
    pvalues_ap.add_argument("--track", choices=(*q2_sources.TRACKS, "both"),
                            default="both", help="which track (default: both)")
    pvalues_ap.add_argument("--workers", type=int, default=1,
                            help="windows to run in parallel (default: 1)")
    pvalues_ap.add_argument("--draws", type=int, default=q2_pvalues.B,
                            help=f"permutations per window (default: {q2_pvalues.B})")
    pvalues_ap.set_defaults(func=_cmd_pvalues)

    multiplicity_ap = sub.add_parser("multiplicity",
                                     help="step 7: Holm, BH screen, DFA sensitivity")
    multiplicity_ap.add_argument("--track", choices=(*q2_sources.TRACKS, "both"),
                                 default="both", help="which track (default: both)")
    multiplicity_ap.set_defaults(func=_cmd_multiplicity)

    evaluate_ap = sub.add_parser("evaluate",
                                 help="step 8: simulated recall and truth isolation")
    evaluate_ap.add_argument("--track", choices=("sim",), default="sim",
                             help="the simulated track is the only one with truth")
    evaluate_ap.set_defaults(func=_cmd_evaluate)

    tests_ap = sub.add_parser("tests",
                              help="step 9: wallet_window_tests.parquet and the summary")
    tests_ap.add_argument("--track", choices=(*q2_sources.TRACKS, "both"),
                          default="both", help="which track (default: both)")
    tests_ap.set_defaults(func=_cmd_tests)

    export_ap = sub.add_parser("export", help="step 9: q2_hashes.json for both tracks")
    export_ap.set_defaults(func=_cmd_export)
    return ap


def main() -> None:
    args = _build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
