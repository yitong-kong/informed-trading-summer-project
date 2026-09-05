# -*- coding: utf-8 -*-
"""Entry point: estimate baseline params and build simulated dataset(s).

Pipeline: estimate_baseline -> generate_null (level 0/1/2) ->
optional inject_h1 -> a single scenario's parquet + sim_market_metadata +
sim_manifest, written under data/sim/<scenario_id>/.

Each invocation builds exactly one dataset, except ``--grid`` which builds the
full evaluation grid through the very same ``build_scenario`` path:

    seeds {2, 42, 100, 1000} x levels {0, 1, 2}
        x (null + the four injection modes x resolved outcome {Yes, No})
        =  12 nulls + 96 H1 datasets

Each H1 dataset carries exactly one injection mode and one resolved outcome
(the side the insider bets; ``No`` makes imbalance drift down after tau). The
No variants get a ``_no`` scenario-id suffix. Nulls are not duplicated per
outcome: the null stream does not depend on ``resolved_outcome`` (it only
enters the market metadata and the injection side), so both outcome variants
pair against the same null. Within one (level, seed) the null and all H1
scenarios share the same base stream, the same random ``tau_info`` and the
same informed wallets (both are drawn from the scenario seed before the mode
branches), so mode and outcome effects are compared paired. See the
table in src/informed_order_flow/sim/README.md for every option's meaning,
default and range.

Usage (example):
    # Estimate baseline params + build the default dataset (Level 0 symmetric null)
    python scripts/05_build_simulated_data.py

    python scripts/05_build_simulated_data.py --estimate-only   # only baseline params
    python scripts/05_build_simulated_data.py --no-estimate     # reuse existing params

    # A Level 1 null with the empirical direction balance and a fixed seed
    python scripts/05_build_simulated_data.py --level 1 --p-long-yes 0.39 --seed 7

    # A Level 2 block-bootstrap null: 30-min blocks, 60 resampled blocks
    python scripts/05_build_simulated_data.py --level 2 --block-minutes 30 --n-blocks 60

    # A Level 1 H1 dataset: a gradual informed episode at a random tau_info
    python scripts/05_build_simulated_data.py --level 1 --h1 --mode additive_trades \
        --total-size 400000 --n-wallets 3

    # The full 4-seed x 3-level x (null + 4 modes x Yes/No) evaluation grid
    python scripts/05_build_simulated_data.py --grid
"""
import argparse
import json

from informed_order_flow.sim import build_scenario, estimate_baseline
from informed_order_flow.sim.run import H0_LEVELS, INJECTION_MODES

DEFAULT_CONTROL_QUESTION = "Maduro out by November 30, 2025?"
BUILD_SPEEDS = ("gradual", "instant")
GRID_SEEDS = (2, 42, 100, 1000)
GRID_OUTCOMES = ("Yes", "No")


def _market(resolved_outcome: str) -> dict:
    """Synthetic market spec; ``resolved_outcome`` fixes the informed side for H1."""
    return {
        "question": "SIM: out by some date?",
        "scheduled_end_date": "2026-03-31T12:00:00Z",
        "resolved_outcome": resolved_outcome,
    }


def _injection(args: argparse.Namespace) -> dict:
    """H1 config carrying every mode's knob (inject_h1 reads what its mode needs)."""
    return {
        "mode": args.mode,
        "tau_frac": args.tau_frac,  # None -> random (seeded) change-point
        "n_wallets": args.n_wallets,
        "total_size": args.total_size,
        "build_speed": args.build_speed,
        "price_impact": args.price_impact,
        "tilt_frac": args.tilt_frac,
        "size_factor": args.size_factor,
    }


def _scenario_id(args: argparse.Namespace) -> str:
    """Default id encodes level + null/H1 mode so distinct datasets never collide."""
    if args.scenario_id:
        return args.scenario_id
    return f"L{args.level}_h1_{args.mode}" if args.h1 else f"L{args.level}_null"


def _grid_config(level: str, seed: int, mode: str | None, outcome: str,
                 args: argparse.Namespace) -> dict:
    """One grid scenario config: the null base (per level/seed) +- one injection.

    Injection knobs are the CLI defaults, identical across the grid, so the
    only varying factors are (seed, level, mode, resolved outcome) --
    everything else is controlled.
    """
    if mode is None:
        sid = f"L{level}_null_s{seed}"
    else:
        sid = f"L{level}_{mode}_s{seed}" + ("_no" if outcome == "No" else "")
    config = {
        "scenario_id": sid,
        "level": level,
        "seed": seed,
        "market": _market(outcome),
        "injection": dict(_injection(args), mode=mode) if mode else None,
    }
    if level in ("0", "1"):
        config["n_trades"] = args.n_trades
        config["p_long_yes"] = args.p_long_yes
    else:
        config["bootstrap"] = {
            "control_question": args.control_question,
            "block_minutes": args.block_minutes,
            "n_blocks": args.n_blocks,
        }
    return config


def _build_grid(args: argparse.Namespace) -> None:
    """Build the full evaluation grid: 12 nulls + 96 single-mode H1 datasets.

    Nulls are outcome-free (one per level/seed); every injection mode is built
    once per resolved outcome (Yes and No), sharing the paired null's stream.
    """
    specs = [(level, seed, None, "Yes")
             for level in H0_LEVELS for seed in GRID_SEEDS]
    specs += [(level, seed, mode, outcome)
              for level in H0_LEVELS
              for seed in GRID_SEEDS
              for mode in INJECTION_MODES
              for outcome in GRID_OUTCOMES]
    print(f"[grid] building {len(specs)} scenarios "
          f"(seeds={GRID_SEEDS}, levels={H0_LEVELS}, "
          f"modes=null+{len(INJECTION_MODES)}, outcomes={GRID_OUTCOMES})")
    for i, (level, seed, mode, outcome) in enumerate(specs, 1):
        m = build_scenario(_grid_config(level, seed, mode, outcome, args))
        tau = m["tau_info_utc"]
        print(f"    [{i:3d}/{len(specs)}] {m['scenario_id']:46s} rows={m['n_rows']:<6d}"
              + (f" tau={tau}" if tau else ""))


def _baseline_summary(params: dict) -> str:
    return json.dumps({"baseline": {
        "fit_rows": params["fit_rows"],
        "arrival_rate_per_sec": round(params["arrival_rate_per_sec"], 5),
        "n_wallets": params["n_wallets"],
        "concentration": params["wallet_concentration"],
    }}, indent=2)


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    g = ap.add_argument_group("general (all scenarios)")
    g.add_argument("--grid", action="store_true",
                   help="build the full evaluation grid (seeds 2/42/100/1000 x "
                        "levels 0/1/2 x null + 4 injection modes x resolved "
                        "outcome Yes/No = 108 datasets) with the default knobs; "
                        "per-scenario options are ignored")
    g.add_argument("--estimate-only", action="store_true",
                   help="only (re)estimate baseline params, then exit")
    g.add_argument("--no-estimate", action="store_true",
                   help="reuse the existing baseline_params.json (skip estimation)")
    g.add_argument("--scenario-id", default=None,
                   help="output dir name under data/sim/ "
                        "(default: L<level>_null or L<level>_h1_<mode>)")
    g.add_argument("--level", choices=H0_LEVELS, default="0",
                   help="H0 null level: 0 i.i.d., 1 non-stationary arrival, "
                        "2 block bootstrap (default: %(default)s)")
    g.add_argument("--seed", type=int, default=1,
                   help="RNG seed; fully determines the dataset (default: %(default)s)")

    n = ap.add_argument_group("Level 0/1 null (ignored for --level 2)")
    n.add_argument("--n-trades", type=int, default=8000,
                   help="exact row count to emit (default: %(default)s)")
    n.add_argument("--p-long-yes", type=float, default=0.5,
                   help="P(long-YES) per trade; 0.5 = symmetric textbook null, "
                        "pass the empirical ~0.39 for an empirical-iid null "
                        "[0..1] (default: %(default)s)")

    b = ap.add_argument_group("Level 2 block bootstrap (only for --level 2)")
    b.add_argument("--control-question", default=DEFAULT_CONTROL_QUESTION,
                   help="real control contract to bootstrap from (default: %(default)r)")
    b.add_argument("--block-minutes", type=int, default=45,
                   help="bootstrap block length in minutes (default: %(default)s)")
    b.add_argument("--n-blocks", type=int, default=None,
                   help="number of blocks to resample; "
                        "default = as many as the source has")

    h = ap.add_argument_group("H1 injection (only when --h1 is set)")
    h.add_argument("--h1", action="store_true",
                   help="inject one parametric informed episode at a known tau_info")
    h.add_argument("--resolved-outcome", choices=("Yes", "No"), default="Yes",
                   help="market truth = the side the insider bets; "
                        "No makes YES price/imbalance drift down after tau "
                        "(default: %(default)s)")
    h.add_argument("--mode", choices=INJECTION_MODES, default="additive_trades",
                   help="injection mechanism (default: %(default)s)")
    h.add_argument("--tau-frac", type=float, default=None,
                   help="change-point location as a fraction of the stream span "
                        "(0..1); default = random in [0.35, 0.70], seeded")
    h.add_argument("--n-wallets", type=int, default=3,
                   help="number of informed wallets the post-tau flow routes to "
                        "(>=1) (default: %(default)s)")
    h.add_argument("--total-size", type=float, default=400_000.0,
                   help="total injected shares; additive_trades only "
                        "(default: %(default)s)")
    h.add_argument("--build-speed", choices=BUILD_SPEEDS, default="gradual",
                   help="additive_trades only: instant=1 order, gradual=40 split "
                        "orders (default: %(default)s)")
    h.add_argument("--price-impact", action=argparse.BooleanOptionalAction, default=True,
                   help="additive_trades only: let the winning side appreciate as "
                        "the insider buys (default: %(default)s)")
    h.add_argument("--tilt-frac", type=float, default=0.5,
                   help="direction_tilt_same_count: fraction of post-tau "
                        "losing-side trades flipped to the win side and marked "
                        "informed (already-winning trades are not relabeled) "
                        "(0..1) (default: %(default)s)")
    h.add_argument("--size-factor", type=float, default=5.0,
                   help="size_tilt only: multiplier on post-tau winning-side trade "
                        "sizes (>=1) (default: %(default)s)")
    return ap


def _validate(ap: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if not 0.0 <= args.p_long_yes <= 1.0:
        ap.error("--p-long-yes must be in [0, 1]")
    if args.n_trades <= 0:
        ap.error("--n-trades must be positive")
    if args.block_minutes <= 0:
        ap.error("--block-minutes must be positive")
    if args.n_blocks is not None and args.n_blocks <= 0:
        ap.error("--n-blocks must be positive")
    if args.tau_frac is not None and not 0.0 < args.tau_frac < 1.0:
        ap.error("--tau-frac must be in (0, 1)")
    if args.n_wallets < 1:
        ap.error("--n-wallets must be >= 1")
    if args.total_size <= 0:
        ap.error("--total-size must be positive")
    if not 0.0 <= args.tilt_frac <= 1.0:
        ap.error("--tilt-frac must be in [0, 1]")
    if args.size_factor < 1.0:
        ap.error("--size-factor must be >= 1")


def main() -> None:
    ap = _build_parser()
    args = ap.parse_args()
    _validate(ap, args)

    if args.estimate_only:
        print(_baseline_summary(estimate_baseline()))
        return

    # Level 0/1 read the parametric baseline; Level 2 bootstraps the real table
    # directly and needs no baseline estimation.
    if not args.no_estimate and (args.grid or args.level in ("0", "1")):
        print(_baseline_summary(estimate_baseline()))

    if args.grid:
        _build_grid(args)
        return

    config = {
        "scenario_id": _scenario_id(args),
        "level": args.level,
        "seed": args.seed,
        "market": _market(args.resolved_outcome),
        "injection": _injection(args) if args.h1 else None,
    }
    if args.level in ("0", "1"):
        config["n_trades"] = args.n_trades
        config["p_long_yes"] = args.p_long_yes
    else:
        config["bootstrap"] = {
            "control_question": args.control_question,
            "block_minutes": args.block_minutes,
            "n_blocks": args.n_blocks,
        }

    m = build_scenario(config)
    print(json.dumps({
        "scenario_id": m["scenario_id"], "level": m["level"],
        "rows": m["n_rows"], "injection_mode": m["injection_mode"],
        "tau_info_utc": m["tau_info_utc"],
    }, indent=2))


if __name__ == "__main__":
    main()
