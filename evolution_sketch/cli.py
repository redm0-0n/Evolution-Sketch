"""Command-line interface for Evolution Sketch."""

from __future__ import annotations

import argparse
from pathlib import Path

from evolution_sketch.config import RunConfig
from evolution_sketch.runner import run_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="Evolution Sketch",
        description=(
            "Approximate an input image with grayscale strokes selected by a "
            "genetic algorithm."
        ),
    )
    parser.add_argument("input", type=Path, help="path to the source image")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("outputs"),
        help="result directory (default: outputs)",
    )
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--max-strokes", type=int, default=10_000)
    parser.add_argument("--time-limit", type=float, default=540.0)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--keep-snapshots", type=int, default=5)
    parser.add_argument("--population", type=int, default=30)
    parser.add_argument("--children", type=int, default=35)
    parser.add_argument("--generations", type=int, default=30)
    parser.add_argument("--tournament-size", type=int, default=5)
    parser.add_argument("--parent-pool", type=int, default=8)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="base random seed; omit for a new random experiment",
    )
    parser.add_argument(
        "--accept-worse-strokes",
        action="store_true",
        help="match the C++ behavior by applying a stroke even if it worsens fitness",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = RunConfig(
        size=args.size,
        runs=args.runs,
        max_strokes=args.max_strokes,
        time_limit_seconds=args.time_limit,
        log_every=args.log_every,
        keep_snapshots=args.keep_snapshots,
        population_size=args.population,
        children=args.children,
        generations=args.generations,
        tournament_size=args.tournament_size,
        parent_pool_size=args.parent_pool,
        accept_worse_strokes=args.accept_worse_strokes,
    )
    try:
        run_experiment(args.input, args.output, config, args.seed)
    except (FileNotFoundError, OSError, ValueError) as error:
        raise SystemExit(f"error: {error}") from error
    return 0
