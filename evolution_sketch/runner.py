"""Experiment orchestration and result persistence."""

from __future__ import annotations

import csv
import json
import shutil
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from evolution_sketch.config import RunConfig
from evolution_sketch.evolution import EvolutionarySearch
from evolution_sketch.imaging import Canvas, TargetImage, write_jpeg


@dataclass(frozen=True, slots=True)
class RunResult:
    run_number: int
    seed: int
    strokes: int
    elapsed_seconds: float
    fitness: float
    output_path: Path


def _write_metadata(
    output_dir: Path,
    input_path: Path,
    config: RunConfig,
    seeds: list[int],
) -> None:
    metadata = {
        "input": str(input_path.resolve()),
        "seeds": seeds,
        "config": config.to_dict(),
    }
    (output_dir / "experiment.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


def _run_once(
    target: TargetImage,
    output_dir: Path,
    run_number: int,
    seed: int,
    config: RunConfig,
) -> RunResult:
    run_dir = output_dir / f"run_{run_number:02d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    canvas = Canvas(target)
    snapshots: deque[np.ndarray] = deque(maxlen=config.keep_snapshots)
    snapshots.append(canvas.image.copy())

    start = time.monotonic()
    strokes = 0
    fitness_sum = 0.0
    fitness_samples = 0
    metrics_path = run_dir / "metrics.csv"

    with metrics_path.open("w", newline="", encoding="utf-8") as metrics_file:
        writer = csv.writer(metrics_file)
        writer.writerow(
            ["elapsed_seconds", "strokes", "fitness", "average_fitness"]
        )

        while strokes < config.max_strokes:
            elapsed = time.monotonic() - start
            if elapsed >= config.time_limit_seconds:
                break

            denominator = max(1, config.max_strokes - 1)
            params = config.parameters_at(strokes / denominator)
            candidate = EvolutionarySearch(canvas, params, config, rng).run()

            if candidate.score > 0.0 or config.accept_worse_strokes:
                canvas.apply(candidate.stroke)

            strokes += 1
            fitness_sum += canvas.fitness
            fitness_samples += 1
            snapshots.append(canvas.image.copy())

            if strokes % config.log_every == 0:
                elapsed = time.monotonic() - start
                average = fitness_sum / fitness_samples
                writer.writerow([elapsed, strokes, canvas.fitness, average])
                metrics_file.flush()
                print(
                    f"run {run_number}/{config.runs}: "
                    f"strokes={strokes}, fitness={canvas.fitness:.4f}, "
                    f"time={elapsed:.1f}s",
                    flush=True,
                )
                fitness_sum = 0.0
                fitness_samples = 0

        elapsed = time.monotonic() - start
        if fitness_samples:
            writer.writerow(
                [elapsed, strokes, canvas.fitness, fitness_sum / fitness_samples]
            )

    for index, snapshot in enumerate(snapshots, start=1):
        write_jpeg(run_dir / f"snapshot_{index:03d}.jpg", snapshot)
    final_path = run_dir / "final.jpg"
    write_jpeg(final_path, canvas.image)
    print(
        f"run {run_number}/{config.runs} finished: strokes={strokes}, "
        f"fitness={canvas.fitness:.4f}, time={elapsed:.1f}s"
    )
    return RunResult(
        run_number=run_number,
        seed=seed,
        strokes=strokes,
        elapsed_seconds=elapsed,
        fitness=canvas.fitness,
        output_path=final_path,
    )


def run_experiment(
    input_path: Path,
    output_dir: Path,
    config: RunConfig,
    seed: int | None = None,
) -> list[RunResult]:
    """Run all experiments and save per-run plus overall best images."""

    config.validate()
    input_path = input_path.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    target = TargetImage.from_file(input_path, config.size)

    seed_sequence = np.random.SeedSequence(seed)
    child_sequences = seed_sequence.spawn(config.runs)
    seeds = [int(sequence.generate_state(1)[0]) for sequence in child_sequences]
    _write_metadata(output_dir, input_path, config, seeds)

    results = [
        _run_once(target, output_dir, index, seeds[index - 1], config)
        for index in range(1, config.runs + 1)
    ]
    best_result = min(results, key=lambda result: result.fitness)
    shutil.copyfile(best_result.output_path, output_dir / "best.jpg")

    summary = {
        "best_run": best_result.run_number,
        "best_fitness": best_result.fitness,
        "runs": [
            {
                "run": result.run_number,
                "seed": result.seed,
                "strokes": result.strokes,
                "elapsed_seconds": result.elapsed_seconds,
                "fitness": result.fitness,
                "output": str(result.output_path),
            }
            for result in results
        ],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(
        f"best result: run={best_result.run_number}, "
        f"fitness={best_result.fitness:.4f}, "
        f"image={output_dir / 'best.jpg'}"
    )
    return results
