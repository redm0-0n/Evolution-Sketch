# Evolution Sketch

Evolution Sketch approximates an input image with thousands of grayscale pencil
and eraser strokes. A genetic algorithm searches for the next stroke that most
reduces the mean squared error (MSE) between the generated canvas and the target.

The repository contains two implementations:

- `generator.c++` — the original single-file C++/OpenCV implementation;
- `evolution_sketch/` — a modular Python implementation with a command-line
  interface, configurable experiments, reproducible seeds, logs and snapshots.

## How the algorithm works

1. The input image is converted to grayscale and resized to a square canvas.
2. A population of random strokes is created.
3. Candidate strokes are evaluated on the part of the image they affect.
4. Tournament selection, crossover and mutation evolve the population.
5. The best improving stroke is applied to the canvas.
6. The process repeats until the stroke or time limit is reached.

Stroke length, pressure and mutation strength gradually decrease during a run:
early iterations explore the image broadly, while later iterations refine small
details.

## Requirements

- Python 3.10 or newer;
- `pip`;
- Windows, Linux or macOS.

No compiler is needed for the Python version.

## Installation

Clone the repository and open a terminal in its directory. On Windows
PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation scripts, either allow scripts for the current
user:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

or run the virtual environment's Python directly without activation:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

On Linux or macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## First test run

Start with a small experiment to confirm that the environment works. Replace
`examples/input.jpg` with the path to your own image if needed:

```powershell
python main.py examples/input.jpg --output outputs/test --runs 1 --size 256 --max-strokes 20 --time-limit 60 --population 10 --children 10 --generations 3 --log-every 5 --seed 42
```

This is a smoke test, not a high-quality generation. It should finish quickly
and create `outputs/test/best.jpg`.

The same command can also be written as `python -m evolution_sketch ...`.

## Full run

The default values reproduce the scale of the C++ configuration: three runs,
up to 10,000 strokes and a 540-second time limit for each run.

```powershell
python main.py path\to\input.jpg --output outputs\my_experiment --seed 42
```

The time limit is checked between strokes. One genetic search can take some
time, so the actual duration may be slightly longer than the specified limit.

Use `python main.py --help` to see every available option.

Important options:

| Option | Default | Meaning |
|---|---:|---|
| `--runs` | `3` | Independent runs |
| `--size` | `512` | Width and height of the working image |
| `--max-strokes` | `10000` | Maximum attempted strokes per run |
| `--time-limit` | `540` | Time limit per run in seconds |
| `--population` | `30` | Population size |
| `--children` | `35` | Children created in each generation |
| `--generations` | `30` | Generations used to select one stroke |
| `--seed` | random | Base seed for reproducible experiments |

By default, the Python implementation rejects strokes that make the image
worse. Pass `--accept-worse-strokes` to reproduce the original C++ behavior.

## Tests

After installing the dependencies, run:

```powershell
python -m unittest
```

## Output structure

```text
outputs/my_experiment/
├── experiment.json       # configuration and seeds
├── summary.json          # metrics for all runs
├── best.jpg              # best final image across all runs
├── run_01/
│   ├── final.jpg
│   ├── metrics.csv
│   └── snapshot_*.jpg
├── run_02/
└── run_03/
```

Output directories are ignored by Git. Copy selected example images into a
separate tracked directory if you want to show them in the README.

## Project structure

```text
Evolution-Sketch/
├── evolution_sketch/
│   ├── cli.py          # command-line arguments
│   ├── config.py       # settings and parameter schedule
│   ├── evolution.py    # selection, crossover and mutation loop
│   ├── imaging.py      # image I/O, drawing and fitness
│   ├── runner.py       # experiment orchestration and outputs
│   └── stroke.py       # stroke chromosome
├── generator.c++       # original single-file C++ version
├── main.py             # Python entry point
└── requirements.txt
```

## Reproducibility

Pass a fixed `--seed`, keep the same dependencies and use the settings stored
in `experiment.json`. Each run receives its own derived seed. Without `--seed`,
a new random experiment is created every time.
