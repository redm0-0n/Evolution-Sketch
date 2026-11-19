"""Configuration and parameter scheduling for the genetic algorithm."""

from __future__ import annotations

from dataclasses import asdict, dataclass


def _lerp_int(start: int, end: int, progress: float) -> int:
    return round((1.0 - progress) * start + progress * end)


def _lerp_float(start: float, end: float, progress: float) -> float:
    return (1.0 - progress) * start + progress * end


@dataclass(frozen=True, slots=True)
class EvolutionParameters:
    """Parameters used while searching for one new stroke."""

    min_length: int
    max_length: int
    min_pressure: int
    max_pressure: int
    coordinate_sigma: int
    shade_sigma: int
    pressure_sigma: int
    eraser_probability: float
    reset_probability: float
    type_flip_probability: float
    population_size: int
    tournament_size: int
    parent_pool_size: int
    children: int
    generations: int


@dataclass(frozen=True, slots=True)
class RunConfig:
    """Top-level experiment settings.

    Defaults mirror the original C++ implementation. Use smaller values for a
    smoke test before starting a long experiment.
    """

    size: int = 512
    runs: int = 3
    max_strokes: int = 10_000
    time_limit_seconds: float = 540.0
    log_every: int = 50
    keep_snapshots: int = 5

    min_length: int = 4
    final_max_length: int = 16
    initial_max_length: int = 48
    pencil_shade_max: int = 220
    eraser_shade_min: int = 200

    final_min_pressure: int = 30
    initial_min_pressure: int = 60
    final_max_pressure: int = 110
    initial_max_pressure: int = 170

    final_coordinate_sigma: int = 4
    initial_coordinate_sigma: int = 10
    final_shade_sigma: int = 10
    initial_shade_sigma: int = 25
    final_pressure_sigma: int = 8
    initial_pressure_sigma: int = 18

    final_eraser_probability: float = 0.20
    initial_eraser_probability: float = 0.45
    final_reset_probability: float = 0.01
    initial_reset_probability: float = 0.04
    final_type_flip_probability: float = 0.01
    initial_type_flip_probability: float = 0.03

    population_size: int = 30
    tournament_size: int = 5
    parent_pool_size: int = 8
    children: int = 35
    generations: int = 30
    accept_worse_strokes: bool = False

    def validate(self) -> None:
        """Raise ValueError when a setting cannot produce a valid run."""

        positive_integer_fields = {
            "size": self.size,
            "runs": self.runs,
            "max_strokes": self.max_strokes,
            "log_every": self.log_every,
            "keep_snapshots": self.keep_snapshots,
            "population_size": self.population_size,
            "tournament_size": self.tournament_size,
            "parent_pool_size": self.parent_pool_size,
            "children": self.children,
            "generations": self.generations,
        }
        for name, value in positive_integer_fields.items():
            if value <= 0:
                raise ValueError(f"{name} must be greater than zero")
        if self.time_limit_seconds <= 0:
            raise ValueError("time_limit_seconds must be greater than zero")
        if self.min_length > self.final_max_length:
            raise ValueError("min_length cannot exceed final_max_length")
        if self.final_max_length > self.initial_max_length:
            raise ValueError("final_max_length cannot exceed initial_max_length")

    def parameters_at(self, progress: float) -> EvolutionParameters:
        """Return smoothly scheduled parameters for progress in [0, 1]."""

        progress = min(1.0, max(0.0, progress))
        return EvolutionParameters(
            min_length=self.min_length,
            max_length=_lerp_int(
                self.initial_max_length, self.final_max_length, progress
            ),
            min_pressure=_lerp_int(
                self.initial_min_pressure, self.final_min_pressure, progress
            ),
            max_pressure=_lerp_int(
                self.initial_max_pressure, self.final_max_pressure, progress
            ),
            coordinate_sigma=_lerp_int(
                self.initial_coordinate_sigma,
                self.final_coordinate_sigma,
                progress,
            ),
            shade_sigma=_lerp_int(
                self.initial_shade_sigma, self.final_shade_sigma, progress
            ),
            pressure_sigma=_lerp_int(
                self.initial_pressure_sigma, self.final_pressure_sigma, progress
            ),
            eraser_probability=_lerp_float(
                self.initial_eraser_probability,
                self.final_eraser_probability,
                progress,
            ),
            reset_probability=_lerp_float(
                self.initial_reset_probability,
                self.final_reset_probability,
                progress,
            ),
            type_flip_probability=_lerp_float(
                self.initial_type_flip_probability,
                self.final_type_flip_probability,
                progress,
            ),
            population_size=self.population_size,
            tournament_size=self.tournament_size,
            parent_pool_size=self.parent_pool_size,
            children=self.children,
            generations=self.generations,
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
