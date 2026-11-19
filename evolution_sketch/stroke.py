"""Stroke chromosome used by the evolutionary algorithm."""

from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np

from evolution_sketch.config import EvolutionParameters, RunConfig


@dataclass(slots=True)
class Stroke:
    x_start: int
    y_start: int
    x_end: int
    y_end: int
    shade: int
    pressure: int
    is_eraser: bool

    @classmethod
    def random(
        cls,
        params: EvolutionParameters,
        config: RunConfig,
        rng: np.random.Generator,
    ) -> "Stroke":
        x_start = int(rng.integers(0, config.size))
        y_start = int(rng.integers(0, config.size))
        x_length = int(rng.integers(params.min_length, params.max_length + 1))
        y_length = int(rng.integers(params.min_length, params.max_length + 1))

        x_direction = -1 if rng.random() < 0.5 else 1
        y_direction = -1 if rng.random() < 0.5 else 1
        is_eraser = bool(rng.random() < params.eraser_probability)
        if is_eraser:
            shade = int(rng.integers(config.eraser_shade_min, 256))
        else:
            shade = int(rng.integers(0, config.pencil_shade_max + 1))

        stroke = cls(
            x_start=x_start,
            y_start=y_start,
            x_end=x_start + x_direction * x_length,
            y_end=y_start + y_direction * y_length,
            shade=shade,
            pressure=int(
                rng.integers(params.min_pressure, params.max_pressure + 1)
            ),
            is_eraser=is_eraser,
        )
        stroke.clamp(config.size)
        return stroke

    def copy(self) -> "Stroke":
        return Stroke(**{field.name: getattr(self, field.name) for field in fields(self)})

    def clamp(self, size: int) -> None:
        self.x_start = int(np.clip(self.x_start, 0, size - 1))
        self.y_start = int(np.clip(self.y_start, 0, size - 1))
        self.x_end = int(np.clip(self.x_end, 0, size - 1))
        self.y_end = int(np.clip(self.y_end, 0, size - 1))
        self.shade = int(np.clip(self.shade, 0, 255))
        self.pressure = int(np.clip(self.pressure, 0, 255))

    def mutate(
        self,
        params: EvolutionParameters,
        config: RunConfig,
        rng: np.random.Generator,
    ) -> "Stroke":
        if rng.random() < params.reset_probability:
            return Stroke.random(params, config, rng)

        result = self.copy()
        result.x_start += int(np.rint(rng.normal(0.0, params.coordinate_sigma)))
        result.y_start += int(np.rint(rng.normal(0.0, params.coordinate_sigma)))
        result.x_end += int(np.rint(rng.normal(0.0, params.coordinate_sigma)))
        result.y_end += int(np.rint(rng.normal(0.0, params.coordinate_sigma)))
        result.shade += int(np.rint(rng.normal(0.0, params.shade_sigma)))
        result.pressure += int(np.rint(rng.normal(0.0, params.pressure_sigma)))
        if rng.random() < params.type_flip_probability:
            result.is_eraser = not result.is_eraser
        result.clamp(config.size)
        return result

    @classmethod
    def crossover(
        cls, first: "Stroke", second: "Stroke", rng: np.random.Generator
    ) -> "Stroke":
        values = {
            field.name: (
                getattr(first, field.name)
                if rng.random() < 0.5
                else getattr(second, field.name)
            )
            for field in fields(cls)
        }
        return cls(**values)

    def shifted(self, x_offset: int, y_offset: int) -> "Stroke":
        result = self.copy()
        result.x_start -= x_offset
        result.x_end -= x_offset
        result.y_start -= y_offset
        result.y_end -= y_offset
        return result
