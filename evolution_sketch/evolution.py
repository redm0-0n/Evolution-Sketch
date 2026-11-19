"""Evolutionary search for the next stroke."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from evolution_sketch.config import EvolutionParameters, RunConfig
from evolution_sketch.imaging import Canvas
from evolution_sketch.stroke import Stroke


@dataclass(slots=True)
class Candidate:
    stroke: Stroke
    score: float


class EvolutionarySearch:
    """Find one high-value stroke for the current canvas."""

    def __init__(
        self,
        canvas: Canvas,
        params: EvolutionParameters,
        config: RunConfig,
        rng: np.random.Generator,
    ) -> None:
        self.canvas = canvas
        self.params = params
        self.config = config
        self.rng = rng
        self.population = [
            self._evaluate(Stroke.random(params, config, rng))
            for _ in range(params.population_size)
        ]
        self._sort_population()

    def _evaluate(self, stroke: Stroke) -> Candidate:
        score = self.canvas.target.stroke_improvement(self.canvas.image, stroke)
        return Candidate(stroke, score)

    def _sort_population(self) -> None:
        self.population.sort(key=lambda candidate: candidate.score, reverse=True)

    def _tournament_winner(self) -> int:
        indices = self.rng.integers(
            0, len(self.population), size=self.params.tournament_size
        )
        return int(
            max(indices, key=lambda index: self.population[int(index)].score)
        )

    def _parent_pool(self) -> list[int]:
        return [
            self._tournament_winner() for _ in range(self.params.parent_pool_size)
        ]

    def _generation(self) -> None:
        parent_pool = self._parent_pool()
        next_population = list(self.population)

        for _ in range(self.params.children):
            first_index = parent_pool[int(self.rng.integers(0, len(parent_pool)))]
            second_index = parent_pool[int(self.rng.integers(0, len(parent_pool)))]
            child = Stroke.crossover(
                self.population[first_index].stroke,
                self.population[second_index].stroke,
                self.rng,
            ).mutate(self.params, self.config, self.rng)
            next_population.append(self._evaluate(child))

        next_population.sort(key=lambda candidate: candidate.score, reverse=True)
        self.population = next_population[: self.params.population_size]

    def run(self) -> Candidate:
        for _ in range(self.params.generations):
            self._generation()
        return self.population[0]
