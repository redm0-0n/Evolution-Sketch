from __future__ import annotations

import unittest

import numpy as np

from evolution_sketch.config import RunConfig
from evolution_sketch.imaging import TargetImage, draw_stroke
from evolution_sketch.stroke import Stroke


class ConfigurationTests(unittest.TestCase):
    def test_parameter_schedule_reaches_both_endpoints(self) -> None:
        config = RunConfig()

        initial = config.parameters_at(0.0)
        final = config.parameters_at(1.0)

        self.assertEqual(initial.max_length, config.initial_max_length)
        self.assertEqual(final.max_length, config.final_max_length)
        self.assertEqual(
            initial.coordinate_sigma, config.initial_coordinate_sigma
        )
        self.assertEqual(final.coordinate_sigma, config.final_coordinate_sigma)


class ImagingTests(unittest.TestCase):
    def test_pencil_stroke_darkens_canvas_and_improves_black_target(self) -> None:
        target = TargetImage(np.zeros((16, 16), dtype=np.uint8))
        canvas = np.full((16, 16), 255, dtype=np.uint8)
        stroke = Stroke(1, 1, 14, 14, shade=0, pressure=255, is_eraser=False)

        improvement = target.stroke_improvement(canvas, stroke)
        draw_stroke(canvas, stroke)

        self.assertGreater(improvement, 0.0)
        self.assertEqual(int(canvas[1, 1]), 0)
        self.assertEqual(int(canvas[14, 14]), 0)


if __name__ == "__main__":
    unittest.main()
