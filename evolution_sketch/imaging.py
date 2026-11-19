"""Image loading, rendering and fitness evaluation."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from evolution_sketch.stroke import Stroke


def read_grayscale(path: Path, size: int) -> np.ndarray:
    """Read an image, supporting non-ASCII paths on Windows."""

    if not path.is_file():
        raise FileNotFoundError(f"Input image does not exist: {path}")
    encoded = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"OpenCV could not decode the input image: {path}")
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)


def write_jpeg(path: Path, image: np.ndarray, quality: int = 95) -> None:
    """Write a JPEG image, supporting non-ASCII paths on Windows."""

    path.parent.mkdir(parents=True, exist_ok=True)
    success, encoded = cv2.imencode(
        ".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality]
    )
    if not success:
        raise OSError(f"OpenCV could not encode image for: {path}")
    encoded.tofile(path)


def mean_squared_error(first: np.ndarray, second: np.ndarray) -> float:
    if first.shape != second.shape:
        raise ValueError(
            f"Images must have the same shape, got {first.shape} and {second.shape}"
        )
    difference = first.astype(np.float64) - second.astype(np.float64)
    return float(np.mean(difference * difference))


def draw_stroke(image: np.ndarray, stroke: Stroke) -> None:
    """Alpha-blend one pencil or eraser stroke into a grayscale image."""

    mask = np.zeros(image.shape, dtype=np.uint8)
    cv2.line(
        mask,
        (stroke.x_start, stroke.y_start),
        (stroke.x_end, stroke.y_end),
        255,
        thickness=1,
        lineType=cv2.LINE_8,
    )
    selected = mask != 0
    alpha = stroke.pressure / 255.0
    destination = 255.0 if stroke.is_eraser else float(stroke.shade)
    current = image[selected].astype(np.float64)
    image[selected] = np.clip(
        np.rint(current + alpha * (destination - current)), 0, 255
    ).astype(np.uint8)


class TargetImage:
    """Preprocessed target and local fitness calculator."""

    def __init__(self, image: np.ndarray) -> None:
        if image.ndim != 2 or image.dtype != np.uint8:
            raise ValueError("Target image must be a uint8 grayscale matrix")
        self.image = image

    @classmethod
    def from_file(cls, path: Path, size: int) -> "TargetImage":
        return cls(read_grayscale(path, size))

    def fitness(self, image: np.ndarray) -> float:
        return mean_squared_error(image, self.image)

    def stroke_improvement(self, image: np.ndarray, stroke: Stroke) -> float:
        """Return local MSE improvement produced by a candidate stroke."""

        height, width = image.shape
        x_start = max(0, min(stroke.x_start, stroke.x_end) - 2)
        y_start = max(0, min(stroke.y_start, stroke.y_end) - 2)
        x_end = min(width - 1, max(stroke.x_start, stroke.x_end) + 2)
        y_end = min(height - 1, max(stroke.y_start, stroke.y_end) + 2)

        image_region = image[y_start : y_end + 1, x_start : x_end + 1].copy()
        target_region = self.image[y_start : y_end + 1, x_start : x_end + 1]
        previous_fitness = mean_squared_error(image_region, target_region)
        draw_stroke(image_region, stroke.shifted(x_start, y_start))
        new_fitness = mean_squared_error(image_region, target_region)
        return previous_fitness - new_fitness


class Canvas:
    """Current generated image and its cached full-image fitness."""

    def __init__(self, target: TargetImage) -> None:
        self.target = target
        self.image = np.full(target.image.shape, 255, dtype=np.uint8)
        self.fitness = target.fitness(self.image)

    def apply(self, stroke: Stroke) -> float:
        draw_stroke(self.image, stroke)
        self.fitness = self.target.fitness(self.image)
        return self.fitness
