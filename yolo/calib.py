"""Road-plane homography calibration utilities."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Sequence

import numpy as np


Point = tuple[float, float]


@dataclass(frozen=True)
class GroundPlane:
    """Map image pixels to metres on the road plane."""

    matrix: tuple[tuple[float, float, float], ...]

    def __post_init__(self) -> None:
        matrix = np.asarray(self.matrix, dtype=float)
        if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
            raise ValueError("homography must be a finite 3x3 matrix")
        if abs(float(np.linalg.det(matrix))) < 1e-12:
            raise ValueError("homography must be invertible")

    def project(self, point: Point) -> Point:
        vector = np.asarray([point[0], point[1], 1.0], dtype=float)
        result = np.asarray(self.matrix, dtype=float) @ vector
        if abs(float(result[2])) < 1e-12:
            raise ValueError("homography projects point to infinity")
        projected = result[:2] / result[2]
        if not np.all(np.isfinite(projected)):
            raise ValueError("homography produced a non-finite point")
        return float(projected[0]), float(projected[1])


def fit_ground_plane(
    image_points: Sequence[Point],
    world_points: Sequence[Point],
) -> GroundPlane:
    """Fit a homography from four or more corresponding points."""
    if len(image_points) != len(world_points) or len(image_points) < 4:
        raise ValueError("at least four corresponding image and world points are required")
    image = np.asarray(image_points, dtype=float)
    world = np.asarray(world_points, dtype=float)
    if image.shape != (len(image_points), 2) or world.shape != image.shape:
        raise ValueError("calibration points must be pairs of numbers")
    if not np.all(np.isfinite(image)) or not np.all(np.isfinite(world)):
        raise ValueError("calibration points must be finite")

    rows: list[list[float]] = []
    for (u, v), (x, y) in zip(image, world):
        rows.extend([
            [-u, -v, -1, 0, 0, 0, u * x, v * x, x],
            [0, 0, 0, -u, -v, -1, u * y, v * y, y],
        ])
    _, singular_values, vh = np.linalg.svd(np.asarray(rows))
    if singular_values[-2] < 1e-12 or singular_values[-2] / max(singular_values[-1], 1e-15) < 1.01:
        raise ValueError("calibration points are degenerate")
    matrix = vh[-1].reshape(3, 3)
    matrix /= matrix[2, 2]
    if not all(isfinite(float(value)) for value in matrix.flat):
        raise ValueError("could not compute a finite homography")
    return GroundPlane(tuple(tuple(float(value) for value in row) for row in matrix))
