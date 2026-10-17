"""Small constant-acceleration Kalman filter for road-plane coordinates."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import numpy as np


@dataclass(frozen=True)
class Kinematics:
    x: float
    y: float
    speed_x: float
    speed_y: float
    acceleration_x: float
    acceleration_y: float
    uncertainty: float

    @property
    def speed(self) -> float:
        return sqrt(self.speed_x**2 + self.speed_y**2)


class ConstantAccelerationFilter:
    """Filter x/y positions with a six-state constant-acceleration model."""

    def __init__(self, measurement_noise: float = 0.75) -> None:
        if measurement_noise <= 0:
            raise ValueError("measurement_noise must be positive")
        self._measurement_noise = measurement_noise
        self._state: np.ndarray | None = None
        self._covariance: np.ndarray | None = None
        self._timestamp: float | None = None

    def update(self, timestamp: float, position: tuple[float, float]) -> Kinematics:
        if self._timestamp is not None and timestamp <= self._timestamp:
            raise ValueError("timestamps must increase")
        if self._state is None:
            self._state = np.array([position[0], position[1], 0, 0, 0, 0], dtype=float)
            self._covariance = np.diag([self._measurement_noise**2, self._measurement_noise**2, 100, 100, 100, 100])
            self._timestamp = timestamp
            return self._result()

        dt = timestamp - self._timestamp
        transition = np.eye(6)
        transition[0, 2], transition[0, 4] = dt, 0.5 * dt * dt
        transition[1, 3], transition[1, 5] = dt, 0.5 * dt * dt
        process = np.diag([0.01, 0.01, 0.1, 0.1, 1.0, 1.0]) * max(dt, 1e-3)
        self._state = transition @ self._state
        self._covariance = transition @ self._covariance @ transition.T + process

        observation = np.zeros((2, 6))
        observation[0, 0] = observation[1, 1] = 1
        noise = np.eye(2) * self._measurement_noise**2
        innovation = np.asarray(position) - observation @ self._state
        gain = self._covariance @ observation.T @ np.linalg.inv(observation @ self._covariance @ observation.T + noise)
        self._state += gain @ innovation
        self._covariance = (np.eye(6) - gain @ observation) @ self._covariance
        self._timestamp = timestamp
        return self._result()

    def _result(self) -> Kinematics:
        assert self._state is not None and self._covariance is not None
        return Kinematics(
            x=float(self._state[0]),
            y=float(self._state[1]),
            speed_x=float(self._state[2]),
            speed_y=float(self._state[3]),
            acceleration_x=float(self._state[4]),
            acceleration_y=float(self._state[5]),
            uncertainty=sqrt(max(float(self._covariance[0, 0] + self._covariance[1, 1]), 0)),
        )
