"""Alert output with an optional Jetson GPIO implementation."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol

from .intrusion import IntrusionEvent

LOGGER = logging.getLogger(__name__)


class AlertSink(Protocol):
    def emit(self, event: IntrusionEvent) -> None: ...

    def close(self) -> None: ...


@dataclass
class ConsoleAlertSink:
    """Portable alert sink used on desktop and Windows."""

    def emit(self, event: IntrusionEvent) -> None:
        LOGGER.warning(
            "intrusion event=%s track=%s at (%.1f, %.1f) t=%.3f",
            event.event_type.value,
            event.track_id,
            event.x,
            event.y,
            event.timestamp,
        )

    def close(self) -> None:
        pass


class GPIOAlertSink:
    def __init__(self, pin: int, pulse_seconds: float = 1.0) -> None:
        try:
            import Jetson.GPIO as gpio
        except ImportError as exc:
            raise RuntimeError(
                "Jetson.GPIO is unavailable; use --alert console on this machine"
            ) from exc
        if pulse_seconds <= 0:
            raise ValueError("pulse_seconds must be positive")
        self._gpio = gpio
        self._pin = pin
        self._pulse_seconds = pulse_seconds
        gpio.setmode(gpio.BOARD)
        gpio.setup(pin, gpio.OUT, initial=gpio.LOW)

    def emit(self, event: IntrusionEvent) -> None:
        import time

        self._gpio.output(self._pin, self._gpio.HIGH)
        time.sleep(self._pulse_seconds)
        self._gpio.output(self._pin, self._gpio.LOW)

    def close(self) -> None:
        self._gpio.output(self._pin, self._gpio.LOW)
        self._gpio.cleanup(self._pin)


def create_alert_sink(kind: str, pin: int, pulse_seconds: float) -> AlertSink:
    if kind == "console":
        return ConsoleAlertSink()
    if kind == "gpio":
        return GPIOAlertSink(pin, pulse_seconds)
    raise ValueError(f"unsupported alert sink: {kind}")
