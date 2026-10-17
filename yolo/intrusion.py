"""Pure-Python intrusion state machine.

This module deliberately has no OpenCV, Ultralytics, or Jetson dependencies so
it can be tested on any development machine.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import hypot
from typing import Mapping, Sequence

from shapely.geometry import Point, Polygon

from .calib import GroundPlane
from .kinematics import ConstantAccelerationFilter, Kinematics


class IntrusionState(Enum):
    SAFE = "SAFE"
    INTRUSION = "INTRUSION"
    STOPPED_INTRUSION = "STOPPED_INTRUSION"


class IntrusionEventType(Enum):
    INTRUSION = "INTRUSION"
    STOPPED_INTRUSION = "STOPPED_INTRUSION"
    TIME_TO_ARRIVAL = "TIME_TO_ARRIVAL"


@dataclass(frozen=True)
class IntrusionEvent:
    track_id: int
    event_type: IntrusionEventType
    timestamp: float
    x: float
    y: float


@dataclass
class _TrackState:
    state: IntrusionState
    last_x: float
    last_y: float
    last_seen: float
    stationary_since: float | None = None
    filter: ConstantAccelerationFilter | None = None
    kinematics: Kinematics | None = None
    arrival_triggered: bool = False


class IntrusionTracker:
    """Emit one event when a track enters a polygon and one when it stops."""

    def __init__(
        self,
        lane_coordinates: Sequence[tuple[float, float]],
        movement_threshold: float = 2.0,
        stopped_time_threshold: float = 3.0,
        timeout: float = 10.0,
        ground_plane: GroundPlane | None = None,
        taper_world: tuple[float, float] | None = None,
        arrival_threshold: float = 3.0,
    ) -> None:
        if len(lane_coordinates) < 3:
            raise ValueError("lane_coordinates must contain at least three points")
        if movement_threshold < 0:
            raise ValueError("movement_threshold must be non-negative")
        if stopped_time_threshold < 0:
            raise ValueError("stopped_time_threshold must be non-negative")
        if timeout < 0:
            raise ValueError("timeout must be non-negative")
        if arrival_threshold <= 0:
            raise ValueError("arrival_threshold must be positive")
        if (ground_plane is None) != (taper_world is None):
            raise ValueError("ground_plane and taper_world must be provided together")

        polygon = Polygon(lane_coordinates)
        if polygon.is_empty or not polygon.is_valid:
            raise ValueError("lane_coordinates must define a valid polygon")

        self._polygon = polygon
        self._movement_threshold = movement_threshold
        self._stopped_time_threshold = stopped_time_threshold
        self._timeout = timeout
        self._ground_plane = ground_plane
        self._taper_world = taper_world
        self._arrival_threshold = arrival_threshold
        self._tracks: dict[int, _TrackState] = {}

    def update(
        self,
        timestamp: float,
        active_tracks: Sequence[Mapping[str, float | int]],
    ) -> list[IntrusionEvent]:
        events: list[IntrusionEvent] = []
        updated_ids: set[int] = set()

        for track in active_tracks:
            track_id = int(track["track_id"])
            x = float(track["bottom_center_x"])
            y = float(track["bottom_center_y"])
            updated_ids.add(track_id)
            inside = self._polygon.covers(Point(x, y))
            previous = self._tracks.get(track_id)
            track_filter = (
                ConstantAccelerationFilter()
                if previous is None and self._ground_plane is not None
                else None
            )
            kinematics = self._update_kinematics(previous, timestamp, (x, y), track_filter)

            if previous is None:
                state = IntrusionState.INTRUSION if inside else IntrusionState.SAFE
                self._tracks[track_id] = _TrackState(
                    state=state,
                    last_x=x,
                    last_y=y,
                    last_seen=timestamp,
                    stationary_since=timestamp if inside else None,
                    filter=track_filter,
                    kinematics=kinematics,
                )
                if inside:
                    events.append(self._event(track_id, IntrusionEventType.INTRUSION, timestamp, x, y))
                continue

            previous.last_seen = timestamp
            previous.kinematics = kinematics
            if kinematics and self._should_trigger_arrival(previous, kinematics):
                previous.arrival_triggered = True
                events.append(self._event(track_id, IntrusionEventType.TIME_TO_ARRIVAL, timestamp, x, y))
            if not inside:
                previous.state = IntrusionState.SAFE
                previous.stationary_since = None
                previous.last_x, previous.last_y = x, y
                continue

            if previous.state is IntrusionState.SAFE:
                previous.state = IntrusionState.INTRUSION
                previous.stationary_since = timestamp
                events.append(self._event(track_id, IntrusionEventType.INTRUSION, timestamp, x, y))
                previous.last_x, previous.last_y = x, y
                continue

            moving = hypot(x - previous.last_x, y - previous.last_y) > self._movement_threshold
            if moving:
                previous.state = IntrusionState.INTRUSION
                previous.stationary_since = timestamp
            elif previous.stationary_since is None:
                previous.stationary_since = timestamp

            if previous.state is not IntrusionState.STOPPED_INTRUSION:
                duration = timestamp - previous.stationary_since
                if duration >= self._stopped_time_threshold:
                    previous.state = IntrusionState.STOPPED_INTRUSION
                    events.append(
                        self._event(
                            track_id,
                            IntrusionEventType.STOPPED_INTRUSION,
                            timestamp,
                            x,
                            y,
                        )
                    )
                else:
                    previous.state = IntrusionState.INTRUSION

            previous.last_x, previous.last_y = x, y

        for track_id, track in list(self._tracks.items()):
            if track_id not in updated_ids and timestamp - track.last_seen > self._timeout:
                del self._tracks[track_id]
        return events

    def _update_kinematics(
        self,
        previous: _TrackState | None,
        timestamp: float,
        image_position: tuple[float, float],
        new_filter: ConstantAccelerationFilter | None,
    ) -> Kinematics | None:
        if self._ground_plane is None:
            return None
        world_position = self._ground_plane.project(image_position)
        filter_state = new_filter if previous is None else previous.filter
        if filter_state is None:
            filter_state = ConstantAccelerationFilter()
            if previous is not None:
                previous.filter = filter_state
        return filter_state.update(timestamp, world_position)

    def _should_trigger_arrival(self, track: _TrackState, kinematics: Kinematics) -> bool:
        if track.arrival_triggered or self._taper_world is None:
            return False
        distance = self._taper_world[1] - kinematics.y
        closing_speed = kinematics.speed_y
        return distance > 0 and closing_speed > 0 and distance / closing_speed <= self._arrival_threshold

    @staticmethod
    def _event(
        track_id: int,
        event_type: IntrusionEventType,
        timestamp: float,
        x: float,
        y: float,
    ) -> IntrusionEvent:
        return IntrusionEvent(track_id, event_type, timestamp, x, y)

    def get_state(self, track_id: int) -> IntrusionState | None:
        track = self._tracks.get(track_id)
        return track.state if track else None

    def tracked_ids(self) -> set[int]:
        return set(self._tracks)

    def clear(self) -> None:
        self._tracks.clear()
