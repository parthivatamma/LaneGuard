from yolo.intrusion import IntrusionEventType, IntrusionState, IntrusionTracker


def track(track_id: int, x: float, y: float) -> dict[str, float | int]:
    return {
        "track_id": track_id,
        "bottom_center_x": x,
        "bottom_center_y": y,
    }


def test_entry_and_stopped_events_are_emitted_once() -> None:
    tracker = IntrusionTracker([(0, 0), (100, 0), (100, 100), (0, 100)], stopped_time_threshold=3)

    assert tracker.update(0, [track(1, 10, 10)])[0].event_type is IntrusionEventType.INTRUSION
    assert tracker.update(2, [track(1, 10, 10)]) == []
    stopped = tracker.update(3, [track(1, 10, 10)])
    assert len(stopped) == 1
    assert stopped[0].event_type is IntrusionEventType.STOPPED_INTRUSION
    assert tracker.update(4, [track(1, 10, 10)]) == []
    assert tracker.get_state(1) is IntrusionState.STOPPED_INTRUSION


def test_leaving_and_reentering_emits_a_new_intrusion() -> None:
    tracker = IntrusionTracker([(0, 0), (10, 0), (10, 10), (0, 10)])

    tracker.update(0, [track(4, 5, 5)])
    tracker.update(1, [track(4, 20, 20)])
    assert tracker.get_state(4) is IntrusionState.SAFE
    events = tracker.update(2, [track(4, 5, 5)])
    assert [event.event_type for event in events] == [IntrusionEventType.INTRUSION]


def test_stale_tracks_are_purged() -> None:
    tracker = IntrusionTracker([(0, 0), (10, 0), (10, 10), (0, 10)], timeout=2)
    tracker.update(0, [track(8, 20, 20)])
    tracker.update(2.1, [])
    assert tracker.tracked_ids() == set()
