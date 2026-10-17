"""Run YOLO tracking and lane intrusion detection.

The module imports GPU-dependent packages only when the application starts, so
the state machine and its tests work on Windows, macOS, and CI runners.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
from datetime import datetime
from collections import deque
from pathlib import Path
from typing import Any

from .alerts import create_alert_sink
from .calib import GroundPlane
from .intrusion import IntrusionEvent, IntrusionState, IntrusionTracker
from .lcs import event_stamp, load_site

LOGGER = logging.getLogger(__name__)


def load_lane(path: Path) -> list[tuple[float, float]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        points = data["points"]
        result = [(float(point[0]), float(point[1])) for point in points]
    except (OSError, KeyError, TypeError, ValueError, IndexError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid lane polygon file {path}") from exc
    if len(result) < 3:
        raise ValueError(f"lane polygon file {path} contains fewer than three points")
    return result


def load_lane_calibration(path: Path) -> tuple[GroundPlane | None, tuple[float, float] | None, float]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        calibration = data.get("calibration")
        if not calibration:
            return None, None, 3.0
        ground_plane = GroundPlane(tuple(tuple(float(value) for value in row) for row in calibration["homography"]))
        taper = tuple(float(value) for value in calibration["taper_world"])
        if len(taper) != 2:
            raise ValueError("taper_world must contain two coordinates")
        arrival_threshold = float(calibration.get("arrival_threshold", 3.0))
        return ground_plane, (taper[0], taper[1]), arrival_threshold
    except (OSError, KeyError, TypeError, ValueError, IndexError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid lane calibration file {path}") from exc


class EvidenceBuffer:
    """Keep frames before an event and write a short clip after it."""

    def __init__(self, output_dir: Path, fps: float, width: int, height: int, seconds: float, site: dict[str, Any] | None = None) -> None:
        self._output_dir = output_dir
        self._fps = max(fps, 1.0)
        self._frames: deque[Any] = deque(maxlen=max(1, round(self._fps * seconds)))
        self._size = (width, height)
        self._seconds = seconds
        self._pending: list[tuple[float, int, list[Any]]] = []
        self._metadata: list[dict[str, Any]] = []
        self._site = site or {}
        self._run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    def add(self, frame: Any) -> None:
        self._frames.append(frame.copy())
        for index in range(len(self._pending) - 1, -1, -1):
            timestamp, event_index, frames = self._pending[index]
            frames.append(frame.copy())
            if len(frames) >= round(self._fps * self._seconds) * 2 + 1:
                self._write(timestamp, event_index, frames)
                del self._pending[index]

    def trigger(self, event: IntrusionEvent, event_index: int) -> None:
        self._pending.append((event.timestamp, event_index, list(self._frames)))
        self._metadata.append({
                "id": event_index,
                "track_id": event.track_id,
                "type": event.event_type.value,
                "timestamp": event.timestamp,
                "clip": self._clip_name(event_index, event.timestamp),
                **event_stamp(self._site),
            })

    def flush(self) -> None:
        for timestamp, event_index, frames in self._pending:
            self._write(timestamp, event_index, frames)
        self._pending.clear()

    def _write(self, event_timestamp: float, event_index: int, frames: list[Any]) -> Path:
        self._output_dir.mkdir(parents=True, exist_ok=True)
        output = self._output_dir / self._clip_name(event_index, event_timestamp)
        try:
            from imageio_ffmpeg import get_ffmpeg_exe
        except ImportError as exc:
            raise RuntimeError(
                "evidence encoding requires imageio-ffmpeg; install requirements.txt"
            ) from exc

        command = [
            get_ffmpeg_exe(),
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{self._size[0]}x{self._size[1]}",
            "-r",
            str(self._fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output),
        ]
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        try:
            for frame in frames:
                process.stdin.write(frame.tobytes())
            process.stdin.close()
            stderr = process.stderr.read().decode("utf-8", errors="replace")
            return_code = process.wait()
        except Exception:
            process.kill()
            process.wait()
            raise
        if return_code != 0:
            raise RuntimeError(f"FFmpeg could not create evidence video: {stderr[-1000:]}")
        (self._output_dir / "events.json").write_text(
            json.dumps(self._metadata, indent=2) + "\n",
            encoding="utf-8",
        )
        return output

    def _clip_name(self, event_index: int, event_timestamp: float) -> str:
        seconds = max(0, round(event_timestamp * 1000))
        return f"{self._run_id}_event_{event_index:04d}_t{seconds:09d}ms.mp4"


class AnnotatedVideoWriter:
    """Encode the complete processed run as a browser-compatible MP4."""

    def __init__(self, output: Path, fps: float, width: int, height: int) -> None:
        from imageio_ffmpeg import get_ffmpeg_exe

        output.parent.mkdir(parents=True, exist_ok=True)
        self._output = output
        self._process = subprocess.Popen(
            [
                get_ffmpeg_exe(), "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
                "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", str(fps),
                "-i", "-", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", str(output),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

    def write(self, frame: Any) -> None:
        self._process.stdin.write(frame.tobytes())

    def close(self) -> None:
        self._process.stdin.close()
        stderr = self._process.stderr.read().decode("utf-8", errors="replace")
        if self._process.wait() != 0:
            raise RuntimeError(f"FFmpeg could not create annotated video: {stderr[-1000:]}")


def _tracks_from_result(result: Any, classes: set[int]) -> list[dict[str, float | int]]:
    boxes = result.boxes
    if boxes is None or boxes.id is None:
        return []
    xyxy = boxes.xyxy.cpu().tolist()
    ids = boxes.id.int().cpu().tolist()
    class_ids = boxes.cls.int().cpu().tolist()
    tracks: list[dict[str, float | int]] = []
    for box, track_id, class_id in zip(xyxy, ids, class_ids):
        if classes and class_id not in classes:
            continue
        x1, y1, x2, y2 = box
        tracks.append(
            {
                "track_id": int(track_id),
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "bottom_center_x": (x1 + x2) / 2,
                "bottom_center_y": y2,
            }
        )
    return tracks


def _draw_lane_polygon(frame: Any, lane: list[tuple[float, float]], cv2: Any) -> Any:
    """Draw the configured intrusion polygon on an annotated video frame."""
    points = [(round(x), round(y)) for x, y in lane]
    for start, end in zip(points, points[1:] + points[:1]):
        cv2.line(frame, start, end, (0, 255, 0), 3, cv2.LINE_AA)

    label_x, label_y = points[0]
    cv2.putText(
        frame,
        "LANE / INTRUSION ZONE",
        (label_x, max(25, label_y - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    return frame


def _annotate_frame(
    result: Any,
    lane: list[tuple[float, float]],
    active_tracks: list[dict[str, float | int]],
    tracker: IntrusionTracker,
    cv2: Any,
) -> Any:
    annotated = result.plot()
    _draw_lane_polygon(annotated, lane, cv2)
    for track in active_tracks:
        state = tracker.get_state(int(track["track_id"]))
        if state not in (IntrusionState.INTRUSION, IntrusionState.STOPPED_INTRUSION):
            continue
        color = (0, 0, 255) if state is IntrusionState.STOPPED_INTRUSION else (0, 165, 255)
        x1, y1, x2, y2 = (
            round(float(track[key])) for key in ("x1", "y1", "x2", "y2")
        )
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 4)
        cv2.putText(
            annotated,
            state.value.replace("_", " "),
            (x1, max(25, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
            cv2.LINE_AA,
        )
    return annotated


def run(args: argparse.Namespace) -> int:
    try:
        import cv2
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "the runtime pipeline requires opencv-python and ultralytics; "
            "install dependencies from requirements.txt"
        ) from exc

    lane = load_lane(args.lane)
    ground_plane, taper_world, arrival_threshold = load_lane_calibration(args.lane)
    model = YOLO(str(args.model))
    source: str | int = int(args.source) if args.source.isdecimal() else args.source
    if isinstance(source, str):
        source_path = Path(source).expanduser()
        if source_path.exists() and not source_path.is_file():
            raise RuntimeError(f"source must be a video file, not a directory: {source_path}")
        if not source_path.exists() and "://" not in source:
            raise RuntimeError(f"video source does not exist: {source_path}")
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"could not open source: {args.source}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = round(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        raise RuntimeError("video source did not provide a valid frame size")
    evidence = EvidenceBuffer(
        args.evidence_dir, fps, width, height, args.evidence_seconds,
        load_site(args.lane),
    )
    annotated_writer = AnnotatedVideoWriter(args.annotated_video, fps, width, height)
    tracker = IntrusionTracker(
        lane,
        movement_threshold=args.movement_threshold,
        stopped_time_threshold=args.stopped_seconds,
        timeout=args.track_timeout,
        ground_plane=ground_plane,
        taper_world=taper_world,
        arrival_threshold=arrival_threshold,
    )
    alerts = create_alert_sink(args.alert, args.gpio_pin, args.gpio_pulse_seconds)
    event_index = 0
    try:
        frame_number = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            timestamp = frame_number / fps
            results = model.track(
                frame,
                persist=True,
                tracker=args.tracker,
                classes=sorted(args.classes) if args.classes else None,
                verbose=False,
            )
            active_tracks = _tracks_from_result(results[0], args.classes)
            events = tracker.update(timestamp, active_tracks)

            annotated = _annotate_frame(results[0], lane, active_tracks, tracker, cv2)
            annotated_writer.write(annotated)
            evidence.add(annotated)
            for event in events:
                alerts.emit(event)
                event_index += 1
                evidence.trigger(event, event_index)

            if args.show:
                cv2.imshow("YOLO intrusion detection", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            frame_number += 1
    finally:
        capture.release()
        annotated_writer.close()
        evidence.flush()
        alerts.close()
        if args.show:
            cv2.destroyAllWindows()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="video path, camera index, or RTSP URL")
    parser.add_argument("--model", required=True, type=Path, help="YOLO .pt or TensorRT .engine")
    parser.add_argument("--lane", required=True, type=Path, help="polygon JSON created by polygon_tool")
    parser.add_argument("--tracker", default="bytetrack.yaml")
    parser.add_argument("--classes", nargs="*", type=int, default=[2, 3, 5, 7])
    parser.add_argument("--movement-threshold", type=float, default=2.0)
    parser.add_argument("--stopped-seconds", type=float, default=3.0)
    parser.add_argument("--track-timeout", type=float, default=10.0)
    parser.add_argument("--evidence-dir", type=Path, default=Path("evidence"))
    parser.add_argument(
        "--annotated-video",
        type=Path,
        default=Path("evidence") / "annotated_full.mp4",
        help="output path for the complete annotated run",
    )
    parser.add_argument("--evidence-seconds", type=float, default=5.0)
    parser.add_argument("--alert", choices=("console", "gpio"), default="console")
    parser.add_argument("--gpio-pin", type=int, default=7)
    parser.add_argument("--gpio-pulse-seconds", type=float, default=1.0)
    parser.add_argument("--show", action="store_true")
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    raise SystemExit(run(build_parser().parse_args()))


if __name__ == "__main__":
    main()
