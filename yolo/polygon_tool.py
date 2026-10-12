"""Select and save a lane polygon from the first video frame."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def save_polygon(points: list[tuple[int, int]], path: Path) -> None:
    if len(points) < 3:
        raise ValueError("at least three points are required")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"points": points}, indent=2) + "\n", encoding="utf-8")


def select_polygon(source: str, output: Path) -> None:
    import cv2

    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"could not open source: {source}")
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"could not read the first frame from: {source}")

    points: list[tuple[int, int]] = []
    window = "Click lane points; press Enter to save, Esc to cancel"

    def on_click(event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))

    cv2.namedWindow(window)
    cv2.setMouseCallback(window, on_click)
    while True:
        display = frame.copy()
        for point in points:
            cv2.circle(display, point, 5, (0, 255, 255), -1)
        if len(points) > 1:
            for start, end in zip(points, points[1:]):
                cv2.line(display, start, end, (0, 255, 255), 2)
        cv2.imshow(window, display)
        key = cv2.waitKey(30) & 0xFF
        if key in (13, 10):
            save_polygon(points, output)
            break
        if key == 27:
            raise RuntimeError("polygon selection cancelled")
    cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("-o", "--output", type=Path, default=Path("lane.json"))
    args = parser.parse_args()
    select_polygon(args.source, args.output)


if __name__ == "__main__":
    main()
