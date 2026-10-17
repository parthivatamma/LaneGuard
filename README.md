# YOLO intrusion pipeline

This repository implements the Option B prototype described in
[`Design.md`](./Design.md). The intrusion state machine is dependency-light and
can be tested on any machine; the video runner loads OpenCV and Ultralytics
only when it is executed.

## Install

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

If `python` is not on `PATH`, use the installed interpreter directly:

```powershell
& "D:\Apps\python\python.exe" -m venv .venv
.\.venv\Scripts\Activate.ps1
```

For CUDA-enabled PyTorch, install the matching wheel from the PyTorch index
before installing the remaining requirements.

## Run

Create a polygon from the first frame:

```powershell
python -m yolo.polygon_tool data\video\workzone_01.mp4 --output lane.json
```

Run tracking:

```powershell
python -m yolo.app --source data\video\workzone_01.mp4 `
  --model data\models\yolo11n_4060ti_fp16.engine --lane lane.json --show
```

The displayed window shows:

- `0.XX` next to a vehicle: the model's confidence that the detected object
  belongs to that class. For example, `car 0.87` means an estimated 87%
  confidence, not an 87% chance that the vehicle is accurately tracked.
- Colored boxes: each color identifies a tracked object ID. The color is
  chosen by Ultralytics from the track ID, so the same vehicle should normally
  keep the same color while it remains tracked. A new color or ID can appear
  after an object is lost and re-identified.
- Green polygon labeled `LANE / INTRUSION ZONE`: the exact polygon loaded from
  `lane.json`. An object's bottom-center point (the center of the bottom edge
  of its box) is tested against this polygon.
- Optional ground-plane calibration: in the dashboard, click **Pick four
  corners**, select a known pavement rectangle clockwise, enter its real width
  and length in feet, and save it. The calibration is stored beside the
  polygon in `lane.json`. The runner then filters each track in metres and can
  emit a `TIME_TO_ARRIVAL` event when its filtered forward closing time to the
  configured taper is at most three seconds.

On a Jetson, use the GStreamer source string described in `Design.md` and
`--alert gpio`. Desktop and Windows runs use the console sink by default.

## Review dashboard

The runner saves event clips and an `events.json` manifest in `evidence`.
Each new run uses unique names such as
`20261011_181500_event_0001_t000001234ms.mp4`, avoiding collisions with
previous runs. Clips contain the same YOLO boxes, confidence labels, lane
polygon, and orange/red intrusion markings as the live detector window.
Start the local dashboard. It now includes the detector controls, so no
separate detector command is required:

```powershell
python -m yolo.web `
  --port 8765
```

Open <http://127.0.0.1:8765/> in a browser, enter the source video, click
**Load video**, draw at least three polygon points, and click **Save polygon and
start**. The dashboard automatically uses `yolo11n.pt`, `lane.json`, and
`evidence\` from the project folder. The page lists intrusion events and opens
each saved clip. The source video has a normal browser timeline for scrolling
and seeking. Use **Stop detection** to stop the pipeline.

The right-hand **Caltrans closures** panel can load the public Lane Closure
System feed by district, search the returned closures, and save a selected
closure as `lane.site.json`. Selected closure metadata is stamped into new
event records. The feed is cached for five minutes; if Caltrans is unavailable,
local video detection remains usable.

The linked-closure card also accepts a local operator name, crew/unit, and
notes. These values are for your installation's audit trail and are not an
official Caltrans identity or submission.

After processing finishes, click **Show full annotated video** to watch or seek
through the entire run with vehicle boxes, confidence labels, tracking IDs,
the lane polygon, and intrusion highlights. It is saved as
`evidence\annotated_full.mp4`.

If Windows reports that port 8000 is unavailable, choose another local port:

```powershell
python -m yolo.web `
  --video "data\video\workzone_01.mp4" `
  --evidence-dir "evidence" `
  --port 8765
```

Then open <http://127.0.0.1:8765/>.

## Tests

```powershell
python -m pytest
```
