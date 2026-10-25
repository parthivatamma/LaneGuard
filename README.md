# LANEGUARD // Edge Workzone Intrusion Telemetry

> A deterministic, low-latency computer vision pipeline executing real-time polygon breach detection, homographic kinematics, and hardware-level GPIO alerting across Nvidia Jetson Nano and desktop TensorRT architectures.

![Architecture](https://img.shields.io/badge/Architecture-Edge_TensorRT_%7C_State_Machine-black?style=flat-square)
![Telemetry](https://img.shields.io/badge/Telemetry-Caltrans_LCS_%7C_GPIO_Sink-white?style=flat-square)
![Stack](https://img.shields.io/badge/Stack-YOLOv11_%7C_OpenCV_%7C_GStreamer-black?style=flat-square)

## Overview

**LaneGuard** is not just a bounding-box wrapper; it is an autonomous workzone perimeter defense and forensic logging engine. It decouples the core intrusion state machine from heavy inference bindings—allowing zero-dependency logic verification on any host while deferring `OpenCV` and `Ultralytics` memory allocation strictly to active execution nodes across `FP16` TensorRT engines (`RTX 4060 Ti`) and embedded `GStreamer` pipelines (`Jetson Nano`).

Engineered for deterministic roadside reliability, the system features homographic ground-plane velocity projection, sub-3-second Time-to-Arrival (`TTA`) breach interception, and automated forensic clip extraction stamped with live Caltrans Lane Closure System (`LCS`) metadata.

## Architecture & State Routing Matrix

Roadside edge compute is heavily constrained by video decode overhead, thermal throttling, and intermittent cellular uplinks in active highway workzones. LaneGuard mitigates this by isolating the geometric state machine from the frame grabber and caching external telemetry locally. If upstream Caltrans APIs go dark, the pipeline degrades gracefully to cached or local-only audit logging without dropping a single inference frame.

| State | Kinematics / Geometry | Caltrans LCS Feed | Alert Sink | Pipeline Action |
| :--- | :--- | :--- | :--- | :--- |
| **01** | `CLEAR` (Exterior to `lane.json`) | `SYNCED` (5m TTL Cache) | `IDLE` | Continuous multi-ID tracking; zero I/O write overhead |
| **02** | `TTA <= 3.0s` (Closing on Taper) | `SYNCED` / `CACHED` | `WARN` (`Console` / `GPIO`) | Emits `TIME_TO_ARRIVAL` pre-breach kinematics interrupt |
| **03** | `BREACH` (Bottom-Center Inside) | `SYNCED` (`lane.site.json`) | `ACTIVE` (`--alert gpio`) | Triggers physical GPIO pin high; stamps closure metadata |
| **04** | `BREACH` (Bottom-Center Inside) | `OFFLINE` (API Timeout) | `ACTIVE` (`Fallback Sink`) | Local-only forensic dump to `evidence\`; zero frame drop |

## Core Engineering Features

- **Homographic Ground-Plane Kinematics:** Converts raw 2D pixel trajectories into real-world metric coordinates via a 4-point clockwise perspective transform (`lane.json`). Filters noisy bounding-box jitter in meters to calculate real-time forward closing velocity and trigger `TIME_TO_ARRIVAL` alerts when a vehicle is `<= 3.0s` from the cone taper.

- **Bottom-Center Polygon Interception:** Rather than evaluating full bounding-box overlap (which triggers false positives from adjacent highway lanes), the state machine isolates the bottom-center anchor point of each tracked vehicle ID and tests it directly against the serialized `LANE / INTRUSION ZONE` vertices.

- **Collision-Free Forensic Clip Ring:** Automatically slices and serializes high-definition incident video (`YYYYMMDD_HHMMSS_event_XXXX_tXXXXXXXXXms.mp4`) alongside an append-only `events.json` manifest and full-run `annotated_full.mp4` replay, preserving bounding boxes, confidence scores, and orange/red breach telemetry.

- **Fault-Tolerant Caltrans LCS Synchronization:** Polls and caches the public Caltrans Lane Closure System feed by district with a strict 5-minute TTL. Binds official closure records and local operator audit strings (`crew/unit`, `operator`, `notes`) into `lane.site.json` while maintaining full offline detection autonomy if external network links sever.

## Tech Stack

**Inference & Vision Engine**
- `YOLOv11n` (`FP16` `.engine` / `.pt`) for high-framerate vehicle classification and persistent multi-object ID tracking.
- `OpenCV` for homography matrix transformations, polygon point-in-contour math, and frame annotation.
- `PyTorch` (CUDA-accelerated) for tensor execution and hardware-bound model loading.

**Edge & Hardware Interfacing (Jetson Nano / Desktop)**
- `GStreamer` for hardware-accelerated CSI/RTSP camera ingestion on Nvidia Jetson silicon.
- `GPIO Sink` (`--alert gpio`) for physical workzone siren/strobe actuation upon state machine breach.

**Control Center & Telemetry (`yolo.web`)**
- Built-in local web server (`Port 8765`) for interactive 4-corner calibration, polygon authoring, and timeline scrubbing.
- `pytest` for headless, zero-GPU validation of the decoupled intrusion state machine.

## Installation & Boot Sequence

### 1. Environment Provisioning (Windows / Workstation)
Initialize the isolated virtual environment and bind CUDA-accelerated dependencies:

```powershell
# Initialize and activate the virtual environment
python -m venv .venv
.venv\Scripts\activate

# If Python is unbound from PATH, invoke the binary directly:
# & "D:\Apps\python\python.exe" -m venv .venv; .\.venv\Scripts\Activate.ps1

# Install core pipeline dependencies (install matching CUDA PyTorch wheel prior if on GPU)
pip install -r requirements.txt
```

### 2. Headless CLI Calibration & TensorRT Execution
Extract the initial frame to define the intrusion boundary, then ignite the `FP16` TensorRT engine:

```powershell
# Author the intrusion zone polygon from the initial video frame
python -m yolo.polygon_tool data\video\workzone_01.mp4 --output lane.json

# Execute the live tracking and intrusion pipeline against a compiled TensorRT engine
python -m yolo.app --source data\video\workzone_01.mp4 `
  --model data\models\yolo11n_4060ti_fp16.engine --lane lane.json --show
```

### 3. Edge Deployment (Nvidia Jetson Nano)
Route the hardware-accelerated `GStreamer` camera pipeline directly into the physical GPIO alert sink:

```bash
# Execute with hardware GPIO actuation enabled (refer to Design.md for CSI GStreamer string)
python3 -m yolo.app --source "<GSTREAMER_PIPELINE_STRING>" \
  --model data/models/yolo11n_fp16.engine --lane lane.json --alert gpio
```

### 4. Interactive Command & Review Dashboard
Boot the unified web control server to calibrate ground-plane homography, sync Caltrans District feeds, and review forensic clips:

```powershell
# Launch the integrated detector and review server (Fallback port 8765 avoids Windows Port 8000 locks)
python -m yolo.web `
  --video "data\video\workzone_01.mp4" `
  --evidence-dir "evidence" `
  --port 8765
```

```powershell
# Verify the decoupled state machine integrity via unit test suite
python -m pytest
```

Navigate to `http://127.0.0.1:8765/` to draw intrusion polygons, bind `lane.site.json` Caltrans records, and inspect `evidence\events.json` incident telemetry.
