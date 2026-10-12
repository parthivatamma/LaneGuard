"""Caltrans Lane Closure System client and lane-site metadata helpers."""

from __future__ import annotations

import json
import logging
import math
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any

log = logging.getLogger("laneguard.lcs")
LCS_URL = "https://cwwp2.dot.ca.gov/data/d{d}/lcs/lcsStatusD{d:02d}.json"
DISTRICTS = {
    1: "Eureka", 2: "Redding", 3: "Marysville / Sacramento", 4: "Bay Area",
    5: "San Luis Obispo", 6: "Fresno", 7: "Los Angeles", 8: "San Bernardino",
    9: "Bishop", 10: "Stockton", 11: "San Diego", 12: "Orange County",
}
CACHE_TTL_S = 300
_cache: dict[int, tuple[float, list[dict[str, Any]]]] = {}
_cache_lock = threading.Lock()


def _s(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def flatten(record: dict[str, Any]) -> dict[str, Any]:
    """Flatten one LCS response record to fields used by Lane Guard."""
    raw = record.get("lcs", record)
    location = raw.get("location", {})
    begin, end = location.get("begin", {}), location.get("end", {})
    closure = raw.get("closure", {})
    timestamp = closure.get("closureTimestamp", {})

    def flag(name: str) -> tuple[bool, str]:
        block = closure.get(name, {})
        prefix = name[0].upper() + name[1:]
        return (
            _s(block.get(f"is{prefix}")) == "true",
            _s(block.get(f"{name}Timestamp", {}).get(f"{name}Epoch")),
        )

    is1097, t1097 = flag("code1097")
    is1098, t1098 = flag("code1098")
    is1022, t1022 = flag("code1022")
    status = "cancelled" if is1022 else "picked_up" if is1098 else "active" if is1097 else "planned"

    def postmile(prefix: str, milepost: str, alignment: str) -> str:
        suffix = f" {alignment}" if alignment else ""
        return f"{prefix}{milepost}{suffix}".strip()

    return {
        "index": _s(raw.get("index")), "closureID": _s(closure.get("closureID")),
        "logNumber": _s(closure.get("logNumber")), "district": _s(begin.get("beginDistrict")),
        "county": _s(begin.get("beginCounty")), "route": _s(begin.get("beginRoute")),
        "direction": _s(location.get("travelFlowDirection")),
        "beginPostmile": postmile(_s(begin.get("beginPostmilePrefix")), _s(begin.get("beginPostmile")), _s(begin.get("beginAlignment"))),
        "endPostmile": postmile(_s(end.get("endPostmilePrefix")), _s(end.get("endPostmile")), _s(end.get("endAlignment"))),
        "beginMilepost": _s(begin.get("beginMilepost")), "endMilepost": _s(end.get("endMilepost")),
        "beginLocation": _s(begin.get("beginLocationName")), "endLocation": _s(end.get("endLocationName")),
        "nearbyPlace": _s(begin.get("beginNearbyPlace")),
        "beginLat": _float(begin.get("beginLatitude")), "beginLon": _float(begin.get("beginLongitude")),
        "endLat": _float(end.get("endLatitude")), "endLon": _float(end.get("endLongitude")),
        "facility": _s(closure.get("facility")), "typeOfClosure": _s(closure.get("typeOfClosure")),
        "typeOfWork": _s(closure.get("typeOfWork")), "durationOfClosure": _s(closure.get("durationOfClosure")),
        "lanesClosed": _s(closure.get("lanesClosed")), "totalExistingLanes": _s(closure.get("totalExistingLanes")),
        "estimatedDelay": _s(closure.get("estimatedDelay")),
        "start": f"{_s(timestamp.get('closureStartDate'))} {_s(timestamp.get('closureStartTime'))}".strip(),
        "end": f"{_s(timestamp.get('closureEndDate'))} {_s(timestamp.get('closureEndTime'))}".strip(),
        "startEpoch": _int(timestamp.get("closureStartEpoch")), "endEpoch": _int(timestamp.get("closureEndEpoch")),
        "status": status, "code1097Epoch": _int(t1097), "code1098Epoch": _int(t1098), "code1022Epoch": _int(t1022),
    }


def fetch_district(district: int, force: bool = False, timeout: float = 20.0) -> list[dict[str, Any]]:
    if district not in DISTRICTS:
        raise ValueError(f"district must be 1..12, got {district}")
    with _cache_lock:
        cached = _cache.get(district)
        if cached and not force and time.time() - cached[0] < CACHE_TTL_S:
            return cached[1]
    request = urllib.request.Request(
        LCS_URL.format(d=district), headers={"User-Agent": "laneguard/0.1"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = json.load(response)
    records = raw.get("data", raw if isinstance(raw, list) else [])
    flattened = [flatten(record) for record in records]
    with _cache_lock:
        _cache[district] = (time.time(), flattened)
    log.info("LCS district %d: %d closures", district, len(flattened))
    return flattened


def search(
    records: list[dict[str, Any]], query: str = "", status: str = "",
    lane_only: bool = False, limit: int = 100,
) -> list[dict[str, Any]]:
    query = query.lower().strip()
    allowed = {"Lane", "One-Way Traffic", "Alternating", "Alternating Lanes", "Moving"}
    result = []
    fields = ("route", "county", "nearbyPlace", "beginLocation", "endLocation", "typeOfWork", "closureID", "index")
    for record in records:
        if status and record["status"] != status:
            continue
        if lane_only and record["typeOfClosure"] not in allowed:
            continue
        if query and not any(query in _s(record[field]).lower() for field in fields):
            continue
        result.append({**record, "distance_mi": None})
    result.sort(key=lambda item: (item["status"] != "active", item["startEpoch"] or 0))
    return result[:limit]


def site_path(lane_path: Path) -> Path:
    lane_path = Path(lane_path)
    return lane_path.with_name(lane_path.stem + ".site.json")


def load_site(lane_path: Path) -> dict[str, Any]:
    try:
        return json.loads(site_path(lane_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def save_site(lane_path: Path, site: dict[str, Any]) -> dict[str, Any]:
    saved = {**site, "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    path = site_path(lane_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(saved, indent=2) + "\n", encoding="utf-8")
    return saved


def event_stamp(site: dict[str, Any]) -> dict[str, Any]:
    if not site:
        return {}
    lcs = site.get("lcs", {})
    keys = ("index", "closureID", "logNumber", "district", "county", "route", "direction",
            "beginPostmile", "endPostmile", "typeOfWork", "typeOfClosure", "lanesClosed",
            "facility", "start", "end")
    return {"lcs": {key: lcs.get(key) for key in keys if key in lcs},
            "crew": site.get("crew", ""), "operator": site.get("operator", ""),
            "notes": site.get("notes", ""), "camera_lat": site.get("camera_lat"),
            "camera_lon": site.get("camera_lon")}
