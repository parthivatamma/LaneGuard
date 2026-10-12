"""Serve a local event-review dashboard.

Usage:
    python -m yolo.web --video data/video/workzone_01.mp4 --evidence-dir evidence
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import subprocess
import sys
import threading
import urllib.error
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import BinaryIO
from urllib.parse import parse_qs, unquote, urlparse

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_VIDEO = PROJECT_ROOT / "data" / "video" / "workzone_01.mp4"


HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Intrusion detection dashboard</title>
<style>
:root{--bg:#111417;--panel:#1d2125;--panel2:#24292e;--line:#343a40;--text:#edf0f2;--muted:#9aa3ab;--green:#45c878;--red:#ef4444;--amber:#f59e0b}
*{box-sizing:border-box}body{font-family:Inter,ui-sans-serif,system-ui,-apple-system,sans-serif;background:var(--bg);color:var(--text);margin:0;font-size:13px}
main{max-width:1500px;margin:auto;padding:0 8px 24px}
.topbar{height:36px;margin:0 -8px 7px;padding:0 8px;display:flex;align-items:center;gap:14px;background:#202428;border-bottom:1px solid #30363b}
.brand{font-size:15px;font-weight:800;letter-spacing:.03em}.brand span{font-weight:400;color:#8f999f}
.top-status{background:#d63d39;color:#fff;border-radius:12px;padding:3px 10px;font-size:10px;font-weight:800;letter-spacing:.05em}
.metrics{margin-left:auto;display:flex;gap:18px;color:#cbd0d4;font-size:11px}.metric b{color:#fff;margin-right:4px}
.layout{display:grid;grid-template-columns:minmax(0,2.7fr) minmax(285px,1fr);gap:7px;align-items:start}
video{width:100%;display:block;aspect-ratio:16/9;object-fit:contain;background:#050607;border:1px solid #30363b;border-radius:3px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:4px;padding:10px}
.layout>.card{padding:7px}.layout aside{padding:8px}
h1{display:none}h2{font-size:12px;text-transform:uppercase;letter-spacing:.08em;margin:0 0 9px;color:#cbd0d4}
button{background:#30363b;color:#fff;border:1px solid #41484f;border-radius:3px;padding:7px 10px;margin:3px;cursor:pointer;font-size:12px}
button:hover{background:#414950}.event{display:block;width:100%;text-align:left;background:#202529;border:0;border-top:1px solid #363c41;border-radius:0;margin:0;padding:9px 7px}
.event strong{display:block;color:#dfe4e7;text-transform:none}.event strong:before{content:"alarm";display:inline-block;background:#e84d48;color:#fff;border-radius:2px;padding:2px 5px;margin-right:7px;font-size:10px;text-transform:uppercase}.event:hover{background:#2a3035}
.muted{color:var(--muted);font-size:.86rem}input{width:100%;box-sizing:border-box;background:#15181b;color:#fff;border:1px solid #454c52;border-radius:3px;padding:8px;margin:4px 0 10px}
.primary{background:#277c4e;border-color:#379866}.danger{background:#9f302f;border-color:#c84440}#status{font-weight:700;color:var(--green);margin-left:7px}
#alarm{display:none;background:#b91c1c;color:white;font-size:1rem;font-weight:700;padding:10px;border-radius:4px;margin:8px 0}
#alarm.active{display:flex;justify-content:space-between;animation:flash .8s infinite}
@keyframes flash{50%{background:#f59e0b;color:#111827}}
#polygonCanvas{display:block;width:100%;height:auto;background:#000;border-radius:3px;cursor:crosshair;touch-action:none}
#control{margin:0 0 8px;display:flex;align-items:center;gap:4px;flex-wrap:wrap}.control-label{flex:1 1 320px}
#control input{margin:3px 0 0}.control-help{display:none}#log{max-height:48px;overflow:auto;margin:5px 0 0}
.player-tools{display:flex;gap:7px;align-items:center;flex-wrap:wrap}.player-tools a{color:#9ee6b7}
#downloadAnnotated{margin-left:auto}
aside h2{border-bottom:1px solid var(--line);padding-bottom:8px}.side-card{background:var(--panel2);border:1px solid var(--line);border-radius:3px;padding:9px;margin-bottom:8px}.side-card p{margin:5px 0;line-height:1.35}
.control-drawer{margin-top:8px}.control-drawer h2{display:inline-block;margin-right:12px}.control-drawer form{display:inline-flex;width:calc(100% - 150px)}
#polygonCard{margin-top:8px;max-width:calc(100% - 300px)}
.lcs-results{max-height:230px;overflow:auto;margin-top:6px}.lcs-result{display:block;width:100%;text-align:left;margin:2px 0;background:#202529}
.lcs-fields{display:grid;grid-template-columns:1fr 1fr;gap:5px}.lcs-fields input{margin:0}.closure-detail{font-size:11px;line-height:1.35}.closure-detail b{color:#45c878}
@media(max-width:640px){.layout{grid-template-columns:1fr}.metrics{display:none}.topbar{gap:8px}}
</style>
</head>
<body><main>
<header class="topbar"><div class="brand">LANE GUARD <span>Prototype</span></div><div class="top-status" id="topStatus">ALARM</div><div class="metrics"><span class="metric"><b id="metricFps">--</b> fps</span><span class="metric"><b id="metricLatency">--</b> ms infer</span><span class="metric"><b id="metricTracks">--</b> tracks</span><span class="metric"><b id="metricEvents">0</b> events</span></div></header>
<div id="alarm"><span id="alarmText">INTRUSION DETECTED</span><button id="ack">Acknowledge</button></div>
<div class="layout">
<section class="card live-card"><video id="video" controls preload="metadata">
<source src="/source" type="video/mp4"></video>
<div class="player-tools"><button id="showAnnotated" type="button">Show full annotated video</button><a id="download" href="/source" download>Download source</a><a id="downloadAnnotated" href="/annotated" download>Download overlay</a></div>
<p id="playerHelp" class="muted">Use the video timeline to scroll through the video. After detection finishes, this player switches automatically to the full video with overlays. Select an event to open its saved evidence clip.</p>
</section>
<aside class="card">
<div class="side-card"><h2>Lane polygon</h2><p><button id="editLane" type="button">Edit lane</button> <button id="testAlarm" type="button">Test alarm</button></p><p class="muted">Click <b>Edit lane</b> to load a frame and redraw the closed intrusion polygon.</p></div>
<div class="side-card"><h2>Active closure</h2><div id="activeClosure"><p><strong id="closureState">inactive</strong> <span class="muted">No linked closure</span></p><p class="muted">Select a Caltrans closure below to link it to this camera.</p></div><div class="lcs-fields"><input id="operatorName" placeholder="Operator name"><input id="crewName" placeholder="Crew / unit"><input id="operatorNotes" placeholder="Operator notes"></div><p><button id="linkClosure" type="button">Link closure</button> <button id="unlinkClosure" type="button">Unlink</button></p><p class="muted">Local operator metadata; not an official Caltrans record.</p></div>
<div class="side-card"><h2>Caltrans closures</h2><select id="lcsDistrict"><option value="4">District 4 · Bay Area</option><option value="1">District 1 · Eureka</option><option value="2">District 2 · Redding</option><option value="3">District 3 · Marysville / Sacramento</option><option value="5">District 5 · San Luis Obispo</option><option value="6">District 6 · Fresno</option><option value="7">District 7 · Los Angeles</option><option value="8">District 8 · San Bernardino</option><option value="9">District 9 · Bishop</option><option value="10">District 10 · Stockton</option><option value="11">District 11 · San Diego</option><option value="12">District 12 · Orange County</option></select><input id="lcsSearch" placeholder="Search route, county, or location"><button id="loadClosures" type="button">Load closures</button><div id="lcsResults" class="lcs-results muted">Load a district to choose the active closure.</div></div>
<h2>Events</h2><div id="events">Loading...</div></aside>
</div>
<section class="card control-drawer" id="controlCard"><h2>Detection control</h2>
<form id="control">
<label class="control-label">Video source
<input id="source" name="source" autocomplete="off" value="__SOURCE__" placeholder="data\\video\\workzone_01.mp4" required></label>
<button id="loadVideo" type="button">Load video</button>
<button class="primary" id="savePolygon" type="button">Save polygon and start</button>
<button class="danger" id="stop" type="button">Stop detection</button>
<span id="status">Stopped</span>
</form><p class="muted control-help">Defaults: <code>yolo11n.pt</code>, <code>lane.json</code>, and <code>evidence\\</code> in the project folder.</p>
<pre id="log" class="muted"></pre></section>
<section class="card" id="polygonCard"><h2>Draw lane polygon</h2>
<p class="muted">Load the source video, then click directly on the displayed frame. Coordinates use the video's native pixel space.</p>
<canvas id="polygonCanvas"></canvas><br>
<button id="loadFrame">Load current video frame</button>
<button id="clearPolygon">Clear points</button>
<span id="polygonStatus" class="muted"></span></section>
<script>
const video=document.getElementById("video"), list=document.getElementById("events"), download=document.getElementById("download"), downloadAnnotated=document.getElementById("downloadAnnotated"), sourceInput=document.getElementById("source"), playerHelp=document.getElementById("playerHelp");
const alarm=document.getElementById("alarm"), alarmText=document.getElementById("alarmText"), canvas=document.getElementById("polygonCanvas"), ctx=canvas.getContext("2d");
let polygon=[], latestEvent=0, eventsInitialized=false, frameImage=null, showingAnnotated=false;
if(sourceInput.value==="__PROJECT_ROOT__"||sourceInput.value==="")sourceInput.value="__DEFAULT_SOURCE__";
function time(s){return `${Math.floor(s/60)}:${String(Math.floor(s%60)).padStart(2,"0")}`}
async function loadEvents(){const events=await fetch("/events.json?"+Date.now()).then(r=>r.json());document.getElementById("metricEvents").textContent=events.length;
 list.innerHTML="";
 if(!events.length){list.textContent="No events recorded yet.";return}
 if(!eventsInitialized){latestEvent=Math.max(...events.map(e=>e.id||0));eventsInitialized=true}
 events.slice().reverse().forEach(e=>{
   const b=document.createElement("button"); b.className="event";
   b.innerHTML=`<strong>${e.type.replaceAll("_"," ")} · Track ${e.track_id}</strong><span class="muted">${time(e.timestamp)} · ${e.clip}</span>`;
   b.onclick=()=>{const url="/evidence/"+encodeURIComponent(e.clip);video.src=url;download.href=url;download.textContent="Download clip";video.load();video.play()};
   list.appendChild(b);
   if(e.id>latestEvent){latestEvent=e.id;alarmText.textContent=`${e.type.replaceAll("_"," ")} · Track ${e.track_id}`;alarm.classList.add("active")}
 });
}
async function status(){const s=await fetch("/api/status").then(r=>r.json());document.getElementById("status").textContent=s.running?"Running":"Stopped";document.getElementById("topStatus").textContent=s.running?"LIVE":"READY";document.getElementById("topStatus").style.background=s.running?"#277c4e":"#4b5563";document.getElementById("log").textContent=s.log||s.error||"";if(s.annotated_ready&&!s.running&&!showingAnnotated){showingAnnotated=true;video.src="/annotated?"+Date.now();video.load();playerHelp.textContent="Full processed video with YOLO boxes, tracking IDs, lane polygon, and intrusion overlays.";loadEvents()}else loadEvents();loadSite()}
async function loadSite(){const r=await fetch("/api/site");const data=await r.json();const box=document.getElementById("activeClosure");if(!data.ok||!data.site||!data.site.lcs){box.innerHTML='<p><strong id="closureState">inactive</strong> <span class="muted">No linked closure</span></p><p class="muted">Select a Caltrans closure below to link it to this camera.</p>';return}const c=data.site.lcs;document.getElementById("operatorName").value=data.site.operator||"";document.getElementById("crewName").value=data.site.crew||"";document.getElementById("operatorNotes").value=data.site.notes||"";box.innerHTML=`<p><strong id="closureState" style="color:#45c878">active</strong> <strong>SR-${c.route||"?"}</strong> · ${c.direction||"Both"}</p><div class="closure-detail">${c.county||""} · ${c.beginPostmile||""} to ${c.endPostmile||""}<br>${c.typeOfClosure||""} · ${c.typeOfWork||""}<br><span class="muted">${c.start||""} → ${c.end||""}</span><br><span class="muted">Operator: ${data.site.operator||"Not assigned"} · Crew: ${data.site.crew||"Not assigned"}</span></div>`}
async function loadVideoSource(){const source=document.querySelector("[name=source]").value.trim();const r=await fetch("/api/source",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({source})});const result=await r.json();document.getElementById("polygonStatus").textContent=result.message;if(result.ok){showingAnnotated=false;video.src="/source?"+Date.now();download.href="/source";download.textContent="Download source";video.load();video.addEventListener("loadedmetadata",loadFrame,{once:true})}}
document.getElementById("loadVideo").onclick=loadVideoSource;
document.getElementById("stop").onclick=async()=>{await fetch("/api/stop",{method:"POST"});status()};
document.getElementById("showAnnotated").onclick=()=>{showingAnnotated=true;video.src="/annotated?"+Date.now();video.load();playerHelp.textContent="Full processed video with YOLO boxes, tracking IDs, lane polygon, and intrusion overlays.";};
document.getElementById("ack").onclick=()=>alarm.classList.remove("active");
document.getElementById("editLane").onclick=()=>document.getElementById("loadFrame").click();
document.getElementById("testAlarm").onclick=()=>{alarmText.textContent="TEST ALARM";alarm.classList.add("active")};
document.getElementById("loadClosures").onclick=async()=>{const box=document.getElementById("lcsResults");box.textContent="Loading Caltrans closures...";const q=encodeURIComponent(document.getElementById("lcsSearch").value);const district=document.getElementById("lcsDistrict").value;const r=await fetch(`/api/lcs?district=${district}&q=${q}`);const data=await r.json();if(!data.ok){box.textContent=data.error;return}box.innerHTML="";if(!data.closures.length){box.textContent="No matching closures.";return}data.closures.forEach(c=>{const b=document.createElement("button");b.className="lcs-result";b.innerHTML=`<b>${c.status} · SR-${c.route||"?"}</b><br><span class="muted">${c.county||""} · ${c.beginLocation||c.nearbyPlace||""} · ${c.typeOfClosure||""}</span>`;b.onclick=async()=>{const saved=await fetch("/api/site",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({lcs:c})}).then(r=>r.json());box.textContent=saved.ok?"Closure selected. Add operator details, then click Link closure.":saved.error||"Could not save closure";loadSite()};box.appendChild(b)})};
document.getElementById("linkClosure").onclick=async()=>{const current=await fetch("/api/site").then(r=>r.json());if(!current.site||!current.site.lcs){document.getElementById("activeClosure").textContent="Select a Caltrans closure first.";return}const payload={...current.site,operator:document.getElementById("operatorName").value.trim(),crew:document.getElementById("crewName").value.trim(),notes:document.getElementById("operatorNotes").value.trim()};await fetch("/api/site",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(payload)});loadSite()};
document.getElementById("unlinkClosure").onclick=async()=>{await fetch("/api/site",{method:"DELETE"});loadSite()};
function redraw(){if(!frameImage)return;ctx.clearRect(0,0,canvas.width,canvas.height);ctx.drawImage(frameImage,0,0);if(polygon.length){ctx.strokeStyle="#00ff00";ctx.lineWidth=Math.max(2,canvas.width/500);ctx.beginPath();polygon.forEach((p,i)=>i?ctx.lineTo(p.x,p.y):ctx.moveTo(p.x,p.y));if(polygon.length>2)ctx.closePath();ctx.stroke()}polygon.forEach((p,i)=>{ctx.fillStyle="#00ff00";ctx.beginPath();ctx.arc(p.x,p.y,Math.max(4,canvas.width/250),0,Math.PI*2);ctx.fill();ctx.fillStyle="#111827";ctx.font=`${Math.max(14,canvas.width/80)}px sans-serif`;ctx.fillText(String(i+1),p.x+8,p.y-8)})}
function loadFrame(){if(!video.videoWidth||!video.videoHeight){document.getElementById("polygonStatus").textContent="The video has not provided a frame yet.";return}canvas.width=video.videoWidth;canvas.height=video.videoHeight;frameImage=document.createElement("canvas");frameImage.width=canvas.width;frameImage.height=canvas.height;frameImage.getContext("2d").drawImage(video,0,0,canvas.width,canvas.height);polygon=[];redraw();document.getElementById("polygonStatus").textContent=`Frame size: ${canvas.width} × ${canvas.height}. Click points on this frame.`}
document.getElementById("loadFrame").onclick=()=>{if(!video.videoWidth){document.getElementById("polygonStatus").textContent="Click Load video first.";return}loadFrame()};
canvas.onclick=e=>{if(!frameImage)return;const r=canvas.getBoundingClientRect();const x=Math.max(0,Math.min(canvas.width,Math.round((e.clientX-r.left)*canvas.width/r.width)));const y=Math.max(0,Math.min(canvas.height,Math.round((e.clientY-r.top)*canvas.height/r.height)));polygon.push({x,y});redraw();document.getElementById("polygonStatus").textContent=`Point ${polygon.length}: (${x}, ${y})`};
document.getElementById("clearPolygon").onclick=()=>{polygon=[];redraw()};
document.getElementById("savePolygon").onclick=async()=>{if(polygon.length<3){document.getElementById("polygonStatus").textContent="Choose at least three points.";return}const source=document.querySelector("[name=source]").value;const r=await fetch("/api/lane",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({points:polygon.map(p=>[p.x,p.y])})});const saved=await r.json();if(saved.ok){const started=await fetch("/api/start",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({})}).then(r=>r.json());document.getElementById("polygonStatus").textContent=started.error||"Polygon saved; detection started.";showingAnnotated=false;video.src="/source?"+Date.now();download.href="/source";download.textContent="Download source";video.load();playerHelp.textContent="Detection is running. The player will switch to the full overlaid video when processing finishes.";status()}else document.getElementById("polygonStatus").textContent=saved.message};
status();setInterval(status,2000);
</script>
</main></body></html>"""


class DashboardHandler(SimpleHTTPRequestHandler):
    def __init__(self, request: object, client_address: object, server: "DashboardServer") -> None:
        self.dashboard = server
        super().__init__(request, client_address, server)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            html = HTML.replace("__SOURCE__", str(self.dashboard.video or "")).replace(
                "__MODEL__", str(self.dashboard.model or "")
            ).replace("__LANE__", str(self.dashboard.lane or "")).replace(
                "__EVIDENCE__", str(self.dashboard.evidence_dir)
            ).replace("__PROJECT_ROOT__", str(PROJECT_ROOT)).replace(
                "__DEFAULT_SOURCE__", str(DEFAULT_VIDEO)
            )
            self._send_bytes("text/html; charset=utf-8", html.encode())
            return
        if path == "/api/status":
            self._send_json(self.dashboard.detector.status())
            return
        if path == "/api/lcs":
            try:
                from .lcs import fetch_district, search
                query = parse_qs(urlparse(self.path).query)
                district = int(query.get("district", ["4"])[0])
                closures = search(fetch_district(district), query.get("q", [""])[0])
                self._send_json({"ok": True, "closures": closures})
            except (OSError, ValueError, TimeoutError, json.JSONDecodeError, urllib.error.URLError) as exc:
                self._send_json({"ok": False, "error": f"Caltrans feed unavailable: {exc}"})
            return
        if path == "/api/site":
            from .lcs import load_site
            self._send_json({"ok": True, "site": load_site(self.dashboard.lane)})
            return
        if path == "/events.json":
            manifest = self.dashboard.evidence_dir / "events.json"
            data = manifest.read_bytes() if manifest.exists() else b"[]"
            self._send_bytes("application/json", data)
            return
        if path == "/source":
            if self.dashboard.video is None:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            self._serve_file(self.dashboard.video)
            return
        if path == "/annotated":
            self._serve_file(self.dashboard.evidence_dir / "annotated_full.mp4")
            return
        if path.startswith("/evidence/"):
            requested = Path(unquote(path.removeprefix("/evidence/")))
            if requested.name != requested.as_posix():
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            self._serve_file(self.dashboard.evidence_dir / requested.name)
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        if path == "/api/source":
            try:
                source = str(json.loads(body.decode("utf-8"))["source"]).strip()
                if not source:
                    raise ValueError("video source is required")
                if not source.isdecimal():
                    source_path = Path(source).expanduser()
                    if not source_path.is_absolute():
                        source_path = PROJECT_ROOT / source_path
                    if not source_path.is_file():
                        raise ValueError(f"video does not exist: {source_path}")
                    source = str(source_path.resolve())
                self.dashboard.source_arg = source
                self.dashboard.video = None if source.isdecimal() else Path(source).expanduser().resolve()
                self._send_json({"ok": True, "message": "Video loaded"})
            except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as exc:
                self._send_json({"ok": False, "message": f"Could not load video: {exc}"})
            return
        if path == "/api/lane":
            try:
                values = json.loads(body.decode("utf-8"))
                lane_path = self.dashboard.lane
                points = values["points"]
                if len(points) < 3:
                    raise ValueError("at least three points are required")
                normalized = [[float(point[0]), float(point[1])] for point in points]
                lane_path.parent.mkdir(parents=True, exist_ok=True)
                lane_path.write_text(
                    json.dumps({"points": normalized}, indent=2) + "\n",
                    encoding="utf-8",
                )
                self._send_json({"ok": True, "message": f"Saved {lane_path}"})
            except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as exc:
                self._send_json({"ok": False, "message": f"Could not save polygon: {exc}"})
            return
        if path == "/api/site":
            try:
                from .lcs import save_site
                values = json.loads(body.decode("utf-8"))
                if not isinstance(values.get("lcs"), dict):
                    raise ValueError("a selected LCS closure is required")
                site = save_site(self.dashboard.lane, values)
                self._send_json({"ok": True, "site": site})
            except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as exc:
                self._send_json({"ok": False, "error": f"Could not save site metadata: {exc}"})
            return
        values = {
            key: item[0]
            for key, item in parse_qs(body.decode()).items()
            if item
        }
        if path == "/api/start":
            self._send_json(self.dashboard.detector.start(values))
            return
        if path == "/api/stop":
            self.dashboard.detector.stop()
            self._send_json(self.dashboard.detector.status())
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_DELETE(self) -> None:
        if urlparse(self.path).path == "/api/site":
            from .lcs import site_path
            try:
                path = site_path(self.dashboard.lane)
                if path.exists():
                    path.unlink()
                self._send_json({"ok": True})
            except OSError as exc:
                self._send_json({"ok": False, "error": f"Could not unlink closure: {exc}"})
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def _serve_file(self, path: Path) -> None:
        if not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        size = path.stat().st_size
        range_header = self.headers.get("Range")
        if not range_header:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(size))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
            with path.open("rb") as file:
                self._copy(file)
            return
        try:
            start_text, end_text = range_header.removeprefix("bytes=").split("-", 1)
            start = int(start_text or 0)
            end = int(end_text) if end_text else size - 1
            if start < 0 or end < start or end >= size:
                raise ValueError
        except ValueError:
            self.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
            return
        with path.open("rb") as file:
            file.seek(start)
            data = file.read(end - start + 1)
        self.send_response(HTTPStatus.PARTIAL_CONTENT)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()
        self.wfile.write(data)

    def _copy(self, file: BinaryIO) -> None:
        while True:
            chunk = file.read(1024 * 1024)
            if not chunk:
                break
            self.wfile.write(chunk)

    def _send_bytes(self, content_type: str, data: bytes) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_json(self, data: dict[str, object]) -> None:
        import json

        self._send_bytes("application/json; charset=utf-8", json.dumps(data).encode())

    def log_message(self, format: str, *args: object) -> None:
        return


class DashboardServer(ThreadingHTTPServer):
    def __init__(
        self,
        address: tuple[str, int],
        video: Path | None,
        model: Path | None,
        lane: Path | None,
        evidence_dir: Path,
    ) -> None:
        self.video = (video or DEFAULT_VIDEO).resolve()
        if not self.video.is_file():
            self.video = None
        self.source_arg = str(self.video or DEFAULT_VIDEO)
        self.model = (model or PROJECT_ROOT / "yolo11n.pt").resolve()
        self.lane = (lane or PROJECT_ROOT / "lane.json").resolve()
        self.evidence_dir = (evidence_dir if evidence_dir.is_absolute() else PROJECT_ROOT / evidence_dir).resolve()
        self.detector = DetectorProcess(self)
        super().__init__(address, DashboardHandler)


class DetectorProcess:
    def __init__(self, dashboard: DashboardServer) -> None:
        self.dashboard = dashboard
        self.process: subprocess.Popen[str] | None = None
        self.log_lines: list[str] = []
        self.error: str | None = None
        self._lock = threading.Lock()

    def start(self, values: dict[str, str]) -> dict[str, object]:
        with self._lock:
            if self.process and self.process.poll() is None:
                return self.status()
            source = values.get("source", "").strip() or self.dashboard.source_arg
            if not source:
                self.error = "Load a video before starting detection."
                return self.status()
            if not source.isdecimal():
                source_path = Path(source).expanduser()
                if not source_path.is_absolute():
                    source_path = PROJECT_ROOT / source_path
                source_path = source_path.resolve()
                if not source_path.is_file():
                    self.error = f"Video source is not a file: {source_path}"
                    return self.status()
                source = str(source_path)
            model = self.dashboard.model
            lane = self.dashboard.lane
            evidence = self.dashboard.evidence_dir
            if not source or not model.is_file() or not lane.is_file():
                missing = [
                    str(path)
                    for path in (model, lane)
                    if not path.is_file()
                ]
                self.error = (
                    "Source is required and model/lane files must exist on the dashboard host. "
                    f"Missing: {', '.join(missing) or 'video source'}"
                )
                return self.status()
            evidence.mkdir(parents=True, exist_ok=True)
            annotated_video = evidence / "annotated_full.mp4"
            try:
                if annotated_video.exists():
                    annotated_video.unlink()
            except OSError as exc:
                self.error = f"Cannot replace annotated video {annotated_video}: {exc}"
                return self.status()
            self.dashboard.video = Path(source).expanduser().resolve() if not source.isdecimal() else None
            self.dashboard.source_arg = source
            self.dashboard.model = model.resolve()
            self.dashboard.lane = lane.resolve()
            self.dashboard.evidence_dir = evidence.resolve()
            command = [
                sys.executable,
                "-m",
                "yolo.app",
                "--source",
                source,
                "--model",
                str(model),
                "--lane",
                str(lane),
                "--evidence-dir",
                str(evidence),
                "--annotated-video",
                str(annotated_video),
                "--stopped-seconds",
                values.get("stopped_seconds", "3"),
            ]
            self.log_lines = []
            self.error = None
            self.process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=str(Path.cwd()),
            )
            threading.Thread(target=self._read_output, daemon=True).start()
            return self.status()

    def _read_output(self) -> None:
        if not self.process or not self.process.stdout:
            return
        for line in self.process.stdout:
            with self._lock:
                self.log_lines.append(line.rstrip())
                self.log_lines = self.log_lines[-30:]

    def stop(self) -> None:
        with self._lock:
            if self.process and self.process.poll() is None:
                self.process.terminate()

    def status(self) -> dict[str, object]:
        running = self.process is not None and self.process.poll() is None
        annotated_video = self.dashboard.evidence_dir / "annotated_full.mp4"
        return {
            "running": running,
            "annotated_ready": annotated_video.is_file() and annotated_video.stat().st_size > 0,
            "log": "\n".join(self.log_lines),
            "error": self.error,
            "source": self.dashboard.source_arg,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--lane", type=Path)
    parser.add_argument("--evidence-dir", type=Path, default=Path("evidence"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if args.video and not args.video.is_file():
        parser.error(f"video does not exist: {args.video}")
    if args.model and not args.model.is_file():
        parser.error(f"model does not exist: {args.model}")
    if args.lane and not args.lane.is_file():
        parser.error(f"lane file does not exist: {args.lane}")
    args.evidence_dir.mkdir(parents=True, exist_ok=True)
    try:
        server = DashboardServer(
            (args.host, args.port),
            args.video,
            args.model,
            args.lane,
            args.evidence_dir,
        )
    except OSError as exc:
        if getattr(exc, "winerror", None) in (10013, 10048) or exc.errno in (13, 98):
            parser.error(
                f"could not bind {args.host}:{args.port}; the port is busy or reserved. "
                f"Try --port {args.port + 1}."
            )
        raise
    print(f"Dashboard: http://{args.host}:{args.port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
