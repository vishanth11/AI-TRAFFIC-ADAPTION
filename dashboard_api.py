"""FastAPI dashboard API — REST + WebSocket.

Start with:
    uvicorn dashboard_api:app --host 0.0.0.0 --port 8000 --reload

Or run alongside the traffic controller:
    python dashboard_api.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
for p in (str(SRC), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from dashboard_state import state

app = FastAPI(title="AI Traffic Adaptation Dashboard API", version="1.0.0")


# -----------------------------------------------------------------------
# Seed state from the most recent JSONL run log (so dashboard is never empty)
# -----------------------------------------------------------------------

def _seed_from_latest_jsonl():
    """Load the most recent runtime.jsonl into DashboardState."""
    runs_dir = ROOT / "results" / "final_runs"
    if not runs_dir.exists():
        return

    # Find newest runtime.jsonl
    jsonl_files = sorted(runs_dir.glob("*/runtime.jsonl"), key=lambda p: p.stat().st_mtime)
    if not jsonl_files:
        return
    latest = jsonl_files[-1]

    decisions = []
    junctions: dict = {}
    logs = []
    max_time = 0.0
    mode = "normal"
    has_emergency = False

    with open(latest, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            event = record.get("event")
            logs.append(record)

            if event == "system_start":
                mode = record.get("mode", "normal")

            elif event == "decision":
                decisions.append(record)
                sim_time = float(record.get("time", 0))
                if sim_time > max_time:
                    max_time = sim_time

                jid = record.get("junction")
                if jid:
                    safety = record.get("safety", {})
                    executed = record.get("executed", {})
                    if record.get("emergency_active"):
                        has_emergency = True
                    junctions[jid] = {
                        "junction_id": jid,
                        "current_phase_index": executed.get("phase"),
                        "current_phase_state": executed.get("state"),
                        "remaining_time": None,
                        "last_decision": record.get("selected"),
                        "decision_method": record.get("method"),
                        "safety_approved": safety.get("approved"),
                        "safety_checks": safety.get("checks"),
                        "emergency_active": record.get("emergency_active", False),
                        "emergency_vehicle_id": record.get("emergency_vehicle_id"),
                        "simulation_time": sim_time,
                    }

    # Populate state (most recent first)
    for record in reversed(decisions):
        state.add_decision(record)
    for jid, jdata in junctions.items():
        state.update_junction(jid, jdata)
    for log in reversed(logs):
        state.logs.appendleft(log)

    # Derive summary stats from decisions
    emergency_count = sum(1 for d in decisions if d.get("emergency_active"))
    state.update_system(
        status="stopped",
        sim_time=max_time,
        mode=mode,
        prediction_available=True,
    )
    state.emergency_vehicles = (
        [{"vehicle_id": "ambulance (historical)", "source": "log"}]
        if has_emergency else []
    )
    state.started_at = logs[0].get("timestamp") if logs else None
    print(f"[dashboard] Seeded from {latest.name}: {len(decisions)} decisions, "
          f"{len(junctions)} junctions, sim_time={max_time}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------
# WebSocket connection manager
# -----------------------------------------------------------------------

class ConnectionManager:
    def __init__(self):
        self._clients: list[WebSocket] = []
        self._lock = asyncio.Lock()

    async def connect(self, ws: WebSocket):
        await ws.accept()
        async with self._lock:
            self._clients.append(ws)

    async def disconnect(self, ws: WebSocket):
        async with self._lock:
            if ws in self._clients:
                self._clients.remove(ws)

    async def broadcast(self, data: dict):
        message = json.dumps(data)
        async with self._lock:
            dead = []
            for ws in self._clients:
                try:
                    await ws.send_text(message)
                except Exception:
                    dead.append(ws)
            for ws in dead:
                self._clients.remove(ws)


manager = ConnectionManager()


# -----------------------------------------------------------------------
# Background task — push live state to all WebSocket clients
# -----------------------------------------------------------------------

async def _broadcast_loop():
    while True:
        await asyncio.sleep(1.0)
        if manager._clients:
            payload = {
                "type": "live_update",
                "status": state.snapshot(),
                "junctions": state.junctions_snapshot(),
                "emergencies": state.emergencies_snapshot(),
                "safety": state.safety_snapshot(),
                "latest_decision": state.decisions_snapshot()[:1],
                "logs": state.logs_snapshot()[:20],
            }
            await manager.broadcast(payload)


@app.on_event("startup")
async def startup():
    _seed_from_latest_jsonl()
    asyncio.create_task(_broadcast_loop())


# -----------------------------------------------------------------------
# REST endpoints
# -----------------------------------------------------------------------

@app.get("/api/status")
def get_status():
    return state.snapshot()


@app.get("/api/junctions")
def get_junctions():
    return state.junctions_snapshot()


@app.get("/api/junctions/{junction_id}")
def get_junction(junction_id: str):
    data = state.junction_snapshot(junction_id)
    if data is None:
        return JSONResponse(status_code=404, content={"detail": "Junction not found"})
    return data


@app.get("/api/decisions")
def get_decisions():
    return state.decisions_snapshot()


@app.get("/api/safety")
def get_safety():
    return state.safety_snapshot()


@app.get("/api/emergencies")
def get_emergencies():
    return state.emergencies_snapshot()


@app.get("/api/metrics")
def get_metrics():
    return state.metrics_snapshot()


@app.get("/api/logs")
def get_logs():
    return state.logs_snapshot()


@app.get("/api/comparison")
def get_comparison():
    return state.comparison_snapshot()


# -----------------------------------------------------------------------
# WebSocket endpoint
# -----------------------------------------------------------------------

@app.websocket("/ws/live")
async def websocket_live(ws: WebSocket):
    await manager.connect(ws)
    # Send initial full state immediately on connect
    await ws.send_text(json.dumps({
        "type": "initial_state",
        "status": state.snapshot(),
        "junctions": state.junctions_snapshot(),
        "decisions": state.decisions_snapshot(),
        "emergencies": state.emergencies_snapshot(),
        "safety": state.safety_snapshot(),
        "metrics": state.metrics_snapshot(),
        "logs": state.logs_snapshot()[:50],
    }))
    try:
        while True:
            # Keep connection alive; client can send pings
            await ws.receive_text()
    except WebSocketDisconnect:
        await manager.disconnect(ws)
    except Exception:
        await manager.disconnect(ws)


# -----------------------------------------------------------------------
# Entry point for standalone run
# -----------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("dashboard_api:app", host="0.0.0.0", port=8000, reload=False)
