import fastapi
import uvicorn
from pydantic import BaseModel
import csv
import os
import numpy as np
import subprocess
import sys
import shutil
import time
import threading
import tempfile
from fastapi.responses import HTMLResponse, FileResponse
from fastapi import (
    HTTPException,
    Header,
    Depends,
    Request,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.security import HTTPBasic, HTTPBasicCredentials
import secrets
import json
import base64
import queue
import asyncio
import jwt
from jwt import ExpiredSignatureError, InvalidTokenError
import logging
from logging.handlers import RotatingFileHandler
import traceback
from smart_scheduler import SmartTaskScheduler

app = fastapi.FastAPI()

LOG_DIR = os.environ.get("MASTER_LOG_DIR", "logs")
try:
    os.makedirs(LOG_DIR, exist_ok=True)
except Exception:
    pass
logger = logging.getLogger("bersim.master")
logger.setLevel(logging.INFO)
log_path = os.path.join(LOG_DIR, "master.log")
handler = RotatingFileHandler(log_path, maxBytes=5 * 1024 * 1024, backupCount=5)
formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
handler.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(handler)
try:
    logging.getLogger().addHandler(handler)
except Exception:
    pass


def _handle_uncaught(exc_type, exc_value, exc_tb):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_tb)
        return
    try:
        logger.exception("Uncaught exception", exc_info=(exc_type, exc_value, exc_tb))
    except Exception:
        try:
            print("Uncaught exception:", exc_type, exc_value)
        except Exception:
            pass
    try:
        save_state()
    except Exception:
        try:
            logger.exception("Failed to save state during uncaught exception")
        except Exception:
            pass


import sys as _sys_for_hook

_sys_for_hook.excepthook = _handle_uncaught


PYTHON_EXEC = sys.executable


API_KEY = os.environ.get("CLUSTER_API_KEY", "supersecretkey")


def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")


BASIC_USER = os.environ.get("MASTER_BASIC_USER")
BASIC_PASS = os.environ.get("MASTER_BASIC_PASS")
security = HTTPBasic()


def verify_basic_auth(credentials: HTTPBasicCredentials = Depends(security)):

    if not BASIC_USER or not BASIC_PASS:
        return True
    correct_user = secrets.compare_digest(credentials.username, BASIC_USER)
    correct_pass = secrets.compare_digest(credentials.password, BASIC_PASS)
    if not (correct_user and correct_pass):
        raise HTTPException(
            status_code=401,
            detail="Unauthorized",
            headers={"WWW-Authenticate": "Basic"},
        )
    return True


EBNO_RANGE = [round(x, 1) for x in np.arange(0.0, 12.1, 0.1)]
COARSE_EBNO_RANGE = [round(x, 1) for x in np.arange(0.0, 12.1, 1.0)]
SYSTEMS = ["Uncoded"]
for r in ["1/2", "2/3", "3/4", "5/6", "7/8"]:
    SYSTEMS.append(f"Conv Rate {r }")
for d in [1, 2, 4, 5, 8, 16]:
    SYSTEMS.append(f"RS Depth {d }")
for r in ["1/2", "2/3", "3/4", "5/6", "7/8"]:
    for d in [1, 2, 4, 5, 8, 16]:
        SYSTEMS.append(f"Concat {r } D{d }")

# Initialize early to prevent NameError in save_state during exception handling
results = {s: {} for s in SYSTEMS}
fer_results = {s: {} for s in SYSTEMS}

os.makedirs("data", exist_ok=True)


MAX_FRAMES = int(os.environ.get("SIM_MAX_FRAMES", "10000000"))


if os.path.exists("ber_simulation_results.csv") and not os.path.exists(
    "data/ber_simulation_results.csv"
):
    shutil.move("ber_simulation_results.csv", "data/ber_simulation_results.csv")
    print("Migrated existing ber_simulation_results.csv to data/ folder.")
if os.path.exists("high_res_ber_curve.png") and not os.path.exists(
    "data/high_res_ber_curve.png"
):
    shutil.move("high_res_ber_curve.png", "data/high_res_ber_curve.png")
    print("Migrated existing high_res_ber_curve.png to data/ folder.")

plot_version = 1

data_lock = threading.Lock()


plot_lock = threading.Lock()
plot_request_pending = False
is_plotting = False

task_cond = threading.Condition()

# Initialize early to prevent NameError in save_state during exception handling
pending_tasks = []
halted_systems = set()  # Systems where no more Eb/No points should run


def _task_key(task: dict):
    try:
        return (task.get("system"), round(float(task.get("ebno")), 1))
    except Exception:
        return None


def _dedupe_task_list(tasks: list):
    seen = set()
    deduped = []
    for task in tasks:
        key = _task_key(task)
        if key is None or key in seen:
            continue
        seen.add(key)
        normalized = dict(task)
        normalized["system"] = key[0]
        normalized["ebno"] = key[1]
        deduped.append(normalized)
    return deduped


STATE_FILE = os.path.join("data", "master_state.json")
STATE_LOADED = False
MASTER_JWT_SECRET = os.environ.get("MASTER_JWT_SECRET", API_KEY)
MASTER_UI_TOKEN_TTL = int(os.environ.get("MASTER_UI_TOKEN_TTL", "300"))


ws_connections = set()
ws_lock = threading.Lock()
workers_lock = threading.Lock()
ASYNC_LOOP = None
scheduler = SmartTaskScheduler()

# Refinement task tracking to prevent rapid regeneration
refinement_assignment_history = {}  # {(system, ebno): timestamp}
refinement_cooldown_seconds = (
    60  # Don't regenerate same refinement task within 1 minute
)


def create_ui_token(username: str = "ui"):
    now = int(time.time())
    payload = {
        "sub": username,
        "iat": now,
        "exp": now + MASTER_UI_TOKEN_TTL,
        "type": "ui",
    }
    token = jwt.encode(payload, MASTER_JWT_SECRET, algorithm="HS256")
    if isinstance(token, bytes):
        token = token.decode()
    return token


def verify_api_or_ui_token(
    x_api_key: str = Header(None), authorization: str = Header(None)
):
    if x_api_key and x_api_key == API_KEY:
        return True
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization.split(" ", 1)[1]
        try:
            payload = jwt.decode(token, MASTER_JWT_SECRET, algorithms=["HS256"])
            if payload.get("type") == "ui":
                return True
        except ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="UI token expired")
        except InvalidTokenError:
            raise HTTPException(status_code=401, detail="Invalid UI token")
        except Exception:
            raise HTTPException(status_code=401, detail="Invalid UI token")
    raise HTTPException(status_code=401, detail="Unauthorized")


def broadcast_event(ev: dict):
    try:
        if ASYNC_LOOP is not None:
            asyncio.run_coroutine_threadsafe(_broadcast_ws(ev), ASYNC_LOOP)
    except Exception:
        pass


async def _broadcast_ws(ev: dict):
    data = json.dumps(ev)
    to_remove = []
    with ws_lock:
        conns = list(ws_connections)
    for ws in conns:
        try:
            await ws.send_text(data)
        except Exception:
            to_remove.append(ws)
    if to_remove:
        with ws_lock:
            for ws in to_remove:
                try:
                    ws_connections.remove(ws)
                except Exception:
                    pass


def _plot_worker():
    global is_plotting, plot_request_pending, plot_version
    while True:
        with plot_lock:
            if not plot_request_pending:
                is_plotting = False
                return
            plot_request_pending = False
        try:
            print("[PLOT] Generating plots in background...")
            proc_res = subprocess.run(
                [PYTHON_EXEC, "plot_results.py"],
                capture_output=True,
                text=True,
                timeout=300,
            )
            if proc_res.returncode != 0:
                print(f"[PLOT ERROR] Plotting failed!\n{proc_res .stderr }")
            else:
                plot_version += 1
                print(f"[PLOT] Plot updated, version {plot_version }")

                try:
                    broadcast_event({"type": "plot", "plot_version": plot_version})
                except Exception:
                    pass
        except subprocess.TimeoutExpired:
            print("[PLOT ERROR] Plotting process timed out after 300 seconds")
        except Exception as e:
            print(f"[PLOT ERROR] {e }")


def schedule_plot():
    global plot_request_pending, is_plotting
    with plot_lock:
        plot_request_pending = True
        if is_plotting:
            return
        is_plotting = True
        t = threading.Thread(target=_plot_worker, daemon=True)
        t.start()


@app.on_event("startup")
async def startup_event():
    global ASYNC_LOOP
    try:
        ASYNC_LOOP = asyncio.get_event_loop()
    except Exception:
        ASYNC_LOOP = None


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):

    token = websocket.query_params.get("token")
    if not token:
        await websocket.close(code=1008)
        return
    try:
        payload = jwt.decode(token, MASTER_JWT_SECRET, algorithms=["HS256"])
        if payload.get("type") != "ui":
            await websocket.close(code=1008)
            return
    except ExpiredSignatureError:
        await websocket.close(code=1008)
        return
    except InvalidTokenError:
        await websocket.close(code=1008)
        return
    except Exception:
        await websocket.close(code=1008)
        return

    await websocket.accept()
    with ws_lock:
        ws_connections.add(websocket)
    try:
        while True:
            try:
                msg = await asyncio.wait_for(websocket.receive_text(), timeout=60)
            except asyncio.TimeoutError:
                try:
                    await websocket.send_text(json.dumps({"type": "ping"}))
                except Exception:
                    break
            except WebSocketDisconnect:
                break
            except Exception:
                await asyncio.sleep(0.1)
    finally:
        with ws_lock:
            try:
                ws_connections.remove(websocket)
            except Exception:
                pass


def save_state():
    try:
        os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
        with data_lock:
            state = {
                "pending_tasks": pending_tasks,
                "results": {
                    s: {str(k): v for k, v in results[s].items()} for s in results
                },
                "fer_results": {
                    s: {str(k): v for k, v in fer_results[s].items()}
                    for s in fer_results
                },
                "max_frames": MAX_FRAMES,
                "cancelled_tasks": [list(t) for t in cancelled_tasks],
                "halted_systems": list(halted_systems),
                "timestamp": time.time(),
            }

        dirpath = os.path.dirname(STATE_FILE) or "."
        with tempfile.NamedTemporaryFile(
            "w", delete=False, dir=dirpath, prefix="state_", suffix=".json"
        ) as tf:
            json.dump(state, tf)
            tf.flush()
            os.fsync(tf.fileno())
            tmpname = tf.name
        try:
            os.replace(tmpname, STATE_FILE)
        except Exception:

            with open(STATE_FILE, "w") as f:
                json.dump(state, f)
    except Exception as e:
        print(f"[STATE ERROR] Could not save state: {e }")


def load_state():
    if not os.path.exists(STATE_FILE):
        return None
    try:
        with open(STATE_FILE, "r") as f:
            state = json.load(f)
        with data_lock:
            if "results" in state:
                for s, m in state["results"].items():
                    if s in results:
                        try:
                            results[s] = {float(k): v for k, v in m.items()}
                        except Exception:
                            results[s] = {}
            if "fer_results" in state:
                for s, m in state["fer_results"].items():
                    if s in fer_results:
                        try:
                            fer_results[s] = {float(k): v for k, v in m.items()}
                        except Exception:
                            fer_results[s] = {}
            if "max_frames" in state:
                global MAX_FRAMES
                try:
                    MAX_FRAMES = int(state["max_frames"])
                except Exception:
                    pass
        global cancelled_tasks
        if "cancelled_tasks" in state:
            try:
                cancelled_tasks = set(tuple(t) for t in state["cancelled_tasks"])
            except Exception:
                pass
        if "halted_systems" in state:
            try:
                halted_systems.update(state["halted_systems"])
            except Exception:
                pass
        return state
    except Exception as e:
        print(f"[STATE ERROR] Could not load state: {e }")
        return None


if os.path.exists("data/ber_simulation_results.csv"):
    print("Loading existing progress from CSV...")
    with open("data/ber_simulation_results.csv", "r") as f:
        reader = csv.reader(f)
        headers = next(reader)
        for row in reader:
            if not row:
                continue
            ebno = round(float(row[0]), 1)
            for i, val in enumerate(row[1:]):
                if val.strip():
                    sys_name = headers[i + 1]
                    if sys_name in results:
                        try:
                            v = float(val)
                            # Keep exact zeros so frontier knowledge survives CSV reloads.
                            if v >= 0 and np.isfinite(v) and abs(v - 1.0e-07) > 1e-15:
                                results[sys_name][ebno] = v
                        except:
                            pass

if os.path.exists("data/fer_simulation_results.csv"):
    with open("data/fer_simulation_results.csv", "r") as f:
        reader = csv.reader(f)
        headers = next(reader)
        for row in reader:
            if not row:
                continue
            ebno = round(float(row[0]), 1)
            for i, val in enumerate(row[1:]):
                if val.strip():
                    sys_name = headers[i + 1]
                    if sys_name in fer_results:
                        try:
                            v = float(val)
                            # Keep exact zeros so frontier knowledge survives CSV reloads.
                            if v >= 0 and np.isfinite(v) and abs(v - 1.0e-07) > 1e-15:
                                fer_results[sys_name][ebno] = v
                        except:
                            pass


state = load_state()
if state:
    STATE_LOADED = True
    print(f"[STATE] Loaded saved state from {STATE_FILE }")
else:
    STATE_LOADED = False

# Auto-fill any missing higher Eb/No points for systems that already hit 0.0
for system in SYSTEMS:
    first_zero = float("inf")
    if system in results:
        zeros = [
            float(e)
            for e, b in results[system].items()
            if b == 0.0 or fer_results.get(system, {}).get(e, -1) == 0.0
        ]
        if zeros:
            first_zero = min(zeros)
            for e in EBNO_RANGE:
                if e > first_zero:
                    if e not in results[system]:
                        results[system][e] = 0.0
                    if e not in fer_results[system]:
                        fer_results[system][e] = 0.0

if os.path.exists("data/ber_simulation_results.csv") and os.path.exists(
    "plot_results.py"
):
    try:
        if not STATE_LOADED:
            proc_res = subprocess.run(
                [PYTHON_EXEC, "plot_results.py"], capture_output=True, text=True
            )
            if proc_res.returncode != 0:
                print(
                    f"[PLOT ERROR] Could not generate initial plot:\n{proc_res .stderr }"
                )
            else:
                plot_version += 1
        else:

            schedule_plot()
    except Exception as e:
        print(f"[PLOT ERROR] {e }")


if STATE_LOADED and state and isinstance(state.get("pending_tasks"), list):

    for t in state.get("pending_tasks", []):
        try:
            sname = t.get("system")
            eb = float(t.get("ebno"))
            # Skip halted systems and tasks with existing results
            if (
                sname in SYSTEMS
                and sname not in halted_systems
                and eb not in results.get(sname, {})
            ):
                pending_tasks.append({"system": sname, "ebno": eb})
        except Exception:
            pass

pending_tasks[:] = _dedupe_task_list(pending_tasks)

# Remove pending tasks for halted systems and prune useless high Eb/No tasks
new_pending = []
for t in pending_tasks:
    sys_name = t.get("system")
    ebno = float(t.get("ebno", 0))

    if sys_name in halted_systems:
        continue

    first_zero = float("inf")
    if sys_name in results:
        zeros = [float(e) for e, b in results[sys_name].items() if b == 0.0]
        if zeros:
            first_zero = min(zeros)

    if ebno > first_zero:
        continue

    new_pending.append(t)
pending_tasks[:] = new_pending

present = set(_task_key(p) for p in pending_tasks if _task_key(p) is not None)
for system in SYSTEMS:
    if system in halted_systems:
        continue  # Skip adding tasks for halted systems

    first_zero = float("inf")
    if system in results:
        zeros = [float(e) for e, b in results[system].items() if b == 0.0]
        if zeros:
            first_zero = min(zeros)

    for ebno in COARSE_EBNO_RANGE:
        if ebno > first_zero:
            continue
        key = (system, round(ebno, 1))
        if ebno not in results[system] and key not in present:
            pending_tasks.append({"system": system, "ebno": ebno})
            present.add(key)
for system in SYSTEMS:
    if system in halted_systems:
        continue  # Skip adding tasks for halted systems

    first_zero = float("inf")
    if system in results:
        zeros = [float(e) for e, b in results[system].items() if b == 0.0]
        if zeros:
            first_zero = min(zeros)

    for ebno in EBNO_RANGE:
        if ebno > first_zero:
            continue
        key = (system, round(ebno, 1))
        if ebno not in results[system] and key not in present:
            pending_tasks.append({"system": system, "ebno": ebno})
            present.add(key)

total_tasks = len(SYSTEMS) * len(EBNO_RANGE)
active_workers = {}
cancelled_tasks = set()
SERVER_START_TIME = time.time()
INITIAL_COMPLETED = total_tasks - len(pending_tasks)
cluster_paused = False
pause_start_time = 0
total_paused_time = 0

print(f"Master initialized. {len (pending_tasks )} tasks remaining.")


def clean_dead_workers():
    now = time.time()
    dead = []
    with workers_lock:
        for wid, info in active_workers.items():
            if info["task"] is not None and (now - info["last_seen"]) > 600:
                dead.append(wid)

    # Build pending task keys once, with proper lock
    with data_lock:
        pending_keys = {_task_key(t) for t in pending_tasks if _task_key(t) is not None}

    with workers_lock:
        for wid in dead:
            task = active_workers[wid]["task"]
            print(
                f"[WARN] Worker {wid } timed out. Re-queueing {task ['system']} @ {task ['ebno']} dB"
            )
            task_key = _task_key(task)
            if task_key is not None and task_key not in pending_keys:
                with data_lock:
                    sys_name = task["system"]
                    ebno = float(task["ebno"])
                    first_zero = float("inf")
                    if sys_name in results:
                        zeros = [
                            float(e) for e, b in results[sys_name].items() if b == 0.0
                        ]
                        if zeros:
                            first_zero = min(zeros)
                    if ebno <= first_zero and sys_name not in halted_systems:
                        pending_tasks.insert(0, task)
            del active_workers[wid]
    if dead:
        try:
            save_state()
        except Exception:
            pass
        try:
            with task_cond:
                task_cond.notify_all()
        except Exception:
            pass


class ResultPayload(BaseModel):
    worker_id: str
    system: str
    ebno: float
    ber: float
    fer: float = 0.0
    frames: int


class StatusPayload(BaseModel):
    worker_id: str
    system: str = ""
    ebno: float = 0.0
    frames: int
    fe: int
    fps: int = 0


class MaxFramesPayload(BaseModel):
    max_frames: int


def generate_refinement_tasks(limit=5):
    """
    Generate refinement tasks from previously completed work.
    Focuses on points near the first known zero-BER frontier.
    Returns list of refinement tasks or empty list if none available.

    IMPORTANT: Prevents duplicate assignments by checking pending_tasks.
    IMPORTANT: Respects cooldown to avoid rapid re-generation of same tasks.
    """
    refinement_candidates = []
    now = time.time()
    frontier_window = 0.6

    # Get snapshot of active workers safely
    with workers_lock:
        active_set = set(
            (info["task"]["system"], round(info["task"]["ebno"], 1))
            for info in active_workers.values()
            if info["task"]
        )

    with data_lock:
        # Build set of (system, ebno) already assigned or pending
        pending_set = set(
            _task_key(t) for t in pending_tasks if _task_key(t) is not None
        )
        already_assigned = pending_set | active_set

        # Clean up old cooldown entries
        for key in list(refinement_assignment_history.keys()):
            if now - refinement_assignment_history[key] > refinement_cooldown_seconds:
                del refinement_assignment_history[key]

        # Look for points near a known zero-BER frontier (precision refinement work)
        for system in SYSTEMS:
            if system in halted_systems:
                continue

            system_results = results.get(system, {})
            zero_points = sorted(
                float(ebno) for ebno, ber in system_results.items() if ber == 0.0
            )
            if not zero_points:
                continue

            frontier = zero_points[0]
            for ebno, ber in system_results.items():
                key = (system, round(ebno, 1))

                # Skip if already pending or active
                if key in already_assigned:
                    continue

                # Skip if recently assigned as refinement task (cooldown)
                if key in refinement_assignment_history:
                    continue

                # Only refine points that are close to the known frontier.
                if abs(float(ebno) - frontier) > frontier_window:
                    continue

                # Keep only low-confidence frontier-adjacent points.
                if ber <= 0 or not np.isfinite(ber):
                    continue

                if ber <= 0.5:
                    refinement_candidates.append(
                        {
                            "system": system,
                            "ebno": float(ebno),
                            "current_ber": ber,
                            "distance": abs(float(ebno) - frontier),
                        }
                    )

    if not refinement_candidates:
        return []

    # Sort by distance to the frontier first, then system and Eb/No.
    refinement_candidates.sort(key=lambda x: (x["distance"], x["system"], x["ebno"]))

    # Limit to avoid flooding queue - max 3 refinements at a time
    refinement_candidates = refinement_candidates[: min(limit, 3)]

    # Convert to task format (add marker to distinguish from initial tasks)
    # Also record these as recently assigned
    refinement_tasks = []
    for c in refinement_candidates:
        key = (c["system"], round(c["ebno"], 1))
        refinement_assignment_history[key] = now
        refinement_tasks.append(
            {
                "system": c["system"],
                "ebno": c["ebno"],
                "type": "refinement",
                "note": f"Frontier refinement near zero point (current BER: {c['current_ber']:.2e})",
            }
        )

    return refinement_tasks


@app.get("/get_task", dependencies=[Depends(verify_api_key)])
def get_task(worker_id: str):

    clean_dead_workers()
    if cluster_paused:
        if worker_id in active_workers:
            with workers_lock:
                if worker_id in active_workers:
                    active_workers[worker_id]["task"] = None
                    active_workers[worker_id]["last_seen"] = time.time()
        return {"status": "wait"}

    wait_seconds = 30
    with task_cond:
        end_time = time.time() + wait_seconds
        while not pending_tasks and not cluster_paused:
            remaining = end_time - time.time()
            if remaining <= 0:
                break
            task_cond.wait(timeout=remaining)

        if cluster_paused:
            if worker_id in active_workers:
                with workers_lock:
                    if worker_id in active_workers:
                        active_workers[worker_id]["task"] = None
                        active_workers[worker_id]["last_seen"] = time.time()
            return {"status": "wait"}

        if not pending_tasks:

            # Try to generate refinement tasks for precision work
            refinement_tasks = generate_refinement_tasks(limit=5)
            if refinement_tasks:
                with data_lock:
                    pending_tasks.extend(refinement_tasks)
                print(
                    f"[REFINE] Generated {len(refinement_tasks)} refinement tasks for precision work"
                )
                # Notify all waiting workers that new tasks are available
                try:
                    with task_cond:
                        task_cond.notify_all()
                except Exception:
                    pass
                # Fall through to grab one of these new tasks
            else:
                if worker_id in active_workers:
                    with workers_lock:
                        if worker_id in active_workers:
                            active_workers[worker_id]["task"] = None
                            active_workers[worker_id]["last_seen"] = time.time()
                return {"status": "wait"}

        # Skip halted systems - remove ALL tasks for halted systems
        with data_lock:
            # First, do aggressive cleanup for any halted system tasks
            removed_count = 0
            new_pending = []
            for t in pending_tasks:
                if t.get("system") in halted_systems:
                    removed_count += 1
                else:
                    new_pending.append(t)
            if removed_count > 0:
                pending_tasks[:] = new_pending
                print(f"[CLEANUP] Removed {removed_count} tasks from halted systems")

            if not pending_tasks:
                if worker_id in active_workers:
                    with workers_lock:
                        if worker_id in active_workers:
                            active_workers[worker_id]["task"] = None
                            active_workers[worker_id]["last_seen"] = time.time()
                return {"status": "wait"}

            # Use smart scheduler to select best task for this worker
            task = scheduler.select_task_for_worker(
                worker_id, pending_tasks, results, fer_results, active_workers
            )

            if task is None:
                # Smart scheduler couldn't find suitable task (e.g., bootstrap mode)
                task_result = None
            else:
                # Remove selected task from pending list by matching system and ebno
                # (Use filtering instead of dict.remove() to handle extra fields)
                initial_len = len(pending_tasks)
                pending_tasks[:] = [
                    t
                    for t in pending_tasks
                    if not (
                        t.get("system") == task.get("system")
                        and round(t.get("ebno", 0), 1) == round(task.get("ebno", 0), 1)
                    )
                ]

                if len(pending_tasks) == initial_len:
                    # Task wasn't found in pending_tasks (shouldn't happen, but handle gracefully)
                    task_result = None
                else:
                    task_result = task

        # All data_lock work done, check result
        if task_result is None:
            if worker_id in active_workers:
                with workers_lock:
                    if worker_id in active_workers:
                        active_workers[worker_id]["task"] = None
                        active_workers[worker_id]["last_seen"] = time.time()
            return {"status": "wait"}

        task = task_result

        with workers_lock:
            active_workers[worker_id] = {
                "task": task,
                "frames": 0,
                "fe": 0,
                "fps": 0,
                "last_seen": time.time(),
            }

        # Log refinement tasks
        if task.get("type") == "refinement":
            note = task.get("note", "")
            print(
                f"[REFINE] Worker {worker_id} assigned refinement task: {task['system']} @ {task['ebno']} dB ({note})"
            )

        try:
            save_state()
        except Exception:
            pass
        return {"status": "ok", "task": task, "max_frames": MAX_FRAMES}


@app.post("/api/toggle_pause", dependencies=[Depends(verify_api_or_ui_token)])
def toggle_pause():
    global cluster_paused, pause_start_time, total_paused_time
    cluster_paused = not cluster_paused
    if cluster_paused:
        pause_start_time = time.time()
    else:
        total_paused_time += time.time() - pause_start_time

    try:
        save_state()
    except Exception:
        pass
    try:
        broadcast_event({"type": "pause", "paused": cluster_paused})
    except Exception:
        pass

    try:
        with task_cond:
            task_cond.notify_all()
    except Exception:
        pass
    return {"status": "ok", "paused": cluster_paused}


@app.post("/api/halt_system", dependencies=[Depends(verify_api_or_ui_token)])
def halt_system(system: str):
    if system not in SYSTEMS:
        raise HTTPException(status_code=400, detail=f"Unknown system: {system}")
    halted_systems.add(system)
    # Remove pending tasks for this system
    pending_tasks[:] = [t for t in pending_tasks if t["system"] != system]
    print(f"[HALT] System {system } halted. No more tasks will be assigned.")
    try:
        with task_cond:
            task_cond.notify_all()
    except Exception:
        pass
    try:
        broadcast_event({"type": "system_halted", "system": system})
    except Exception:
        pass
    return {"status": "ok", "system": system, "halted": True}


@app.post("/api/unhalt_system", dependencies=[Depends(verify_api_or_ui_token)])
def unhalt_system(system: str):
    if system not in SYSTEMS:
        raise HTTPException(status_code=400, detail=f"Unknown system: {system}")
    if system in halted_systems:
        halted_systems.remove(system)
        print(f"[UNHALT] System {system } unhalted.")
        try:
            broadcast_event({"type": "system_unhalted", "system": system})
        except Exception:
            pass
    return {"status": "ok", "system": system, "halted": False}


@app.post("/api/ui_token")
def ui_token_route(credentials: HTTPBasicCredentials = Depends(security)):

    if not BASIC_USER or not BASIC_PASS:
        raise HTTPException(status_code=401, detail="Basic auth not configured")
    if not (
        secrets.compare_digest(credentials.username, BASIC_USER)
        and secrets.compare_digest(credentials.password, BASIC_PASS)
    ):
        raise HTTPException(status_code=401, detail="Unauthorized")
    token = create_ui_token(credentials.username)
    return {"token": token, "expires_in": MASTER_UI_TOKEN_TTL}


@app.post("/api/set_max_frames", dependencies=[Depends(verify_api_or_ui_token)])
def set_max_frames(payload: MaxFramesPayload):
    global MAX_FRAMES
    if payload.max_frames <= 0:
        raise HTTPException(status_code=400, detail="max_frames must be positive")
    MAX_FRAMES = payload.max_frames

    try:
        with open("data/master_config.json", "w") as f:
            json.dump({"max_frames": MAX_FRAMES}, f)
    except Exception:
        pass
    try:
        save_state()
    except Exception:
        pass
    try:
        broadcast_event({"type": "max_frames", "max_frames": MAX_FRAMES})
    except Exception:
        pass
    return {"status": "ok", "max_frames": MAX_FRAMES}


@app.post("/update_status", dependencies=[Depends(verify_api_key)])
def update_status(stat: StatusPayload):
    with workers_lock:
        if stat.worker_id in active_workers:
            active_workers[stat.worker_id].update(
                {
                    "frames": stat.frames,
                    "fe": stat.fe,
                    "fps": stat.fps,
                    "last_seen": time.time(),
                }
            )
        elif stat.system != "":

            print(
                f"[INFO] Re-attaching orphaned worker {stat .worker_id } doing {stat .system } @ {stat .ebno } dB"
            )
            active_workers[stat.worker_id] = {
                "task": {"system": stat.system, "ebno": stat.ebno},
                "frames": stat.frames,
                "fe": stat.fe,
                "fps": stat.fps,
                "last_seen": time.time(),
            }
            pending_tasks[:] = [
                t
                for t in pending_tasks
                if not (t["system"] == stat.system and t["ebno"] == stat.ebno)
            ]

    # Update smart scheduler with worker performance
    if stat.fps > 0:
        scheduler.update_worker_fps(stat.worker_id, stat.fps)

    try:
        save_state()
    except Exception:
        pass
    try:
        broadcast_event(
            {
                "type": "worker_update",
                "worker_id": stat.worker_id,
                "frames": stat.frames,
                "fe": stat.fe,
                "fps": stat.fps,
                "task": {"system": stat.system, "ebno": stat.ebno},
            }
        )
    except Exception:
        pass
    return {"status": "ok"}


@app.get("/check_task_cancelled", dependencies=[Depends(verify_api_key)])
def check_task_cancelled(worker_id: str, system: str, ebno: float):
    """Check if a task has been marked for cancellation."""
    with workers_lock:
        task_key = (worker_id, system, float(ebno))
        is_cancelled = task_key in cancelled_tasks
    return {"cancelled": is_cancelled}


@app.post("/api/cancel_task", dependencies=[Depends(verify_api_or_ui_token)])
def cancel_task(system: str, ebno: float):
    """Cancel a task and re-queue it. Slaves will detect cancellation on next check."""
    global cancelled_tasks
    with workers_lock:
        # Find which worker is running this task
        worker_to_cancel = None
        for wid, info in active_workers.items():
            if (
                info["task"]
                and info["task"]["system"] == system
                and info["task"]["ebno"] == ebno
            ):
                worker_to_cancel = wid
                break

        if worker_to_cancel:
            task = active_workers[worker_to_cancel]["task"]
            print(
                f"[CANCEL] Marking task {system } @ {ebno } dB (worker {worker_to_cancel }) for cancellation"
            )
            cancelled_tasks.add((worker_to_cancel, system, float(ebno)))
            active_workers[worker_to_cancel]["task"] = None
        else:
            return {
                "status": "not_found",
                "message": f"No active worker running {system } @ {ebno } dB",
            }

    if worker_to_cancel:
        with data_lock:
            task_key = _task_key(task)
            if task_key is not None and task_key not in set(
                _task_key(t) for t in pending_tasks if _task_key(t) is not None
            ):
                pending_tasks.insert(0, task)
        try:
            save_state()
        except Exception:
            pass
        try:
            with task_cond:
                task_cond.notify_all()
        except Exception:
            pass
        return {
            "status": "ok",
            "message": f"Task {system } @ {ebno } dB cancelled and re-queued",
        }


@app.post("/submit_result", dependencies=[Depends(verify_api_key)])
def submit_result(payload: ResultPayload):
    global plot_version
    global cancelled_tasks

    # Minimal lock section: update data only
    with data_lock:
        results[payload.system][payload.ebno] = payload.ber
        fer_results[payload.system][payload.ebno] = payload.fer

        # Check for zero error floor to prune useless higher Eb/No tasks
        hit_zero = payload.ber == 0.0 or payload.fer == 0.0
        if hit_zero:
            pending_tasks[:] = [
                t
                for t in pending_tasks
                if not (
                    t["system"] == payload.system and float(t["ebno"]) > payload.ebno
                )
            ]

            # Auto-fill the rest of the curve with 0.0 so it is marked as completely done
            for e in EBNO_RANGE:
                if e > payload.ebno:
                    if e not in results[payload.system]:
                        results[payload.system][e] = 0.0
                    if e not in fer_results[payload.system]:
                        fer_results[payload.system][e] = 0.0

        # Copy results for CSV writing (outside lock)
        results_copy = {
            s: {float(k): v for k, v in results[s].items()} for s in results
        }
        fer_results_copy = {
            s: {float(k): v for k, v in fer_results[s].items()} for s in fer_results
        }

    # All subsequent operations happen OUTSIDE data_lock
    print(
        f"[RESULT] Worker {payload .worker_id } finished {payload .system } @ {payload .ebno } dB -> BER: {payload .ber :.4e} | FER: {payload .fer :.4e}"
    )
    print(f"  -> Frames: {payload .frames}/{MAX_FRAMES}")

    if hit_zero:
        print(
            f"  -> Zero error frames detected for {payload .system } at {payload.ebno} dB. Pruning higher Eb/No points."
        )
        with workers_lock:
            for wid, info in active_workers.items():
                task = info.get("task")
                if (
                    task
                    and task.get("system") == payload.system
                    and float(task.get("ebno")) > payload.ebno
                ):
                    print(
                        f"[CANCEL] Actively cancelling useless task {task['system']} @ {task['ebno']} dB on worker {wid}"
                    )
                    cancelled_tasks.add((wid, task["system"], float(task["ebno"])))

    # Update worker status (uses workers_lock, not data_lock)
    if payload.worker_id in active_workers:
        with workers_lock:
            if payload.worker_id in active_workers:
                active_workers[payload.worker_id]["task"] = None
                active_workers[payload.worker_id]["last_seen"] = time.time()
            # Clean up cancelled_tasks entry if it exists
            cancelled_tasks.discard(
                (payload.worker_id, payload.system, float(payload.ebno))
            )

    # Notify waiting workers (outside all locks)
    try:
        with task_cond:
            task_cond.notify_all()
    except Exception:
        pass

    # Heavy I/O operations - spawn in background threads to avoid blocking request
    def _async_csv_and_state():
        try:
            _save_csv_data(results_copy, fer_results_copy)
        except Exception as e:
            print(f"[CSV ERROR] {e }")
        try:
            save_state()
        except Exception:
            pass

    def _async_plot():
        try:
            if os.path.exists("plot_results.py"):
                schedule_plot()
        except Exception as e:
            print(f"[PLOT ERROR] Could not schedule plotting: {e }")

    # Spawn background threads (request handler returns immediately)
    threading.Thread(target=_async_csv_and_state, daemon=True).start()
    threading.Thread(target=_async_plot, daemon=True).start()

    # Broadcast events (outside locks)
    try:
        broadcast_event(
            {
                "type": "system_auto_halted",
                "system": payload.system,
                "ebno": payload.ebno,
                "reason": "zero_error_frames",
            }
        )
    except Exception:
        pass

    try:
        if hit_zero:
            broadcast_event(
                {
                    "type": "system_auto_halted",
                    "system": payload.system,
                    "ebno": payload.ebno,
                    "reason": "zero_error_frames",
                }
            )
    except Exception:
        pass
    return {"status": "ok"}


def _save_csv_data(results_snapshot, fer_results_snapshot):
    """Write CSV files from snapshot data (call outside of locks)"""

    # Preprocess data to handle floor values and track minimums
    def preprocess_results(snapshot):
        processed = {}
        for system in SYSTEMS:
            processed[system] = {
                "min_value": float("inf"),
                "hit_floor": False,
                "values": {},
            }
            if system not in snapshot:
                continue
            for ebno in EBNO_RANGE:
                val = snapshot[system].get(ebno, "")
                if val == "":
                    continue
                try:
                    v = float(val) if isinstance(val, str) else val
                    # Skip invalid values; keep exact zeros so the zero frontier survives reloads.
                    if not np.isfinite(v):
                        processed[system]["hit_floor"] = True
                        continue
                    if v < 0:
                        processed[system]["hit_floor"] = True
                        continue
                    # Skip floor values (1e-7 quantization)
                    if abs(v - 1.0e-07) < 1e-15:
                        processed[system]["hit_floor"] = True
                        continue
                    if v == 0:
                        processed[system]["hit_floor"] = True
                        processed[system]["values"][ebno] = 0.0
                        continue
                    # Track minimum real value
                    if v < processed[system]["min_value"]:
                        processed[system]["min_value"] = v
                    # Once we hit floor, use minimum for subsequent values
                    if processed[system]["hit_floor"] and processed[system][
                        "min_value"
                    ] != float("inf"):
                        processed[system]["values"][ebno] = processed[system][
                            "min_value"
                        ]
                    else:
                        processed[system]["values"][ebno] = v
                except Exception as e:
                    pass
        return processed

    try:
        ber_processed = preprocess_results(results_snapshot)
        fer_processed = preprocess_results(fer_results_snapshot)
    except Exception as e:
        print(f"[CSV ERROR] Preprocessing failed: {e}")
        return

    with open("data/ber_simulation_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Eb/No (dB)"] + SYSTEMS)
        for ebno in EBNO_RANGE:
            row = [f"{ebno :.1f}"]
            for system in SYSTEMS:
                val = ber_processed[system]["values"].get(ebno, "")
                if val != "":
                    row.append(f"{val :.6e}")
                else:
                    row.append("")
            writer.writerow(row)

    with open("data/fer_simulation_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Eb/No (dB)"] + SYSTEMS)
        for ebno in EBNO_RANGE:
            row = [f"{ebno :.1f}"]
            for system in SYSTEMS:
                val = fer_processed[system]["values"].get(ebno, "")
                if val != "":
                    row.append(f"{val :.6e}")
                else:
                    row.append("")
            writer.writerow(row)


def save_csv():
    with data_lock:
        results_copy = {
            s: {float(k): v for k, v in results[s].items()} for s in results
        }
        fer_results_copy = {
            s: {float(k): v for k, v in fer_results[s].items()} for s in fer_results
        }
    _save_csv_data(results_copy, fer_results_copy)


@app.get("/plot.png")
def get_plot():
    if os.path.exists("data/high_res_ber_curve.png"):
        return FileResponse(
            "data/high_res_ber_curve.png",
            headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"},
        )
    return fastapi.Response(status_code=404)


@app.get("/fer_plot.png")
def get_fer_plot():
    if os.path.exists("data/high_res_fer_curve.png"):
        return FileResponse(
            "data/high_res_fer_curve.png",
            headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"},
        )
    return fastapi.Response(status_code=404)


@app.get("/api/status")
def get_api_status():
    try:
        clean_dead_workers()
    except Exception as e:
        print(f"[ERROR] clean_dead_workers failed: {e}")

    try:
        # Quick status snapshot
        with workers_lock:
            active_count = len([w for w in active_workers.values() if w["task"]])
            workers_snapshot = dict(active_workers)

        with data_lock:
            pending_count = len(pending_tasks)
            # Build filtered pending tasks list for dashboard
            filtered_pending = [
                {
                    "system": t.get("system"),
                    "ebno": t.get("ebno"),
                    "type": t.get("type"),
                }
                for t in pending_tasks[:50]
                if t.get("system") not in halted_systems
            ]

        completed = total_tasks - pending_count - active_count
        session_completed = max(0, completed - INITIAL_COMPLETED)
        session_time = time.time() - SERVER_START_TIME - total_paused_time
        if cluster_paused:
            session_time -= time.time() - pause_start_time
        eta_seconds = (
            (session_time / session_completed * (total_tasks - completed))
            if session_completed > 0
            else -1
        )

        return {
            "total_tasks": total_tasks,
            "completed": completed,
            "pending": pending_count,
            "pending_tasks": filtered_pending,
            "workers": workers_snapshot,
            "plot_version": plot_version,
            "eta_seconds": eta_seconds,
            "paused": cluster_paused,
            "max_frames": MAX_FRAMES,
        }
    except Exception as e:
        print(f"[ERROR] /api/status failed: {e}")
        import traceback

        traceback.print_exc()
        return {
            "total_tasks": total_tasks,
            "completed": 0,
            "pending": 0,
            "pending_tasks": [],
            "workers": {},
            "plot_version": plot_version,
            "eta_seconds": -1,
            "paused": cluster_paused,
            "max_frames": MAX_FRAMES,
        }


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def public_page():
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>BERSim Cluster - Public Status</title>
        <style>
            body { font-family: Arial, Helvetica, sans-serif; background: #1e1e1e; color: #ddd; margin: 0; padding: 20px; }
            h1 { color: #4ec9b0; text-align: center; }
            .row { display:flex; gap: 20px; justify-content:center; margin-bottom:16px; }
            .card { background:#252526; padding:12px 18px; border-radius:8px; border:1px solid #333; min-width:160px; text-align:center; }
            .value { font-size:20px; font-weight:700; color:#cfeee6; margin-top:6px; }
            .plots { display:flex; gap:12px; margin-top:18px; }
            .plots img { width:100%; max-width:720px; border-radius:6px; background:white; }
            .controls { text-align:center; margin-top:12px; }
            button { padding:8px 12px; border-radius:6px; border:none; background:#007acc; color:white; cursor:pointer; }
        </style>
        <script>
            let lastPlotVersion = -1;
            async function fetchStatusPublic(){
                try{
                    const res = await fetch('/api/status');
                    if(!res.ok) return;
                    const data = await res.json();
                    document.getElementById('stat-completed').innerText = data.completed + ' / ' + data.total_tasks;
                    document.getElementById('stat-pending').innerText = data.pending;
                    document.getElementById('stat-workers').innerText = Object.keys(data.workers).length;
                    document.getElementById('stat-maxframes').innerText = (data.max_frames || 10000000).toLocaleString();
                    let etaStr = 'Calculating...';
                    if (data.completed === data.total_tasks) etaStr = 'Done!';
                    else if (data.paused) etaStr = 'Paused';
                    else if (data.eta_seconds >= 0){ let s = Math.round(data.eta_seconds); let h=Math.floor(s/3600); let m=Math.floor((s%3600)/60); let sec=s%60; etaStr = h>0?`${h}h ${m}m ${sec}s`:`${m}m ${sec}s`; }
                    document.getElementById('stat-eta').innerText = etaStr;
                    if (data.plot_version !== lastPlotVersion){
                        document.getElementById('ber-plot').src = '/plot.png?v=' + data.plot_version + '&t=' + Date.now();
                        document.getElementById('fer-plot').src = '/fer_plot.png?v=' + data.plot_version + '&t=' + Date.now();
                        document.getElementById('ber-plot').style.display = 'block';
                        document.getElementById('fer-plot').style.display = 'block';
                        lastPlotVersion = data.plot_version;
                    }
                }catch(e){ }
            }
            window.onload = function(){ fetchStatusPublic(); setInterval(fetchStatusPublic, 2000); };
        </script>
    </head>
    <body>
        <h1>BERSim Cluster — Public Status</h1>
        <div class="row">
            <div class="card">Tasks Completed<div class="value" id="stat-completed">-</div></div>
            <div class="card">Tasks Queued<div class="value" id="stat-pending">-</div></div>
            <div class="card">Active Workers<div class="value" id="stat-workers">-</div></div>
            <div class="card">ETA<div class="value" id="stat-eta">-</div></div>
            <div class="card">Max Frames<div class="value" id="stat-maxframes">-</div></div>
        </div>
            <div class="controls"><button onclick="location.href='/admin'">Login (admin)</button></div>
        <div class="plots">
            <img id="ber-plot" src="/plot.png" alt="BER plot" style="display:none">
            <img id="fer-plot" src="/fer_plot.png" alt="FER plot" style="display:none">
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html)


@app.get(
    "/admin", response_class=HTMLResponse, dependencies=[Depends(verify_basic_auth)]
)
def dashboard():
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>BERSim Cluster Dashboard</title>
        <style>
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #1e1e1e; color: #d4d4d4; margin: 0; padding: 20px; }
            h1 { color: #569cd6; text-align: center; }
            .stats-container { display: flex; justify-content: center; gap: 20px; margin-bottom: 20px; }
            .stat-box { background-color: #252526; padding: 15px 25px; border-radius: 8px; text-align: center; border: 1px solid #333; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }
            .stat-value { font-size: 24px; font-weight: bold; color: #4ec9b0; margin-top: 10px; }
            table { width: 100%; border-collapse: collapse; margin-top: 10px; background-color: #252526; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }
            th, td { padding: 12px 15px; text-align: left; border-bottom: 1px solid #3e3e42; }
            th { background-color: #333333; color: #c586c0; font-weight: 600; }
            tr:hover { background-color: #2a2d2e; }
            th { cursor: pointer; position: relative; user-select: none; }
            th::after { content: ''; border: 4px solid transparent; position: absolute; right: 10px; top: 50%; }
            th.sort-asc::after { border-bottom-color: #c586c0; transform: translateY(-50%); }
            th.sort-desc::after { border-top-color: #c586c0; transform: translateY(-25%); }
            .progress-bar { width: 100%; background-color: #3e3e42; border-radius: 4px; overflow: hidden; height: 10px; margin-top: 5px; }
            .progress-fill { background-color: #007acc; height: 100%; transition: width 0.5s; }
        </style>
        <script>
            let lastPlotVersion = -1;
            let ws = null;
            let sortState = { column: 6, asc: true }; // Default: sort by Last Ping, ascending

            function initWebSocket() {
                const proto = location.protocol === 'https:' ? 'wss' : 'ws';
                try {
                    const token = window.UI_TOKEN || '';
                    ws = new WebSocket(proto + '://' + location.host + '/ws?token=' + encodeURIComponent(token || ''));
                    ws.onopen = () => console.log('WS connected');
                    ws.onmessage = (e) => {
                        try { const d = JSON.parse(e.data); handleWsEvent(d); } catch (err) { }
                    };
                        ws.onclose = () => { console.log('WS closed, reconnecting in 2s'); setTimeout(() => fetchUiToken().then(initWebSocket), 2000); };
                        ws.onerror = (e) => { console.log('WS error', e); ws.close(); };
                } catch (e) {
                    setTimeout(initWebSocket, 2000);
                }
            }

            function handleWsEvent(data) {
                if (!data || !data.type) return;
                switch (data.type) {
                    case 'plot':
                        if (data.plot_version !== lastPlotVersion) {
                            const newBerPlot = new Image();
                            newBerPlot.onload = () => { const berPlot = document.getElementById('ber-plot'); berPlot.src = '/plot.png?v=' + data.plot_version + '&t=' + Date.now(); berPlot.style.display = 'block'; };
                            newBerPlot.onerror = () => { document.getElementById('ber-plot').style.display = 'none'; };
                            newBerPlot.src = '/plot.png?v=' + data.plot_version + '&t=' + Date.now();

                            const newFerPlot = new Image();
                            newFerPlot.onload = () => { const ferPlot = document.getElementById('fer-plot'); ferPlot.src = '/fer_plot.png?v=' + data.plot_version + '&t=' + Date.now(); ferPlot.style.display = 'block'; };
                            newFerPlot.onerror = () => { document.getElementById('fer-plot').style.display = 'none'; };
                            newFerPlot.src = '/fer_plot.png?v=' + data.plot_version + '&t=' + Date.now();

                            lastPlotVersion = data.plot_version;
                        }
                        break;
                    case 'result':
                    case 'worker_update':
                    case 'pause':
                        fetchStatus();
                        break;
                    case 'max_frames':
                        const mf = document.getElementById('max-frames-input'); if (mf) mf.value = data.max_frames; break;
                }
            }

            async function fetchUiToken() {
                try {
                    const res = await fetch('/api/ui_token', { method: 'POST', credentials: 'include' });
                    if (!res.ok) {
                        return null;
                    }
                    const j = await res.json();
                    window.UI_TOKEN = j.token;
                    if (j.expires_in) {
                        const refresh = Math.max(30, Math.floor(j.expires_in * 0.75));
                        if (window.UI_TOKEN_REFRESHER) clearTimeout(window.UI_TOKEN_REFRESHER);
                        window.UI_TOKEN_REFRESHER = setTimeout(fetchUiToken, refresh * 1000);
                    }
                    return j.token;
                } catch (e) {
                    console.warn(e);
                    return null;
                }
            }

            async function togglePause() {
                const token = window.UI_TOKEN || await fetchUiToken();
                await fetch('/api/toggle_pause', {
                    method: 'POST',
                    headers: { 'Authorization': 'Bearer ' + (token || '') },
                    credentials: 'include'
                });
                fetchStatus();
            }

            function sortTable(colIndex) {
                if (sortState.column === colIndex) {
                    sortState.asc = !sortState.asc;
                } else {
                    sortState.column = colIndex;
                    sortState.asc = true;
                }
                fetchStatus();
            }

            async function fetchStatus() {
                const token = window.UI_TOKEN || null;
                const headers = token ? { 'Authorization': 'Bearer ' + token } : {};
                const res = await fetch('/api/status', { credentials: 'include', headers: headers });
                const data = await res.json();
                const maxFrames = data.max_frames || 10000000;
                // Update input if present
                let mfInput = document.getElementById('max-frames-input');
                if (mfInput) mfInput.value = maxFrames;
                
                document.getElementById('stat-completed').innerText = data.completed + " / " + data.total_tasks;
                document.getElementById('stat-pending').innerText = data.pending;
                document.getElementById('stat-workers').innerText = Object.keys(data.workers).length;
                
                let pauseBtn = document.getElementById('pause-btn');
                if (data.paused) {
                    pauseBtn.innerHTML = "&#9658; Resume Cluster";
                    pauseBtn.style.backgroundColor = "#28a745";
                } else {
                    pauseBtn.innerHTML = "&#10074;&#10074; Pause Cluster";
                    pauseBtn.style.backgroundColor = "#cca700";
                }

                let etaStr = "Calculating...";
                if (data.completed === data.total_tasks) {
                    etaStr = "Done!";
                } else if (data.paused) {
                    etaStr = "Paused";
                } else if (data.eta_seconds >= 0) {
                    let s = Math.round(data.eta_seconds);
                    let h = Math.floor(s / 3600);
                    let m = Math.floor((s % 3600) / 60);
                    let sec = s % 60;
                    etaStr = h > 0 ? `${h}h ${m}m ${sec}s` : `${m}m ${sec}s`;
                } else {
                    etaStr = "Calculating...";
                }
                document.getElementById('stat-eta').innerText = etaStr;
                
                let workersArray = Object.entries(data.workers).map(([wid, info]) => {
                    let workerEtaSeconds = -1; // -1 indicates no ETA (idle or calculating)
                    if (info.task && info.fps > 0) {
                        let remaining = Math.max(0, maxFrames - info.frames);
                        workerEtaSeconds = Math.round(remaining / info.fps);
                    }
                    return {
                        wid,
                        ...info,
                        lastSeenSeconds: Math.round(Date.now()/1000 - info.last_seen),
                        workerEtaSeconds: workerEtaSeconds
                    };
                });

                workersArray.sort((a, b) => {
                    let valA, valB;
                    switch (sortState.column) {
                        case 0: valA = a.wid; valB = b.wid; break;
                        case 1: valA = a.task ? a.task.system : ''; valB = b.task ? b.task.system : ''; break;
                        case 2: valA = a.task ? a.frames : -1; valB = b.task ? b.frames : -1; break;
                        case 3: valA = a.task ? a.fe : -1; valB = b.task ? b.fe : -1; break;
                        case 4: valA = a.task ? a.fps : -1; valB = b.task ? b.fps : -1; break;
                        case 5: valA = a.workerEtaSeconds; valB = b.workerEtaSeconds; break;
                        case 6: valA = a.lastSeenSeconds; valB = b.lastSeenSeconds; break;
                        default: return 0;
                    }

                    let comparison = 0;
                    if (valA > valB) comparison = 1;
                    else if (valA < valB) comparison = -1;

                    if (sortState.column === 5) { // Special handling for Task ETA: push idle/calculating to bottom
                        if (valA === -1 && valB !== -1) return 1;
                        if (valA !== -1 && valB === -1) return -1;
                    }
                    return sortState.asc ? comparison : -comparison;
                });

                let tbody = document.getElementById('workers-tbody');
                tbody.innerHTML = '';
                let totalFps = 0;
                for (const info of workersArray) {
                    let row = document.createElement('tr');
                    if (info.task && info.task.type === 'refinement') {
                        row.style.backgroundColor = '#3a3d2e'; // A subtle highlight for refinement tasks
                    }
                    let taskStr = info.task ? `${info.task.system} @ ${info.task.ebno} dB` : "<i>Idle</i>";
                    let pct = info.task ? Math.min(100, (info.frames / maxFrames) * 100).toFixed(1) : 0;
                    let frameStr = info.task ? `${info.frames.toLocaleString()} / ${maxFrames.toLocaleString()}` : "-";
                    let feStr = info.task ? info.fe : "-";
                    let fpsStr = info.task && info.fps ? info.fps.toLocaleString() : "-";
                    
                    if (info.task && info.fps) totalFps += info.fps;
                    
                    let workerEtaStr = "-";
                    if (info.workerEtaSeconds > 0) {
                        let s = info.workerEtaSeconds;
                        let h = Math.floor(s / 3600);
                        let m = Math.floor((s % 3600) / 60);
                        let sec = s % 60;
                        workerEtaStr = h > 0 ? `${h}h ${m}m ${sec}s` : `${m}m ${sec}s`;
                    } else if (info.task) {
                        workerEtaStr = "<i>Calculating...</i>";
                    }
                    
                    row.innerHTML = `
                        <td>${info.wid}</td>
                        <td>${taskStr}${info.task && info.task.type === 'refinement' ? ' <span style="color: #d7ba7d; font-style: italic;">(Refine)</span>' : ''}</td>
                        <td>${frameStr} <div class="progress-bar"><div class="progress-fill" style="width: ${pct}%"></div></div></td>
                        <td style="color: ${info.fe > 0 ? '#f48771' : '#d4d4d4'}">${feStr}</td>
                        <td>${fpsStr}</td>
                        <td>${workerEtaStr}</td>
                        <td>${info.lastSeenSeconds}s ago</td>
                    `;
                    tbody.appendChild(row);
                }
                
                document.getElementById('stat-fps').innerText = totalFps > 0 ? totalFps.toLocaleString() + " FPS" : "-";
                
                // Update pending tasks table
                let tasksArray = data.pending_tasks || [];
                let tasksTbody = document.getElementById('tasks-tbody');
                if (tasksTbody) {
                    tasksTbody.innerHTML = '';
                    tasksArray.forEach(task => {
                        let row = document.createElement('tr');
                        if (task.type === 'refinement') {
                            row.style.backgroundColor = '#3a3d2e';
                        }
                        row.innerHTML = `
                            <td>${task.system}${task.type === 'refinement' ? ' <span style="color: #d7ba7d; font-style: italic;">(Refine)</span>' : ''}</td>
                            <td>${task.ebno ? task.ebno.toFixed(1) : '-'} dB</td>
                        `;
                        tasksTbody.appendChild(row);
                    });
                    let tasksCountEl = document.getElementById('tasks-count');
                    if (tasksCountEl) {
                        tasksCountEl.innerText = data.pending + ' tasks queued (showing first 50)';
                    }
                }
                
                document.querySelectorAll('th').forEach((th, i) => {
                    th.classList.remove('sort-asc', 'sort-desc');
                    if (i === sortState.column) {
                        th.classList.add(sortState.asc ? 'sort-asc' : 'sort-desc');
                    }
                });

                if (data.plot_version !== lastPlotVersion) {
                    // Preload the new images to prevent flickering
                    const newBerPlot = new Image();
                    newBerPlot.onload = () => {
                        const berPlot = document.getElementById('ber-plot');
                        berPlot.src = newBerPlot.src;
                        berPlot.style.display = 'block'; // Ensure it's visible once loaded
                    };
                    newBerPlot.onerror = () => {
                        document.getElementById('ber-plot').style.display = 'none'; // Hide if new image fails to load
                    };
                    newBerPlot.src = '/plot.png?v=' + data.plot_version + '&t=' + Date.now();

                    const newFerPlot = new Image();
                    newFerPlot.onload = () => {
                        const ferPlot = document.getElementById('fer-plot');
                        ferPlot.src = newFerPlot.src;
                        ferPlot.style.display = 'block';
                    };
                    newFerPlot.onerror = () => {
                        document.getElementById('fer-plot').style.display = 'none';
                    };
                    newFerPlot.src = '/fer_plot.png?v=' + data.plot_version + '&t=' + Date.now();

                    lastPlotVersion = data.plot_version;
                }
            }

            async function setMaxFrames() {
                const val = parseInt(document.getElementById('max-frames-input').value);
                if (!val || val <= 0) return alert('Please enter a positive integer');
                const token = window.UI_TOKEN || await fetchUiToken();
                await fetch('/api/set_max_frames', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json', 'Authorization': 'Bearer ' + (token || '') },
                    credentials: 'include',
                    body: JSON.stringify({ max_frames: val })
                });
                fetchStatus();
            }
            setInterval(fetchStatus, 1000);
            window.onload = async () => { await fetchUiToken(); initWebSocket(); fetchStatus(); };
        </script>
    </head>
    <body>
        <h1>Distributed BerSim Cluster Master</h1>
        <div style="text-align: center; margin-bottom: 20px;">
            <button id="pause-btn" onclick="togglePause()" style="padding: 10px 20px; font-size: 16px; color: white; border: none; border-radius: 4px; cursor: pointer; background-color: #4ec9b0; font-weight: bold; transition: background-color 0.3s; box-shadow: 0 4px 6px rgba(0,0,0,0.3);">Loading...</button>
            <div style="margin-top:10px;">
                <input id="max-frames-input" type="number" min="1" value="10000000" style="width:160px; padding:6px; margin-left:12px; margin-right:8px;">
                <button onclick="setMaxFrames()" style="padding:6px 10px;">Set Max Frames</button>
            </div>
        </div>
        <div class="stats-container">
            <div class="stat-box">Tasks Completed<div class="stat-value" id="stat-completed">-</div></div>
            <div class="stat-box">Tasks Queued<div class="stat-value" id="stat-pending">-</div></div>
            <div class="stat-box">Active Workers<div class="stat-value" id="stat-workers">-</div></div>
            <div class="stat-box">Total Speed<div class="stat-value" id="stat-fps">-</div></div>
            <div class="stat-box">ETA<div class="stat-value" id="stat-eta">-</div></div>
        </div>
        <div style="display: flex; gap: 20px; margin-bottom: 20px; justify-content: center;">
            <div style="flex: 1; text-align: center; background-color: #252526; padding: 15px; border-radius: 8px; border: 1px solid #333; box-shadow: 0 4px 6px rgba(0,0,0,0.3);">
                <img id="ber-plot" src="/plot.png" alt="BER Curve Plot" style="width: 100%; border-radius: 4px; display: none; margin: 0 auto; background: white;">
            </div>
            <div style="flex: 1; text-align: center; background-color: #252526; padding: 15px; border-radius: 8px; border: 1px solid #333; box-shadow: 0 4px 6px rgba(0,0,0,0.3);">
                <img id="fer-plot" src="/fer_plot.png" alt="FER Curve Plot" style="width: 100%; border-radius: 4px; display: none; margin: 0 auto; background: white;">
            </div>
        </div>
        <table>
            <thead>
                <tr id="table-headers">
                    <th onclick="sortTable(0)">Worker ID</th>
                    <th onclick="sortTable(1)">Current Task</th>
                    <th onclick="sortTable(2)">Frames Processed</th>
                    <th onclick="sortTable(3)">Frame Errors (FE)</th>
                    <th onclick="sortTable(4)">FPS</th>
                    <th onclick="sortTable(5)">Task ETA</th>
                    <th onclick="sortTable(6)">Last Ping</th>
                </tr>
            </thead>
            <tbody id="workers-tbody"></tbody>
        </table>
        <h2 style="margin-top: 30px; color: #569cd6;">Pending Tasks Queue</h2>
        <div style="margin-bottom: 15px; color: #d4d4d4; font-size: 14px;" id="tasks-count">Loading...</div>
        <table>
            <thead>
                <tr>
                    <th>System</th>
                    <th>Eb/No (dB)</th>
                </tr>
            </thead>
            <tbody id="tasks-tbody"></tbody>
        </table>
    </body>
    </html>
    """
    return HTMLResponse(content=html)


if __name__ == "__main__":
    try:
        logger.info("Starting Master server")
    except Exception:
        pass
    try:
        uvicorn.run(app, host="0.0.0.0", port=8000)
    except Exception as e:
        try:
            logger.exception("Master server crashed: %s", e)
        except Exception:
            print("Master server crashed:", e)
        try:
            save_state()
        except Exception:
            try:
                logger.exception("Failed to save state after crash")
            except Exception:
                pass
        raise
