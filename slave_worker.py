import json
import threading
import random
import requests
import subprocess
import time
import sys
import uuid
import os
import socket
import argparse
import signal
import shutil
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dotenv import load_dotenv

load_dotenv()

MASTER_URL = os.environ.get("MASTER_URL", "test")
API_KEY = os.environ.get("CLUSTER_API_KEY", "test")
HEADERS = {"X-API-Key": API_KEY, "Connection": "keep-alive"}
WORKER_ID = socket.gethostname()


def make_session():
    s = requests.Session()
    s.headers.update(HEADERS)
    retries = Retry(total=5, backoff_factor=1, status_forcelist=(502, 503, 504), allowed_methods=frozenset(['GET', 'POST']))
    adapter = HTTPAdapter(max_retries=retries)
    s.mount('http://', adapter)
    s.mount('https://', adapter)
    return s


RESULT_QUEUE_DIR = os.environ.get("RESULT_QUEUE_DIR", os.path.join(os.getcwd(), "failed_results"))
RESULT_RETRY_MAX = int(os.environ.get("RESULT_RETRY_MAX", "5"))
RESULT_RETRY_BACKOFF = float(os.environ.get("RESULT_RETRY_BACKOFF", "1.0"))
RESULT_FLUSH_INTERVAL = int(os.environ.get("RESULT_FLUSH_INTERVAL", "30"))

def ensure_queue_dir():
    try:
        os.makedirs(RESULT_QUEUE_DIR, exist_ok=True)
    except Exception:
        pass

def queue_result(payload):
    ensure_queue_dir()
    fname = f"{int(time.time())}-{uuid.uuid4().hex[:8]}.json"
    path = os.path.join(RESULT_QUEUE_DIR, fname)
    try:
        with open(path, "w") as fh:
            json.dump(payload, fh)
        print(f"Queued failed result to {path}")
    except Exception as e:
        print(f"Failed to write queued result to disk: {e}")

def submit_with_retries(session, endpoint, payload, max_retries=RESULT_RETRY_MAX, backoff_factor=RESULT_RETRY_BACKOFF, timeout=10):
    attempt = 0
    while True:
        try:
            r = session.post(endpoint, json=payload, timeout=timeout)
            r.raise_for_status()
            return True
        except Exception as exc:
            attempt += 1
            if attempt > max_retries:
                print(f"submit_with_retries: final failure after {max_retries} attempts: {exc}")
                return False
            sleep_time = backoff_factor * (2 ** (attempt - 1))
            jitter = random.uniform(0, min(1, sleep_time * 0.1))
            sleep_time += jitter
            print(f"submit_with_retries: attempt {attempt}/{max_retries} failed, sleeping {sleep_time:.1f}s")
            time.sleep(sleep_time)

def flush_queue(session):
    ensure_queue_dir()
    try:
        files = sorted(os.listdir(RESULT_QUEUE_DIR))
    except OSError:
        return
    for fname in files:
        path = os.path.join(RESULT_QUEUE_DIR, fname)
        if not os.path.isfile(path):
            continue
        try:
            with open(path, "r") as fh:
                payload = json.load(fh)
        except Exception as e:
            print(f"Failed to read queued result {path}: {e}")
            try:
                os.remove(path)
            except Exception:
                pass
            continue
        success = submit_with_retries(session, f"{MASTER_URL}/submit_result", payload)
        if success:
            try:
                os.remove(path)
                print(f"Flushed queued result {path}")
            except Exception:
                pass
        else:
            print(f"Leaving queued result for later retry: {path}")

def start_flush_worker(session, stop_event, interval=RESULT_FLUSH_INTERVAL):
    def worker():
        while not stop_event.wait(interval):
            try:
                flush_queue(session)
            except Exception as e:
                print("Error while flushing queue:", e)
    t = threading.Thread(target=worker, daemon=True)
    t.start()
    return t


def main():
    parser = argparse.ArgumentParser(description="BERSim Cluster Slave Worker")
    parser.add_argument(
        "-t", "--threads", type=int, default=0,
        help="Number of threads for the simulator to use (0 for auto-detect)."
    )
    args = parser.parse_args()

    import socket
    hostname = socket.gethostname()
    print(f"Slave Worker [{hostname}-{WORKER_ID}] starting... Connecting to {MASTER_URL}")
    if args.threads > 0:
        print(f"Thread count explicitly set to {args.threads}.")

    session = make_session()

    def find_simulator():
        sim_env = os.environ.get("SIMULATOR_PATH") or os.environ.get("SIMULATOR_CMD")
        if sim_env:
            if os.path.exists(sim_env) and os.access(sim_env, os.X_OK):
                print(f"Using simulator from SIMULATOR_PATH: {sim_env}")
                return sim_env
            abs_try = os.path.join(os.getcwd(), sim_env)
            if os.path.exists(abs_try) and os.access(abs_try, os.X_OK):
                print(f"Using simulator from SIMULATOR_PATH (cwd-joined): {abs_try}")
                return abs_try
            which_env = shutil.which(sim_env)
            if which_env:
                print(f"Using simulator from PATH resolving SIMULATOR_PATH: {which_env}")
                return which_env
            print(f"SIMULATOR_PATH set but not found/executable: {sim_env}; will try fallback locations.")

        pref = os.path.join(os.getcwd(), 'build', 'simulator')
        if os.path.exists(pref) and os.access(pref, os.X_OK):
            return pref
        if os.path.exists('./simulator') and os.access('./simulator', os.X_OK):
            return './simulator'
        which = shutil.which('simulator')
        if which:
            return which
        return './simulator'

    SIMULATOR_CMD = find_simulator()
    print(f"Using simulator executable: {SIMULATOR_CMD}")

    stop = False
    stop_event = threading.Event()

    def _signal_handler(signum, frame):
        nonlocal stop
        stop = True
        stop_event.set()
        print("Received shutdown signal, exiting gracefully...")

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # Start background flush worker and try an immediate flush of queued results
    start_flush_worker(session, stop_event)
    try:
        flush_queue(session)
    except Exception:
        pass

    while not stop:
        try:
            r = session.get(f"{MASTER_URL}/get_task?worker_id={hostname}-{WORKER_ID}", timeout=70)
            r.raise_for_status()
            try:
                data = r.json()
            except ValueError:
                print("Invalid JSON from Master server; response:", r.text)
                time.sleep(5)
                continue
        except requests.exceptions.RequestException as e:
            print(f"Cannot reach Master server or HTTP error: {e}. Retrying in 5 seconds...")
            time.sleep(5)
            continue

        if not isinstance(data, dict):
            print("Unexpected response from Master (not JSON object):", data)
            time.sleep(5)
            continue

        if data.get("status") == "wait":
            print(f"\rNo task available. Slave [{hostname}-{WORKER_ID}] waiting...", end="", flush=True)
            continue

        if data.get("status") == "done":
            print("All cluster tasks completed! Slave resting.")
            break

        task = data.get("task")
        if not task:
            print("Missing 'task' in Master response:", data)
            time.sleep(5)
            continue
        sys_name = task["system"]
        ebno = task["ebno"]
        max_frames = data.get("max_frames", os.environ.get("SIM_MAX_FRAMES", "10000000"))
        print(f"\n--- Assigned Task: {sys_name} @ {ebno} dB (max frames: {max_frames}) ---")

        env = os.environ.copy()
        env["SIM_MAX_FRAMES"] = str(max_frames)
        command = [SIMULATOR_CMD, sys_name, str(ebno)]
        if args.threads > 0:
            command.extend(["--threads", str(args.threads)])

        proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)

        result_ber = 1.0
        result_fer = 1.0
        result_frames = 0
        buffer = ""
        last_ping = time.time()

        while True:
            char = proc.stdout.read(1)
            if not char:
                break

            buffer += char

            if char in ('\r', '\n'):
                line = buffer.strip()
                if "~> Time:" in line and "Frames:" in line:
                    sys.stdout.write("\r    " + line + "   ")
                    sys.stdout.flush()
                    try:
                        parts = line.split("|")
                        frames = int([p for p in parts if "Frames:" in p][0].split(":")[1].split("/")[0].strip())
                        fe = int([p for p in parts if "FE:" in p][0].split(":")[1].split("/")[0].strip())
                        fps_part = [p for p in parts if "FPS:" in p]
                        fps = int(fps_part[0].split(":")[1].strip()) if fps_part else 0

                        now = time.time()
                        if now - last_ping > 2.0:
                            try:
                                session.post(
                                    f"{MASTER_URL}/update_status",
                                    json={
                                        "worker_id": f"{hostname}-{WORKER_ID}",
                                        "system": sys_name,
                                        "ebno": ebno,
                                        "frames": frames,
                                        "fe": fe,
                                        "fps": fps,
                                    },
                                    timeout=5,
                                )
                            except Exception:
                                pass
                            
                            # Check if task was cancelled
                            try:
                                cancel_check = session.get(
                                    f"{MASTER_URL}/check_task_cancelled",
                                    params={
                                        "worker_id": f"{hostname}-{WORKER_ID}",
                                        "system": sys_name,
                                        "ebno": ebno,
                                    },
                                    timeout=5,
                                )
                                if cancel_check.status_code == 200:
                                    cancel_data = cancel_check.json()
                                    if cancel_data.get("cancelled"):
                                        print(f"\n[CANCEL] Task {sys_name} @ {ebno} dB has been cancelled. Terminating subprocess...")
                                        proc.terminate()
                                        try:
                                            proc.wait(timeout=5)
                                        except subprocess.TimeoutExpired:
                                            print("[CANCEL] Subprocess did not terminate; killing it.")
                                            proc.kill()
                                            proc.wait()
                                        print(f"[CANCEL] Task cancelled. Requesting new task...")
                                        break
                            except Exception:
                                pass
                            
                            last_ping = now
                    except Exception:
                        pass

                elif line:
                    if "|" in line and "Time:" not in line and "Eb/No" not in line:
                        sys.stdout.write("\r" + line + " " * 40 + "\n")
                        sys.stdout.flush()
                        parts = [p.strip() for p in line.split("|")]
                        try:
                            if len(parts) >= 5 and sys_name in parts[0] and str(ebno) in parts[1]:
                                result_ber, result_fer, result_frames = float(parts[2]), float(parts[3]), int(parts[4])
                            elif len(parts) >= 4 and str(ebno) in parts[0]:
                                result_ber, result_fer, result_frames = float(parts[1]), float(parts[2]), int(parts[3])
                        except ValueError:
                            pass
                    else:
                        print(line)
                buffer = ""

        proc.wait()
        if proc.returncode != 0:
            print(f"\n[ERROR] C++ simulator crashed with exit code {proc.returncode}! Skipping upload.")
            time.sleep(5)
        else:
            payload = {
                "worker_id": f"{hostname}-{WORKER_ID}",
                "system": sys_name,
                "ebno": ebno,
                "ber": result_ber,
                "fer": result_fer,
                "frames": result_frames,
            }
            success = submit_with_retries(session, f"{MASTER_URL}/submit_result", payload, timeout=10)
            if not success:
                print("Failed to POST results to Master after retries; queuing result for later retry.")
                queue_result(payload)
            else:
                print("Result submitted to Master successfully.")


if __name__ == "__main__":
    main()