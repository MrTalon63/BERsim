import requests 
import subprocess 
import time 
import sys 
import uuid 
import os 
import requests
import subprocess
import time
import sys
import uuid
import os
import argparse
import signal
import shutil
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

MASTER_URL = os.environ.get("MASTER_URL", "test")
API_KEY = os.environ.get("CLUSTER_API_KEY", "test")
HEADERS = {"X-API-Key": API_KEY, "Connection": "keep-alive"}
WORKER_ID = str(uuid.uuid4())[:8]


def make_session():
    s = requests.Session()
    s.headers.update(HEADERS)
    retries = Retry(total=5, backoff_factor=1, status_forcelist=(502, 503, 504), allowed_methods=frozenset(['GET', 'POST']))
    adapter = HTTPAdapter(max_retries=retries)
    s.mount('http://', adapter)
    s.mount('https://', adapter)
    return s


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

    def _signal_handler(signum, frame):
        nonlocal stop
        stop = True
        print("Received shutdown signal, exiting gracefully...")

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    while not stop:
        try:
            r = session.get(f"{MASTER_URL}/get_task?worker_id={hostname}-{WORKER_ID}", timeout=70)
            data = r.json()
        except Exception:
            print("Cannot reach Master server. Retrying in 5 seconds...")
            time.sleep(5)
            continue

        if data.get("status") == "wait":
            print(f"\rNo task available. Slave [{hostname}-{WORKER_ID}] waiting...", end="", flush=True)
            continue

        if data.get("status") == "done":
            print("All cluster tasks completed! Slave resting.")
            break

        task = data["task"]
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
            try:
                session.post(
                    f"{MASTER_URL}/submit_result",
                    json={
                        "worker_id": f"{hostname}-{WORKER_ID}",
                        "system": sys_name,
                        "ebno": ebno,
                        "ber": result_ber,
                        "fer": result_fer,
                        "frames": result_frames,
                    },
                    timeout=10,
                )
            except Exception:
                print("Failed to POST results to Master; will retry after short delay.")
                time.sleep(5)
                try:
                    session.post(
                        f"{MASTER_URL}/submit_result",
                        json={
                            "worker_id": f"{hostname}-{WORKER_ID}",
                            "system": sys_name,
                            "ebno": ebno,
                            "ber": result_ber,
                            "fer": result_fer,
                            "frames": result_frames,
                        },
                        timeout=10,
                    )
                except Exception:
                    print("Second attempt failed; giving up on this result.")


if __name__ == "__main__":
    main()