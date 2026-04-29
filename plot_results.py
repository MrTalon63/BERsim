import csv
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import shutil
import json

os.makedirs("data", exist_ok=True)

if os.path.exists("ber_simulation_results.csv") and not os.path.exists(
    "data/ber_simulation_results.csv"
):
    shutil.move("ber_simulation_results.csv", "data/ber_simulation_results.csv")
if os.path.exists("high_res_ber_curve.png"):
    shutil.move("high_res_ber_curve.png", "data/high_res_ber_curve.png")

# Read MAX_FRAMES from environment or config
MAX_FRAMES = int(os.environ.get("SIM_MAX_FRAMES", "10000000"))
try:
    if os.path.exists("data/master_config.json"):
        with open("data/master_config.json", "r") as f:
            cfg = json.load(f)
            MAX_FRAMES = cfg.get("max_frames", MAX_FRAMES)
except Exception:
    pass


def generate_curve(csv_path, out_path, title, ylabel, chart_type="ber"):
    if not os.path.exists(csv_path):
        return

    print(f"Generating plot from {csv_path}...")
    all_data = {}

    with open(csv_path, mode="r") as file:
        reader = csv.reader(file)
        headers = next(reader, [])
        for h in headers[1:]:
            all_data[h] = {"x": [], "y": []}
        for row in reader:
            if not row:
                continue
            try:
                ebno = float(row[0])
            except Exception:
                continue
            for i, h in enumerate(headers[1:]):
                if i + 1 < len(row):
                    val = row[i + 1].strip()
                    if val == "":
                        continue
                    try:
                        v = float(val)
                    except Exception:
                        continue
                    if not np.isfinite(v) or v <= 0.0:
                        continue
                    all_data[h]["x"].append(ebno)
                    all_data[h]["y"].append(v)

    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
    has_data = False

    for key, data in all_data.items():
        if len(data["y"]) > 0:
            x_data = data["x"]
            y_data = data["y"]
            has_data = True
            if "Uncoded" in key:
                ax.semilogy(x_data, y_data, "k--", label=key)
            elif "Concat" in key:
                ax.semilogy(x_data, y_data, linestyle="-.", label=key)
            elif "RS" in key:
                ax.semilogy(x_data, y_data, linestyle=":", label=key)
            elif "LDPC" in key:
                ax.semilogy(x_data, y_data, linestyle="-", linewidth=2, label=key)
            else:
                ax.semilogy(x_data, y_data, marker="o", markersize=3, label=key)

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Eb/No (dB)", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)

    if has_data:
        ax.legend(fontsize=6, loc="center left", bbox_to_anchor=(1, 0.5), ncol=2)

    all_positive = []
    for key, data in all_data.items():
        for y in data["y"]:
            if y > 0 and np.isfinite(y):
                all_positive.append(y)

    # Calculate theoretical floor based on MAX_FRAMES
    # FER floor: ~1/MAX_FRAMES (1 frame error in all frames)
    # BER floor: much lower since errors spread across many bits
    if chart_type.lower() == "fer":
        theoretical_floor = 1.0 / MAX_FRAMES
    else:  # BER - can go much lower
        theoretical_floor = 1e-10

    if all_positive:
        ymin = min(all_positive)
        ybottom = max(ymin / 10.0, theoretical_floor)
    else:
        ybottom = theoretical_floor

    ax.set_ylim([ybottom, 1])
    ax.set_xlim([0.0, 12.0])
    ax.set_xticks(np.arange(0, 13, 1))

    # Add subtitle with MAX_FRAMES info
    fig.text(
        0.5,
        0.02,
        f"Max Frames per Point: {MAX_FRAMES:,}",
        ha="center",
        fontsize=9,
        style="italic",
        color="gray",
    )

    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved: {out_path}")


generate_curve(
    "data/ber_simulation_results.csv",
    "data/high_res_ber_curve.png",
    "BER vs Eb/No for CCSDS FEC Standards over AWGN",
    "Bit Error Rate (BER)",
    "ber",
)
generate_curve(
    "data/fer_simulation_results.csv",
    "data/high_res_fer_curve.png",
    "FER vs Eb/No for CCSDS FEC Standards over AWGN",
    "Frame Error Rate (FER)",
    "fer",
)
