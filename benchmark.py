import os
import cv2
import numpy as np
import time
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.stats import ttest_1samp, ttest_ind, t
import math

from detector.onnx_detector import ONNXDetector
from detector.yolo_detector import YOLODetector  # ← 반드시 YOLODetector 정의되어 있어야 함

# --------------------------------------------------
# Settings
# --------------------------------------------------
ONNX_MODEL_PATH = "./model/best.onnx"
YOLO_MODEL_PATH = "./model/best.pt"
IMG_PATH = "./image.jpg"

OUTPUT_DIR = "./result"
PLOT_DIR = "./result/plots"
os.makedirs(PLOT_DIR, exist_ok=True)

SAMPLE_SIZE = 30       # inference per batch
SAMPLE_COUNT = 100     # number of batches


# --------------------------------------------------
# Load Model & Image
# --------------------------------------------------
onnx_model = ONNXDetector(ONNX_MODEL_PATH)
yolo_model = YOLODetector(YOLO_MODEL_PATH)

img = cv2.imread(IMG_PATH)
if img is None:
    raise FileNotFoundError("IMG_PATH invalid")

# Warm-up
for _ in range(8):
    onnx_model.detect(img)
    yolo_model.detect(img)


# ==================================================
# ============ Benchmark Function ==================
# ==================================================
def run_benchmark(detector, name):
    print(f"\nRunning {name} Benchmark...")
    records = []
    all_times = []

    for batch in tqdm(range(SAMPLE_COUNT), desc=f"Benchmark {name}"):
        times = []
        for _ in range(SAMPLE_SIZE):
            start = time.perf_counter()
            detector.detect(img)
            end = time.perf_counter()
            times.append((end - start) * 1000)
            all_times.append(times[-1])

        avg_ms = np.mean(times)
        records.append({"batch": batch, "avg_inference_ms": avg_ms})

    df = pd.DataFrame(records)
    df.to_csv(f"{OUTPUT_DIR}/{name}_benchmark.csv", index=False)
    print(f"-> Saved {name} CSV to {OUTPUT_DIR}/{name}_benchmark.csv")
    return df


# --------------------------------------------------
# Run ONNX Benchmark
# --------------------------------------------------
onnx_df = run_benchmark(onnx_model, "onnx")
onnx_means = onnx_df["avg_inference_ms"].values


# --------------------------------------------------
# Run YOLO Benchmark
# --------------------------------------------------
yolo_df = run_benchmark(yolo_model, "yolo")
yolo_means = yolo_df["avg_inference_ms"].values


# ==================================================
# =========== Two-sample t-test (ONNX vs YOLO) =====
# ==================================================
print("\n\n========== ONNX vs YOLO Two-Sample t-test ==========")
print("H0: μ_ONNX = μ_YOLO")
print("H1: μ_ONNX < μ_YOLO")

t_stat, p_two_sided = ttest_ind(onnx_means, yolo_means, equal_var=False)

# Left-tailed test
if t_stat < 0:
    p_one_sided = p_two_sided / 2
else:
    p_one_sided = 1 - p_two_sided / 2

reject_H0 = p_one_sided < 0.05

ttest_result = {
    "onnx_mean": onnx_means.mean(),
    "yolo_mean": yolo_means.mean(),
    "t_statistic": t_stat,
    "p_value_one_sided(H1: onnx < yolo)": p_one_sided,
    "reject_H0_p<0.05": reject_H0
}

pd.DataFrame([ttest_result]).to_csv(f"{PLOT_DIR}/onnx_vs_yolo_ttest.csv", index=False)

print(f"ONNX mean: {onnx_means.mean():.4f} ms")
print(f"YOLO mean: {yolo_means.mean():.4f} ms")
print(f"t-stat: {t_stat:.5f}")
print(f"one-sided p: {p_one_sided:.6f}")

if reject_H0:
    print("Result: Reject H0 → ONNX significantly faster.")
else:
    print("Result: Fail to reject H0 → No significant difference.")


# ==================================================
# =============== Combined Plots ===================
# ==================================================

# ------- BOX PLOT -------
plt.figure(figsize=(10,6))
plt.boxplot([onnx_means, yolo_means], labels=["ONNX", "YOLO"])
plt.title("ONNX vs YOLO - Batch Mean Inference Time (Boxplot)")
plt.ylabel("Inference Time (ms)")
plt.grid(True, linestyle="--")
plt.savefig(f"{PLOT_DIR}/onnx_vs_yolo_boxplot.png")
plt.close()


# ------- HISTOGRAM -------
plt.figure(figsize=(10,6))
plt.hist(onnx_means, bins=30, color='skyblue', alpha=0.6, label='ONNX')
plt.hist(yolo_means, bins=30, color='salmon', alpha=0.6, label='YOLO')
plt.title("ONNX vs YOLO - Batch Mean Inference Time Distribution")
plt.xlabel("Inference Time (ms)")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True, linestyle="--")
plt.savefig(f"{PLOT_DIR}/onnx_vs_yolo_histogram.png")
plt.close()


# ------- LINE PLOT -------
plt.figure(figsize=(12,6))
plt.plot(onnx_means, label="ONNX", alpha=0.8)
plt.plot(yolo_means, label="YOLO", alpha=0.8)
plt.title("ONNX vs YOLO - Batch Mean Inference Time")
plt.xlabel("Batch Index")
plt.ylabel("Avg Inference Time (ms)")
plt.legend()
plt.grid(True, linestyle="--")
plt.savefig(f"{PLOT_DIR}/onnx_vs_yolo_lineplot.png")
plt.close()

print("\nSaved comparison plots to:", PLOT_DIR)
print("Benchmark Completed.\n")
