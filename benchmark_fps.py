import time
import torch
from ultralytics import YOLO

WEIGHT = "/home/jiao/users/xiexy/ultralytics-main/GC10_RETRAIN_FOR_PAPER/final_weights/gc10_lcrb_rcsfusion_map50_7418_seed0_best.pt"

DEVICE = "cuda:0"
IMG_SIZE = 640
WARMUP = 100
ITERS = 1000

# -----------------------------
# Load model
# -----------------------------
model = YOLO(WEIGHT)
net = model.model.to(DEVICE)

net.eval()
net.float()  # FP32

# Print model information
model.info()

# Dummy input: batch=1, 3x640x640
x = torch.randn(
    1, 3, IMG_SIZE, IMG_SIZE,
    device=DEVICE,
    dtype=torch.float32
)

# -----------------------------
# Warm-up
# -----------------------------
print("\nWarming up...")

with torch.inference_mode():
    for _ in range(WARMUP):
        _ = net(x)

torch.cuda.synchronize()

# -----------------------------
# Benchmark
# -----------------------------
print("Benchmarking...")

times = []

with torch.inference_mode():
    for _ in range(ITERS):
        torch.cuda.synchronize()
        start = time.perf_counter()

        _ = net(x)

        torch.cuda.synchronize()
        end = time.perf_counter()

        times.append((end - start) * 1000)

# -----------------------------
# Results
# -----------------------------
avg_latency = sum(times) / len(times)
fps = 1000.0 / avg_latency

print("\n========== FPS BENCHMARK ==========")
print(f"GPU:       {torch.cuda.get_device_name(0)}")
print(f"Precision: FP32")
print(f"Input:     1 x 3 x {IMG_SIZE} x {IMG_SIZE}")
print(f"Warmup:    {WARMUP}")
print(f"Iterations:{ITERS}")
print(f"Latency:   {avg_latency:.3f} ms/image")
print(f"FPS:       {fps:.2f}")
print("===================================")