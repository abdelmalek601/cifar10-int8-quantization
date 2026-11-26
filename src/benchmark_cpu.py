import torch
import time
import csv
import statistics
from torch.ao.quantization import fuse_modules, get_default_qconfig, prepare, convert
from train import SmallCNN
from quantize import FUSE

torch.backends.quantized.engine = "onednn"
torch.set_num_threads(14)
torch.manual_seed(123)

with open("results/cpu_latency.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["seed", "batch_size", "precision", "run", "latency_ms"])

    for seed in range(5):
        fp32 = SmallCNN().eval()
        fp32.load_state_dict(torch.load(
            f"models/fp32_seed{seed}.pt", map_location="cpu"))

        int8 = fuse_modules(fp32, FUSE)
        int8.qconfig = get_default_qconfig("onednn")
        int8 = prepare(int8)

        with torch.no_grad():
            int8(torch.randn(1, 3, 32, 32))

        int8 = convert(int8)
        int8.load_state_dict(torch.load(
            f"models/int8_seed{seed}.pt", map_location="cpu"))

        for batch_size in [1, 128]:
            x = torch.randn(batch_size, 3, 32, 32)

            times32 = []
            times8 = []

            with torch.inference_mode():

                # Warm-up
                for _ in range(20):
                    fp32(x)
                    int8(x)

                # Benchmark
                for run in range(100):
                    start = time.perf_counter()
                    fp32(x)
                    t32 = (time.perf_counter() - start) * 1000

                    start = time.perf_counter()
                    int8(x)
                    t8 = (time.perf_counter() - start) * 1000

                    times32.append(t32)
                    times8.append(t8)

                    writer.writerow([seed, batch_size, "FP32", run, t32])
                    writer.writerow([seed, batch_size, "INT8", run, t8])

            m32 = statistics.median(times32)
            m8 = statistics.median(times8)

            print(f"Seed {seed} | Batch {batch_size}")
            print(f"FP32: {m32:.3f} ms")
            print(f"INT8: {m8:.3f} ms")
            print(f"Speedup: {m32 / m8:.2f}x")
            print()