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

with open("results/cpu_latency_fused.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["seed", "batch_size", "model", "run", "latency_ms"])

    for seed in range(5):

        fp32 = SmallCNN().eval()
        fp32.load_state_dict(torch.load(
            f"models/fp32_seed{seed}.pt", map_location="cpu"))

        fused = fuse_modules(fp32, FUSE)

        int8 = fuse_modules(fp32, FUSE)
        int8.qconfig = get_default_qconfig("onednn")
        int8 = prepare(int8)

        with torch.no_grad():
            int8(torch.randn(1, 3, 32, 32))

        int8 = convert(int8)
        int8.load_state_dict(torch.load(
            f"models/int8_seed{seed}.pt", map_location="cpu"))

        models = [
            ("FP32", fp32),
            ("Fused FP32", fused),
            ("INT8", int8)
        ]

        for batch in [1, 128]:

            x = torch.randn(batch, 3, 32, 32)
            times = {name: [] for name, model in models}

            with torch.inference_mode():

                for _ in range(20):
                    for name, model in models:
                        model(x)

                for run in range(100):
                    for i in range(3):
                        name, model = models[(run + i) % 3]

                        start = time.perf_counter()
                        model(x)
                        ms = (time.perf_counter() - start) * 1000

                        times[name].append(ms)
                        writer.writerow([seed, batch, name, run, ms])

            print(f"\nSeed {seed} | Batch {batch}")

            for name, model in models:
                median = statistics.median(times[name])
                print(f"{name}: {median:.3f} ms")