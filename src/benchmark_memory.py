import torch
import psutil
import csv
import sys
import subprocess
import threading
import json
import gc

from torch.ao.quantization import fuse_modules, get_default_qconfig, prepare, convert
from train import SmallCNN
from quantize import FUSE


def measure(seed, batch, precision):

    torch.set_num_threads(14)
    torch.manual_seed(123)
    torch.backends.quantized.engine = "onednn"

    process = psutil.Process()

    def memory():
        return process.memory_info().rss / (1024 ** 2)

    gc.collect()
    baseline = memory()

    model = SmallCNN().eval()

    if precision == "INT8":
        model = fuse_modules(model, FUSE)
        model.qconfig = get_default_qconfig("onednn")
        model = prepare(model)

        with torch.no_grad():
            model(torch.randn(1, 3, 32, 32))

        model = convert(model)
        model.load_state_dict(torch.load(
            f"models/int8_seed{seed}.pt", map_location="cpu"))

    else:
        model.load_state_dict(torch.load(
            f"models/fp32_seed{seed}.pt", map_location="cpu"))

        if precision == "Fused FP32":
            model = fuse_modules(model, FUSE)

    model.eval()
    gc.collect()

    loaded = memory()
    peak = [loaded]
    stop = threading.Event()

    def monitor():
        while not stop.is_set():
            peak[0] = max(peak[0], memory())
            stop.wait(0.002)

    x = torch.randn(batch, 3, 32, 32)

    thread = threading.Thread(target=monitor)
    thread.start()

    try:
        with torch.inference_mode():
            for _ in range(20):
                model(x)

            runs = 1000 if batch == 1 else 100

            for _ in range(runs):
                model(x)

    finally:
        peak[0] = max(peak[0], memory())
        stop.set()
        thread.join()

    return [
        seed, batch, precision,
        baseline, loaded, peak[0],
        peak[0] - baseline
    ]


if __name__ == "__main__":

    if len(sys.argv) > 1:
        result = measure(
            int(sys.argv[1]),
            int(sys.argv[2]),
            sys.argv[3]
        )
        print(json.dumps(result))

    else:
        with open("results/cpu_memory.csv", "w", newline="") as f:
            writer = csv.writer(f)

            writer.writerow([
                "seed", "batch_size", "precision",
                "baseline_mib", "loaded_mib",
                "peak_rss_mib", "additional_mib"
            ])

            with open("results/memory_warnings.txt", "w") as log:
                for seed in range(5):
                    for batch in [1, 128]:
                        for precision in ["FP32", "Fused FP32", "INT8"]:

                            output = subprocess.run(
                                [sys.executable, __file__,
                                 str(seed), str(batch), precision],
                                stdout=subprocess.PIPE,
                                stderr=log,
                                text=True,
                                check=True
                            )

                            row = json.loads(
                                output.stdout.strip().splitlines()[-1]
                            )

                            writer.writerow(row)
                            f.flush()

                            print(
                                f"Seed {seed} | Batch {batch} | "
                                f"{precision} | Peak: {row[5]:.2f} MiB | "
                                f"Additional: {row[6]:.2f} MiB"
                            )