
import csv
import json
import os
import subprocess
import sys


def measure(seed, precision, batch):
    import torch
    import tensorrt as trt

    torch.cuda.init()
    stream = torch.cuda.Stream()
    stream.synchronize()

    def memory_mib():
        free, total = torch.cuda.mem_get_info()
        return (total - free) / (1024 ** 2)

    baseline = memory_mib()

    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)

    path = f"models/engines/{precision}_seed{seed}.engine"

    with open(path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())

    if engine is None:
        raise RuntimeError("Engine loading failed")

    context = engine.create_execution_context()

    if not context.set_input_shape(
        "input", (batch, 3, 32, 32)
    ):
        raise RuntimeError("Invalid input shape")

    loaded = memory_mib()

    with torch.cuda.stream(stream):
        x = torch.randn(
            batch, 3, 32, 32, device="cuda"
        )

        y = torch.empty(
            tuple(context.get_tensor_shape("output")),
            device="cuda",
            dtype=torch.float32
        )

        context.set_tensor_address("input", x.data_ptr())
        context.set_tensor_address("output", y.data_ptr())

    stream.synchronize()

    peak = max(loaded, memory_mib())

    for run in range(40):
        with torch.cuda.stream(stream):
            ok = context.execute_async_v3(
                stream.cuda_stream
            )

        if not ok:
            raise RuntimeError("Inference failed")

        stream.synchronize()
        peak = max(peak, memory_mib())

    return [
        seed,
        precision.upper(),
        batch,
        round(baseline, 2),
        round(loaded, 2),
        round(peak, 2),
        round(peak - baseline, 2)
    ]


if __name__ == "__main__":

    if len(sys.argv) > 1:
        result = measure(
            int(sys.argv[1]),
            sys.argv[2],
            int(sys.argv[3])
        )
        print(json.dumps(result))

    else:
        os.makedirs("results", exist_ok=True)

        with open(
            "results/gpu_memory.csv",
            "w", newline=""
        ) as f:

            writer = csv.writer(f)

            writer.writerow([
                "seed", "precision", "batch_size",
                "baseline_mib", "loaded_mib",
                "peak_observed_mib", "additional_mib"
            ])

            for seed in range(5):
                for batch in [1, 128]:
                    for precision in ["fp32", "fp16", "int8"]:

                        p = subprocess.run(
                            [
                                sys.executable,
                                __file__,
                                str(seed),
                                precision,
                                str(batch)
                            ],
                            capture_output=True,
                            text=True
                        )

                        if p.returncode != 0:
                            print(p.stderr)
                            raise RuntimeError(
                                "Memory measurement failed"
                            )

                        row = json.loads(
                            p.stdout.strip().splitlines()[-1]
                        )

                        writer.writerow(row)
                        f.flush()

                        print(
                            f"Seed {seed} | "
                            f"{precision.upper()} | "
                            f"Batch {batch} | "
                            f"Additional: {row[6]:.2f} MiB",
                            flush=True
                        )

        print("GPU memory benchmark completed.")
