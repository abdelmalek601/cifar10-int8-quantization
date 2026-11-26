
import csv
import statistics
import torch
import tensorrt as trt

logger = trt.Logger(trt.Logger.WARNING)
runtime = trt.Runtime(logger)

torch.cuda.init()
stream = torch.cuda.Stream()

with open("results/gpu_latency.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "seed", "precision", "batch_size",
        "run", "latency_ms"
    ])

    for seed in range(5):
        for batch in [1, 128]:

            for precision in ["fp32", "fp16", "int8"]:

                path = (
                    f"models/engines/"
                    f"{precision}_seed{seed}.engine"
                )

                with open(path, "rb") as file:
                    engine = runtime.deserialize_cuda_engine(
                        file.read()
                    )

                assert engine is not None

                context = engine.create_execution_context()

                assert context.set_input_shape(
                    "input", (batch, 3, 32, 32)
                )

                with torch.cuda.stream(stream):
                    x = torch.randn(
                        batch, 3, 32, 32,
                        device="cuda"
                    )

                    y = torch.empty(
                        tuple(context.get_tensor_shape("output")),
                        device="cuda",
                        dtype=torch.float32
                    )

                    assert context.set_tensor_address(
                        "input", x.data_ptr()
                    )
                    assert context.set_tensor_address(
                        "output", y.data_ptr()
                    )

                    # Warm-up
                    for _ in range(20):
                        assert context.execute_async_v3(
                            stream.cuda_stream
                        )

                stream.synchronize()

                times = []

                # Benchmark: 100 runs
                for run in range(100):

                    start = torch.cuda.Event(
                        enable_timing=True
                    )
                    end = torch.cuda.Event(
                        enable_timing=True
                    )

                    with torch.cuda.stream(stream):
                        start.record(stream)

                        assert context.execute_async_v3(
                            stream.cuda_stream
                        )

                        end.record(stream)

                    end.synchronize()

                    ms = start.elapsed_time(end)
                    times.append(ms)

                    writer.writerow([
                        seed, precision.upper(),
                        batch, run, ms
                    ])

                f.flush()

                median = statistics.median(times)
                p90 = sorted(times)[89]

                print(
                    f"Seed {seed} | "
                    f"{precision.upper()} | "
                    f"Batch {batch} | "
                    f"Median: {median:.4f} ms | "
                    f"P90: {p90:.4f} ms",
                    flush=True
                )

                del context, engine

print("GPU benchmark completed.")
