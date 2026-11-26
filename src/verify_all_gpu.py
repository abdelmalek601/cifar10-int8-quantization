import os
import json
import torch
import tensorrt as trt
from collections import Counter

logger = trt.Logger(trt.Logger.WARNING)
runtime = trt.Runtime(logger)

torch.cuda.init()

os.makedirs("results/gpu_verification", exist_ok=True)

for seed in range(5):

    for precision in ["fp32", "fp16", "int8"]:

        path = f"models/engines/{precision}_seed{seed}.engine"

        with open(path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())

        assert engine is not None
        context = engine.create_execution_context()

        for batch in [1, 128]:

            assert context.set_input_shape(
                "input", (batch, 3, 32, 32)
            )

            x = torch.randn(
                batch, 3, 32, 32, device="cuda"
            )

            y = torch.empty(
                tuple(context.get_tensor_shape("output")),
                device="cuda",
                dtype=torch.float32
            )

            assert engine.get_tensor_dtype("input") == trt.float32
            assert engine.get_tensor_dtype("output") == trt.float32

            context.set_tensor_address("input", x.data_ptr())
            context.set_tensor_address("output", y.data_ptr())

            assert context.execute_async_v3(
                torch.cuda.current_stream().cuda_stream
            )

            torch.cuda.synchronize()

            inspector = engine.create_engine_inspector()
            inspector.execution_context = context

            info = inspector.get_engine_information(
                trt.LayerInformationFormat.JSON
            )

            filename = (
                f"results/gpu_verification/"
                f"{precision}_seed{seed}_batch{batch}.json"
            )

            with open(filename, "w") as f:
                f.write(info)

            layers = json.loads(info)["Layers"]

            types = Counter(
                layer["Weights"]["Type"]
                for layer in layers
                if "Weights" in layer
            )

            print(
                f"Seed {seed} | {precision.upper()} | "
                f"Batch {batch} | {dict(types)}"
            )