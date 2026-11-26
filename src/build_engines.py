import os
import torch
import tensorrt as trt

os.makedirs("models/engines", exist_ok=True)

logger = trt.Logger(trt.Logger.WARNING)

def build_engine(seed, precision):

    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)

    path = f"models/onnx/fp32_seed{seed}.onnx"

    if not parser.parse_from_file(path):
        for i in range(parser.num_errors):
            print(parser.get_error(i))
        raise RuntimeError("ONNX parsing failed")

    config = builder.create_builder_config()

    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, 1 << 30
    )

    config.clear_flag(trt.BuilderFlag.TF32)

    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED

    profile = builder.create_optimization_profile()

    profile.set_shape(
        "input",
        (1, 3, 32, 32),
        (128, 3, 32, 32),
        (128, 3, 32, 32)
    )

    config.add_optimization_profile(profile)

    if precision == "FP16":
        config.set_flag(trt.BuilderFlag.FP16)

    print(f"Building seed {seed} | {precision}", flush=True)

    engine = builder.build_serialized_network(network, config)

    if engine is None:
        raise RuntimeError("Engine build failed")

    path = f"models/engines/{precision.lower()}_seed{seed}.engine"

    with open(path, "wb") as f:
        f.write(engine)

    print(f"SUCCESS | {precision} | {os.path.getsize(path)} bytes")


for seed in range(1, 5):
    for precision in ["FP32", "FP16"]:
        build_engine(seed, precision)