import os
import torch
import torchvision
import torchvision.transforms as T
import tensorrt as trt
from torch.utils.data import DataLoader, Subset

logger = trt.Logger(trt.Logger.WARNING)

tf = T.Compose([
    T.ToTensor(),
    T.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2470, 0.2435, 0.2616)
    )
])

train = torchvision.datasets.CIFAR10(
    "data", train=True, download=True, transform=tf
)

idx = torch.randperm(
    len(train),
    generator=torch.Generator().manual_seed(123)
)[:500].tolist()

loader = DataLoader(
    Subset(train, idx), batch_size=100
)


class Calibrator(trt.IInt8EntropyCalibrator2):

    def __init__(self, seed):
        super().__init__()
        self.seed = seed
        self.data = iter(loader)
        self.buffer = torch.empty(
            (100, 3, 32, 32),
            dtype=torch.float32,
            device="cuda"
        )

    def get_batch_size(self):
        return 1

    def get_batch(self, names):
        try:
            x, _ = next(self.data)
            self.buffer.copy_(x)
            torch.cuda.synchronize()
            return [int(self.buffer.data_ptr())]
        except StopIteration:
            return None

    def read_calibration_cache(self):
        return None

    def write_calibration_cache(self, cache):
        path = f"models/engines/int8_seed{self.seed}.cache"
        with open(path, "wb") as f:
            f.write(cache)


def build_int8(seed):

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
    config.set_flag(trt.BuilderFlag.INT8)

    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED

    profile = builder.create_optimization_profile()

    profile.set_shape(
        "input",
        (1, 3, 32, 32),
        (128, 3, 32, 32),
        (128, 3, 32, 32)
    )

    config.add_optimization_profile(profile)

    calibration_profile = builder.create_optimization_profile()

    calibration_profile.set_shape(
        "input",
        (100, 3, 32, 32),
        (100, 3, 32, 32),
        (100, 3, 32, 32)
    )

    config.set_calibration_profile(calibration_profile)

    calibrator = Calibrator(seed)
    config.int8_calibrator = calibrator

    print(f"Building seed {seed} | INT8", flush=True)

    engine = builder.build_serialized_network(network, config)

    if engine is None:
        raise RuntimeError("Engine build failed")

    path = f"models/engines/int8_seed{seed}.engine"

    with open(path, "wb") as f:
        f.write(engine)

    print(f"SUCCESS | INT8 | {os.path.getsize(path)} bytes")


os.makedirs("models/engines", exist_ok=True)

for seed in range(5):
    path = f"models/engines/int8_seed{seed}.engine"

    if os.path.exists(path):
        print(f"Seed {seed}: already exists, skipping")
    else:
        build_int8(seed)