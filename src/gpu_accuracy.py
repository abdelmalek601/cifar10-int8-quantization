import os
import csv
import torch
import torchvision
import torchvision.transforms as T
import tensorrt as trt
from torch.utils.data import DataLoader

logger = trt.Logger(trt.Logger.WARNING)
runtime = trt.Runtime(logger)

stream = torch.cuda.Stream()

tf = T.Compose([
    T.ToTensor(),
    T.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2470, 0.2435, 0.2616)
    )
])

test = torchvision.datasets.CIFAR10(
    "data", train=False, download=True, transform=tf
)

loader = DataLoader(test, batch_size=128, shuffle=False)


def evaluate_engine(engine):

    context = engine.create_execution_context()
    correct = 0

    assert engine.get_tensor_dtype("input") == trt.float32
    assert engine.get_tensor_dtype("output") == trt.float32

    with torch.inference_mode():

        for x, labels in loader:

            with torch.cuda.stream(stream):

                x = x.to("cuda").contiguous()

                assert context.set_input_shape(
                    "input", tuple(x.shape)
                )

                y = torch.empty(
                    tuple(context.get_tensor_shape("output")),
                    device="cuda",
                    dtype=torch.float32
                )

                context.set_tensor_address("input", x.data_ptr())
                context.set_tensor_address("output", y.data_ptr())

                assert context.execute_async_v3(
                    stream.cuda_stream
                )

                predictions = y.argmax(dim=1)

            stream.synchronize()

            correct += (
                predictions.cpu() == labels
            ).sum().item()

    return correct / len(test)


with open("results/gpu_accuracy_size.csv", "w", newline="") as f:

    writer = csv.writer(f)

    writer.writerow([
        "seed", "precision", "accuracy", "engine_bytes"
    ])

    for seed in range(5):

        for precision in ["fp32", "fp16", "int8"]:

            path = f"models/engines/{precision}_seed{seed}.engine"

            with open(path, "rb") as file:
                engine = runtime.deserialize_cuda_engine(file.read())

            assert engine is not None

            accuracy = evaluate_engine(engine)
            size = os.path.getsize(path)

            writer.writerow([
                seed, precision.upper(), accuracy, size
            ])

            f.flush()

            print(
                f"Seed {seed} | {precision.upper()} | "
                f"Accuracy: {accuracy * 100:.2f}% | "
                f"Size: {size} bytes"
            )

            del engine