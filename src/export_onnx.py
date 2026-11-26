import torch
import onnx
import os
from train import SmallCNN

os.makedirs("models/onnx", exist_ok=True)

for seed in range(5):

    model = SmallCNN().eval()

    model.load_state_dict(torch.load(
        f"models/fp32_seed{seed}.pt",
        map_location="cpu"
    ))

    x = torch.randn(1, 3, 32, 32)

    path = f"models/onnx/fp32_seed{seed}.onnx"

    torch.onnx.export(
        model,
        x,
        path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch"},
            "output": {0: "batch"}
        },
        opset_version=17,
        dynamo=False
    )

    onnx.checker.check_model(onnx.load(path))

    print(f"Seed {seed}: ONNX export successful")