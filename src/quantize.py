import csv, os
import torch, torchvision, torchvision.transforms as T
from torch.utils.data import DataLoader, Subset
from torch.ao.quantization import fuse_modules, get_default_qconfig, prepare, convert
from train import SmallCNN, evaluate

torch.backends.quantized.engine = "onednn"

FUSE = [["conv1", "bn1", "relu1"], ["conv2", "bn2", "relu2"], ["conv3", "bn3", "relu3"]]

if __name__ == "__main__":
    tf = T.Compose([T.ToTensor(),
                    T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])
    train = torchvision.datasets.CIFAR10("data", train=True, download=True, transform=tf)
    test = torchvision.datasets.CIFAR10("data", train=False, download=True, transform=tf)

    # calibration: 500 random TRAIN images (never test images)
    idx = torch.randperm(len(train), generator=torch.Generator().manual_seed(123))[:500].tolist()
    calib_loader = DataLoader(Subset(train, idx), batch_size=100)
    test_loader = DataLoader(test, batch_size=256, shuffle=False)

    os.makedirs("results", exist_ok=True)
    rows = []
    for seed in range(5):
        fp32 = SmallCNN()
        fp32.load_state_dict(torch.load(f"models/fp32_seed{seed}.pt", map_location="cpu"))
        fp32.eval()

        fused = fuse_modules(fp32, FUSE)               # merge conv+bn+relu
        fused.qconfig = get_default_qconfig("onednn")  # int8 settings
        prepared = prepare(fused)                      # insert observers
        with torch.no_grad():
            for x, _ in calib_loader:                  # calibration pass
                prepared(x)
        int8 = convert(prepared)                       # real INT8 model

        torch.save(int8.state_dict(), f"models/int8_seed{seed}.pt")
        acc32 = evaluate(fp32, test_loader, "cpu")
        acc8 = evaluate(int8, test_loader, "cpu")
        s32 = os.path.getsize(f"models/fp32_seed{seed}.pt")
        s8 = os.path.getsize(f"models/int8_seed{seed}.pt")
        rows.append([seed, acc32, acc8, acc32 - acc8, s32, s8])
        print(f"seed {seed}: fp32={acc32:.4f} int8={acc8:.4f} size {s32} -> {s8} bytes")

    with open("results/quant_accuracy_size.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "fp32_acc", "int8_acc", "acc_drop", "fp32_bytes", "int8_bytes"])
        w.writerows(rows)