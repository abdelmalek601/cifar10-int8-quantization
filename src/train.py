import argparse, csv, os, random
import numpy as np
import torch, torch.nn as nn
import torchvision, torchvision.transforms as T

class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.quant = torch.ao.quantization.QuantStub()
        self.conv1, self.bn1, self.relu1 = nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU()
        self.conv2, self.bn2, self.relu2 = nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU()
        self.conv3, self.bn3, self.relu3 = nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(128 * 4 * 4, 10)
        self.dequant = torch.ao.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.pool(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool(self.relu3(self.bn3(self.conv3(x))))
        x = self.fc(torch.flatten(x, 1))
        return self.dequant(x)

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def get_loaders(bs=128):
    norm = T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    tr = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.ToTensor(), norm])
    te = T.Compose([T.ToTensor(), norm])
    train = torchvision.datasets.CIFAR10("data", train=True, download=True, transform=tr)
    test = torchvision.datasets.CIFAR10("data", train=False, download=True, transform=te)
    return (torch.utils.data.DataLoader(train, bs, shuffle=True, num_workers=2),
            torch.utils.data.DataLoader(test, 256, shuffle=False, num_workers=2))

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval(); correct = 0
    for x, y in loader:
        correct += (model(x.to(device)).argmax(1).cpu() == y).sum().item()
    return correct / len(loader.dataset)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=20)
    a = ap.parse_args()
    set_seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, test_loader = get_loaders()
    model = SmallCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    loss_fn = nn.CrossEntropyLoss()
    for ep in range(a.epochs):
        model.train()
        for x, y in train_loader:
            opt.zero_grad()
            loss_fn(model(x.to(device)), y.to(device)).backward()
            opt.step()
        sched.step()
        print(f"epoch {ep+1}/{a.epochs} done")
    acc = evaluate(model, test_loader, device)
    print(f"seed {a.seed} test accuracy: {acc:.4f}")
    os.makedirs("models", exist_ok=True); os.makedirs("results", exist_ok=True)
    torch.save(model.state_dict(), f"models/fp32_seed{a.seed}.pt")
    new = not os.path.exists("results/train_log.csv")
    with open("results/train_log.csv", "a", newline="") as f:
        w = csv.writer(f)
        if new: w.writerow(["seed", "epochs", "test_acc"])
        w.writerow([a.seed, a.epochs, acc])