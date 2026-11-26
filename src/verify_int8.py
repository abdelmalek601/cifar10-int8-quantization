import torch
from torch.ao.quantization import fuse_modules, get_default_qconfig, prepare, convert
from torch.profiler import profile, ProfilerActivity
from train import SmallCNN
from quantize import FUSE

torch.backends.quantized.engine = "onednn"

model = SmallCNN().eval()
model = fuse_modules(model, FUSE)
model.qconfig = get_default_qconfig("onednn")
model = prepare(model)

x = torch.randn(1, 3, 32, 32)

with torch.no_grad():
    model(x)

model = convert(model)
model.load_state_dict(torch.load("models/int8_seed0.pt", map_location="cpu"))
model.eval()

print("Backend:", torch.backends.quantized.engine)
print("Weight dtype:", model.conv1.weight().dtype)
print(model)

with torch.no_grad():
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        with torch.backends.mkldnn.verbose(torch.backends.mkldnn.VERBOSE_ON):
            model(x)

print(prof.key_averages().table(sort_by="cpu_time_total"))