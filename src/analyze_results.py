
import csv
import statistics as stats
from collections import defaultdict
from pathlib import Path

folder = Path("results")
report = []

def load(filename):
    with open(folder / filename, newline="") as f:
        return list(csv.DictReader(f))

def analyze(title, filename, groups, column,
            use_median=False, multiplier=1):

    data = defaultdict(lambda: defaultdict(list))

    for row in load(filename):
        key = tuple(row[g] for g in groups)
        seed = int(row["seed"])
        value = float(row[column]) * multiplier
        data[key][seed].append(value)

    report.append("\n" + title)

    for key, seeds in sorted(data.items()):
        values = []

        for seed in sorted(seeds):
            samples = seeds[seed]
            value = (stats.median(samples) if use_median
                     else stats.mean(samples))
            values.append(value)

        avg = stats.mean(values)
        sd = stats.stdev(values) if len(values) > 1 else 0

        line = (
            f"{' | '.join(key)}: "
            f"Mean = {avg:.4f}, SD = {sd:.4f}, "
            f"Seeds = {len(values)}"
        )
        report.append(line)

cpu = load("quant_accuracy_size.csv")

report.append("CPU ACCURACY (%)")
for name in ["fp32_acc", "int8_acc"]:
    values = [float(r[name]) * 100 for r in cpu]
    report.append(
        f"{name}: Mean = {stats.mean(values):.4f}, "
        f"SD = {stats.stdev(values):.4f}"
    )

analyze("GPU ACCURACY (%)",
        "gpu_accuracy_size.csv",
        ["precision"], "accuracy", multiplier=100)

analyze("CPU LATENCY (ms)",
        "cpu_latency_fused.csv",
        ["batch_size", "model"], "latency_ms",
        use_median=True)

analyze("GPU LATENCY (ms)",
        "gpu_latency.csv",
        ["batch_size", "precision"], "latency_ms",
        use_median=True)

analyze("CPU ADDITIONAL MEMORY (MiB)",
        "cpu_memory.csv",
        ["batch_size", "precision"], "additional_mib")

analyze("GPU ADDITIONAL MEMORY (MiB)",
        "gpu_memory.csv",
        ["batch_size", "precision"], "additional_mib")

text = "\n".join(report)
print(text)

(folder / "summary_statistics.txt").write_text(text)

print("\nAnalysis saved successfully.")
