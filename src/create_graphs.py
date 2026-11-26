
import csv
import os
import statistics as stats
from collections import defaultdict

import matplotlib.pyplot as plt

os.makedirs("figures", exist_ok=True)

precisions = ["FP32", "FP16", "INT8"]


def read_csv(filename):
    with open("results/" + filename, newline="") as f:
        return list(csv.DictReader(f))


def calculate(filename, column, batch=None, latency=False):
    rows = read_csv(filename)
    grouped = defaultdict(list)

    for row in rows:
        if batch is not None:
            if int(row["batch_size"]) != batch:
                continue

        seed = int(row["seed"])
        precision = row["precision"].upper()

        grouped[(precision, seed)].append(float(row[column]))

    means = []
    deviations = []

    for precision in precisions:
        per_seed = []

        for seed in range(5):
            values = grouped[(precision, seed)]

            if latency:
                value = stats.median(values)
            else:
                value = stats.mean(values)

            per_seed.append(value)

        means.append(stats.mean(per_seed))
        deviations.append(stats.stdev(per_seed))

    return means, deviations


def save_figure(name):
    plt.tight_layout()
    plt.savefig("figures/" + name + ".png", dpi=300)
    plt.savefig("figures/" + name + ".pdf")
    plt.close()
    print("Saved:", name)


def bar_chart(name, title, ylabel, means, deviations):
    plt.figure(figsize=(7, 4.5))

    bars = plt.bar(
        precisions, means,
        yerr=deviations,
        capsize=5
    )

    plt.title(title)
    plt.ylabel(ylabel)
    plt.ylim(0, max(
        m + s for m, s in zip(means, deviations)
    ) * 1.25)

    plt.grid(axis="y", alpha=0.25)
    plt.gca().set_axisbelow(True)

    for bar, value in zip(bars, means):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(means) * 0.06,
            f"{value:.3f}",
            ha="center"
        )

    save_figure(name)


# Figure 1: GPU Accuracy
means, deviations = calculate(
    "gpu_accuracy_size.csv", "accuracy"
)

means = [v * 100 for v in means]
deviations = [v * 100 for v in deviations]

plt.figure(figsize=(7, 4.5))
plt.errorbar(
    precisions, means,
    yerr=deviations,
    fmt="o",
    capsize=5
)
plt.title("GPU Classification Accuracy")
plt.ylabel("Accuracy (%)")
plt.ylim(81.5, 83.3)
plt.grid(axis="y", alpha=0.25)

save_figure("gpu_accuracy")


# Figures 2-3: GPU Latency
for batch in [1, 128]:
    means, deviations = calculate(
        "gpu_latency.csv",
        "latency_ms",
        batch=batch,
        latency=True
    )

    bar_chart(
        f"gpu_latency_batch{batch}",
        f"GPU Inference Latency - Batch {batch}",
        "Latency (ms)",
        means, deviations
    )


# Figures 4-5: GPU Memory
for batch in [1, 128]:
    means, deviations = calculate(
        "gpu_memory.csv",
        "additional_mib",
        batch=batch
    )

    bar_chart(
        f"gpu_memory_batch{batch}",
        f"GPU Additional Memory - Batch {batch}",
        "Memory (MiB)",
        means, deviations
    )

print("\nAll GPU figures generated successfully!")
