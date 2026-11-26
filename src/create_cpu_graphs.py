
import csv
import os
import statistics as stats
from collections import defaultdict

import matplotlib.pyplot as plt

os.makedirs("figures", exist_ok=True)


def read_csv(filename):
    with open("results/" + filename, newline="") as f:
        return list(csv.DictReader(f))


def calculate(filename, column, labels, label_field,
              batch=None, use_median=False, factor=1):

    data = defaultdict(lambda: defaultdict(list))

    for row in read_csv(filename):

        if batch is not None:
            if int(row["batch_size"]) != batch:
                continue

        label = row[label_field]
        seed = int(row["seed"])

        data[label][seed].append(
            float(row[column]) * factor
        )

    means = []
    deviations = []

    for label in labels:
        values = []

        for seed in sorted(data[label]):
            samples = data[label][seed]

            if use_median:
                values.append(stats.median(samples))
            else:
                values.append(stats.mean(samples))

        means.append(stats.mean(values))
        deviations.append(stats.stdev(values))

    return means, deviations


def make_graph(name, title, ylabel, labels,
               means, deviations, accuracy=False):

    plt.figure(figsize=(7, 4.5))

    if accuracy:
        plt.errorbar(
            labels, means,
            yerr=deviations,
            fmt="o",
            capsize=5
        )
        plt.ylim(81.5, 83.3)

    else:
        plt.bar(
            labels, means,
            yerr=deviations,
            capsize=5
        )

        plt.ylim(
            0,
            max(m + s for m, s in
                zip(means, deviations)) * 1.25
        )

    plt.title(title)
    plt.ylabel(ylabel)
    plt.grid(axis="y", alpha=0.25)
    plt.gca().set_axisbelow(True)
    plt.tight_layout()

    plt.savefig("figures/" + name + ".png", dpi=300)
    plt.savefig("figures/" + name + ".pdf")
    plt.close()

    print("Saved:", name)


# Figure 1: CPU Accuracy

cpu = read_csv("quant_accuracy_size.csv")

labels = ["FP32", "INT8"]

means = [
    stats.mean(float(r["fp32_acc"]) * 100 for r in cpu),
    stats.mean(float(r["int8_acc"]) * 100 for r in cpu)
]

deviations = [
    stats.stdev(float(r["fp32_acc"]) * 100 for r in cpu),
    stats.stdev(float(r["int8_acc"]) * 100 for r in cpu)
]

make_graph(
    "cpu_accuracy",
    "CPU Classification Accuracy",
    "Accuracy (%)",
    labels, means, deviations,
    accuracy=True
)


# Figure 2: CPU Model Size

means = [
    stats.mean(float(r["fp32_bytes"]) / 1024 for r in cpu),
    stats.mean(float(r["int8_bytes"]) / 1024 for r in cpu)
]

deviations = [
    stats.stdev(float(r["fp32_bytes"]) / 1024 for r in cpu),
    stats.stdev(float(r["int8_bytes"]) / 1024 for r in cpu)
]

make_graph(
    "cpu_model_size",
    "Stored CPU Model Size",
    "Size (KiB)",
    labels, means, deviations
)


# Figures 3-4: CPU Latency

models = ["FP32", "Fused FP32", "INT8"]

for batch in [1, 128]:

    means, deviations = calculate(
        "cpu_latency_fused.csv",
        "latency_ms",
        models,
        "model",
        batch=batch,
        use_median=True
    )

    make_graph(
        f"cpu_latency_batch{batch}",
        f"CPU Inference Latency - Batch {batch}",
        "Latency (ms)",
        models, means, deviations
    )


# Figures 5-6: CPU Memory

for batch in [1, 128]:

    means, deviations = calculate(
        "cpu_memory.csv",
        "additional_mib",
        models,
        "precision",
        batch=batch
    )

    make_graph(
        f"cpu_memory_batch{batch}",
        f"CPU Additional Memory - Batch {batch}",
        "Memory (MiB)",
        models, means, deviations
    )


print("\nAll CPU figures generated successfully!")
