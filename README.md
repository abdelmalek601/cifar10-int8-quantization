\# INT8 Quantization of a Small CNN: CPU and GPU Trade-offs



\## Project Overview



This repository contains the source code, experimental results,

verification reports, and figures supporting a research paper

conducted for the Architecture of Computers course at

Vistula University.



The study investigates the effects of post-training INT8

quantization on classification accuracy, stored model size,

inference latency, and additional runtime memory.



\## Author



\- Abdelmalek Khadraoui

\- Vistula University

\- Architecture of Computers

\- November 2026



\## Experimental Hardware



\- CPU: Intel Core i9 (12th generation)

\- GPU: NVIDIA RTX A2000 Laptop GPU (8 GB)

\- Operating System: Windows



\## Software and Dependencies



\- Python

\- PyTorch

\- Torchvision

\- oneDNN

\- ONNX

\- NVIDIA TensorRT

\- NumPy

\- Matplotlib

\- psutil



The original dependency versions are recorded in:



\- `results/requirements\_cpu.txt`

\- `results/requirements\_gpu.txt`



Separate CPU and GPU Python environments were used.



\## Dataset and Model



\- Dataset: CIFAR-10

\- Training images: 50,000

\- Test images: 10,000

\- Image resolution: 32 x 32

\- Number of classes: 10

\- CNN architecture: three convolutional blocks

\- Output channels: 32, 64, 128

\- Training epochs: 20

\- Training batch size: 128

\- Optimizer: Adam

\- Initial learning rate: 0.001

\- Random seeds: 0, 1, 2, 3, 4

\- Calibration images: 500



\## Repository Structure



\- `src/` - Python source code

\- `results/` - Experimental results and verification reports

\- `figures/` - Experimental figures in PDF and PNG formats

\- `README.md` - Documentation and reproduction instructions

\- `.gitignore` - Excluded files and directories



The dataset, trained models, TensorRT engines, and virtual

environments are excluded from the repository.



\## Reproducing the Experiments



Run all commands from the repository root using Windows CMD.



\### Step 1: CPU Environment



Create and activate a separate Python environment.



Install the dependencies recorded in

`results/requirements\_cpu.txt`, using the appropriate

PyTorch CPU package index.



\### Step 2: Train the FP32 Models



```cmd

for %s in (0 1 2 3 4) do python src\\train.py --seed %s --epochs 20

```



The CIFAR-10 dataset is downloaded automatically.



\### Step 3: CPU Quantization and Benchmarking



```cmd

python src\\quantize.py

python src\\verify\_int8.py

python src\\benchmark\_cpu.py

python src\\benchmark\_fused.py

python src\\benchmark\_memory.py

```



\### Step 4: GPU Environment



Create a separate Python environment with CUDA-enabled

PyTorch and NVIDIA TensorRT.



Refer to `results/requirements\_gpu.txt` for the original

dependency versions.



A compatible NVIDIA GPU and driver are required.



\### Step 5: GPU Engine Generation



```cmd

python src\\export\_onnx.py

python src\\build\_engines.py

python src\\build\_int8.py

```



\### Step 6: GPU Verification and Benchmarking



```cmd

python src\\verify\_all\_gpu.py

python src\\gpu\_accuracy.py

python src\\gpu\_latency.py

python src\\gpu\_memory.py

```



\### Step 7: Statistical Analysis and Figures



```cmd

python src\\analyze\_results.py

python src\\create\_graphs.py

python src\\create\_cpu\_graphs.py

```



These scripts generate statistical summaries and

experimental figures.



The complete workflow has not yet been independently

tested from a clean repository clone.



\## Main Results



| Metric | CPU | GPU |

|---|---:|---:|

| FP32 accuracy | 82.300% | 82.300% |

| INT8 accuracy | 82.344% | 82.320% |

| INT8 speedup (batch 128) | 2.59x | 4.92x |

| Stored size reduction | 72.8% | 43.6% |

| Additional memory reduction (batch 128) | 68.0% | 33.9% |



Speedups and reductions are relative to the corresponding

standard FP32 baseline.



INT8 provided substantial latency improvements at batch

size 128 without an observed average accuracy loss.



However, fused FP32 was faster than INT8 on the CPU

at batch size 1.



\## Experimental Figures



\### CPU Inference Latency - Batch 1



!\[CPU Batch 1](figures/cpu\_latency\_batch1.png)



\### CPU Inference Latency - Batch 128



!\[CPU Batch 128](figures/cpu\_latency\_batch128.png)



\### GPU Inference Latency - Batch 128



!\[GPU Batch 128](figures/gpu\_latency\_batch128.png)



\### GPU Additional Runtime Memory - Batch 128



!\[GPU Memory](figures/gpu\_memory\_batch128.png)



All 11 experimental figures are available in the

`figures/` directory.



\## Execution Verification



CPU INT8 execution was checked through quantized operator

inspection and oneDNN runtime diagnostics.



GPU precision was checked using TensorRT engine inspection

and inference runs.



One INT8 engine retained FP32 execution in its final

fully connected layer.



Verification reports are available in:



\- `results/int8\_verification.txt`

\- `results/gpu\_verification/`



\## Limitations



\- One CNN architecture and one dataset.

\- Experiments conducted on a single laptop.

\- Five independently trained models.

\- GPU latency excludes host-device data transfers.

\- Runtime memory measurements include execution overhead.

\- Sampled memory measurements may miss temporary peaks.

\- CPU and GPU memory measurements use different methods.

\- One TensorRT INT8 engine uses mixed precision.

\- Results may vary with hardware and software versions.



\## Declaration of AI Assistance



Claude and ChatGPT were used to assist with experimental

planning, Python code development, debugging, statistical

analysis, and academic writing.



The author executed the experiments on the specified

hardware, collected the reported measurements, and

checked the generated outputs.



Responsibility for verifying references, interpreting

results, and ensuring the accuracy of the final submission

remains with the author.



\## Academic Purpose



This repository supports the reproducibility and

transparency of the associated research paper.

