<p align="center">
  <b>matcha</b>
</p>

<h3 align="center">Energy observability for AI workloads</h3>

<div align="center">

[![PyPI version](https://img.shields.io/pypi/v/matcha-gpu?color=4ade80)](https://pypi.org/project/usematcha/)
[![PyPI Downloads](https://static.pepy.tech/badge/matcha-gpu/month)](https://pepy.tech/projects/usematcha)
[![License](https://img.shields.io/badge/license-Apache%202.0-4ade80)](https://opensource.org/licenses/Apache-2.0)

</div>

<p align="center">
  Measure GPU energy consumption of any training run. Zero overhead. Zero code changes.
</p>

---

## Install

```bash
pip install usematcha
```

Requires an NVIDIA GPU with drivers installed.

## Quick Start

Prefix your training command with `matcha run`:

```bash
matcha run torchrun --standalone --nproc_per_node=1 train_gpt.py
```

Your training runs at full speed. Matcha appends one line at the end:

```
matcha_energy gpus:NVIDIA H100 80GB HBM3 total:364722J (101.31Wh) duration:746.0s avg_power:489W peak_power:700W samples:7449
```

No code changes. No config files. Works with any training script.

## Commands

### `matcha run` - Total energy, zero overhead

Launches your command, polls GPU power in the background, prints a summary when it finishes. Your training runs natively - no stdout interception, no performance impact.

```bash
matcha run python train.py
matcha run torchrun --standalone --nproc_per_node=1 train_gpt.py
matcha run deepspeed --num_gpus=4 train.py --deepspeed ds_config.json
```

### `matcha wrap` - Per-step energy breakdown

Parses stdout for step markers (`step 10`, `iter 10`, `step:10/1000`, `[10/1000]`, etc.) and appends energy data to each step line.

```bash
matcha wrap torchrun --standalone --nproc_per_node=1 train_gpt.py
```

Output:

```
step:1/20000 train_loss:6.9357 train_time:438ms step_avg:438.01ms energy:106.7J/step avg_power:354W peak_power:427W
step:2/20000 train_loss:16.7414 train_time:833ms step_avg:416.60ms energy:154.0J/step avg_power:508W peak_power:533W
step:3/20000 train_loss:8.7524 train_time:1258ms step_avg:419.23ms energy:221.8J/step avg_power:551W peak_power:565W
...
matcha_energy gpus:NVIDIA H100 80GB HBM3 total:97271J (27.02Wh) duration:202.9s avg_power:479W peak_power:701W samples:2025
```

### `matcha monitor` - Live GPU power

```bash
matcha monitor
matcha monitor --gpus 0 --window 2.0
```

## Multi-GPU

Matcha auto-detects all GPUs and sums power across them. No flags needed.

```bash
# 8xH100 - automatically polls all 8 GPUs
matcha run torchrun --standalone --nproc_per_node=8 train_gpt.py

# Specific GPUs only
matcha run --gpus 0,1,2,3 torchrun ...

# Single GPU
matcha run --gpus 0 torchrun ...
```

## How It Works

Matcha runs a background thread that polls GPU power via NVML at 100ms intervals. Energy is computed using trapezoidal integration of instantaneous power readings. Your training process runs natively - Matcha never touches your stdout, your model, or your training loop.

## Tested On

- NVIDIA H100 80GB HBM3 - verified zero overhead across 4 benchmark modes
- Works with `torchrun`, `deepspeed`, `accelerate`, or plain `python`
- Compatible with PyTorch and any framework that runs on NVIDIA GPUs

## Why

```
10-minute H100 training run:
  Energy cost:   $0.01 (101 Wh @ $0.12/kWh)
  Compute cost:  $0.48 (RunPod @ $2.90/hr)

  → Compute is 48x the energy cost
  → Optimizing energy/step = faster training = less rental time
```

## Built by

[Keeya Labs](https://keeyalabs.com) · [Docs](https://usematcha.dev)

## License

Apache 2.0
