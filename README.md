<p align="center">
  <b>matcha</b>
</p>

<h3 align="center">GPU energy observability for AI training</h3>

<div align="center">

[![PyPI version](https://img.shields.io/pypi/v/usematcha?color=4ade80)](https://pypi.org/project/usematcha/)
[![Python versions](https://img.shields.io/pypi/pyversions/usematcha?color=4ade80)](https://pypi.org/project/usematcha/)
[![PyPI Downloads](https://static.pepy.tech/badge/usematcha/month)](https://pepy.tech/projects/usematcha)
[![License](https://img.shields.io/badge/license-Apache%202.0-4ade80)](https://opensource.org/licenses/Apache-2.0)

</div>

<p align="center">
  Measure GPU energy per training run and per step. No code changes. Structured output for any observability stack.
</p>

---

## Install

```bash
pip install usematcha
```

Requires Linux + an NVIDIA GPU with drivers installed. Python 3.9+.

## Quick start

Prefix your training command with `matcha run`:

```bash
matcha run torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Your training runs at full speed. matcha prints one line at the end:

```
matcha_energy gpus:8x NVIDIA H100 80GB HBM3 total:778168J (216.16Wh) duration:203.1s avg_power:3832W peak_power:4120W samples:2031
```

No code changes. No config files. Works with any training script.

---

## Commands

### `matcha run` — total energy, zero overhead

Launches your command, polls GPU power in the background, prints a summary when it finishes. Training runs natively — no stdout interception, no performance impact.

```bash
matcha run python train.py
matcha run torchrun --standalone --nproc_per_node=8 train_gpt.py
matcha run deepspeed --num_gpus=8 train.py --deepspeed ds_config.json
```

### `matcha wrap` — per-step energy

Parses your training stdout for step markers (`step 10`, `iter 10`, `step:10/1000`, `[10/1000]`, etc.) and appends energy data to each step line.

```bash
matcha wrap torchrun --standalone --nproc_per_node=8 train_gpt.py
```

```
step:1/20000 train_loss:6.9357 train_time:612ms step_avg:612.00ms energy:2354.0J/step avg_power:3847W peak_power:4120W
step:2/20000 train_loss:16.7414 train_time:831ms step_avg:721.50ms energy:3012.6J/step avg_power:3625W peak_power:3998W
step:3/20000 train_loss:8.7524 train_time:1258ms step_avg:783.40ms energy:3472.8J/step avg_power:3610W peak_power:3890W
...
matcha_energy gpus:8x NVIDIA H100 80GB HBM3 total:778168J (216.16Wh) duration:203.1s avg_power:3832W peak_power:4120W samples:2031
```

### `matcha monitor` — live per-GPU dashboard

A drop-in replacement for running `watch nvidia-smi` in a second terminal. Shows per-GPU power, utilization, temperature, memory, and a running total.

```bash
matcha monitor
matcha monitor --gpus 0,1,2,3 --interval 500
```

---

## Structured output for observability

matcha emits JSONL records (`session_start`, `step`, `session_end`) ready to stream into ClickHouse, Grafana, or any logging pipeline. Enable with `--json` / `--output` on `run` or `wrap`:

```bash
matcha wrap --output run.jsonl \
    --label team=capacity --label config=lr_3e-4 --label seed=42 \
    torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Training stdout passes through untouched. Records are appended to `run.jsonl`:

```jsonl
{"type":"session_start","ts":"2026-04-17T15:19:18.004Z","run_id":"3f7a9b1c4e2d","matcha_version":"0.2.0","hostname":"h100-node-4","driver_version":"535.104.12","interval_ms":100,"gpus":[{"idx":0,"uuid":"GPU-f364...","name":"NVIDIA H100 80GB HBM3"}, ... ],"cmd":["torchrun","--standalone","--nproc_per_node=8","train_gpt.py"],"labels":{"team":"capacity","config":"lr_3e-4","seed":"42"}}
{"type":"step","ts":"2026-04-17T15:19:18.616Z","run_id":"3f7a9b1c4e2d","step":1,"step_gap":1,"energy_j":2354.0,"energy_per_step_j":2354.0,"duration_s":0.612,"avg_power_w":3847.0,"peak_power_w":4120.0,"gpus":[{"idx":0,"energy_j":323.5,"avg_power_w":528.6,"peak_power_w":540.0}, ... ]}
{"type":"session_end","ts":"2026-04-17T15:22:41.104Z","run_id":"3f7a9b1c4e2d","total_energy_j":778168.0,"energy_wh":216.16,"duration_s":203.1,"avg_power_w":3832.0,"peak_power_w":4120.0,"total_samples":2031,"total_steps":20000,"energy_per_step_j":38.91,"gpus":[ ... ]}
```

Ingest example — ClickHouse:

```bash
cat run.jsonl | clickhouse-client --query "INSERT INTO energy_steps FORMAT JSONEachRow"
```

### Flags

| Flag | Description |
| --- | --- |
| `--json` | Emit structured JSONL records. |
| `--output PATH` | Write JSONL to a file (implies `--json`). Required for `wrap --json`. |
| `--label KEY=VALUE` | Attach a label to the run. Repeatable. |
| `--run-id ID` | Stable run identifier. Also honors `MATCHA_RUN_ID`. Auto-generated if unset. |
| `--gpus` | `all`, a single index (`0`), or a list (`0,1,2,3`). Default: all visible GPUs. |
| `--interval` | NVML sampling interval in ms. Default: 100. |

---

## Multi-GPU

matcha auto-detects every visible GPU and reports summed totals plus per-GPU breakdowns in structured output. The per-GPU arrays make straggler detection a one-query affair.

```bash
# 8xH100 — auto-detects all 8
matcha run torchrun --standalone --nproc_per_node=8 train_gpt.py

# Subset
matcha run --gpus 0,1,2,3 torchrun ...

# Single GPU
matcha run --gpus 0 torchrun ...
```

Each step record carries a `gpus: [{idx, energy_j, avg_power_w, peak_power_w}, ...]` array alongside the totals. Useful for:

- **Straggler detection** — one rank consistently drawing ~30% less power usually means a stuck collective, a thermal-throttled card, or a PCIe link degraded to Gen3.
- **DP / PP / TP fingerprinting** — the per-GPU power pattern over time tells you what parallelism strategy is actually running.
- **Rank-0 asymmetry** — expected overhead from checkpoint I/O and collective origins, good to confirm it's bounded.

---

## How it works

matcha runs a background thread that polls GPU power via NVML at a configurable interval (100 ms default). Energy is computed by trapezoidal integration of instantaneous power readings. The training process runs natively — matcha never touches stdin, stdout (in `run` mode), your model, or your training loop.

- **`run`** does not intercept the child's stdout — it's as close to zero-overhead as reasonable.
- **`wrap`** pipes the child's stdout to detect step boundaries, then appends energy data inline or emits structured records.
- **`monitor`** samples directly without launching a child process.

---

## Compatibility

- **Hardware:** verified on NVIDIA H100. Works with any GPU supported by NVML (A100, H100, L4, L40S, Blackwell).
- **Frameworks:** framework-agnostic — `torchrun`, `deepspeed`, `accelerate`, or plain `python`.
- **Multi-node:** matcha runs per-node and emits per-node records; aggregate with `labels.node=...` or `hostname` downstream.

---

## Why

Frontier training runs burn hundreds of MWh. The gap between teams that optimize for energy-per-step and those that don't is measured in millions of dollars per training run. matcha makes that number visible without changing your training code.

```
8xH100 training run — 1 hour:
  Energy cost:   $0.26 (2.16 kWh @ $0.12/kWh)
  Compute cost:  $23.20 (RunPod @ $23.20/hr)

  → Optimizing energy per step == faster training == less rental time
```

---

## Built by

[Keeya Labs](https://keeyalabs.com) · [Docs](https://usematcha.dev) · [GitHub](https://github.com/keeyalabs/usematcha)

## License

Apache 2.0 — see [LICENSE](LICENSE).
