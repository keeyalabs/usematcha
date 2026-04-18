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
  Measure GPU energy per training run and per step.<br>
  No code changes. Structured output for any observability stack.
</p>

---

## Install

```bash
pip install usematcha
```

Requires Linux + an NVIDIA GPU with drivers installed. Python 3.9+.

---

## Quick start

Prefix your training command with `matcha run`:

```bash
matcha run torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Training runs at full speed. One line at the end:

```
matcha_energy gpus:8x NVIDIA H100 80GB HBM3 total:778168J (216.16Wh) duration:203.1s avg_power:3832W peak_power:4120W samples:2031
```

No code changes. No config files. Works with any training script.

---

## Commands

### `matcha run`

Total energy for a whole command. Zero overhead — matcha doesn't touch stdout.

```bash
matcha run python train.py
matcha run torchrun --standalone --nproc_per_node=8 train_gpt.py
```

<br>

### `matcha wrap`

Per-step energy. matcha parses the training stdout for step markers (`step 10`, `iter 10`, `[10/1000]`, etc.) and appends energy per step.

```bash
matcha wrap torchrun --standalone --nproc_per_node=8 train_gpt.py
```

```
step:1/20000 train_loss:6.9357 train_time:612ms step_avg:612.00ms energy:2354.0J/step avg_power:3847W peak_power:4120W
step:2/20000 train_loss:16.7414 train_time:831ms step_avg:721.50ms energy:3012.6J/step avg_power:3625W peak_power:3998W
...
matcha_energy gpus:8x NVIDIA H100 80GB HBM3 total:778168J (216.16Wh) duration:203.1s avg_power:3832W peak_power:4120W samples:2031
```

<br>

### `matcha monitor`

A drop-in replacement for running `watch nvidia-smi` in a second terminal.

```bash
matcha monitor
matcha monitor --gpus 0,1,2,3 --interval 500
```

---

## Observability

matcha plugs into any stack you already have. Three outputs, pick one or several:

<br>

### JSONL for files & ingestion

```bash
matcha wrap --output run.jsonl \
    --label team=capacity --label config=lr_3e-4 \
    torchrun --standalone --nproc_per_node=8 train_gpt.py
```

`run.jsonl` contains `session_start`, one `step` record per step, and `session_end`. Per-GPU breakdowns included.

```jsonl
{"type":"step","step":1,"energy_j":2354.0,"duration_s":0.612,"avg_power_w":3847,"peak_power_w":4120,
 "train_metrics":{"train_loss":6.9357,"train_time_ms":612,"step_avg_ms":612.0},
 "gpus":[{"idx":0,"energy_j":323.5,"avg_power_w":528.6,"peak_power_w":540.0}, ... ]}
```

Stream into ClickHouse, DuckDB, Loki, or anything that reads JSON lines:

```bash
cat run.jsonl | clickhouse-client --query "INSERT INTO energy_steps FORMAT JSONEachRow"
```

<br>

### Prometheus (pull)

```bash
matcha wrap --prometheus :9400 torchrun train.py
```

Exposes `/metrics` for any Prometheus scraper. Both GPU-live gauges and step-level metrics.

```
matcha_step_energy_joules{run_id="...",gpu="0"}      2354.0
matcha_step_duration_seconds{run_id="..."}           0.612
matcha_step_peak_power_watts{run_id="..."}           4120
matcha_step_gpu_energy_deviation_ratio{gpu="3"}     -0.18   # straggler signal
matcha_metric_train_loss{run_id="..."}               6.9357
matcha_metric_step_avg_ms{run_id="..."}              612.0
```

One-line alerts:

```yaml
- alert: GpuStraggler
  expr: matcha_step_gpu_energy_deviation_ratio < -0.15
  for: 5m
```

<br>

### OpenTelemetry / OTLP (push)

```bash
pip install 'usematcha[otlp]'

matcha wrap \
    --otlp https://otlp-gateway.grafana.net/otlp \
    --otlp-header "Authorization=Basic <token>" \
    torchrun train.py
```

Pushes the same metric set to Grafana Cloud, Honeycomb, Datadog, or any OTel collector. Metric names match the Prometheus endpoint so dashboards port across deployments.

<br>

### Training metrics (automatic)

In `wrap` mode, matcha parses numeric fields from your stdout — `train_loss:6.94`, `lr=1.23e-4`, HuggingFace `{'loss': 2.3, ...}` — and surfaces them alongside energy. Works out of the box for nanoGPT, modded-nanogpt, parameter-golf, DeepSpeed, HF Trainer.

No config. No code changes. In Grafana:

```
matcha_metric_train_loss        +
matcha_step_energy_joules       → efficiency curve: J per unit loss reduction
```

---

## Flags

| Flag | Description |
| --- | --- |
| `--output PATH` | Write JSONL records to a file. |
| `--json` | Emit JSONL to stdout/stderr (use `--output` when running `wrap`). |
| `--prometheus :PORT` | Expose a Prometheus `/metrics` endpoint. |
| `--otlp URL` | Push metrics to an OTLP/HTTP collector. |
| `--otlp-header K=V` | Auth header for OTLP (repeatable). |
| `--label KEY=VALUE` | Attach a label to the run (repeatable). |
| `--run-id ID` | Stable run identifier. Honors `MATCHA_RUN_ID`. |
| `--gpus` | `all`, a single index (`0`), or a list (`0,1,2,3`). |
| `--interval` | Peak-power poll interval in ms. Default: 100. |

---

## Multi-GPU

matcha auto-detects every visible GPU and reports summed totals plus per-GPU breakdowns. The per-GPU arrays make straggler detection a one-query affair:

- **Straggler** — one rank consistently drawing ~30% less power usually means a stuck collective, a thermal-throttled card, or a PCIe link degraded to Gen3.
- **DP / PP / TP fingerprinting** — the per-GPU power pattern tells you what parallelism strategy is actually running.

```bash
matcha run torchrun --standalone --nproc_per_node=8 train_gpt.py
matcha run --gpus 0,1,2,3 torchrun ...
matcha run --gpus 0 torchrun ...
```

---

## How it works

matcha reads energy directly from NVML's hardware accumulator (`nvmlDeviceGetTotalEnergyConsumption`, Volta+). Per-step and session energy are exact counter deltas — millijoule-precise, no integration error. A background poller (default 100 ms) plus boundary reads at each step transition track peak power. Pre-Volta GPUs fall back to trapezoidal integration of polled samples.

The training process runs natively — matcha never touches your model or your training loop.

- **`run`** does not intercept stdout. Close to zero-overhead as reasonable.
- **`wrap`** pipes stdout to detect step boundaries, then appends energy data inline or emits structured records.
- **`monitor`** samples directly without launching a child process.

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
