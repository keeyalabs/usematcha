<h1 align="center">
  <img src="https://raw.githubusercontent.com/keeyalabs/usematcha/main/docs/matcha.svg" alt="matcha" width="280">
</h1>

<p align="center">
  <b>GPU energy observability for AI training.</b>
</p>

<p align="center">
  <a href="https://pypi.org/project/usematcha/"><img src="https://img.shields.io/pypi/v/usematcha?color=4ade80&label=pypi" alt="PyPI"></a>
  &nbsp;
  <a href="https://pypi.org/project/usematcha/"><img src="https://img.shields.io/pypi/pyversions/usematcha?color=4ade80" alt="Python versions"></a>
  &nbsp;
  <a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/license-Apache%202.0-4ade80" alt="License"></a>
</p>

<p align="center">
  Measure energy per training run and per step — from NVML's hardware counter, not sampled power.
  Zero-code CLI, Python API, and HuggingFace Trainer callback.
  Structured output for any observability stack.
</p>

---

## Install

```bash
pip install usematcha
```

Linux, Python 3.9+, NVIDIA GPU with drivers installed.

## Quickstart

```bash
matcha run torchrun --standalone --nproc_per_node=8 train_gpt.py
```

```
matcha_energy gpus:8x NVIDIA H100 80GB HBM3 total:778168J (216.16Wh) duration:203.1s avg_power:3832W peak_power:4120W samples:2031
```

No code changes. No config files. Works with any training script.

---

## Three ways to use it

matcha exposes **one measurement engine** through three surfaces. All three read the same NVML hardware counter and emit the same `StepResult` / `SessionResult` shape.

**CLI** — zero-code, wraps any training command.

```bash
matcha run  python train.py                         # total energy
matcha wrap python train.py                         # per-step energy
matcha monitor                                      # live dashboard
```

See [docs/playbooks/cli](https://docs.usematcha.dev/playbooks/cli) for `diff`, JSONL output, and multi-run comparison.

**Python API** — opt-in, for framework integrations and notebook work.

```python
import matcha

with matcha.session() as s:
    for i in range(num_steps):
        with s.step(i):
            train_step()

print(s.result.total_energy_j, s.result.energy_wh)
```

See [docs/playbooks/python-api](https://docs.usematcha.dev/playbooks/python-api) for explicit lifecycle, custom metrics, and multi-GPU details.

**HuggingFace Trainer callback** — drop-in for the `Trainer` loop.

```python
from matcha.callbacks import StepEnergyCallback

trainer = Trainer(model=model, args=args, callbacks=[StepEnergyCallback()])
trainer.train()
```

Per-step energy flows into the Trainer's log dict — visible in stdout, TensorBoard, and WandB automatically. Install with `pip install 'usematcha[hf]'`.

See [docs/playbooks/huggingface](https://docs.usematcha.dev/playbooks/huggingface) for DDP, failure modes, and config.

---

## Observability

Structured output plugs into the stack you already have.

- **[JSONL](https://docs.usematcha.dev/playbooks/jsonl)** — `--output run.jsonl` writes `session_start` / `step` / `session_end` records with per-GPU breakdowns. Stream into ClickHouse, DuckDB, or any log pipeline.
- **[Prometheus](https://docs.usematcha.dev/playbooks/prometheus)** — `--prometheus :9400` exposes a `/metrics` endpoint with step-level and GPU-live gauges, plus training metrics auto-extracted from stdout.
- **[OpenTelemetry](https://docs.usematcha.dev/playbooks/otlp)** — `--otlp URL` pushes the same metric set to Grafana Cloud, Honeycomb, Datadog, or any OTel collector. Install with `pip install 'usematcha[otlp]'`.

Metric names match across Prometheus and OTLP so dashboards port between deployments.

---

## Multi-GPU

matcha auto-detects every visible GPU and reports summed totals plus a per-GPU breakdown in every record. The per-GPU arrays make straggler detection a one-query affair — one rank consistently drawing ~30% less power usually means a stuck collective, a thermally throttled card, or a PCIe link degraded to Gen3.

```bash
matcha run --gpus 0,1,2,3 torchrun ...
```

---

## How it works

matcha reads energy directly from NVML's hardware accumulator (`nvmlDeviceGetTotalEnergyConsumption`, Volta+). Per-step and session energy are exact counter deltas — millijoule-precise, no integration error. A background poller plus boundary reads at each step transition track peak power. Pre-Volta GPUs fall back to trapezoidal integration. Training runs natively; matcha never touches your model or training loop.

Full design in [ARCHITECTURE.md](https://github.com/keeyalabs/usematcha/blob/main/ARCHITECTURE.md).

---

<p align="center">
  <a href="https://docs.usematcha.dev">Documentation</a>
  &nbsp;·&nbsp;
  <a href="https://github.com/keeyalabs/usematcha/blob/main/CHANGELOG.md">Changelog</a>
  &nbsp;·&nbsp;
  <a href="https://github.com/keeyalabs/usematcha/blob/main/ARCHITECTURE.md">Architecture</a>
  &nbsp;·&nbsp;
  <a href="https://github.com/keeyalabs/usematcha/blob/main/CONTRIBUTING.md">Contributing</a>
  &nbsp;·&nbsp;
  <a href="https://github.com/keeyalabs/usematcha/blob/main/SECURITY.md">Security</a>
</p>

<p align="center">
  Built by <a href="https://keeyalabs.com">Keeya Labs</a>. Apache 2.0.
</p>
