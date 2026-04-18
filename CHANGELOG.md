# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.3] — 2026-04-17

### Added
- **Prometheus `/metrics` endpoint** via `--prometheus [HOST]:PORT` on `run` and `wrap`. Exposes step-level metrics (`matcha_step_energy_joules`, `matcha_step_duration_seconds`, `matcha_step_peak_power_watts`, `matcha_step_gpu_energy_deviation_ratio` for straggler detection) plus running session counters. GPU-live gauges (`matcha_gpu_power_watts`, `matcha_gpu_energy_joules_total`) are also exposed for teams without DCGM-exporter. All user `--label KEY=VALUE` values become Prometheus labels.
- **OTLP/HTTP push** via `--otlp URL [--otlp-header K=V] [--otlp-interval MS]` on `run` and `wrap`. Pushes the same metric set as the Prometheus endpoint to any OTel-compatible backend (Grafana Cloud, Honeycomb, Datadog, self-hosted collector). Requires the optional dependencies: `pip install 'usematcha[otlp]'`.
- **Training metrics auto-extracted from stdout** in `wrap`. Numeric `key:value`, `key=value`, and HuggingFace-style `'key': value` pairs are parsed from step lines and surfaced as `matcha_metric_<key>` in Prometheus/OTLP and as a `train_metrics` block in JSONL step records. Works out of the box for parameter-golf / nanoGPT / modded-nanogpt / DeepSpeed / HF Trainer outputs. Unit suffixes (`ms`, `W`, `J`, `%`) are preserved in the key (`matcha_metric_train_time_ms`). NaN / Inf / non-numeric values are skipped. Matcha's own output fields (`energy`, `avg_power`, ...) are blacklisted to avoid re-ingestion if outputs are chained.

### Changed
- `matcha monitor` rendering stripped of ANSI colors and emojis for cleaner terminal capture and consistent rendering across shells. ASCII progress bar (`[####------]`) retained for at-a-glance power/TDP ratio. Live in-place refresh unchanged.

## [0.2.2] — 2026-04-17

### Fixed
- **Per-step peak power could report below average power** when step duration was shorter than the sampling interval. `begin_step` and `end_step` now read `nvmlDeviceGetPowerUsage` at each boundary in addition to the background sampler, guaranteeing ≥2 data points per step window. Peak is also floored at the counter-derived average (physical invariant: peak ≥ avg).
- Same floor applied to session-level peak.

### Changed
- Default `--interval` returned to 100 ms. The 500 ms default in 0.2.1 was chosen to reduce overhead that turned out to be pod variance, not matcha overhead — dense polling is free and gives a richer peak signal.

## [0.2.1] — 2026-04-17

### Changed
- **Energy now read from NVML's hardware accumulator** (`nvmlDeviceGetTotalEnergyConsumption`, Volta+). Per-step and session energy are exact counter deltas — no trapezoidal integration error, no per-step edge bias, and no measurable overhead from the previous tight polling loop.
- Default `--interval` raised from 100 ms to 500 ms. It now controls only peak-power polling; energy accuracy is independent of it.
- `session_start` and `session_end` JSONL records now include `energy_source` (`"counter"` or `"polled"`).

### Fixed
- Measurable per-step overhead (~30-50% observed on single-H100 RunPod instances) caused by 100 ms NVML polling. Counter-based energy eliminates it.

### Compatibility
- Falls back to trapezoidal integration of polled power samples if the hardware counter is unavailable (pre-Volta GPUs). Public schema and CLI flags unchanged.

## [0.2.0] — 2026-04-17

### Added
- **Structured JSONL output** on `matcha run` and `matcha wrap`.
  - `--json` emits `session_start`, `step`, and `session_end` records.
  - `--output PATH` writes JSONL to a file. Required for `wrap --json` (training stdout is preserved).
  - `--label KEY=VALUE` (repeatable) attaches arbitrary labels to the run.
  - `--run-id ID` sets a stable identifier; also honors `MATCHA_RUN_ID`. Auto-generates a UUID if unset.
- **Per-GPU breakdown** in every step and session record: `gpus: [{idx, energy_j, avg_power_w, peak_power_w}, ...]`. Useful for straggler detection and parallelism fingerprinting on multi-GPU runs.
- **Run metadata** captured at `session_start`: `hostname`, `driver_version`, GPU UUIDs/indices/names, sampling interval, and the launched command.
- `matcha monitor` reworked into a live per-GPU dashboard (cursor-redrawn) showing power / TDP, utilization, temperature, memory, and cumulative energy with a clean Ctrl+C summary.
- Type information published via PEP 561 (`py.typed` marker).

### Changed
- Default `matcha monitor` refresh interval is now 500 ms (was 100 ms) for a more readable live view.
- `SessionResult.avg_power_w` is now `total_energy_j / duration_s` (consistent with integrated energy) rather than the arithmetic mean of samples.
- Clearer error message when NVML initialization fails.
- `__version__` is now derived from package metadata, eliminating drift against `pyproject.toml`.

### Removed
- The programmatic Python API (`matcha.init`, `matcha.watch`, `matcha.Meter`). `matcha` is now a pure CLI tool; the API contradicted the "zero code changes" pitch and was not documented.

### Fixed
- License metadata is now consistent across `pyproject.toml`, `LICENSE`, and source files (Apache 2.0 throughout, with SPDX identifiers on source files).

## [0.1.0] — 2025-10

Initial public release. `matcha run`, `matcha wrap`, `matcha monitor`.
