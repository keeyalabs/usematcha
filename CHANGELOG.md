# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] — 2026-04-20

### Added
- **Multi-vendor hardware backends.** matcha's measurement engine is now vendor-agnostic: the same `session_start` / `step` / `session_end` schema, the same Prometheus and OTLP metric names, and the same `matcha_energy` summary line are produced on NVIDIA (NVML), AMD (`rocm-smi`), Intel (`xpu-smi`), and Apple Silicon (IOReport). Hosts auto-detect at `start()`; override with `MATCHA_BACKEND=nvml|rocm|intel|apple` for multi-vendor machines. Backend is tagged into every record (`backend` field on `SessionResult`, `matcha.backend` OTel resource attribute, `backend` Prometheus label). NVIDIA remains the fastest path (millijoule-precise hardware energy counter, microsecond reads). ROCm and Intel backends run their vendor CLI under a cached refresher thread so sampler latency stays flat; energy is derived by trapezoidal integration of polled power until the `amdsmi` / Level Zero counter paths land. **Apple Silicon reads Darwin's IOReport framework directly via ctypes** — cumulative millijoule GPU energy counters, same semantic class as NVML's `nvmlDeviceGetTotalEnergyConsumption`, so `energy_source="counter"` on M-series too. **No sudo, no `powermetrics` subprocess, no extra pip deps** (stdlib ctypes against `/usr/lib/libIOReport.dylib`). Step-boundary energy reads force a fresh IOReport sample, so per-step attribution is **counter-exact** — short steps (≤100ms) are attributed correctly without smearing across refresher ticks, matching NVIDIA's microsecond-precise step boundaries. The background refresher runs at 20 Hz (50ms) by default — a ~3% one-core cost that keeps `read_power_w` fresh for peak detection during long steps; tunable via `AppleSiliconBackend(refresh_ms=…)` (floor 5ms).
- **Python API** — `matcha.session()` context manager and `matcha.Session` class. Thin facade over the internal power sampler with lifecycle methods (`start` / `stop`, `step_begin` / `step_end`, `with s.step(i):` sugar), `update_metrics` passthrough for user-supplied scalars, and read-only introspection (`gpu_name`, `energy_source`, `last_step`, `steps_completed`, `result`, ...). Safe to construct on CPU-only hosts and in CI — the backend is only resolved and initialized at `start()`. A module-level guard raises a clean error if a second session is started in the same process. Public API is exposed via `__getattr__` on the `matcha` package, so `import matcha` stays driver-free (~35ms cold import on Python 3.12).
- **HuggingFace Trainer callback** — `matcha.callbacks.StepEnergyCallback`, behind the new optional extra `pip install 'usematcha[hf]'`. Measures only on the local-zero DDP rank (so 8-rank hosts don't 8x-count). Injects `matcha/energy_j`, `matcha/energy_step_j`, `matcha/power_avg_w`, `matcha/power_peak_w` into the Trainer's `logs` dict — the values flow into stdout, TensorBoard, and WandB automatically. Emits the standard `matcha_energy` summary line on `on_train_end`. Fails closed: if matcha can't init (no GPU of any vendor, NVML/rocm-smi/xpu-smi/IOReport flake), the callback logs a warning and becomes a no-op — training is never affected. Externally-owned `Session` objects passed via `session=...` are respected (callback does not stop them). Dangling step windows from crashed hooks are cleaned up before `stop()` so session totals stay clean.

### Notes
- The programmatic Python API was intentionally removed in 0.2.0 with the reasoning that it conflicted with the CLI's "zero code changes" pitch. It returns in 0.3.0 as an **additive, opt-in** surface so framework integrations (HuggingFace Trainer today; Lightning / Ray / Accelerate planned) have a real code path. The CLI remains the primary entry point and stays fully zero-code — nothing about `matcha run`, `matcha wrap`, `matcha diff`, or `matcha monitor` has changed.

### Compatibility
- Fully additive. No CLI flags removed, no breaking engine changes. JSONL / Prometheus / OTLP schemas gain one field (`backend` on session records, `backend` label on Prom series, `matcha.backend` on OTel resources) — existing consumers ignore it cleanly.
- `SessionResult` gains a `backend: str` field (`"nvml"` | `"rocm"` | `"intel"` | `"apple"`).
- `nvidia-ml-py` stays a default dependency (pure-Python, installs on every platform). New no-op extras `usematcha[amd]`, `usematcha[intel]`, `usematcha[mac]` exist for intent declaration and forward-compat — AMD/Intel shell out to vendor CLIs today; Apple Silicon uses stdlib ctypes against IOReport with zero pip deps.
- New optional extra: `pip install 'usematcha[hf]'` pulls `transformers>=4.30`. Core install is unchanged.

## [0.2.4] — 2026-04-17

### Added
- **`matcha diff`** — compare runs recorded as JSONL. With two files, emits a side-by-side view with absolute and percent deltas on duration, energy, power, steps, and auto-extracted training metrics (train_loss, step_avg_ms, etc.). With three or more, pivots to a sweep table — one column per run, `*` marks the best value per row, column headers read from `--label config=...` if set else the filename stem. Layout is fully content-driven: GPU names normalize to short form (`8x H100`), energy auto-scales to J/kJ/MJ/GJ based on magnitude, and columns widen to fit longest header or value — no truncation.
- **Straggler signal in diffs** — `max |per-gpu deviation from median|` surfaced as a row, so regressions in load balance across runs are visible alongside efficiency wins.

### Fixed
- **Off-by-one step attribution in `matcha wrap`.** Training scripts print `step:N/...` *after* step N's work finishes, so the window between `step:N-1` and `step:N` lines is step N's work, not step N-1's. Previously that window was labeled N-1 and the energy was appended to the N-1 line. Now the window is attributed to the newly-finished step and the inline energy suffix prints on the line that named it. First step line prints raw (no measurement window yet). End-of-stream partial windows are dropped rather than emitted as phantom steps — session totals in `session_end` still include that time, so no energy is lost. Removed `last_line` and `pending_metrics` bookkeeping that became unnecessary once attribution lined up with the line being printed.

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
