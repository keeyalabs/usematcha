# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
