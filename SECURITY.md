# Security

## Reporting a vulnerability

Please email **jay@keeyalabs.com** with:

- A description of the issue and how to reproduce it.
- The matcha version you're running (`matcha --version`) and your Python / OS / NVIDIA driver versions.
- Whether you believe the issue is exploitable in a multi-tenant environment.
- Any proof-of-concept code, if applicable.

We aim to acknowledge reports within **2 business days** and issue a fix or documented mitigation within **30 days** for confirmed issues. Coordinated disclosure is welcome — we'll agree on a public-disclosure date with the reporter.

**Please do not open a public GitHub issue** for suspected vulnerabilities.

## Supported versions

Only the most recent minor release line (currently **`0.3.x`**) receives security fixes. Earlier lines (`0.0.x`, `0.1.x`, `0.2.x`) are unsupported.

| Version | Supported |
| --- | --- |
| `0.3.x` | ✓ |
| `< 0.3` | ✗ — upgrade to `0.3.x` |

## What matcha accesses

matcha is a measurement tool. The full list of what it reads:

- **NVML counters on the local host**, via `nvidia-ml-py`. This includes power draw, energy counters (`nvmlDeviceGetTotalEnergyConsumption`), temperature, utilization, memory, and device metadata (name, UUID, driver version). No kernel-mode access. No PCIe probing.
- **The training process's stdout** (only when invoked via `matcha wrap`). Used to detect step markers and numeric `key:value` pairs. matcha never sends stdout anywhere; only the extracted numeric values surface in matcha's own output.

matcha does **not** access:

- Model weights, gradients, optimizer state.
- Training data, batch contents, dataset files.
- GPU device memory contents.
- The filesystem, beyond writing to paths the user explicitly provides (`--output`).
- The network, by default (see next section).

## Outbound network traffic

By default, matcha makes **no network calls**. Two opt-in paths:

| Path | What it does | Your responsibility |
| --- | --- | --- |
| `--prometheus [HOST]:PORT` | Starts a local HTTP server exposing `/metrics`. | Bind to `127.0.0.1` unless you want remote scraping. Use a reverse proxy or an agent sidecar for remote access rather than binding to `0.0.0.0`. |
| `--otlp URL` | POSTs metric batches to the URL you provide, at `--otlp-interval` (default 10s). | Use `https://` endpoints. Rotate any auth tokens passed via `--otlp-header` regularly — they're transmitted on every batch. |

Neither path transmits stdout contents, filesystem paths, or anything about the training code beyond the numeric metrics described above.

## Dependencies

- **Required**: `nvidia-ml-py>=12.0`
- **Optional** (`usematcha[otlp]`): `opentelemetry-sdk>=1.20`, `opentelemetry-exporter-otlp-proto-http>=1.20`
- **Optional** (`usematcha[hf]`): `transformers>=4.30`

We pin minimum versions; we do not pin maximums. Run `pip-audit` against your install before deploying into security-sensitive environments. Our CI runs `pip-audit` and `bandit` on every PR against the main branch.

## Hardening recommendations

For shared / multi-tenant GPU infrastructure:

- Bind the Prometheus endpoint to `127.0.0.1` and scrape via a local agent, not a routable address.
- Keep OTLP auth tokens out of CI logs — pass them via secrets, not plain environment variables that get echoed.
- Treat matcha output (energy traces, step durations) as potentially side-channel-sensitive if you run untrusted workloads on the same host. Energy traces can reveal coarse-grained properties of the workload shape.
- If you run matcha as a long-lived process (e.g. inside a sidecar container), cap its memory and CPU via your orchestrator — matcha's sample buffer grows linearly with run duration, and there's no automatic eviction in the 0.3.x line.
