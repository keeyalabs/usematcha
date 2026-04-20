# Architecture

matcha is one NVML sampler exposed through four layers of increasing convenience. Each layer is independently replaceable; consumers at any layer see a stable contract.

```
┌─────────────────────────────────────────────────────────────────────┐
│ User entry points                                                   │
│                                                                     │
│   CLI:       matcha run │ matcha wrap │ matcha diff │ matcha        │
│                                                      monitor        │
│   Library:   matcha.session() │ matcha.Session                      │
│   Callback:  matcha.callbacks.StepEnergyCallback (HF Trainer)       │
└─────────────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────────────┐
│ Public API (matcha/_api.py)                                         │
│   Session facade — lifecycle, concurrent-session guard,             │
│   deferred NVML binding, safe pre-start introspection.              │
└─────────────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────────────┐
│ Engine (matcha/_engine.py)                                          │
│   PowerSampler — NVML counter (Volta+) or polled fallback,          │
│   per-step + per-GPU energy, peak-power polling thread,             │
│   user-metric passthrough for exporters.                            │
└─────────────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────────────┐
│ NVML (nvidia-ml-py)                                                 │
└─────────────────────────────────────────────────────────────────────┘
```

## File map

| Path | Role |
| --- | --- |
| `matcha/__init__.py` | Lazy loader — keeps `import matcha` pynvml-free and transformers-free |
| `matcha/_api.py` | Public `Session` facade; concurrent-session guard; gpu-spec normalization |
| `matcha/_engine.py` | `PowerSampler` — counter + polled fallback; per-step / per-GPU / session aggregation |
| `matcha/cli.py` | `matcha run / wrap / diff / monitor`, flag parsing, subprocess plumbing |
| `matcha/callbacks/__init__.py` | Lazy gate that translates missing framework deps into friendly install hints |
| `matcha/callbacks/hf.py` | HuggingFace `TrainerCallback` adapter (optional extra: `[hf]`) |
| `matcha/exporters/jsonl.py` | Structured record writers (`session_start` / `step` / `session_end`) |
| `matcha/exporters/prometheus.py` | `/metrics` HTTP endpoint |
| `matcha/exporters/otlp.py` | OTLP/HTTP push exporter (optional extra: `[otlp]`) |
| `matcha/commands/diff.py` | `matcha diff` — pairwise and sweep comparisons over JSONL |
| `matcha/commands/monitor.py` | `matcha monitor` — live per-GPU TTY view |
| `matcha/commands/stdout_metrics.py` | Stdout train-metric extraction (numeric key:value, HF dict form, ...) |

## Data flow

**Run without steps** (`matcha run`, `with matcha.session():`)

```
start()  ─▶  nvmlInit, counter snapshot, start poll thread
           │
           │  (background thread samples power every interval_ms)
           │
stop()   ─▶  counter diff, drain samples, nvmlShutdown → SessionResult
```

**Run with steps** (`matcha wrap`, `with s.step(i):`, `StepEnergyCallback`)

```
step_begin() ─▶  counter snapshot, mark_window_start
                 │
                 │  (work happens; poll thread keeps going)
                 │
step_end(i)  ─▶  counter diff for this window, integrate polled samples for
                 peak/avg, emit StepResult(i, energy_j, avg_w, peak_w, per_gpu)
```

All step and session results carry an `energy_source` field — `"counter"` when every monitored GPU supports the hardware energy accumulator, `"polled"` when any doesn't. Consumers (`exporters.jsonl`, `exporters.prometheus`, `exporters.otlp`, `commands.diff`) do not branch on this — the schema is identical in either mode.

## Invariants

1. **`energy_source` is honest.** `"counter"` requires every monitored GPU to expose `nvmlDeviceGetTotalEnergyConsumption`. A mixed configuration downgrades to `"polled"` rather than silently misreporting.
2. **One active session per process.** NVML is process-global. `matcha._api` rejects concurrent sessions with a clean `RuntimeError` instead of letting NVML fail in a confusing way.
3. **Callbacks measure only on the local-zero DDP rank.** 8-rank hosts would 8x-count without this guard. Cross-host aggregation is a separate problem, handled at the Prometheus / OTLP layer.
4. **Callbacks never raise from hooks.** Instrumentation must never break training. If matcha can't init, the callback logs once and becomes a no-op for the remainder of the run.
5. **`import matcha` is cheap.** No `pynvml`, no `transformers`, no engine code loads at module import. Public symbols live behind `__getattr__`. Cold-import budget: <50ms on Python 3.12 for `import matcha` and `import matcha.callbacks`.
6. **Library mode is silent by default.** The CLI does human output. The library emits nothing to stdout unless the caller asks.
7. **Counter and polled paths share one schema.** JSONL / Prometheus / OTLP records do not shape-shift between energy modes.
8. **`matcha wrap` and in-process callbacks are mutually exclusive.** Both would double-instrument the same GPU set. The failure mode is a clean "session already active" error rather than silent double-counting.

## Release validation

Off-hardware CI (Linux, Python 3.9 – 3.12) covers:

- Public API surface tests (name / signature / error paths).
- Callback tests with faked `transformers.TrainerCallback` driving every hook, including the disabled / no-GPU / non-zero-rank paths.
- Stdout parser tests against captured real-world training logs (modded-nanoGPT, parameter-golf, HF Trainer, DeepSpeed).
- Import-leak tests: `import matcha` must not pull pynvml or transformers.

On-hardware validation (not CI-gated) — run before each release tag:

- Single-GPU H100 on RunPod: parameter-golf end-to-end, counter-vs-polled agreement, JSONL schema check.
- Multi-GPU (when available): per-GPU deviation sanity check, summed-total cross-check against external power measurement.

Tag → push to main → GitHub Actions runs the release workflow (Trusted Publishing + Sigstore attestations) on the versioned tag. No PyPI tokens live on contributor machines.

## Non-goals

| | Why not |
| --- | --- |
| **Non-NVIDIA GPUs** (AMD ROCm, Intel XPU, Apple Metal) | Different vendor APIs are fundamentally different tools. Each deserves a focused project, not a plugin. |
| **Cross-host aggregation** | Prometheus / OTLP / your log pipeline already does this well. Duplicating it would create two sources of truth. |
| **Intrusive training hooks** | matcha measures; it does not modify gradients, batching, or schedules. That's a different class of tool. |
| **Energy-based alerts / limits** | Policies belong in your monitoring stack (Prometheus alertmanager, Grafana alerts, PagerDuty). matcha's job is to emit the data they act on. |
| **Profiling at sub-step granularity** | Use `nsys`, `nvprof`, or PyTorch Profiler. They instrument the right layer of abstraction; matcha deliberately doesn't. |
