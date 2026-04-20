# SPDX-License-Identifier: Apache-2.0
"""matcha.exporters.prometheus — Prometheus text-exposition endpoint.

Exposes `/metrics` over HTTP for Prometheus scrapers. Reads live from
the active backend on each scrape; the handler is stateless beyond a
reference to the running ``PowerSampler``.

Pull-only. For push (Grafana Cloud remote_write, OTLP), see
``exporters.otlp``.
"""

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, Optional

from .._engine import PowerSampler
from .. import __version__


_NAMESPACE = "matcha"


def _escape_label(v: str) -> str:
    return v.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def _fmt_labels(labels: Dict[str, str]) -> str:
    if not labels:
        return ""
    parts = [f'{k}="{_escape_label(str(v))}"' for k, v in labels.items()]
    return "{" + ",".join(parts) + "}"


def _render(sampler: PowerSampler, run_id: str, user_labels: Dict[str, str]) -> str:
    """Build Prometheus text-exposition-format output from a live sample."""
    lines = []

    def metric(name, mtype, help_text):
        lines.append(f"# HELP {_NAMESPACE}_{name} {help_text}")
        lines.append(f"# TYPE {_NAMESPACE}_{name} {mtype}")

    def emit(name, labels, value):
        lines.append(f"{_NAMESPACE}_{name}{_fmt_labels(labels)} {value}")

    # The ``backend`` label lets multi-vendor fleets slice by vendor
    # ("where did AMD regress?") without having to join on name.
    base = {"run_id": run_id, "backend": sampler.backend_name, **user_labels}

    metric("info", "gauge", "matcha build information")
    emit("info", {**base, "version": __version__, "energy_source": sampler.energy_source}, 1)

    backend = sampler.backend
    if backend is None or backend.device_count == 0:
        return "\n".join(lines) + "\n"

    n = backend.device_count
    powers, utils, temps, mem_used, energies_j = [], [], [], [], []
    for i in range(n):
        powers.append(backend.read_power_w(i))
        utils.append(backend.read_utilization_pct(i))
        temps.append(backend.read_temperature_c(i))
        mem_used.append(backend.read_memory_used_bytes(i))
        if backend.has_energy_counter:
            try:
                energies_j.append(backend.read_energy_mj(i) / 1000.0)
            except Exception:
                energies_j.append(0.0)
        else:
            energies_j.append(0.0)

    metric("gpu_power_watts", "gauge", "Instantaneous GPU power draw in watts")
    for idx, name, uuid, p in zip(sampler.gpu_indices, sampler.gpu_names, sampler.gpu_uuids, powers):
        emit("gpu_power_watts", {**base, "gpu": str(idx), "name": name, "uuid": uuid}, p)

    metric("gpu_utilization_ratio", "gauge", "GPU SM utilization (0-1)")
    for idx, u in zip(sampler.gpu_indices, utils):
        emit("gpu_utilization_ratio", {**base, "gpu": str(idx)}, u / 100.0)

    metric("gpu_temperature_celsius", "gauge", "GPU die temperature in degrees Celsius")
    for idx, t in zip(sampler.gpu_indices, temps):
        emit("gpu_temperature_celsius", {**base, "gpu": str(idx)}, t)

    metric("gpu_memory_used_bytes", "gauge", "GPU memory used in bytes")
    for idx, m in zip(sampler.gpu_indices, mem_used):
        emit("gpu_memory_used_bytes", {**base, "gpu": str(idx)}, m)

    if backend.has_energy_counter:
        metric("gpu_energy_joules_total", "counter",
               "Cumulative GPU energy consumption since driver boot, in joules (hardware counter)")
        for idx, e in zip(sampler.gpu_indices, energies_j):
            emit("gpu_energy_joules_total", {**base, "gpu": str(idx)}, e)

    total_power = sum(powers)
    metric("power_watts", "gauge", "Total instantaneous power across all sampled GPUs")
    emit("power_watts", base, total_power)

    # Step-level metrics — matcha's differentiator vs DCGM-exporter.
    # Exposed whenever `wrap` has observed at least one step.
    last = sampler.last_step
    if last is not None:
        metric("step_number", "gauge", "Most recent completed training step number")
        emit("step_number", base, last.step)

        metric("step_energy_joules", "gauge", "Energy consumed in the most recent step (J)")
        emit("step_energy_joules", base, round(last.energy_j, 3))

        metric("step_duration_seconds", "gauge", "Wallclock duration of the most recent step (s)")
        emit("step_duration_seconds", base, round(last.duration_s, 4))

        metric("step_avg_power_watts", "gauge", "Average power during the most recent step (W)")
        emit("step_avg_power_watts", base, round(last.avg_power_w, 1))

        metric("step_peak_power_watts", "gauge", "Peak power during the most recent step (W)")
        emit("step_peak_power_watts", base, round(last.peak_power_w, 1))

        metric("step_gpu_energy_joules", "gauge", "Per-GPU energy in the most recent step (J)")
        for g in last.per_gpu:
            emit("step_gpu_energy_joules", {**base, "gpu": str(g.idx)}, round(g.energy_j, 3))

        metric("step_gpu_peak_power_watts", "gauge",
               "Per-GPU peak power in the most recent step (W)")
        for g in last.per_gpu:
            emit("step_gpu_peak_power_watts", {**base, "gpu": str(g.idx)}, round(g.peak_power_w, 1))

        # Straggler signal: each GPU's deviation from the per-step median.
        # Makes one-line Prom alerts possible: e.g. alert if deviation < -0.15 for 5m.
        if len(last.per_gpu) > 1:
            energies = sorted(g.energy_j for g in last.per_gpu)
            mid = len(energies) // 2
            median = (energies[mid] if len(energies) % 2
                      else 0.5 * (energies[mid - 1] + energies[mid]))
            metric("step_gpu_energy_deviation_ratio", "gauge",
                   "Per-GPU (energy - median) / median for the most recent step")
            for g in last.per_gpu:
                dev = (g.energy_j - median) / median if median > 0 else 0.0
                emit("step_gpu_energy_deviation_ratio",
                     {**base, "gpu": str(g.idx)}, round(dev, 4))

    metric("steps_total", "counter", "Total number of training steps observed")
    emit("steps_total", base, sampler.steps_completed)

    metric("session_energy_joules_total", "counter",
           "Cumulative energy attributed to observed training steps, since session start (J)")
    emit("session_energy_joules_total", base, round(sampler.session_step_energy_j, 3))

    if sampler._session_start_t is not None:
        import time as _t
        metric("session_duration_seconds", "gauge",
               "Elapsed time since matcha session started (s)")
        emit("session_duration_seconds", base,
             round(_t.monotonic() - sampler._session_start_t, 3))

    # User-provided training metrics parsed from stdout (train_loss, lr, mfu, ...).
    # Emitted as one gauge per distinct key so Prom queries stay natural:
    # `matcha_metric_train_loss` rather than filtering a catchall by a label.
    for key, val in sorted(sampler.last_train_metrics.items()):
        name = f"metric_{key}"
        metric(name, "gauge", f"Training metric '{key}' parsed from stdout")
        emit(name, base, val)

    return "\n".join(lines) + "\n"


class PromServer:
    """Tiny HTTP server exposing /metrics for Prometheus scrapers."""

    def __init__(
        self,
        sampler: PowerSampler,
        run_id: str,
        labels: Dict[str, str],
        bind: str = ":9400",
    ):
        self._sampler = sampler
        self._run_id = run_id
        self._labels = dict(labels)
        host, _, port_s = bind.rpartition(":")
        self._host = host or "0.0.0.0"
        self._port = int(port_s)
        self._httpd: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    def start(self) -> str:
        sampler = self._sampler
        run_id = self._run_id
        labels = self._labels

        class _Handler(BaseHTTPRequestHandler):
            def log_message(self, *a, **kw):
                pass

            def do_GET(self):
                if self.path.rstrip("/") in ("/metrics", ""):
                    body = _render(sampler, run_id, labels).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                else:
                    self.send_response(404)
                    self.end_headers()

        self._httpd = ThreadingHTTPServer((self._host, self._port), _Handler)
        actual_port = self._httpd.server_address[1]
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()
        return f"http://{self._host}:{actual_port}/metrics"

    def stop(self) -> None:
        if self._httpd:
            self._httpd.shutdown()
            self._httpd.server_close()
        if self._thread:
            self._thread.join(timeout=2.0)
