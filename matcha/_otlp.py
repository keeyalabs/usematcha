# SPDX-License-Identifier: Apache-2.0
"""matcha._otlp — OTLP push metrics export.

Pushes the same metrics exposed by `_prometheus` to an OTel collector
over OTLP/HTTP. Uses the OpenTelemetry SDK; installed as an optional
extra (`pip install usematcha[otlp]`).

Metric names are kept identical to the Prometheus endpoint so dashboards
and alerts transfer between scrape and push deployments.
"""

import time
import warnings
from typing import Dict, List, Optional, Set

try:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        import pynvml
except ImportError:
    pynvml = None

from ._engine import PowerSampler
from . import __version__


def _import_otel():
    try:
        from opentelemetry import metrics
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
        from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
        from opentelemetry.sdk.resources import Resource
    except ImportError as e:
        raise SystemExit(
            "matcha: --otlp requires the optional OTLP dependencies. "
            "Install with: pip install 'usematcha[otlp]'"
        ) from e
    return metrics, MeterProvider, PeriodicExportingMetricReader, OTLPMetricExporter, Resource


def _parse_headers(pairs: Optional[List[str]]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not pairs:
        return out
    for p in pairs:
        if "=" not in p:
            raise SystemExit(f"matcha: --otlp-header expects KEY=VALUE, got: {p!r}")
        k, v = p.split("=", 1)
        out[k.strip()] = v
    return out


class OtlpExporter:
    """Periodic OTLP/HTTP metrics push, callback-driven from a PowerSampler."""

    def __init__(
        self,
        sampler: PowerSampler,
        run_id: str,
        labels: Dict[str, str],
        endpoint: str,
        headers: Optional[List[str]] = None,
        interval_ms: int = 10000,
    ):
        self._sampler = sampler
        self._run_id = run_id
        self._labels = dict(labels)
        self._endpoint = endpoint
        self._headers = _parse_headers(headers)
        self._interval_ms = interval_ms
        self._provider = None
        self._meter = None
        self._base_attrs: Dict[str, str] = {}
        self._registered_keys: Set[str] = set()

    def start(self) -> str:
        (metrics, MeterProvider, PeriodicExportingMetricReader,
         OTLPMetricExporter, Resource) = _import_otel()

        # OTLP/HTTP: endpoint must include the metrics path.
        url = self._endpoint.rstrip("/")
        if not url.endswith("/v1/metrics"):
            url = url + "/v1/metrics"

        exporter = OTLPMetricExporter(endpoint=url, headers=self._headers or None)
        reader = PeriodicExportingMetricReader(
            exporter, export_interval_millis=self._interval_ms
        )

        resource_attrs = {
            "service.name": "matcha",
            "service.version": __version__,
            "matcha.run_id": self._run_id,
            "matcha.energy_source": self._sampler.energy_source,
            **{f"matcha.label.{k}": v for k, v in self._labels.items()},
        }
        resource = Resource.create(resource_attrs)

        self._provider = MeterProvider(resource=resource, metric_readers=[reader])
        meter = self._provider.get_meter("matcha", __version__)
        self._meter = meter

        sampler = self._sampler
        base_attrs = {"run_id": self._run_id, **self._labels}
        self._base_attrs = base_attrs

        # ---- GPU-live gauges (redundant with DCGM if they have it, cheap otherwise) ----
        def _power_cb(_options):
            from opentelemetry.metrics import Observation
            obs = []
            for i, h in enumerate(sampler._handles):
                try:
                    w = pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0
                except pynvml.NVMLError:
                    w = 0.0
                obs.append(Observation(w, {
                    **base_attrs,
                    "gpu": str(sampler.gpu_indices[i]),
                    "name": sampler.gpu_names[i],
                }))
            return obs

        def _energy_cb(_options):
            from opentelemetry.metrics import Observation
            obs = []
            for i, h in enumerate(sampler._handles):
                try:
                    e = pynvml.nvmlDeviceGetTotalEnergyConsumption(h) / 1000.0
                except pynvml.NVMLError:
                    e = 0.0
                obs.append(Observation(e, {
                    **base_attrs, "gpu": str(sampler.gpu_indices[i])
                }))
            return obs

        meter.create_observable_gauge(
            "matcha_gpu_power_watts", callbacks=[_power_cb],
            description="Instantaneous GPU power draw", unit="W",
        )
        meter.create_observable_counter(
            "matcha_gpu_energy_joules_total", callbacks=[_energy_cb],
            description="Cumulative GPU energy (NVML hardware counter)", unit="J",
        )

        # ---- Step-level (the differentiator) ----
        def _step_cb(attr_key: str):
            from opentelemetry.metrics import Observation

            def _cb(_options):
                last = sampler.last_step
                if last is None:
                    return []
                value = getattr(last, attr_key)
                return [Observation(float(value), base_attrs)]
            return _cb

        meter.create_observable_gauge(
            "matcha_step_energy_joules", callbacks=[_step_cb("energy_j")],
            description="Energy of the most recent training step", unit="J",
        )
        meter.create_observable_gauge(
            "matcha_step_duration_seconds", callbacks=[_step_cb("duration_s")],
            description="Wallclock duration of the most recent step", unit="s",
        )
        meter.create_observable_gauge(
            "matcha_step_avg_power_watts", callbacks=[_step_cb("avg_power_w")],
            description="Average power during the most recent step", unit="W",
        )
        meter.create_observable_gauge(
            "matcha_step_peak_power_watts", callbacks=[_step_cb("peak_power_w")],
            description="Peak power during the most recent step", unit="W",
        )
        meter.create_observable_gauge(
            "matcha_step_number", callbacks=[_step_cb("step")],
            description="Most recent completed step number",
        )

        def _step_gpu_energy_cb(_options):
            from opentelemetry.metrics import Observation
            last = sampler.last_step
            if last is None:
                return []
            return [
                Observation(g.energy_j, {**base_attrs, "gpu": str(g.idx)})
                for g in last.per_gpu
            ]

        def _step_gpu_dev_cb(_options):
            from opentelemetry.metrics import Observation
            last = sampler.last_step
            if last is None or len(last.per_gpu) < 2:
                return []
            energies = sorted(g.energy_j for g in last.per_gpu)
            mid = len(energies) // 2
            median = (energies[mid] if len(energies) % 2
                      else 0.5 * (energies[mid - 1] + energies[mid]))
            if median <= 0:
                return []
            return [
                Observation((g.energy_j - median) / median,
                            {**base_attrs, "gpu": str(g.idx)})
                for g in last.per_gpu
            ]

        meter.create_observable_gauge(
            "matcha_step_gpu_energy_joules", callbacks=[_step_gpu_energy_cb],
            description="Per-GPU energy of the most recent step", unit="J",
        )
        meter.create_observable_gauge(
            "matcha_step_gpu_energy_deviation_ratio", callbacks=[_step_gpu_dev_cb],
            description="Per-GPU (energy - median) / median — straggler signal",
        )

        def _steps_cb(_options):
            from opentelemetry.metrics import Observation
            return [Observation(sampler.steps_completed, base_attrs)]

        def _session_energy_cb(_options):
            from opentelemetry.metrics import Observation
            return [Observation(sampler.session_step_energy_j, base_attrs)]

        def _session_duration_cb(_options):
            from opentelemetry.metrics import Observation
            if sampler._session_start_t is None:
                return []
            return [Observation(time.monotonic() - sampler._session_start_t, base_attrs)]

        meter.create_observable_counter(
            "matcha_steps_total", callbacks=[_steps_cb],
            description="Total number of training steps observed",
        )
        meter.create_observable_counter(
            "matcha_session_energy_joules_total", callbacks=[_session_energy_cb],
            description="Cumulative energy across observed steps", unit="J",
        )
        meter.create_observable_gauge(
            "matcha_session_duration_seconds", callbacks=[_session_duration_cb],
            description="Elapsed session time", unit="s",
        )

        return url

    def note_key(self, key: str) -> None:
        """Register a new training-metric key as an observable gauge.

        Called from the wrap loop the first time a stdout-parsed metric
        appears. The OTel SDK picks up new instruments on the next export
        cycle, so there's no need to pre-declare anything.
        """
        if not self._meter or key in self._registered_keys:
            return
        self._registered_keys.add(key)

        from opentelemetry.metrics import Observation
        sampler = self._sampler
        base_attrs = self._base_attrs
        captured = key

        def _cb(_options):
            val = sampler.last_train_metrics.get(captured)
            if val is None:
                return []
            return [Observation(val, base_attrs)]

        self._meter.create_observable_gauge(
            f"matcha_metric_{captured}",
            callbacks=[_cb],
            description=f"Training metric '{captured}' parsed from stdout",
        )

    def stop(self) -> None:
        if self._provider:
            try:
                self._provider.force_flush(timeout_millis=2000)
            except Exception:
                pass
            try:
                self._provider.shutdown()
            except Exception:
                pass
