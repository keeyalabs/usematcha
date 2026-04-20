# SPDX-License-Identifier: Apache-2.0
"""matcha._engine — vendor-agnostic GPU energy sampling.

The engine is the same on every host: open a backend, run a low-rate
power poller, bracket each step with boundary power + energy reads,
and assemble ``StepResult`` / ``SessionResult`` records.

Vendor-specific logic lives entirely in ``matcha._backends.*`` — the
engine only sees the ``Backend`` protocol. Swapping NVIDIA for AMD,
Intel, or Apple Silicon changes nothing here.

Energy source
-------------

When the active backend reports ``has_energy_counter=True`` (NVML on
Volta+), per-step and session energy are exact counter deltas —
millijoule-precise, no integration error, zero per-step overhead.
Otherwise energy is trapezoidal integration of the polled power
series. The ``energy_source`` field on ``SessionResult`` reports which
path was taken.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

from ._backends import Backend, detect


Sample = Tuple[float, List[float]]  # (monotonic_ts, [w_gpu0, w_gpu1, ...])


@dataclass
class GpuStats:
    idx: int
    energy_j: float
    avg_power_w: float
    peak_power_w: float


@dataclass
class StepResult:
    step: int
    energy_j: float
    duration_s: float
    avg_power_w: float
    peak_power_w: float
    per_gpu: List[GpuStats] = field(default_factory=list)


@dataclass
class SessionResult:
    gpu_name: str
    total_energy_j: float
    total_duration_s: float
    avg_power_w: float
    peak_power_w: float
    total_samples: int
    energy_source: str = "counter"  # "counter" | "polled"
    backend: str = "nvml"
    per_gpu: List[GpuStats] = field(default_factory=list)

    @property
    def energy_wh(self) -> float:
        return self.total_energy_j / 3600.0


def _integrate_series(samples: List[Tuple[float, float]]) -> Tuple[float, float]:
    """Trapezoidal over (t, w) samples. Returns (energy_j, peak_w).

    Used when the backend has no hardware energy counter.
    """
    if len(samples) < 2:
        return 0.0, (samples[0][1] if samples else 0.0)
    energy_j = 0.0
    peak_w = 0.0
    for i in range(1, len(samples)):
        t0, p0 = samples[i - 1]
        t1, p1 = samples[i]
        energy_j += 0.5 * (p0 + p1) * (t1 - t0)
        if p0 > peak_w:
            peak_w = p0
        if p1 > peak_w:
            peak_w = p1
    return energy_j, peak_w


def _polled_stats(
    samples: List[Sample],
    gpu_indices: List[int],
    duration_s: float,
    running_peak_total_w: float,
) -> Tuple[float, float, float, List[GpuStats]]:
    """Fallback: derive totals + per-GPU stats from polled power samples."""
    per_gpu: List[GpuStats] = []
    total_energy_j = 0.0
    for g, idx in enumerate(gpu_indices):
        series = [(t, ws[g]) for t, ws in samples if g < len(ws)]
        e_j, peak_w = _integrate_series(series)
        avg_w = (e_j / duration_s) if duration_s > 0 else (series[0][1] if series else 0.0)
        per_gpu.append(GpuStats(idx=idx, energy_j=e_j, avg_power_w=avg_w, peak_power_w=peak_w))
        total_energy_j += e_j

    total_avg_w = total_energy_j / duration_s if duration_s > 0 else 0.0
    return total_energy_j, total_avg_w, running_peak_total_w, per_gpu


class PowerSampler:
    """GPU energy + peak-power tracker, vendor-agnostic.

    The sampler delegates every hardware read to a ``Backend``
    (``matcha._backends``). Energy is read from the backend's hardware
    counter when available; otherwise the engine integrates polled
    power to derive energy. Peak power always comes from a low-rate
    background poller plus per-step boundary reads — the boundary
    reads guarantee ≥2 data points per step even when the step is
    shorter than the sampling interval.
    """

    def __init__(
        self,
        gpu_indices: Union[int, List[int]] = -1,
        interval_ms: int = 100,
        backend: Optional[Backend] = None,
    ):
        self._gpu_spec = gpu_indices
        self._interval_s = interval_ms / 1000.0
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._running = False

        # Backend is resolved at ``start()`` time (not construction) to
        # mirror the old pynvml-lazy semantics: constructing a sampler
        # on a CPU-only CI host must stay side-effect-free.
        self._backend: Optional[Backend] = backend

        self._samples: List[Sample] = []
        self._peak_total_w = 0.0
        self._per_gpu_peak_w: List[float] = []
        self._session_start_t: Optional[float] = None

        self._session_start_energy_mj: List[int] = []
        self._step_start_t: Optional[float] = None
        self._step_start_energy_mj: Optional[List[int]] = None
        self._step_start_powers: Optional[List[float]] = None
        self._step_start_sample_idx: int = 0

        self.gpu_name: str = ""
        self.gpu_indices: List[int] = []
        self.gpu_names: List[str] = []
        self.gpu_uuids: List[str] = []
        self.driver_version: str = ""

        self.last_step: Optional[StepResult] = None
        self.steps_completed: int = 0
        self.session_step_energy_j: float = 0.0

        # Training metrics parsed from the wrapped process's stdout
        # (train_loss, lr, mfu, ...). Updated by the wrap loop via
        # update_train_metrics. Sticky: old keys persist until overwritten
        # so Prom/OTLP gauges stay continuous across sparse step lines.
        self.last_train_metrics: Dict[str, float] = {}

    # ---- public introspection (unchanged shape) --------------------------

    @property
    def backend(self) -> Optional[Backend]:
        return self._backend

    @property
    def backend_name(self) -> str:
        return self._backend.name if self._backend else ""

    @property
    def energy_source(self) -> str:
        return (
            "counter"
            if (self._backend is not None and self._backend.has_energy_counter)
            else "polled"
        )

    # ---- compatibility shim: some callers still read ``_handles`` --------
    # Exporters that used to index into ``sampler._handles`` now use
    # positional indices (0..device_count-1) against the backend. This
    # property keeps any lingering ``len(sampler._handles)`` checks
    # working until those call sites are fully migrated.
    @property
    def _handles(self) -> List[int]:
        return list(range(self._backend.device_count)) if self._backend else []

    def update_train_metrics(self, metrics: Dict[str, float]) -> None:
        if metrics:
            self.last_train_metrics = {**self.last_train_metrics, **metrics}

    # ---- lifecycle -------------------------------------------------------

    def start(self) -> None:
        if self._backend is None:
            self._backend = detect()
        self._backend.init(self._gpu_spec)

        self.gpu_indices = self._backend.device_indices
        self.gpu_names = self._backend.device_names
        self.gpu_uuids = self._backend.device_uuids
        self.driver_version = self._backend.driver_version

        n = self._backend.device_count
        self._per_gpu_peak_w = [0.0] * n

        first = self.gpu_names[0] if self.gpu_names else self._backend.name
        self.gpu_name = f"{n}x {first}" if n > 1 else first

        if self._backend.has_energy_counter:
            try:
                self._session_start_energy_mj = self._read_energy_mj()
            except Exception:
                # Backend advertised a counter but the first read
                # failed — demote to polled integration for the rest
                # of the session rather than dying mid-run.
                self._session_start_energy_mj = []

        self._session_start_t = time.monotonic()
        self._running = True
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def _read_per_gpu_power(self) -> List[float]:
        b = self._backend
        assert b is not None
        return [b.read_power_w(i) for i in range(b.device_count)]

    def _read_energy_mj(self) -> List[int]:
        b = self._backend
        assert b is not None
        return [b.read_energy_mj(i) for i in range(b.device_count)]

    def _sample_loop(self) -> None:
        while self._running:
            powers = self._read_per_gpu_power()
            ts = time.monotonic()
            total_w = sum(powers)
            with self._lock:
                self._samples.append((ts, powers))
                if total_w > self._peak_total_w:
                    self._peak_total_w = total_w
                for i, w in enumerate(powers):
                    if i < len(self._per_gpu_peak_w) and w > self._per_gpu_peak_w[i]:
                        self._per_gpu_peak_w[i] = w
            time.sleep(self._interval_s)

    def begin_step(self) -> None:
        t = time.monotonic()
        has_counter = self._backend is not None and self._backend.has_energy_counter
        energy_mj = self._read_energy_mj() if has_counter and self._session_start_energy_mj else None
        powers = self._read_per_gpu_power()
        with self._lock:
            self._step_start_t = t
            self._step_start_energy_mj = energy_mj
            self._step_start_powers = powers
            self._step_start_sample_idx = len(self._samples)

    def end_step(self, step: int) -> StepResult:
        end_t = time.monotonic()
        has_counter = self._backend is not None and self._backend.has_energy_counter
        end_energy_mj = self._read_energy_mj() if has_counter and self._session_start_energy_mj else None
        end_powers = self._read_per_gpu_power()
        with self._lock:
            start_t = self._step_start_t
            start_energy = self._step_start_energy_mj
            start_powers = self._step_start_powers
            start_idx = self._step_start_sample_idx
            window_samples = self._samples[start_idx:]
            self._step_start_t = None
            self._step_start_energy_mj = None
            self._step_start_powers = None

        duration_s = end_t - start_t if start_t else 0.0

        # Boundary power reads guarantee ≥2 data points for peak even when
        # step duration is shorter than the background sampling interval.
        augmented = []
        if start_powers is not None and start_t is not None:
            augmented.append((start_t, start_powers))
        augmented.extend(window_samples)
        augmented.append((end_t, end_powers))

        n = self._backend.device_count if self._backend else 0
        per_gpu_peak = [0.0] * n
        step_peak_total_w = 0.0
        for _, ws in augmented:
            total = 0.0
            for i, w in enumerate(ws):
                if i < len(per_gpu_peak) and w > per_gpu_peak[i]:
                    per_gpu_peak[i] = w
                total += w
            if total > step_peak_total_w:
                step_peak_total_w = total

        if has_counter and start_energy is not None and end_energy_mj is not None:
            per_gpu: List[GpuStats] = []
            total_energy_j = 0.0
            for i, idx in enumerate(self.gpu_indices):
                e_j = max(0.0, (end_energy_mj[i] - start_energy[i]) / 1000.0)
                avg_w = e_j / duration_s if duration_s > 0 else 0.0
                # Invariant: peak ≥ avg. If our sparse sampling hasn't seen a
                # value that high, the true peak was at least the average.
                peak_w = max(per_gpu_peak[i], avg_w)
                per_gpu.append(GpuStats(
                    idx=idx, energy_j=e_j, avg_power_w=avg_w, peak_power_w=peak_w
                ))
                total_energy_j += e_j
            total_avg_w = total_energy_j / duration_s if duration_s > 0 else 0.0
            total_peak_w = max(step_peak_total_w, total_avg_w)
        else:
            total_energy_j, total_avg_w, total_peak_w, per_gpu = _polled_stats(
                window_samples, self.gpu_indices, duration_s, step_peak_total_w
            )

        result = StepResult(
            step=step,
            energy_j=total_energy_j,
            duration_s=duration_s,
            avg_power_w=total_avg_w,
            peak_power_w=total_peak_w,
            per_gpu=per_gpu,
        )
        self.last_step = result
        self.steps_completed += 1
        self.session_step_energy_j += total_energy_j
        return result

    def stop(self) -> SessionResult:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

        end_t = time.monotonic()
        has_counter = self._backend is not None and self._backend.has_energy_counter
        end_energy_mj = self._read_energy_mj() if has_counter and self._session_start_energy_mj else None
        duration_s = end_t - self._session_start_t if self._session_start_t else 0.0

        if has_counter and end_energy_mj is not None and self._session_start_energy_mj:
            per_gpu: List[GpuStats] = []
            total_energy_j = 0.0
            for i, idx in enumerate(self.gpu_indices):
                e_j = max(0.0, (end_energy_mj[i] - self._session_start_energy_mj[i]) / 1000.0)
                avg_w = e_j / duration_s if duration_s > 0 else 0.0
                polled_peak = self._per_gpu_peak_w[i] if i < len(self._per_gpu_peak_w) else 0.0
                peak = max(polled_peak, avg_w)
                per_gpu.append(GpuStats(
                    idx=idx, energy_j=e_j, avg_power_w=avg_w, peak_power_w=peak
                ))
                total_energy_j += e_j
            total_avg_w = total_energy_j / duration_s if duration_s > 0 else 0.0
            total_peak_w = max(self._peak_total_w, total_avg_w)
        else:
            total_energy_j, total_avg_w, total_peak_w, per_gpu = _polled_stats(
                self._samples, self.gpu_indices, duration_s, self._peak_total_w
            )

        if self._backend is not None:
            try:
                self._backend.shutdown()
            except Exception:
                pass

        return SessionResult(
            gpu_name=self.gpu_name,
            total_energy_j=total_energy_j,
            total_duration_s=duration_s,
            avg_power_w=total_avg_w,
            peak_power_w=total_peak_w,
            total_samples=len(self._samples),
            energy_source=self.energy_source,
            backend=self._backend.name if self._backend else "",
            per_gpu=per_gpu,
        )
