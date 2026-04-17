# SPDX-License-Identifier: Apache-2.0
"""matcha._engine — GPU energy sampling.

Energy is read from NVML's hardware accumulator
(`nvmlDeviceGetTotalEnergyConsumption`, supported Volta+). Per-step and
session energy are exact counter deltas — no integration error, no
overhead from a tight polling loop.

A background thread still polls power at a low rate (default 500 ms) for
peak-power tracking and per-GPU peak stats. If the hardware counter is
unavailable, energy falls back to trapezoidal integration of the polled
power samples.
"""

import threading
import time
import warnings
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

try:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        import pynvml
except ImportError:
    pynvml = None


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
    per_gpu: List[GpuStats] = field(default_factory=list)

    @property
    def energy_wh(self) -> float:
        return self.total_energy_j / 3600.0


def _integrate_series(samples: List[Tuple[float, float]]) -> Tuple[float, float]:
    """Trapezoidal over (t, w) samples. Returns (energy_j, peak_w).

    Used only when the hardware energy counter is unavailable.
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
    """GPU energy + peak-power tracker.

    Energy is read directly from NVML's hardware accumulator
    (millijoule-precise, no integration error, zero overhead per step).
    Peak power is tracked by a low-rate background poller.
    """

    def __init__(self, gpu_indices: Union[int, List[int]] = -1, interval_ms: int = 100):
        if pynvml is None:
            raise ImportError("matcha requires nvidia-ml-py: pip install nvidia-ml-py")

        self._gpu_spec = gpu_indices
        self._interval_s = interval_ms / 1000.0
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._handles: List = []

        self._samples: List[Sample] = []
        self._peak_total_w = 0.0
        self._per_gpu_peak_w: List[float] = []
        self._session_start_t: Optional[float] = None

        self._energy_counter_available = False
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

    @property
    def energy_source(self) -> str:
        return "counter" if self._energy_counter_available else "polled"

    def start(self) -> None:
        try:
            pynvml.nvmlInit()
        except Exception as e:
            raise RuntimeError(
                "matcha: NVML init failed — is an NVIDIA GPU present "
                f"with drivers installed? ({e})"
            )

        count = pynvml.nvmlDeviceGetCount()
        if self._gpu_spec == -1:
            indices = list(range(count))
        elif isinstance(self._gpu_spec, int):
            indices = [self._gpu_spec]
        else:
            indices = list(self._gpu_spec)

        self._handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in indices]
        self.gpu_indices = indices
        self._per_gpu_peak_w = [0.0] * len(self._handles)

        def _decode(x):
            return x.decode("utf-8") if isinstance(x, bytes) else x

        for h in self._handles:
            self.gpu_names.append(_decode(pynvml.nvmlDeviceGetName(h)))
            try:
                self.gpu_uuids.append(_decode(pynvml.nvmlDeviceGetUUID(h)))
            except pynvml.NVMLError:
                self.gpu_uuids.append("")

        try:
            self.driver_version = _decode(pynvml.nvmlSystemGetDriverVersion())
        except pynvml.NVMLError:
            self.driver_version = ""

        first = self.gpu_names[0]
        self.gpu_name = f"{len(self._handles)}x {first}" if len(self._handles) > 1 else first

        try:
            self._session_start_energy_mj = self._read_energy_mj()
            self._energy_counter_available = True
        except pynvml.NVMLError:
            self._energy_counter_available = False
            self._session_start_energy_mj = []

        self._session_start_t = time.monotonic()
        self._running = True
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def _read_per_gpu_power(self) -> List[float]:
        powers: List[float] = []
        for h in self._handles:
            try:
                powers.append(pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0)
            except pynvml.NVMLError:
                powers.append(0.0)
        return powers

    def _read_energy_mj(self) -> List[int]:
        return [pynvml.nvmlDeviceGetTotalEnergyConsumption(h) for h in self._handles]

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
        energy_mj = self._read_energy_mj() if self._energy_counter_available else None
        powers = self._read_per_gpu_power()
        with self._lock:
            self._step_start_t = t
            self._step_start_energy_mj = energy_mj
            self._step_start_powers = powers
            self._step_start_sample_idx = len(self._samples)

    def end_step(self, step: int) -> StepResult:
        end_t = time.monotonic()
        end_energy_mj = self._read_energy_mj() if self._energy_counter_available else None
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

        per_gpu_peak = [0.0] * len(self._handles)
        step_peak_total_w = 0.0
        for _, ws in augmented:
            total = 0.0
            for i, w in enumerate(ws):
                if i < len(per_gpu_peak) and w > per_gpu_peak[i]:
                    per_gpu_peak[i] = w
                total += w
            if total > step_peak_total_w:
                step_peak_total_w = total

        if self._energy_counter_available and start_energy is not None and end_energy_mj is not None:
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

        return StepResult(
            step=step,
            energy_j=total_energy_j,
            duration_s=duration_s,
            avg_power_w=total_avg_w,
            peak_power_w=total_peak_w,
            per_gpu=per_gpu,
        )

    def stop(self) -> SessionResult:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

        end_t = time.monotonic()
        end_energy_mj = self._read_energy_mj() if self._energy_counter_available else None
        duration_s = end_t - self._session_start_t if self._session_start_t else 0.0

        if self._energy_counter_available and end_energy_mj is not None:
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

        try:
            pynvml.nvmlShutdown()
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
            per_gpu=per_gpu,
        )
