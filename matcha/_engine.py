"""matcha._engine — GPU power sampling."""

import threading
import time
import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

try:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        import pynvml
except ImportError:
    pynvml = None


@dataclass
class StepResult:
    step: int
    energy_j: float
    duration_s: float
    avg_power_w: float
    peak_power_w: float


@dataclass
class SessionResult:
    gpu_name: str
    total_energy_j: float
    total_duration_s: float
    avg_power_w: float
    peak_power_w: float
    total_samples: int

    @property
    def energy_wh(self) -> float:
        return self.total_energy_j / 3600.0


def _integrate(samples: List[Tuple[float, float]]) -> Tuple[float, float]:
    """Trapezoidal energy integration over (timestamp_s, power_w) samples.

    Returns (energy_j, peak_w).
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


class PowerSampler:
    """Background-thread sampler of total GPU power (summed across devices).

    Continuous samples feed end-of-run totals. Optional per-step windows
    (begin_step / end_step) give per-step energy for `matcha wrap`.
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

        self._samples: List[Tuple[float, float]] = []
        self._peak_w = 0.0
        self._session_start_t: Optional[float] = None

        self._step_samples: Optional[List[Tuple[float, float]]] = None
        self._step_start_t: Optional[float] = None

        self.gpu_name: str = ""
        self.gpu_indices: List[int] = []
        self.gpu_names: List[str] = []
        self.gpu_uuids: List[str] = []
        self.driver_version: str = ""

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

        self._session_start_t = time.monotonic()
        self._running = True
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def _read_total_power(self) -> float:
        total_w = 0.0
        for h in self._handles:
            try:
                total_w += pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0
            except pynvml.NVMLError:
                pass
        return total_w

    def _sample_loop(self) -> None:
        while self._running:
            power_w = self._read_total_power()
            ts = time.monotonic()
            with self._lock:
                self._samples.append((ts, power_w))
                if power_w > self._peak_w:
                    self._peak_w = power_w
                if self._step_samples is not None:
                    self._step_samples.append((ts, power_w))
            time.sleep(self._interval_s)

    def begin_step(self) -> None:
        with self._lock:
            self._step_start_t = time.monotonic()
            self._step_samples = []

    def end_step(self, step: int) -> StepResult:
        with self._lock:
            samples = self._step_samples or []
            start_t = self._step_start_t
            self._step_samples = None
            self._step_start_t = None

        duration_s = time.monotonic() - start_t if start_t else 0.0
        energy_j, peak_w = _integrate(samples)
        avg_power_w = (energy_j / duration_s) if duration_s > 0 else (
            samples[0][1] if samples else 0.0
        )

        return StepResult(
            step=step,
            energy_j=energy_j,
            duration_s=duration_s,
            avg_power_w=avg_power_w,
            peak_power_w=peak_w,
        )

    def stop(self) -> SessionResult:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

        duration_s = time.monotonic() - self._session_start_t if self._session_start_t else 0.0
        energy_j, _ = _integrate(self._samples)
        avg_power_w = energy_j / duration_s if duration_s > 0 else 0.0

        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass

        return SessionResult(
            gpu_name=self.gpu_name,
            total_energy_j=energy_j,
            total_duration_s=duration_s,
            avg_power_w=avg_power_w,
            peak_power_w=self._peak_w,
            total_samples=len(self._samples),
        )
