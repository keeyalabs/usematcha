"""
matcha._engine — GPU power sampling engine.

Copyright (c) 2025 Keeya Labs. All rights reserved.
This source code is proprietary and confidential.
Unauthorized copying or distribution is strictly prohibited.
"""

import threading
import time
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Union

import warnings
try:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        import pynvml
    _HAS_NVML = True
except ImportError:
    _HAS_NVML = False


@dataclass
class StepResult:
    """Energy measurement for a single training step."""
    step: int
    energy_j: float
    duration_s: float
    avg_power_w: float
    peak_power_w: float
    samples: int

    @property
    def energy_mwh(self) -> float:
        return self.energy_j / 3.6

    @property
    def energy_kwh(self) -> float:
        return self.energy_j / 3_600_000

    def __repr__(self) -> str:
        return (
            f"Step {self.step}: {self.energy_j:.2f} J "
            f"({self.energy_mwh:.2f} mWh) | "
            f"{self.duration_s:.3f}s | "
            f"avg {self.avg_power_w:.1f}W peak {self.peak_power_w:.1f}W"
        )


@dataclass
class SessionResult:
    """Cumulative energy stats for an entire training run."""
    total_energy_j: float
    total_duration_s: float
    total_steps: int
    avg_power_w: float
    peak_power_w: float
    total_samples: int
    gpu_name: str
    gpu_count: int
    steps: List

    @property
    def energy_wh(self) -> float:
        return self.total_energy_j / 3600

    @property
    def energy_kwh(self) -> float:
        return self.total_energy_j / 3_600_000

    @property
    def j_per_step(self) -> float:
        return self.total_energy_j / self.total_steps if self.total_steps else 0.0


class _Collector:
    """Internal power collection engine. Do not use directly.

    Args:
        _gpus: GPU indices to monitor.
              -1 = auto-detect all GPUs
              int = single GPU index
              list[int] = specific GPU indices
        _iv: sampling interval in milliseconds
    """

    def __init__(self, _gpus: Union[int, List[int]] = -1, _iv: int = 100):
        if not _HAS_NVML:
            raise ImportError("pynvml is required: pip install nvidia-ml-py")

        self._gpu_spec = _gpus
        self._iv = _iv / 1000.0
        self._handles: List = []
        self._gpu_indices: List[int] = []
        self._t: Optional[threading.Thread] = None
        self._r = False
        self._lk = threading.Lock()

        # step-level readings: (timestamp, total_power_w)
        self._ss: Optional[float] = None
        self._sp: List[Tuple[float, float]] = []

        # session-level
        self._t0: Optional[float] = None
        self._pk: float = 0.0
        self._buf: List[Tuple[float, float]] = []
        self._done: List[StepResult] = []

        self.gpu_name: str = ""
        self.gpu_count: int = 0
        self.power_limit_w: float = 0.0

    def start(self):
        pynvml.nvmlInit()

        # Resolve GPU indices
        _dc = pynvml.nvmlDeviceGetCount()

        if self._gpu_spec == -1:
            self._gpu_indices = list(range(_dc))
        elif isinstance(self._gpu_spec, int):
            self._gpu_indices = [self._gpu_spec]
        else:
            self._gpu_indices = list(self._gpu_spec)

        # Get handles
        self._handles = []
        _tdp = 0.0
        for _i in self._gpu_indices:
            _h = pynvml.nvmlDeviceGetHandleByIndex(_i)
            self._handles.append(_h)
            try:
                _tdp += pynvml.nvmlDeviceGetPowerManagementLimit(_h) / 1000.0
            except pynvml.NVMLError:
                pass

        self.gpu_count = len(self._handles)
        self.power_limit_w = _tdp

        # GPU name from first device
        _n = pynvml.nvmlDeviceGetName(self._handles[0])
        self.gpu_name = _n.decode("utf-8") if isinstance(_n, bytes) else _n
        if self.gpu_count > 1:
            self.gpu_name = f"{self.gpu_count}x {self.gpu_name}"

        self._t0 = time.monotonic()
        self._r = True
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()

    def _read_total_power(self) -> float:
        """Read and sum power across all monitored GPUs."""
        _total = 0.0
        for _h in self._handles:
            try:
                _total += pynvml.nvmlDeviceGetPowerUsage(_h) / 1000.0
            except pynvml.NVMLError:
                pass
        return _total

    def _loop(self):
        while self._r:
            _pw = self._read_total_power()
            _ts = time.monotonic()
            with self._lk:
                self._buf.append((_ts, _pw))
                self._pk = max(self._pk, _pw)
                if self._ss is not None:
                    self._sp.append((_ts, _pw))
            time.sleep(self._iv)

    def mark_start(self):
        with self._lk:
            self._ss = time.monotonic()
            self._sp = []

    def mark_end(self, step: int) -> StepResult:
        with self._lk:
            _rd = self._sp
            _s0 = self._ss
            self._ss = None
            self._sp = []

        _s1 = time.monotonic()
        _dur = _s1 - _s0 if _s0 else 0.0

        if len(_rd) < 2:
            _pw = _rd[0][1] if _rd else 0.0
            _ej = _pw * _dur
            _res = StepResult(step=step, energy_j=_ej, duration_s=_dur,
                              avg_power_w=_pw, peak_power_w=_pw, samples=len(_rd))
        else:
            _ej = 0.0
            _mx = 0.0
            _sm = 0.0
            for _i in range(1, len(_rd)):
                _t0, _p0 = _rd[_i - 1]
                _t1, _p1 = _rd[_i]
                _ej += 0.5 * (_p0 + _p1) * (_t1 - _t0)
                _mx = max(_mx, _p0, _p1)
                _sm += _p0
            _sm += _rd[-1][1]
            _mx = max(_mx, _rd[-1][1])

            _res = StepResult(step=step, energy_j=_ej, duration_s=_dur,
                              avg_power_w=_sm / len(_rd), peak_power_w=_mx,
                              samples=len(_rd))

        self._done.append(_res)
        return _res

    def stop(self) -> SessionResult:
        self._r = False
        if self._t:
            self._t.join(timeout=2.0)

        _end = time.monotonic()
        _dur = _end - self._t0 if self._t0 else 0.0

        _rd = self._buf
        _te = 0.0
        if len(_rd) >= 2:
            for _i in range(1, len(_rd)):
                _t0, _p0 = _rd[_i - 1]
                _t1, _p1 = _rd[_i]
                _te += 0.5 * (_p0 + _p1) * (_t1 - _t0)

        _ap = _te / _dur if _dur > 0 else 0.0

        try:
            pynvml.nvmlShutdown()
        except pynvml.NVMLError:
            pass

        return SessionResult(
            total_energy_j=_te, total_duration_s=_dur,
            total_steps=len(self._done), avg_power_w=_ap,
            peak_power_w=self._pk, total_samples=len(_rd),
            gpu_name=self.gpu_name, gpu_count=self.gpu_count,
            steps=self._done,
        )
