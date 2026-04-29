# SPDX-License-Identifier: Apache-2.0
"""matcha._backends.nvml — NVIDIA backend via NVML (pynvml / nvidia-ml-py).

This is matcha's original backend and the fastest path. Every read is
an in-process NVML call (microseconds), and energy comes straight from
``nvmlDeviceGetTotalEnergyConsumption`` — the hardware accumulator on
Volta+. No subprocesses, no integration error.

For pre-Volta GPUs the energy counter call raises; the backend reports
``has_energy_counter=False`` and the engine falls back to trapezoidal
integration of polled power samples.
"""

from __future__ import annotations

import warnings
from typing import List, Union

from ._base import Backend

try:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        import pynvml  # type: ignore
except ImportError:
    pynvml = None  # type: ignore


def _decode(x):
    return x.decode("utf-8") if isinstance(x, bytes) else x


class NvmlBackend(Backend):
    name = "nvml"

    def __init__(self) -> None:
        self._handles: List = []
        self._indices: List[int] = []
        self._names: List[str] = []
        self._uuids: List[str] = []
        self._tdps: List[float] = []
        self._mem_totals: List[int] = []
        self._driver_version: str = ""
        self._energy_counter_available: bool = False
        self._initialized: bool = False

    # ---- lifecycle --------------------------------------------------------

    def is_available(self) -> bool:
        if pynvml is None:
            return False
        # Probing nvmlInit is the most reliable check ("are NVIDIA drivers
        # loaded and functional?"). It's cheap, but not strictly
        # side-effect-free — we shut it down immediately so a successful
        # detect() doesn't leave NVML in an initialized state.
        try:
            pynvml.nvmlInit()
            try:
                pynvml.nvmlDeviceGetCount()
            finally:
                try:
                    pynvml.nvmlShutdown()
                except Exception:
                    pass
            return True
        except Exception:
            return False

    def unavailable_reason(self) -> str:
        if pynvml is None:
            return "nvidia-ml-py not installed"
        return "NVML init failed (no NVIDIA driver?)"

    def init(self, gpu_spec: Union[int, List[int]] = -1) -> None:
        if pynvml is None:
            raise ImportError("matcha: nvidia-ml-py is required for the NVML backend")

        try:
            pynvml.nvmlInit()
        except Exception as e:
            raise RuntimeError(
                "matcha: NVML init failed — is an NVIDIA GPU present "
                f"with drivers installed? ({e})"
            )

        count = pynvml.nvmlDeviceGetCount()
        if gpu_spec == -1:
            indices = list(range(count))
        elif isinstance(gpu_spec, int):
            indices = [gpu_spec]
        else:
            indices = list(gpu_spec)

        self._handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in indices]
        self._indices = indices

        for h in self._handles:
            self._names.append(_decode(pynvml.nvmlDeviceGetName(h)))
            try:
                self._uuids.append(_decode(pynvml.nvmlDeviceGetUUID(h)))
            except pynvml.NVMLError:
                self._uuids.append("")
            try:
                self._tdps.append(pynvml.nvmlDeviceGetPowerManagementLimit(h) / 1000.0)
            except pynvml.NVMLError:
                self._tdps.append(0.0)
            try:
                self._mem_totals.append(pynvml.nvmlDeviceGetMemoryInfo(h).total)
            except pynvml.NVMLError:
                self._mem_totals.append(0)

        try:
            self._driver_version = _decode(pynvml.nvmlSystemGetDriverVersion())
        except pynvml.NVMLError:
            self._driver_version = ""

        # Probe the energy counter once — if the first device supports it,
        # assume all devices on the host do (they share a driver).
        try:
            if self._handles:
                pynvml.nvmlDeviceGetTotalEnergyConsumption(self._handles[0])
                self._energy_counter_available = True
        except pynvml.NVMLError:
            self._energy_counter_available = False

        self._initialized = True

    def shutdown(self) -> None:
        if not self._initialized:
            return
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass
        self._initialized = False

    # ---- metadata --------------------------------------------------------

    @property
    def device_count(self) -> int:
        return len(self._handles)

    @property
    def device_indices(self) -> List[int]:
        return list(self._indices)

    @property
    def device_names(self) -> List[str]:
        return list(self._names)

    @property
    def device_uuids(self) -> List[str]:
        return list(self._uuids)

    @property
    def driver_version(self) -> str:
        return self._driver_version

    @property
    def has_energy_counter(self) -> bool:
        return self._energy_counter_available

    def device_tdp_w(self, i: int) -> float:
        return self._tdps[i] if i < len(self._tdps) else 0.0

    def device_memory_total_bytes(self, i: int) -> int:
        return self._mem_totals[i] if i < len(self._mem_totals) else 0

    # ---- reads ------------------------------------------------------------

    def _handle(self, i: int):
        return self._handles[i]

    def read_power_w(self, i: int) -> float:
        try:
            return pynvml.nvmlDeviceGetPowerUsage(self._handle(i)) / 1000.0
        except pynvml.NVMLError:
            return 0.0

    def read_energy_mj(self, i: int) -> int:
        # Raises pynvml.NVMLError on pre-Volta; caller should have
        # checked has_energy_counter first.
        return pynvml.nvmlDeviceGetTotalEnergyConsumption(self._handle(i))

    def read_utilization_pct(self, i: int) -> int:
        try:
            return pynvml.nvmlDeviceGetUtilizationRates(self._handle(i)).gpu
        except pynvml.NVMLError:
            return 0

    def read_temperature_c(self, i: int) -> int:
        try:
            return pynvml.nvmlDeviceGetTemperature(
                self._handle(i), pynvml.NVML_TEMPERATURE_GPU
            )
        except pynvml.NVMLError:
            return 0

    def read_memory_used_bytes(self, i: int) -> int:
        try:
            return pynvml.nvmlDeviceGetMemoryInfo(self._handle(i)).used
        except pynvml.NVMLError:
            return 0
