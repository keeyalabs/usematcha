# SPDX-License-Identifier: Apache-2.0
"""matcha._backends.intel — Intel GPU backend via ``xpu-smi``.

Intel datacenter / discrete GPUs (Arc, Flex, PVC, Battlemage) expose
observability through ``xpu-smi``, the Intel XPU Manager CLI. This
backend wraps it the same way the ROCm backend wraps ``rocm-smi``:
``xpu-smi stats -j`` for live readings, parsed and cached behind a
refresher thread.

Energy counter
--------------

Ponte Vecchio and newer Arc / Battlemage parts expose a firmware
energy counter through Level Zero (``zetSysmanPowerGetEnergyCounter``).
Exposing that path cleanly requires the ``pyze_l0`` binding and
card-specific probing. For the first release matcha sticks to xpu-smi
power and integrates for energy. On newer ``xpu-smi`` versions the
``stats`` output includes ``GPU Energy Used`` directly — we read it
opportunistically as a polled series rather than a true counter delta,
which is sufficient for per-step attribution.

Hardware coverage
-----------------

``xpu-smi`` is only shipped with Intel's compute runtime install (it is
not part of mesa or intel-gpu-tools). Hosts running Arc cards through
the consumer driver stack typically use ``intel_gpu_top`` instead,
which this backend does not currently speak. Adding ``intel_gpu_top``
as a secondary path is a natural follow-up.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import threading
import time
from typing import List, Optional, Union

from ._base import Backend, BackendUnavailable


_REFRESH_INTERVAL_S = 0.5
_XPU_SMI_TIMEOUT_S = 3.0


def _xpu_smi_path() -> Optional[str]:
    return shutil.which("xpu-smi")


_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _first_float(s) -> float:
    if isinstance(s, (int, float)):
        return float(s)
    if isinstance(s, str):
        m = _NUMBER_RE.search(s)
        if m:
            try:
                return float(m.group(0))
            except ValueError:
                return 0.0
    return 0.0


def _run_xpu_smi(args: List[str]) -> Optional[dict]:
    path = _xpu_smi_path()
    if not path:
        return None
    try:
        out = subprocess.check_output(
            [path, *args, "-j"],
            timeout=_XPU_SMI_TIMEOUT_S,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return None
    try:
        return json.loads(out.decode("utf-8", errors="replace"))
    except Exception:
        return None


class IntelBackend(Backend):
    name = "intel"

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._refresher: Optional[threading.Thread] = None
        self._running = False

        # xpu-smi's device IDs are small ints — we remember the ones we
        # were told to track, not a 0..N positional index, because a
        # host can have non-contiguous device IDs after PCI hot-remove.
        self._xpu_ids: List[int] = []
        self._indices: List[int] = []
        self._names: List[str] = []
        self._uuids: List[str] = []
        self._tdps: List[float] = []
        self._mem_totals: List[int] = []
        self._driver_version: str = ""

        self._powers: List[float] = []
        self._utils: List[int] = []
        self._temps: List[int] = []
        self._mem_used: List[int] = []

        self._initialized = False

    # ---- availability ---------------------------------------------------

    def is_available(self) -> bool:
        return _xpu_smi_path() is not None

    def unavailable_reason(self) -> str:
        return "xpu-smi not found on PATH (install Intel XPU Manager)"

    # ---- lifecycle ------------------------------------------------------

    def init(self, gpu_spec: Union[int, List[int]] = -1) -> None:
        disc = _run_xpu_smi(["discovery"])
        if disc is None:
            raise BackendUnavailable(
                "matcha: xpu-smi is present but `xpu-smi discovery` failed."
            )

        devices = disc.get("device_list") or []
        if not devices:
            raise BackendUnavailable("matcha: xpu-smi reported zero devices")

        all_xpu_ids = [int(d.get("device_id", idx)) for idx, d in enumerate(devices)]
        if gpu_spec == -1:
            indices = list(range(len(devices)))
        elif isinstance(gpu_spec, int):
            indices = [gpu_spec]
        else:
            indices = list(gpu_spec)

        for i in indices:
            if i < 0 or i >= len(devices):
                raise ValueError(
                    f"matcha: intel device index {i} out of range 0..{len(devices) - 1}"
                )

        self._indices = indices
        self._xpu_ids = [all_xpu_ids[i] for i in indices]

        for i in indices:
            d = devices[i]
            self._names.append(str(d.get("device_name") or d.get("name") or "Intel GPU"))
            self._uuids.append(str(d.get("uuid") or d.get("pci_bdf") or ""))
            self._tdps.append(_first_float(d.get("max_power") or d.get("tdp") or 0))
            # xpu-smi discovery often reports memory in MiB strings
            mem_mib = _first_float(d.get("memory_physical_size") or 0)
            self._mem_totals.append(int(mem_mib * 1024 * 1024))

        self._driver_version = str(disc.get("driver_version") or "")

        self._powers = [0.0] * len(self._xpu_ids)
        self._utils = [0] * len(self._xpu_ids)
        self._temps = [0] * len(self._xpu_ids)
        self._mem_used = [0] * len(self._xpu_ids)

        self._refresh_once()

        self._running = True
        self._refresher = threading.Thread(target=self._refresh_loop, daemon=True)
        self._refresher.start()
        self._initialized = True

    def shutdown(self) -> None:
        if not self._initialized:
            return
        self._running = False
        if self._refresher:
            self._refresher.join(timeout=2.0)
        self._initialized = False

    # ---- refresher -----------------------------------------------------

    def _refresh_loop(self) -> None:
        while self._running:
            self._refresh_once()
            time.sleep(_REFRESH_INTERVAL_S)

    def _refresh_once(self) -> None:
        powers, utils, temps, mem_used = [], [], [], []
        for xid in self._xpu_ids:
            stats = _run_xpu_smi(["stats", "-d", str(xid)])
            p = u = t = m = 0
            p = 0.0
            if stats and "device_level" in stats:
                dl = stats["device_level"] or {}
                # ``device_level`` fields mirror the xpu-smi top-line
                # summary card and are stable across versions.
                p = _first_float(dl.get("power"))
                u = int(_first_float(dl.get("gpu_utilization")))
                t = int(_first_float(dl.get("gpu_core_temperature")))
                m = int(_first_float(dl.get("memory_used"))) * 1024 * 1024  # MiB → B
            powers.append(p)
            utils.append(u)
            temps.append(t)
            mem_used.append(m)
        with self._lock:
            self._powers = powers
            self._utils = utils
            self._temps = temps
            self._mem_used = mem_used

    # ---- metadata ------------------------------------------------------

    @property
    def device_count(self) -> int:
        return len(self._indices)

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
        # Level-Zero energy counter integration is deferred; polled
        # power + trapezoidal integration is used on all Intel parts
        # for now.
        return False

    def device_tdp_w(self, i: int) -> float:
        return self._tdps[i] if i < len(self._tdps) else 0.0

    def device_memory_total_bytes(self, i: int) -> int:
        return self._mem_totals[i] if i < len(self._mem_totals) else 0

    # ---- reads ---------------------------------------------------------

    def read_power_w(self, i: int) -> float:
        with self._lock:
            return self._powers[i] if i < len(self._powers) else 0.0

    def read_utilization_pct(self, i: int) -> int:
        with self._lock:
            return self._utils[i] if i < len(self._utils) else 0

    def read_temperature_c(self, i: int) -> int:
        with self._lock:
            return self._temps[i] if i < len(self._temps) else 0

    def read_memory_used_bytes(self, i: int) -> int:
        with self._lock:
            return self._mem_used[i] if i < len(self._mem_used) else 0
