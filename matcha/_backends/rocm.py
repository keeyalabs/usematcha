# SPDX-License-Identifier: Apache-2.0
"""matcha._backends.rocm — AMD backend via ``rocm-smi``.

ROCm ships ``rocm-smi`` as a system binary. This backend invokes it
with ``--json`` to get a structured, version-stable-enough snapshot of
every card's power, utilization, temperature, and memory use.

Energy source
-------------

MI200-class and newer accelerators expose a hardware energy counter
(``rocm-smi --showenergycounter``), but the counter unit and scale
change between ROCm minor versions and cards, and parsing is brittle.
For the first release we report ``has_energy_counter=False`` and rely
on matcha's polled-power integration — correct on every ROCm GPU,
including consumer Radeon cards that lack the counter entirely.

A future version can plug in ``amdsmi`` (the official Python binding
shipped with ROCm 5.7+) for counter-based energy on MI200+.

Refresh model
-------------

``rocm-smi --json`` costs 100–300 ms per invocation. We don't want to
spend that on every sampler tick, so this backend runs its own
refresher thread that polls rocm-smi at ~500 ms and serves all reads
from a cache. The engine's own sampler thread stays responsive.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import threading
import time
from typing import Dict, List, Optional, Union

from ._base import Backend, BackendUnavailable


_REFRESH_INTERVAL_S = 0.5
_ROCM_SMI_TIMEOUT_S = 3.0


def _rocm_smi_path() -> Optional[str]:
    return shutil.which("rocm-smi")


def _run_rocm_smi(args: List[str]) -> Optional[dict]:
    """Run ``rocm-smi <args> --json`` and return the parsed payload.

    Returns None on subprocess failure, timeout, or parse error — the
    caller keeps serving the previous cached snapshot.
    """
    path = _rocm_smi_path()
    if not path:
        return None
    try:
        out = subprocess.check_output(
            [path, *args, "--json"],
            timeout=_ROCM_SMI_TIMEOUT_S,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return None
    try:
        return json.loads(out.decode("utf-8", errors="replace"))
    except Exception:
        return None


_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _first_float(s) -> float:
    """Extract the first numeric token from a rocm-smi scalar string.

    rocm-smi values are often ``"45.123 W"`` or ``"55.0"`` or a bare
    number depending on ROCm version. Accept all shapes.
    """
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


def _find_field(d: Dict, *substrings: str) -> Optional[str]:
    """Find the first dict key that case-insensitively contains all substrings.

    rocm-smi keys are verbose, version-dependent, and unit-suffixed —
    ``"Average Graphics Package Power (W)"``, ``"GPU use (%)"``,
    ``"Temperature (Sensor edge) (C)"``. Substring matching is
    deliberately forgiving.
    """
    needles = [s.lower() for s in substrings]
    for k in d:
        lk = k.lower()
        if all(n in lk for n in needles):
            return k
    return None


class RocmBackend(Backend):
    name = "rocm"

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._refresher: Optional[threading.Thread] = None
        self._running = False

        self._card_keys: List[str] = []  # e.g. ["card0", "card1", ...]
        self._indices: List[int] = []
        self._names: List[str] = []
        self._uuids: List[str] = []
        self._tdps: List[float] = []
        self._mem_totals: List[int] = []
        self._driver_version: str = ""

        # Live readings, parallel to _indices.
        self._powers: List[float] = []
        self._utils: List[int] = []
        self._temps: List[int] = []
        self._mem_used: List[int] = []

        self._initialized = False

    # ---- availability ---------------------------------------------------

    def is_available(self) -> bool:
        path = _rocm_smi_path()
        if not path:
            return False
        # Cheap existence check is enough — actually invoking rocm-smi
        # can be slow, and init() already has to do that.
        return True

    def unavailable_reason(self) -> str:
        return "rocm-smi not found on PATH"

    # ---- lifecycle ------------------------------------------------------

    def init(self, gpu_spec: Union[int, List[int]] = -1) -> None:
        snapshot = _run_rocm_smi(
            ["--showproductname", "--showuniqueid", "--showmeminfo", "vram",
             "--showpower", "--showpowercap", "--showtemp", "--showuse"]
        )
        if snapshot is None:
            raise BackendUnavailable(
                "matcha: rocm-smi is present but did not return a usable "
                "snapshot. Check `rocm-smi --json` manually and ensure the "
                "ROCm kernel module is loaded."
            )

        card_keys = sorted(
            [k for k in snapshot.keys() if k.startswith("card")],
            key=lambda k: int(k[4:]) if k[4:].isdigit() else 0,
        )
        if not card_keys:
            raise BackendUnavailable("matcha: rocm-smi reported zero cards")

        all_indices = list(range(len(card_keys)))
        if gpu_spec == -1:
            indices = all_indices
        elif isinstance(gpu_spec, int):
            indices = [gpu_spec]
        else:
            indices = list(gpu_spec)
        for i in indices:
            if i < 0 or i >= len(card_keys):
                raise ValueError(f"matcha: rocm card index {i} out of range 0..{len(card_keys) - 1}")

        self._card_keys = [card_keys[i] for i in indices]
        self._indices = indices

        for key in self._card_keys:
            card = snapshot.get(key, {})
            name_f = _find_field(card, "card series") or _find_field(card, "product name") \
                or _find_field(card, "card model") or _find_field(card, "name")
            self._names.append(str(card.get(name_f, "AMD GPU")) if name_f else "AMD GPU")

            uid_f = _find_field(card, "unique id") or _find_field(card, "serial")
            self._uuids.append(str(card.get(uid_f, "")) if uid_f else "")

            cap_f = _find_field(card, "max graphics package power") \
                or _find_field(card, "power cap")
            self._tdps.append(_first_float(card.get(cap_f)) if cap_f else 0.0)

            mem_total_f = _find_field(card, "vram total memory") \
                or _find_field(card, "total memory")
            self._mem_totals.append(int(_first_float(card.get(mem_total_f, 0))) if mem_total_f else 0)

        # rocm-smi doesn't expose a standalone driver-version field, but
        # ``rocm-smi --version`` returns the ROCm-SMI version string —
        # good enough for the ``driver_version`` record.
        try:
            v = subprocess.check_output(
                [_rocm_smi_path(), "--version"],
                timeout=_ROCM_SMI_TIMEOUT_S,
                stderr=subprocess.DEVNULL,
            )
            self._driver_version = v.decode("utf-8", errors="replace").strip().splitlines()[-1]
        except Exception:
            self._driver_version = ""

        self._powers = [0.0] * len(self._card_keys)
        self._utils = [0] * len(self._card_keys)
        self._temps = [0] * len(self._card_keys)
        self._mem_used = [0] * len(self._card_keys)

        # Prime the cache once synchronously so the first read doesn't
        # return zeros.
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
        snap = _run_rocm_smi(
            ["--showpower", "--showtemp", "--showuse", "--showmeminfo", "vram"]
        )
        if snap is None:
            return
        powers, utils, temps, mem_used = [], [], [], []
        for key in self._card_keys:
            card = snap.get(key, {})
            p_f = _find_field(card, "average graphics package power") \
                or _find_field(card, "graphics package power") \
                or _find_field(card, "package power") \
                or _find_field(card, "power")
            u_f = _find_field(card, "gpu use") or _find_field(card, "busy")
            t_f = _find_field(card, "temperature", "junction") \
                or _find_field(card, "temperature", "edge") \
                or _find_field(card, "temperature")
            m_f = _find_field(card, "vram total used memory") \
                or _find_field(card, "used memory")
            powers.append(_first_float(card.get(p_f)) if p_f else 0.0)
            utils.append(int(_first_float(card.get(u_f))) if u_f else 0)
            temps.append(int(_first_float(card.get(t_f))) if t_f else 0)
            mem_used.append(int(_first_float(card.get(m_f))) if m_f else 0)
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
        # Reliable counter support requires amdsmi + card-specific unit
        # decoding; deferred until the library-side amdsmi integration.
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
