# SPDX-License-Identifier: Apache-2.0
"""matcha._backends — hardware vendor abstraction layer.

matcha's measurement engine is vendor-agnostic. Each vendor ships a
``Backend`` that exposes a uniform contract for power, energy,
utilization, temperature, and memory reads. The engine (``_engine.py``)
and exporters (``exporters/*.py``) only ever talk to this contract —
adding a new GPU vendor does not touch the rest of the codebase.

Auto-detection order at ``detect()`` time: NVML → ROCm → Intel → Apple
Silicon. Users can force a specific backend with
``MATCHA_BACKEND=nvml|rocm|intel|apple`` (useful in CI and for
multi-vendor hosts).
"""

from __future__ import annotations

import os
from typing import Optional

from ._base import Backend, BackendUnavailable, DeviceInfo


_REGISTRY_ORDER = ("nvml", "rocm", "intel", "apple")


def _load(name: str) -> Backend:
    if name == "nvml":
        from .nvml import NvmlBackend

        return NvmlBackend()
    if name == "rocm":
        from .rocm import RocmBackend

        return RocmBackend()
    if name == "intel":
        from .intel import IntelBackend

        return IntelBackend()
    if name == "apple":
        from .apple import AppleSiliconBackend

        return AppleSiliconBackend()
    raise ValueError(f"unknown backend: {name!r}")


def detect(prefer: Optional[str] = None) -> Backend:
    """Return the first backend whose hardware is available on this host.

    ``prefer`` (or ``MATCHA_BACKEND`` env) forces a specific backend and
    raises if it's unavailable — no silent fallback. Without a preference
    matcha probes each backend in ``_REGISTRY_ORDER`` and returns the
    first one that reports ``is_available()``.

    Each backend's ``is_available()`` check is cheap and side-effect-free
    (no NVML init, no subprocess spawns that stick around) — only
    ``Backend.init()`` opens real resources.
    """
    prefer = prefer or os.environ.get("MATCHA_BACKEND")
    if prefer:
        prefer = prefer.lower()
        b = _load(prefer)
        if not b.is_available():
            raise BackendUnavailable(
                f"matcha: MATCHA_BACKEND={prefer!r} requested but backend is "
                f"not available on this host ({b.unavailable_reason()})"
            )
        return b

    tried = []
    for name in _REGISTRY_ORDER:
        try:
            b = _load(name)
        except Exception as e:
            tried.append(f"{name}: {e}")
            continue
        if b.is_available():
            return b
        tried.append(f"{name}: {b.unavailable_reason()}")

    raise BackendUnavailable(
        "matcha: no supported GPU backend detected. Tried:\n  - "
        + "\n  - ".join(tried)
        + "\nSupported: NVIDIA (NVML), AMD (ROCm), Intel (xpu-smi), "
        "Apple Silicon (IOReport)."
    )


__all__ = ["Backend", "BackendUnavailable", "DeviceInfo", "detect"]
