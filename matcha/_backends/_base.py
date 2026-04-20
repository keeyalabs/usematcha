# SPDX-License-Identifier: Apache-2.0
"""matcha._backends._base — vendor backend protocol.

A backend is the one and only place in matcha that knows how to talk to
a specific GPU vendor's driver / tool stack. The measurement engine
(``matcha._engine.PowerSampler``) and every exporter consume this
interface and nothing else — adding a new vendor is a new file here,
and no changes elsewhere.

Contract summary
----------------

* Lifecycle — ``is_available()`` → ``init(gpu_spec)`` → many reads →
  ``shutdown()``.
* Metadata — after ``init()``, ``device_count``, ``device_indices``,
  ``device_names``, ``device_uuids`` are populated.
* Reads — all per-device reads take a position ``i`` in
  ``[0, device_count)`` and are expected to never raise. On failure
  they return a sentinel (``0.0`` / ``0``) and the engine treats that as
  a missing sample. Only ``read_energy_mj`` is allowed to raise, and
  only when the backend has reported ``has_energy_counter is False`` —
  in which case the engine falls back to trapezoidal integration of the
  polled power samples.
* Thread-safety — backends are consumed by a single sampler thread
  plus occasional calls from exporter/dashboard threads. Backends must
  be safe under concurrent reads after ``init()``.

Backends that need a long-running helper process (Apple Silicon's
``powermetrics`` is the motivating case) own it internally: they spawn
it in ``init()``, parse it in a daemon thread, and tear it down in
``shutdown()``. The engine never learns about it.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import List, Union


class BackendUnavailable(RuntimeError):
    """Raised when a requested backend can't be used on this host."""


@dataclass
class DeviceInfo:
    """Static per-device metadata captured at ``init()``."""

    index: int
    name: str
    uuid: str
    tdp_w: float = 0.0  # 0.0 when the vendor doesn't expose a TDP / power limit
    memory_total_bytes: int = 0


class Backend(abc.ABC):
    """Abstract GPU vendor backend.

    Subclasses implement the read methods for their vendor's driver
    stack. ``is_available()`` is side-effect-free — matcha calls it
    during auto-detection on every registered backend, so it must not
    open driver handles, spawn subprocesses, or log warnings.
    """

    #: Short backend id. Mirrored into ``energy_source`` tags, JSONL
    #: ``backend`` field, and Prometheus ``backend`` label.
    name: str = "unknown"

    # ---- lifecycle --------------------------------------------------------

    @abc.abstractmethod
    def is_available(self) -> bool:
        """Return True iff this backend can run on the current host.

        Must be cheap and side-effect-free — matcha calls this on every
        backend during auto-detection.
        """

    def unavailable_reason(self) -> str:
        """Human-readable one-liner for why ``is_available()`` is False.

        Surfaced in the error message when auto-detect fails, to help
        users diagnose multi-vendor or headless hosts.
        """
        return "not available"

    @abc.abstractmethod
    def init(self, gpu_spec: Union[int, List[int]] = -1) -> None:
        """Open driver resources and populate device metadata.

        ``gpu_spec`` mirrors the CLI's ``--gpus`` semantics:
        ``-1`` → all visible devices; ``int`` → a single device;
        ``list[int]`` → a specific set.
        """

    @abc.abstractmethod
    def shutdown(self) -> None:
        """Release driver resources. Idempotent."""

    # ---- metadata (valid after init()) -----------------------------------

    @property
    @abc.abstractmethod
    def device_count(self) -> int: ...

    @property
    @abc.abstractmethod
    def device_indices(self) -> List[int]: ...

    @property
    @abc.abstractmethod
    def device_names(self) -> List[str]: ...

    @property
    @abc.abstractmethod
    def device_uuids(self) -> List[str]: ...

    @property
    @abc.abstractmethod
    def driver_version(self) -> str: ...

    @property
    @abc.abstractmethod
    def has_energy_counter(self) -> bool:
        """True if the backend has a hardware energy counter.

        When True, ``read_energy_mj()`` returns exact cumulative energy
        and the engine produces counter-based ``energy_source="counter"``
        results. When False, the engine integrates polled power and
        tags results ``energy_source="polled"``.
        """

    def device_tdp_w(self, i: int) -> float:
        """Rated power limit for device ``i`` in watts, or 0.0 if unknown."""
        return 0.0

    def device_memory_total_bytes(self, i: int) -> int:
        return 0

    # ---- per-device reads ------------------------------------------------

    @abc.abstractmethod
    def read_power_w(self, i: int) -> float:
        """Instantaneous GPU package power in watts. Returns 0.0 on failure."""

    def read_energy_mj(self, i: int) -> int:
        """Cumulative energy counter for device ``i`` in millijoules.

        Only meaningful when ``has_energy_counter`` is True. The default
        implementation raises ``NotImplementedError`` so a backend that
        claims a counter but forgets to implement the read fails loudly
        instead of silently returning zeros.
        """
        raise NotImplementedError(
            f"{self.name} backend claimed has_energy_counter but did not "
            "implement read_energy_mj()"
        )

    def read_utilization_pct(self, i: int) -> int:
        """SM / GPU-engine utilization, 0–100. Returns 0 if unavailable."""
        return 0

    def read_temperature_c(self, i: int) -> int:
        """GPU die temperature in degrees Celsius. Returns 0 if unavailable."""
        return 0

    def read_memory_used_bytes(self, i: int) -> int:
        """Used GPU memory in bytes. Returns 0 if unavailable."""
        return 0
