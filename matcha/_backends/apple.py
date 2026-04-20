# SPDX-License-Identifier: Apache-2.0
"""matcha._backends.apple — Apple Silicon backend via IOReport.

Apple doesn't publish an NVML-equivalent, but Darwin ships a private
framework — ``IOReport`` — that has been the ground truth for
on-device energy measurement for a decade. It exposes **cumulative**
per-subsystem energy counters in millijoules (GPU, GPU SRAM, ANE,
CPU, DRAM), updates at high rate, and does not require root. It is
how ``powermetrics``, ``asitop``, ``macmon``, and iStat Menus all
read the same numbers we do.

This backend is the Apple Silicon equivalent of NVML on Volta+:
``has_energy_counter=True``, ``read_energy_mj`` returns an exact
counter, and the engine produces counter-based energy attribution —
no trapezoidal integration, no root requirement, no subprocess
overhead.

Instantaneous power reporting
-----------------------------

IOReport itself only exposes energy; power has to be derived from two
samples. The backend's refresher thread does this implicitly: on each
tick it asks the IOReport session for the energy consumed since the
previous tick, divides by the wall-clock delta, and caches that as
``read_power_w(0)``.

Refresh cadence defaults to **50 ms** on Apple (20 Hz). The per-tick
cost on M-series is ~1.5 ms — dominated by per-channel CFString
conversion in Python, not the CoreFoundation syscalls themselves —
so a 20 Hz refresher uses ~3% of one core. Refresh *only* controls
how fresh ``read_power_w()`` is for peak detection; step-level
energy attribution is independently counter-exact via the
force-fresh path described below. Measured on M4:

* 10 ms (100 Hz): 16% of a core — overkill for peak tracking
* **50 ms (20 Hz): 3% of a core — default sweet spot**
* 100 ms (10 Hz): 1.6% — matcha's engine default; too coarse for
  sub-100ms training-step peak capture

Tune via ``AppleSiliconBackend(refresh_ms=...)``; floor is 5 ms to
avoid GIL-jitter-dominated wall-clock deltas in ``dt = now - last``.

Step-boundary energy reads are **independently counter-exact**: the
engine's ``begin_step`` / ``end_step`` path calls ``read_energy_mj``,
which forces a fresh IOReport sample before returning the cumulative
millijoule counter. No dependence on the refresher's phase — step
attribution has no quantization from the refresh interval.

Private-API caveat
------------------

IOReport is not in Apple's public SDK. The symbol shapes have been
stable across macOS 10.15 → 15.x but could, in principle, break. If
``_ioreport.is_supported()`` returns False at detect-time, this
backend reports unavailable and auto-detect falls through cleanly.

Implementation details of the ctypes bindings live in
``matcha._backends._ioreport``. Reference: Zeus's
``zeus-apple-silicon`` C++ header (Apache 2.0) — signatures and
channel-group names cross-checked against it; this implementation is
otherwise independent and Python-native.
"""

from __future__ import annotations

import platform
import socket
import subprocess
import threading
import time
from typing import List, Optional, Union

from ._base import Backend, BackendUnavailable
from . import _ioreport


# Default refresher cadence. 50 ms = 20 Hz. IOReport per-tick cost is
# ~1.5 ms on Python/ctypes (channel-walk + CFString decode), so 20 Hz
# costs ~3% of a core — a good tradeoff between peak-detection
# resolution and CPU overhead. Step-level energy attribution does not
# depend on this interval (``read_energy_mj`` forces a fresh sample),
# so the knob is purely about how finely we track power bursts.
# Tunable via ``AppleSiliconBackend(refresh_ms=...)`` and floored at
# 5 ms — below that, Python GIL jitter dominates the wall-clock delta
# we divide by to get power, making the reported watts noisy without
# a real accuracy win.
_DEFAULT_REFRESH_MS = 50
_MIN_REFRESH_MS = 5


def _detect_chip_name() -> str:
    """Return a marketing-style chip label (e.g. 'Apple M2 Max').

    Derived from ``machdep.cpu.brand_string``. Falls back to a generic
    label if sysctl output is unavailable.
    """
    try:
        out = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], timeout=2
        )
        name = out.decode("utf-8", errors="replace").strip()
        return name or "Apple Silicon GPU"
    except Exception:
        return "Apple Silicon GPU"


class AppleSiliconBackend(Backend):
    name = "apple"

    def __init__(self, refresh_ms: Optional[int] = None) -> None:
        # Clamp below to avoid pathological refresher behavior. The
        # engine's own ``interval_ms`` still controls the engine-level
        # power sample cadence; this controls the backend-internal
        # IOReport refresh that keeps ``read_power_w`` fresh.
        ms = _DEFAULT_REFRESH_MS if refresh_ms is None else max(_MIN_REFRESH_MS, int(refresh_ms))
        self._refresh_s = ms / 1000.0

        self._session: Optional[_ioreport.IOReportSession] = None
        self._refresher: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.Lock()

        self._chip_name: str = ""
        self._uuid: str = ""
        self._os_version: str = ""

        # Cached readings. The refresher thread updates these; reads
        # are lock-free on the values themselves (Python int/float
        # assignment is atomic), but cumulative_mj is read through
        # the IOReportSession's own lock.
        self._last_power_w: float = 0.0
        self._last_tick_t: Optional[float] = None

        self._initialized = False

    # ---- availability ----------------------------------------------------

    def is_available(self) -> bool:
        if platform.system() != "Darwin":
            return False
        if platform.machine() not in ("arm64", "arm"):
            return False
        return _ioreport.is_supported()

    def unavailable_reason(self) -> str:
        if platform.system() != "Darwin":
            return "not macOS"
        if platform.machine() not in ("arm64", "arm"):
            return "not Apple Silicon (Intel Mac)"
        return (
            "IOReport framework could not be loaded "
            "(/System/Library/PrivateFrameworks/IOReport.framework missing "
            "or incompatible on this macOS version)"
        )

    # ---- lifecycle -------------------------------------------------------

    def init(self, gpu_spec: Union[int, List[int]] = -1) -> None:
        # Apple Silicon exposes a single integrated GPU. gpu_spec is
        # validated but otherwise ignored — there's no such thing as
        # "GPU 0 and GPU 3" on an M-series SoC.
        if isinstance(gpu_spec, list) and any(i != 0 for i in gpu_spec):
            raise ValueError(
                "matcha: Apple Silicon exposes a single GPU — valid "
                "--gpus values are -1 or 0"
            )

        self._chip_name = _detect_chip_name()
        try:
            self._os_version = platform.mac_ver()[0] or ""
        except Exception:
            self._os_version = ""
        # Apple doesn't expose a GPU UUID. A host-scoped pseudo-UUID is
        # good enough for per-run join keys in Prom / OTLP / JSONL.
        try:
            self._uuid = f"apple-silicon:{socket.gethostname()}"
        except Exception:
            self._uuid = "apple-silicon:unknown"

        try:
            session = _ioreport.IOReportSession()
            session.open()
        except _ioreport.IOReportUnavailable as e:
            raise BackendUnavailable(f"matcha: IOReport unavailable — {e}") from e

        self._session = session
        self._last_tick_t = time.monotonic()

        # Warm up the counter with one priming tick so the first
        # read_power_w doesn't report 0.0 for up to one refresh cycle.
        # The priming call also surfaces any runtime symbol breakage
        # immediately, before the background thread hides it.
        try:
            self._tick()
        except Exception as e:
            session.close()
            self._session = None
            raise BackendUnavailable(f"matcha: IOReport priming sample failed — {e}") from e

        self._running = True
        self._refresher = threading.Thread(
            target=self._refresh_loop, daemon=True, name="matcha-apple-ioreport"
        )
        self._refresher.start()
        self._initialized = True

    def shutdown(self) -> None:
        if not self._initialized:
            return
        self._running = False
        if self._refresher:
            self._refresher.join(timeout=2.0)
            self._refresher = None
        if self._session:
            try:
                self._session.close()
            except Exception:
                pass
            self._session = None
        self._initialized = False

    # ---- refresher ------------------------------------------------------

    def _refresh_loop(self) -> None:
        while self._running:
            try:
                self._tick()
            except Exception:
                # Don't kill the thread on a transient IOReport blip —
                # stale cached values are preferable to a silently-dead
                # backend. The next tick has a chance to recover.
                pass
            time.sleep(self._refresh_s)

    def _tick(self) -> None:
        """Capture one sample, update cached power, accumulate energy."""
        if self._session is None:
            return
        now = time.monotonic()
        mj = self._session.energy_delta_mj()
        with self._lock:
            last = self._last_tick_t
            self._last_tick_t = now
        if last is None:
            return
        dt = now - last
        if dt <= 0:
            return
        # Instantaneous-ish power averaged across the tick window:
        # energy in joules / time in seconds = watts.
        self._last_power_w = (mj / 1000.0) / dt

    # ---- metadata -------------------------------------------------------

    @property
    def device_count(self) -> int:
        return 1 if self._initialized else 0

    @property
    def device_indices(self) -> List[int]:
        return [0] if self._initialized else []

    @property
    def device_names(self) -> List[str]:
        return [self._chip_name] if self._initialized else []

    @property
    def device_uuids(self) -> List[str]:
        return [self._uuid] if self._initialized else []

    @property
    def driver_version(self) -> str:
        return self._os_version

    @property
    def has_energy_counter(self) -> bool:
        # IOReport's GPU channels are cumulative mJ counters, same
        # semantic class as NVML's nvmlDeviceGetTotalEnergyConsumption.
        return True

    # ---- reads ----------------------------------------------------------

    def read_power_w(self, i: int) -> float:
        if i != 0 or self._session is None:
            return 0.0
        return self._last_power_w

    def read_energy_mj(self, i: int) -> int:
        """Cumulative GPU energy in mJ, counter-exact as of this call.

        Forces a fresh IOReport sample before returning — the engine
        calls this at step boundaries (``begin_step`` / ``end_step``)
        and at session start/stop, and needs the value to reflect the
        counter *at the exact moment of the call*, not whatever the
        refresher thread last cached. This makes step-level energy on
        Apple precisely counter-exact, same semantic class as reading
        ``nvmlDeviceGetTotalEnergyConsumption`` cold on NVIDIA.

        ``energy_delta_mj()`` takes the IOReportSession lock, so this
        path is safe to interleave with the refresher thread's calls.
        The extra roundtrip at the boundary is sub-millisecond, cheap
        compared to the step work being measured.
        """
        if i != 0 or self._session is None:
            return 0
        try:
            self._session.energy_delta_mj()
        except Exception:
            # A transient IOReport failure shouldn't break step
            # attribution — fall through and return whatever the
            # refresher's last tick accumulated. Worst case we get
            # one quantized step; next boundary call recovers.
            pass
        # Round to integer mJ to match NVML's counter semantics (mJ is
        # the precision NVML exposes too). IOReport's internal
        # precision is finer, but consumers downstream treat the
        # counter as a monotonic integer.
        return int(self._session.cumulative_mj)

    def read_utilization_pct(self, i: int) -> int:
        # Not exposed by the Energy Model group. Apple's GPU
        # utilization is available through IOReport's "GPU Stats"
        # group, but adding it doubles the subscription cost and isn't
        # needed for energy attribution. Leave at 0; the dashboard
        # treats 0 as "unavailable".
        return 0

    def read_temperature_c(self, i: int) -> int:
        # Apple Silicon's GPU die temperature is not exposed in user
        # space without SMC access (IOKit HID interface) — out of
        # scope for this backend.
        return 0

    def read_memory_used_bytes(self, i: int) -> int:
        # Unified memory on Apple Silicon means "GPU-used memory" is
        # not a well-defined quantity.
        return 0
