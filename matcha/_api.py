# SPDX-License-Identifier: Apache-2.0
"""matcha — public Python API.

Typical usage::

    import matcha

    with matcha.session() as s:
        for i in range(num_steps):
            with s.step(i):
                train_step()

    print(s.result.total_energy_j, s.result.energy_wh)

Explicit lifecycle (no context manager)::

    s = matcha.Session(gpus="all").start()
    try:
        for i in range(num_steps):
            s.step_begin()
            train_step()
            s.step_end(i)
    finally:
        result = s.stop()

Session-level only (no per-step instrumentation)::

    with matcha.session() as s:
        train()
    print(s.result.total_energy_j)
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Iterator, List, Mapping, Optional, Union

from ._engine import GpuStats, PowerSampler, SessionResult, StepResult

__all__ = [
    "Session",
    "session",
    "GpuStats",
    "StepResult",
    "SessionResult",
]

GpuSpec = Union[str, int, List[int], None]

# NVML is process-global. Track the one active session to give a clean
# error instead of a confusing NVML-level failure on concurrent use.
_active_lock = threading.Lock()
_active_session: Optional["Session"] = None


def _resolve_gpus(gpus: GpuSpec) -> Union[int, List[int]]:
    if gpus is None or gpus == "all" or gpus == -1:
        return -1
    if isinstance(gpus, int):
        return gpus
    return list(gpus)


def _acquire(s: "Session") -> None:
    global _active_session
    with _active_lock:
        if _active_session is not None:
            raise RuntimeError(
                "matcha: another session is already active in this process. "
                "Call stop() on it first, or use `with matcha.session(): ...` "
                "to scope sessions automatically."
            )
        _active_session = s


def _release(s: "Session") -> None:
    global _active_session
    with _active_lock:
        if _active_session is s:
            _active_session = None


class Session:
    """Handle for a matcha energy-measurement session.

    Prefer ``matcha.session()`` — a context manager that handles start/stop.
    Construct ``Session`` directly only when you need explicit lifecycle
    control (e.g. sessions that span multiple functions).

    Args:
        gpus: Which GPUs to monitor. ``"all"`` (default), an int index, a
            list of int indices, or ``-1`` (= all).
        interval_ms: Background peak-power polling interval in ms. Energy
            is read from the NVML hardware counter and is independent of
            this interval on Volta+ GPUs.
    """

    def __init__(self, gpus: GpuSpec = "all", interval_ms: int = 100):
        # Store config only. Defer PowerSampler instantiation to start() so
        # Session() is safe to construct even when pynvml / NVIDIA drivers
        # aren't present (tests, type-checking, docs builds, CPU hosts).
        self._gpu_spec = _resolve_gpus(gpus)
        self._interval_ms = interval_ms
        self._sampler: Optional[PowerSampler] = None
        self._started = False
        self._stopped = False
        self._result: Optional[SessionResult] = None
        self._step_open = False

    # ---- lifecycle ---------------------------------------------------

    def start(self) -> "Session":
        """Start metering. Returns self so you can chain: ``Session().start()``.

        Raises ImportError if ``nvidia-ml-py`` is not installed and RuntimeError
        if NVML can't initialize (no NVIDIA GPU / drivers).
        """
        if self._started:
            raise RuntimeError("matcha: session already started")
        _acquire(self)
        try:
            self._sampler = PowerSampler(
                gpu_indices=self._gpu_spec, interval_ms=self._interval_ms
            )
            self._sampler.start()
        except Exception:
            _release(self)
            raise
        self._started = True
        return self

    def stop(self) -> SessionResult:
        """Stop metering and return the session summary.

        Idempotent: calling stop() again returns the same cached result
        without touching NVML a second time.
        """
        if self._result is not None:
            return self._result
        if not self._started or self._sampler is None:
            raise RuntimeError("matcha: stop() called before start()")
        try:
            self._result = self._sampler.stop()
        finally:
            self._stopped = True
            _release(self)
        return self._result

    # ---- step API ----------------------------------------------------

    def step_begin(self) -> None:
        """Mark the start of a training step."""
        if not self._started or self._stopped or self._sampler is None:
            raise RuntimeError("matcha: session is not running")
        if self._step_open:
            raise RuntimeError(
                "matcha: step_begin() called twice without a step_end() "
                "in between"
            )
        self._sampler.begin_step()
        self._step_open = True

    def step_end(self, step: int) -> StepResult:
        """Mark the end of a training step and return its energy."""
        if not self._step_open or self._sampler is None:
            raise RuntimeError(
                "matcha: step_end() called without a matching step_begin()"
            )
        self._step_open = False
        return self._sampler.end_step(step)

    @contextmanager
    def step(self, step: int) -> Iterator[None]:
        """Context manager around a single step.

        Example::

            with s.step(i):
                train_step()

        Always closes the step in a finally block, so exceptions in the
        training step don't leak an open measurement window. The step is
        still emitted — the work happened and the energy was spent.
        """
        self.step_begin()
        try:
            yield
        finally:
            self.step_end(step)

    # ---- user metadata ----------------------------------------------

    def update_metrics(self, metrics: Mapping[str, float]) -> None:
        """Attach user-defined scalar metrics (e.g. loss, lr) to the session.

        Sticky: later calls merge with earlier ones so exporters see
        continuous values across sparse log lines. Surfaces in future
        JSONL / Prometheus / OTLP emission paths from the Python API.
        """
        if self._sampler is None:
            raise RuntimeError("matcha: update_metrics() called before start()")
        self._sampler.update_train_metrics(dict(metrics))

    # ---- read-only introspection ------------------------------------
    #
    # All pre-start accesses return safe empty/zero values rather than
    # raising — they're commonly touched by logging / repr code that
    # shouldn't explode just because metering hasn't started yet.

    @property
    def is_running(self) -> bool:
        return self._started and not self._stopped

    @property
    def result(self) -> Optional[SessionResult]:
        """The SessionResult once stop() has run; None before that."""
        return self._result

    @property
    def last_step(self) -> Optional[StepResult]:
        return self._sampler.last_step if self._sampler else None

    @property
    def steps_completed(self) -> int:
        return self._sampler.steps_completed if self._sampler else 0

    @property
    def energy_source(self) -> str:
        """``"counter"`` (hardware) or ``"polled"`` (integrated samples).

        Returns ``"unknown"`` before start().
        """
        return self._sampler.energy_source if self._sampler else "unknown"

    @property
    def gpu_name(self) -> str:
        return self._sampler.gpu_name if self._sampler else ""

    @property
    def gpu_names(self) -> List[str]:
        return list(self._sampler.gpu_names) if self._sampler else []

    @property
    def gpu_indices(self) -> List[int]:
        return list(self._sampler.gpu_indices) if self._sampler else []

    @property
    def gpu_uuids(self) -> List[str]:
        return list(self._sampler.gpu_uuids) if self._sampler else []

    @property
    def driver_version(self) -> str:
        return self._sampler.driver_version if self._sampler else ""

    # ---- context manager --------------------------------------------

    def __enter__(self) -> "Session":
        if not self._started:
            self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if not self._stopped:
            self.stop()


@contextmanager
def session(
    gpus: GpuSpec = "all", interval_ms: int = 100
) -> Iterator[Session]:
    """Context manager: start a matcha session, stop it on exit.

    Example::

        with matcha.session() as s:
            for i in range(num_steps):
                with s.step(i):
                    train_step()
        print(s.result.total_energy_j)
    """
    s = Session(gpus=gpus, interval_ms=interval_ms)
    s.start()
    try:
        yield s
    finally:
        s.stop()
