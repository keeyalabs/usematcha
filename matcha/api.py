"""
matcha — GPU energy metering for AI training workloads.

Copyright (c) 2025 Keeya Labs. All rights reserved.

Usage:
    import matcha

    m = matcha.init()
    for step in range(num_steps):
        m.step_start()
        # ... training ...
        energy = m.step_end(step)
    summary = m.finish()
"""

from contextlib import contextmanager
from typing import Optional

from ._engine import _Collector, StepResult, SessionResult
from ._output import Printer


class Meter:
    """Handle for step-level GPU energy tracking."""

    def __init__(self, _c: _Collector, _p: Optional[Printer] = None):
        self._c = _c
        self._p = _p

    def step_start(self):
        """Call before a training step."""
        self._c.mark_start()

    def step_end(self, step: int) -> StepResult:
        """Call after a training step. Returns energy measurement."""
        r = self._c.mark_end(step)
        if self._p:
            self._p.step(r)
        return r

    def finish(self) -> SessionResult:
        """Stop metering. Returns session summary."""
        s = self._c.stop()
        if self._p:
            self._p.summary(s)
        return s


def init(gpus="all", interval_ms: int = 100, quiet: bool = True, gpu: int = None) -> Meter:
    """
    Start GPU energy metering.

    Args:
        gpus: GPUs to monitor. "all" (default), single int, or list of ints.
        interval_ms: Power sampling interval in ms (default 100)
        quiet: Suppress terminal output (default True)
        gpu: (deprecated) single GPU index, use gpus instead

    Returns:
        Meter with step_start() / step_end() / finish()
    """
    if gpu is not None:
        _g = gpu
    elif gpus == "all":
        _g = -1
    elif isinstance(gpus, int):
        _g = gpus
    else:
        _g = list(gpus)

    c = _Collector(_gpus=_g, _iv=interval_ms)
    c.start()

    p = None if quiet else Printer()
    if p:
        p.header(c.gpu_name, c.power_limit_w, interval_ms)

    return Meter(c, p)


@contextmanager
def watch(gpus="all", interval_ms: int = 100, quiet: bool = True):
    """
    Context manager for energy metering.

    with matcha.watch() as m:
        for step in range(100):
            m.step_start()
            train_step()
            m.step_end(step)
    """
    meter = init(gpus=gpus, interval_ms=interval_ms, quiet=quiet)
    try:
        yield meter
    finally:
        meter.finish()
