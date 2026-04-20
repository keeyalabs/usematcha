# SPDX-License-Identifier: Apache-2.0
"""matcha.callbacks.hf — HuggingFace TrainerCallback integration.

Register on any ``transformers.Trainer`` to get per-step GPU energy in
your existing log stream (stdout, TensorBoard, WandB, ...) plus a
``matcha_energy`` summary line at the end of training.

Example::

    from transformers import Trainer
    from matcha.callbacks import StepEnergyCallback

    trainer = Trainer(
        model=model, args=args, train_dataset=ds,
        callbacks=[StepEnergyCallback()],
    )
    trainer.train()

Only the local-process-zero rank actually measures — NVML is per-process
and every rank on the same host sees the same GPUs, so 8-rank DDP would
8x-count without this guard. That means each host produces one energy
trace. Cross-host aggregation (multi-node training) is not this
callback's job; scrape the per-host Prometheus or JSONL outputs
instead.

Failure policy: if matcha can't init (no NVIDIA GPU, NVML flake,
missing pynvml), the callback logs a warning and becomes a no-op for
the rest of training. We never raise from a hook — energy metering is
instrumentation, not critical training infrastructure.
"""

from __future__ import annotations

import logging
import sys
from typing import Any, List, Optional, Union

# Hard-fail here is fine — this module sits behind a lazy gate in
# matcha.callbacks.__init__, which already surfaces a friendly install
# hint when transformers isn't present.
from transformers import TrainerCallback

from .._api import Session
from .._engine import SessionResult, StepResult

log = logging.getLogger("matcha.callbacks.hf")

GpuSpec = Union[str, int, List[int], None]


class StepEnergyCallback(TrainerCallback):
    """HuggingFace ``TrainerCallback`` that records per-step GPU energy.

    Args:
        gpus: GPUs to monitor. ``"all"`` (default), an int index, a list
            of indices, or ``-1``. Only the local-zero rank measures;
            other ranks no-op.
        interval_ms: Background peak-power polling interval. Energy on
            Volta+ is read from the NVML hardware counter and is
            independent of this value.
        log_prefix: Prefix for metric keys pushed into the Trainer's
            ``logs`` dict. Default ``"matcha/"`` to mirror HF's own
            ``train/``, ``eval/`` convention. Set to ``""`` for bare
            keys if your downstream expects that.
        quiet: If False (default), print a startup line and the final
            ``matcha_energy`` summary to stderr.
        session: Pre-constructed ``matcha.Session`` to reuse. Advanced;
            leave ``None`` for default behavior. When you pass one in,
            the callback will *not* stop it — you own its lifecycle.
    """

    def __init__(
        self,
        gpus: GpuSpec = "all",
        interval_ms: int = 100,
        log_prefix: str = "matcha/",
        quiet: bool = False,
        session: Optional[Session] = None,
    ):
        self._gpus = gpus
        self._interval_ms = interval_ms
        self._log_prefix = log_prefix
        self._quiet = quiet
        self._session: Optional[Session] = session
        self._externally_owned = session is not None

        # Per-instance state, all mutated only from Trainer-owned threads.
        self._active = False       # this rank is the measuring rank
        self._disabled = False     # matcha init failed; no-op from here
        self._step_open = False
        self._cumulative_j = 0.0   # running sum of per-step energy
        self._last_step: Optional[StepResult] = None
        self._session_result: Optional[SessionResult] = None

    # ---- helpers -----------------------------------------------------

    def _should_measure(self, state: Any) -> bool:
        # Default to True if Trainer state lacks the attribute (e.g. a
        # custom state object in tests). Production HF Trainer always
        # sets is_local_process_zero.
        return bool(getattr(state, "is_local_process_zero", True))

    def _say(self, msg: str) -> None:
        if not self._quiet:
            print(msg, file=sys.stderr, flush=True)

    def _disable(self, reason: str) -> None:
        if not self._disabled:
            log.warning("matcha: disabling energy metering — %s", reason)
        self._disabled = True

    # ---- TrainerCallback hooks --------------------------------------

    def on_train_begin(self, args, state, control, **kwargs):
        if not self._should_measure(state):
            return
        self._active = True
        if self._session is None:
            self._session = Session(
                gpus=self._gpus, interval_ms=self._interval_ms
            )
        try:
            if not self._session.is_running:
                self._session.start()
        except Exception as e:
            self._disable(f"start failed: {e}")
            return
        self._say(
            f"matcha: metering {self._session.gpu_name} "
            f"(source={self._session.energy_source}, "
            f"interval={self._interval_ms}ms)"
        )

    def on_step_begin(self, args, state, control, **kwargs):
        if not self._active or self._disabled or self._session is None:
            return
        try:
            if self._step_open:
                # Two on_step_begin without a matching on_step_end — can
                # happen if a prior hook raised. Close the stale window
                # with the last known step index so NVML state is clean.
                prev_step = max(int(getattr(state, "global_step", 1)) - 1, 0)
                try:
                    self._session.step_end(prev_step)
                except Exception:
                    pass
                self._step_open = False
            self._session.step_begin()
            self._step_open = True
        except Exception as e:
            self._disable(f"step_begin failed: {e}")

    def on_step_end(self, args, state, control, **kwargs):
        if (
            not self._active
            or self._disabled
            or not self._step_open
            or self._session is None
        ):
            return
        try:
            r = self._session.step_end(int(state.global_step))
            self._last_step = r
            self._cumulative_j += r.energy_j
            self._step_open = False
        except Exception as e:
            self._disable(f"step_end failed: {e}")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if (
            not self._active
            or self._disabled
            or logs is None
            or self._last_step is None
        ):
            return
        p = self._log_prefix
        r = self._last_step
        # All numeric — TensorBoard / WandB both require scalar values
        # in the logs dict. Rounding matches JSONL emitter precision.
        logs[f"{p}energy_j"] = round(self._cumulative_j, 3)
        logs[f"{p}energy_step_j"] = round(r.energy_j, 3)
        logs[f"{p}power_avg_w"] = round(r.avg_power_w, 1)
        logs[f"{p}power_peak_w"] = round(r.peak_power_w, 1)

    def on_train_end(self, args, state, control, **kwargs):
        if not self._active or self._session is None:
            return
        # Close any dangling step so totals are clean.
        if self._step_open:
            try:
                self._session.step_end(int(state.global_step))
            except Exception:
                pass
            self._step_open = False
        if self._disabled:
            return
        # Respect externally-owned sessions — caller stops them.
        if self._externally_owned:
            self._session_result = self._session.result
            return
        try:
            r = self._session.stop()
        except Exception as e:
            log.warning("matcha: stop failed — %s", e)
            return
        self._session_result = r
        # Match the exact format used by `matcha run` / `matcha wrap` so
        # downstream tools (and humans) see one consistent summary line.
        self._say(
            f"matcha_energy gpus:{r.gpu_name} total:{r.total_energy_j:.0f}J "
            f"({r.energy_wh:.2f}Wh) duration:{r.total_duration_s:.1f}s "
            f"avg_power:{r.avg_power_w:.0f}W peak_power:{r.peak_power_w:.0f}W "
            f"samples:{r.total_samples}"
        )

    # ---- public read-only introspection ----------------------------

    @property
    def last_step(self) -> Optional[StepResult]:
        """Last completed StepResult, or None before the first step."""
        return self._last_step

    @property
    def result(self) -> Optional[SessionResult]:
        """SessionResult once training has ended, else None."""
        return self._session_result

    @property
    def cumulative_energy_j(self) -> float:
        """Running sum of per-step energy (J) since training began."""
        return self._cumulative_j

    @property
    def disabled(self) -> bool:
        """True if matcha failed to init and the callback is a no-op."""
        return self._disabled
