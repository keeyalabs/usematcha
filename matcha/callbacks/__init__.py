# SPDX-License-Identifier: Apache-2.0
"""matcha.callbacks — framework-integration callbacks.

``StepEnergyCallback``
    HuggingFace ``transformers.Trainer`` callback. Pushes per-step
    energy into the Trainer's log stream (WandB / TensorBoard / stdout)
    and prints a ``matcha_energy`` summary at ``on_train_end``.
    Requires ``pip install 'usematcha[hf]'``.

Additional adapters (PyTorch Lightning, Ray Train, standalone
Accelerate, ...) will live here under the same lazy-import pattern so
``import matcha`` stays cheap and no framework becomes a hard dependency
of the core install.
"""

from __future__ import annotations

__all__ = ["StepEnergyCallback"]

_LAZY = {"StepEnergyCallback"}


def __getattr__(name):
    if name == "StepEnergyCallback":
        # Probe transformers explicitly so we surface the install hint
        # instead of a generic ModuleNotFoundError. Any *other* import
        # failure (a real bug in our own adapter) propagates unchanged.
        try:
            import transformers  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "matcha: StepEnergyCallback requires `transformers`. "
                "Install with:  pip install 'usematcha[hf]'"
            ) from e
        from .hf import StepEnergyCallback

        return StepEnergyCallback
    raise AttributeError(f"module 'matcha.callbacks' has no attribute {name!r}")


def __dir__():
    return sorted(_LAZY)
