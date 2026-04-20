# SPDX-License-Identifier: Apache-2.0
"""matcha — GPU energy observability for AI training workloads.

Public API::

    import matcha

    with matcha.session() as s:
        for i in range(num_steps):
            with s.step(i):
                train_step()
    print(s.result.total_energy_j)

See ``matcha.session`` and ``matcha.Session`` for details.
"""

from importlib.metadata import (
    PackageNotFoundError as _PackageNotFoundError,
    version as _pkg_version,
)

try:
    __version__ = _pkg_version("usematcha")
except _PackageNotFoundError:
    __version__ = "0.0.0+unknown"

__all__ = [
    "__version__",
    "session",
    "Session",
    "StepResult",
    "SessionResult",
    "GpuStats",
]

# Lazy-import the public API so `import matcha` stays cheap — no pynvml
# load until a user actually instantiates a Session. Submodules (CLI,
# JSONL, exporters) keep importing _engine directly; this lazy surface
# only gates what's reached through the top-level `matcha.` namespace.
_LAZY = {"session", "Session", "StepResult", "SessionResult", "GpuStats"}


def __getattr__(name):
    if name in _LAZY:
        from . import _api

        return getattr(_api, name)
    raise AttributeError(f"module 'matcha' has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | _LAZY)
