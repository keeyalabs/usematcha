# SPDX-License-Identifier: Apache-2.0
"""matcha — GPU energy observability for AI training workloads."""

from importlib.metadata import PackageNotFoundError, version as _pkg_version

try:
    __version__ = _pkg_version("usematcha")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
