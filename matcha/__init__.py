"""
matcha — GPU energy metering for AI training workloads.

Copyright (c) 2025 Keeya Labs. All rights reserved.

Usage:
    pip install matcha-gpu
    matcha run torchrun --nproc_per_node=1 train.py
    matcha wrap torchrun --nproc_per_node=1 train.py
"""

__version__ = "0.2.3"

from .api import init, watch, Meter
from ._engine import StepResult, SessionResult

__all__ = ["init", "watch", "Meter", "StepResult", "SessionResult", "__version__"]
