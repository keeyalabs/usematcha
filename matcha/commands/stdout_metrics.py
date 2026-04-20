# SPDX-License-Identifier: Apache-2.0
"""matcha.commands.stdout_metrics — extract numeric training metrics from a stdout line.

Parses `key:value` and `key=value` pairs (with optional unit suffix), plus
Python-dict-style `'key': value` used by HuggingFace Trainer. The numeric
values become Prometheus / OTLP gauges and JSONL fields, letting dashboards
correlate training signal (loss, lr, mfu) with matcha's energy signal.

Scope: deliberately conservative. Only handles unambiguous formats — no
space-separated `key value`, no multi-line frames, no string values. The
payoff is: for the parameter-golf / nanoGPT / modded-nanogpt audience
(which logs in `key:value` form) this just works, zero config.
"""

import math
import re
from typing import Dict

# Matches `key:val`, `key=val`, optionally followed by a unit suffix.
# key: starts with a letter, then [A-Za-z0-9_.]. Preceded by line start or
# whitespace / punctuation (so "lr:1e-4" in "loss=2.3,lr:1e-4" parses).
# value: signed decimal, optional scientific notation.
# unit: up to 4 alphabetic chars or %, consumed greedily when present
# (prevents "612ms" from yielding value=612 and leftover "ms").
_KV = re.compile(
    r"(?:^|[\s,(\[{])"
    r"([A-Za-z][A-Za-z0-9_.]*)"
    r"\s*[:=]\s*"
    r"(-?\d+\.?\d*(?:[eE][-+]?\d+)?|-?\.\d+(?:[eE][-+]?\d+)?)"
    r"([A-Za-z%]{0,4})?"
)

# HuggingFace-style dict: `{'loss': 2.3, 'learning_rate': 1.2e-4}`.
_HF = re.compile(
    r"['\"]([A-Za-z_][A-Za-z0-9_]*)['\"]\s*:\s*"
    r"(-?\d+\.?\d*(?:[eE][-+]?\d+)?|-?\.\d+(?:[eE][-+]?\d+)?)"
)

# Keys to ignore:
# - step markers: matcha emits step number itself
# - matcha's own appended fields: if a user ever pipes `matcha wrap` output
#   back into another `matcha wrap`, don't re-ingest our own data
_BLACKLIST = frozenset({
    "step", "iter", "iteration", "warmup_step", "epoch_step",
    "energy", "avg_power", "peak_power", "samples",
    "matcha_energy", "total", "duration",
})


def _sanitize(name: str) -> str:
    """Force into Prometheus/OTel-legal label: [A-Za-z_][A-Za-z0-9_]*."""
    s = re.sub(r"[^A-Za-z0-9_]", "_", name)
    s = re.sub(r"_+", "_", s).strip("_").lower()
    if not s:
        return ""
    if s[0].isdigit():
        s = "_" + s
    return s


def _accept_number(s: str) -> float:
    """Parse float, rejecting NaN / Inf (Prom/OTLP don't like them)."""
    v = float(s)
    if math.isnan(v) or math.isinf(v):
        raise ValueError("non-finite")
    return v


def extract_metrics(line: str) -> Dict[str, float]:
    """Return {sanitized_key: value} for every numeric metric found in the line.

    On duplicate keys in one line, the first occurrence wins (stable ordering
    for the common case where later occurrences are noise).
    """
    out: Dict[str, float] = {}

    for m in _KV.finditer(line):
        key, num, unit = m.group(1), m.group(2), m.group(3) or ""
        if key.lower() in _BLACKLIST:
            continue
        try:
            val = _accept_number(num)
        except ValueError:
            continue
        full = _sanitize(f"{key}_{unit}" if unit else key)
        if full and full not in out:
            out[full] = val

    # HF dict fallback: only apply when the line looks dict-shaped, to keep
    # the fast path cheap and avoid double-counting normal `key:value` lines.
    if "{" in line and "'" in line:
        for m in _HF.finditer(line):
            key, num = m.group(1), m.group(2)
            if key.lower() in _BLACKLIST:
                continue
            try:
                val = _accept_number(num)
            except ValueError:
                continue
            k = _sanitize(key)
            if k and k not in out:
                out[k] = val

    return out
