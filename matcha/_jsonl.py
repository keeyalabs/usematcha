# SPDX-License-Identifier: Apache-2.0
"""matcha._jsonl — structured-record output."""

import datetime
import json
import socket
import sys
from typing import Any, Dict, List, Optional, TextIO

from . import __version__
from ._engine import PowerSampler, SessionResult, StepResult


def now_iso() -> str:
    return (
        datetime.datetime.now(datetime.timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


class JsonlEmitter:
    """Writes one JSON object per line. Line-buffered so tail -f works."""

    def __init__(self, path: Optional[str], default_stream: Optional[TextIO] = None):
        if path:
            self._fh: TextIO = open(path, "a", buffering=1)
            self._owned = True
        else:
            self._fh = default_stream or sys.stdout
            self._owned = False

    def emit(self, record: Dict[str, Any]) -> None:
        self._fh.write(json.dumps(record, separators=(",", ":")) + "\n")
        self._fh.flush()

    def close(self) -> None:
        if self._owned:
            try:
                self._fh.close()
            except Exception:
                pass


def session_start_record(
    run_id: str,
    sampler: PowerSampler,
    cmd: List[str],
    labels: Dict[str, str],
    interval_ms: int,
) -> Dict[str, Any]:
    return {
        "type": "session_start",
        "ts": now_iso(),
        "run_id": run_id,
        "matcha_version": __version__,
        "hostname": socket.gethostname(),
        "driver_version": sampler.driver_version,
        "interval_ms": interval_ms,
        "energy_source": sampler.energy_source,
        "gpus": [
            {"idx": idx, "uuid": uuid, "name": name}
            for idx, uuid, name in zip(
                sampler.gpu_indices, sampler.gpu_uuids, sampler.gpu_names
            )
        ],
        "cmd": cmd,
        "labels": labels,
    }


def _per_gpu_list(per_gpu) -> List[Dict[str, Any]]:
    return [
        {
            "idx": g.idx,
            "energy_j": round(g.energy_j, 3),
            "avg_power_w": round(g.avg_power_w, 1),
            "peak_power_w": round(g.peak_power_w, 1),
        }
        for g in per_gpu
    ]


def step_record(
    run_id: str,
    result: StepResult,
    step_gap: int,
    train_metrics: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    r: Dict[str, Any] = {
        "type": "step",
        "ts": now_iso(),
        "run_id": run_id,
        "step": result.step,
        "step_gap": step_gap,
        "energy_j": round(result.energy_j, 3),
        "energy_per_step_j": round(result.energy_j / max(step_gap, 1), 3),
        "duration_s": round(result.duration_s, 4),
        "avg_power_w": round(result.avg_power_w, 1),
        "peak_power_w": round(result.peak_power_w, 1),
        "gpus": _per_gpu_list(result.per_gpu),
    }
    if train_metrics:
        r["train_metrics"] = train_metrics
    return r


def session_end_record(
    run_id: str, result: SessionResult, total_steps: int
) -> Dict[str, Any]:
    energy_per_step = result.total_energy_j / total_steps if total_steps else 0.0
    return {
        "type": "session_end",
        "ts": now_iso(),
        "run_id": run_id,
        "total_energy_j": round(result.total_energy_j, 3),
        "energy_wh": round(result.energy_wh, 4),
        "duration_s": round(result.total_duration_s, 3),
        "avg_power_w": round(result.avg_power_w, 1),
        "peak_power_w": round(result.peak_power_w, 1),
        "total_samples": result.total_samples,
        "total_steps": total_steps,
        "energy_per_step_j": round(energy_per_step, 3) if total_steps else None,
        "energy_source": result.energy_source,
        "gpus": _per_gpu_list(result.per_gpu),
    }
