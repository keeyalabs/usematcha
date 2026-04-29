# SPDX-License-Identifier: Apache-2.0
"""matcha.commands.monitor — live per-GPU power monitor.

Backend-agnostic: on NVIDIA hosts it reads NVML, on AMD it reads
rocm-smi, on Intel it reads xpu-smi, on Apple Silicon it reads
Darwin's IOReport framework (via stdlib ctypes). All through
``matcha._backends.detect()``.
"""

import sys
import time
from typing import List, Optional

from .._backends import BackendUnavailable, detect


_UP = "\033[F"
_CLR = "\033[K"
_HIDE = "\033[?25l"
_SHOW = "\033[?25h"

_WIDTH = 88


def _bar(ratio: float, width: int = 10) -> str:
    ratio = max(0.0, min(1.0, ratio))
    filled = int(ratio * width)
    return f"[{'#' * filled}{'-' * (width - filled)}]"


def _fmt_elapsed(s: float) -> str:
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    sec = int(s % 60)
    return f"{h:d}h{m:02d}m{sec:02d}s" if h else f"{m:02d}:{sec:02d}"


def _fmt_energy(j: float) -> str:
    if j >= 3_600_000:
        return f"{j / 3_600_000:.3f} kWh"
    if j >= 3600:
        return f"{j / 3600:.2f} Wh"
    if j >= 1000:
        return f"{j / 1000:.2f} kJ"
    return f"{j:.1f} J"


def run(gpu_indices: Optional[List[int]], interval_ms: int) -> int:
    try:
        backend = detect()
    except BackendUnavailable as e:
        print(f"matcha: {e}", file=sys.stderr)
        return 1

    try:
        backend.init(-1 if gpu_indices is None else gpu_indices)
    except Exception as e:
        print(f"matcha: backend {backend.name!r} init failed: {e}", file=sys.stderr)
        return 1

    try:
        indices = backend.device_indices
        names = backend.device_names
        tdps = [backend.device_tdp_w(i) for i in range(backend.device_count)]
        mem_totals_gb = [
            backend.device_memory_total_bytes(i) / (1024 ** 3)
            for i in range(backend.device_count)
        ]
        n_gpu = backend.device_count

        sys.stdout.write(_HIDE)
        print()
        print(f"  matcha monitor - live GPU power  (backend: {backend.name})")
        print(f"  {'-' * _WIDTH}")
        print(
            f"  {'gpu':>3}  {'name':<26}  {'power':>13}  "
            f"{'util':>4}  {'temp':>5}  {'mem':>11}  {'load':<12}"
        )
        print(f"  {'-' * _WIDTH}")
        for _ in range(n_gpu + 2):
            print()

        start = time.monotonic()
        total_energy_j = 0.0
        peak_total_w = 0.0
        last_t = start
        last_total_w = 0.0
        first = True
        dt = interval_ms / 1000.0

        try:
            while True:
                now = time.monotonic()
                powers, utils, temps, mems = [], [], [], []
                for i in range(n_gpu):
                    powers.append(backend.read_power_w(i))
                    utils.append(backend.read_utilization_pct(i))
                    temps.append(backend.read_temperature_c(i))
                    mems.append(backend.read_memory_used_bytes(i) / (1024 ** 3))

                total_w = sum(powers)
                if not first:
                    # Trapezoidal running energy — accurate enough for a
                    # live dashboard regardless of whether the backend
                    # itself has an energy counter.
                    total_energy_j += 0.5 * (last_total_w + total_w) * (now - last_t)
                last_t, last_total_w = now, total_w
                peak_total_w = max(peak_total_w, total_w)
                first = False

                sys.stdout.write(_UP * (n_gpu + 2))

                for idx, name, tdp, p, u, t, m, mt in zip(
                    indices, names, tdps, powers, utils, temps, mems, mem_totals_gb
                ):
                    nm = name[:26]
                    pw = f"{p:>4.0f}W/{tdp:>4.0f}W" if tdp else f"{p:>4.0f}W"
                    ratio = (p / tdp) if tdp else 0.0
                    mem = f"{m:>4.1f}/{mt:>4.1f}G" if mt else (f"{m:>4.1f}G" if m else "  -")
                    sys.stdout.write(
                        f"  {idx:>3}  "
                        f"{nm:<26}  "
                        f"{pw:>13}  "
                        f"{u:>3}%  "
                        f"{t:>4}C  "
                        f"{mem:>11}  "
                        f"{_bar(ratio)}"
                        f"{_CLR}\n"
                    )

                sys.stdout.write(f"  {'-' * _WIDTH}{_CLR}\n")
                elapsed = now - start
                sys.stdout.write(
                    f"  total {total_w:>5.0f}W   "
                    f"peak {peak_total_w:>5.0f}W   "
                    f"elapsed {_fmt_elapsed(elapsed):>10}   "
                    f"energy {_fmt_energy(total_energy_j)}"
                    f"{_CLR}\n"
                )
                sys.stdout.flush()
                time.sleep(dt)
        except KeyboardInterrupt:
            pass
        finally:
            sys.stdout.write(_SHOW)
            elapsed = time.monotonic() - start
            avg = total_energy_j / elapsed if elapsed > 0 else 0.0
            print()
            print(f"  {'-' * _WIDTH}")
            print("  stopped")
            print(f"  duration   {_fmt_elapsed(elapsed)}")
            print(f"  energy     {_fmt_energy(total_energy_j)}")
            print(f"  avg power  {avg:.1f}W")
            print(f"  peak       {peak_total_w:.1f}W")
            print()
    finally:
        try:
            backend.shutdown()
        except Exception:
            pass

    return 0
