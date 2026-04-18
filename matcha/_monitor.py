# SPDX-License-Identifier: Apache-2.0
"""matcha._monitor — live per-GPU power monitor."""

import sys
import time
import warnings
from typing import List, Optional

try:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        import pynvml
except ImportError:
    pynvml = None


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
    if pynvml is None:
        print("matcha: nvidia-ml-py is required. pip install nvidia-ml-py", file=sys.stderr)
        return 1

    try:
        pynvml.nvmlInit()
    except Exception as e:
        print(
            f"matcha: NVML init failed - is an NVIDIA GPU present with drivers installed? ({e})",
            file=sys.stderr,
        )
        return 1

    try:
        if gpu_indices is None:
            gpu_indices = list(range(pynvml.nvmlDeviceGetCount()))

        handles, names, tdps, mem_totals = [], [], [], []
        for i in gpu_indices:
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            handles.append(h)
            n = pynvml.nvmlDeviceGetName(h)
            names.append(n.decode("utf-8") if isinstance(n, bytes) else n)
            try:
                tdps.append(pynvml.nvmlDeviceGetPowerManagementLimit(h) / 1000.0)
            except pynvml.NVMLError:
                tdps.append(0.0)
            try:
                mem_totals.append(pynvml.nvmlDeviceGetMemoryInfo(h).total / (1024 ** 3))
            except pynvml.NVMLError:
                mem_totals.append(0.0)

        n_gpu = len(handles)
        sys.stdout.write(_HIDE)
        print()
        print("  matcha monitor - live GPU power")
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
                for h in handles:
                    try:
                        p = pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0
                    except pynvml.NVMLError:
                        p = 0.0
                    try:
                        u = pynvml.nvmlDeviceGetUtilizationRates(h).gpu
                    except pynvml.NVMLError:
                        u = 0
                    try:
                        t = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
                    except pynvml.NVMLError:
                        t = 0
                    try:
                        m = pynvml.nvmlDeviceGetMemoryInfo(h).used / (1024 ** 3)
                    except pynvml.NVMLError:
                        m = 0.0
                    powers.append(p)
                    utils.append(u)
                    temps.append(t)
                    mems.append(m)

                total_w = sum(powers)
                if not first:
                    total_energy_j += 0.5 * (last_total_w + total_w) * (now - last_t)
                last_t, last_total_w = now, total_w
                peak_total_w = max(peak_total_w, total_w)
                first = False

                sys.stdout.write(_UP * (n_gpu + 2))

                for idx, name, tdp, p, u, t, m, mt in zip(
                    gpu_indices, names, tdps, powers, utils, temps, mems, mem_totals
                ):
                    nm = name[:26]
                    pw = f"{p:>4.0f}W/{tdp:>4.0f}W" if tdp else f"{p:>4.0f}W"
                    ratio = (p / tdp) if tdp else 0.0
                    mem = f"{m:>4.1f}/{mt:>4.1f}G" if mt else f"{m:>4.1f}G"
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
            pynvml.nvmlShutdown()
        except Exception:
            pass

    return 0
