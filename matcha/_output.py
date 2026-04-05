"""
matcha._output — Terminal rendering.

Copyright (c) 2025 Keeya Labs. All rights reserved.
"""

import sys
import time

from ._engine import StepResult, SessionResult

_R = "\033[0m"
_B = "\033[1m"
_D = "\033[2m"
_G = "\033[38;5;114m"
_Y = "\033[38;5;221m"
_C = "\033[38;5;117m"
_W = "\033[38;5;255m"
_X = "\033[38;5;245m"
_E = "\033[38;5;210m"


def _bar(v: float, mx: float, w: int = 12) -> str:
    r = min(v / mx, 1.0) if mx > 0 else 0.0
    f = int(r * w)
    c = _E if r > 0.85 else (_Y if r > 0.6 else _G)
    return f"{c}{'█' * f}{_D}{'░' * (w - f)}{_R}"


class Printer:
    def __init__(self):
        self._n = 0
        self._ej = 0.0

    def header(self, gpu: str, tdp: float, iv: int):
        print()
        print(f"  {_G}{_B}⚡ matcha{_R} {_D}— gpu energy metering{_R}")
        print(f"  {_X}{'─' * 52}{_R}")
        print(f"  {_D}gpu{_R}        {_W}{gpu}{_R}")
        if tdp > 0:
            print(f"  {_D}tdp{_R}        {_W}{tdp:.0f}W{_R}")
        print(f"  {_D}sampling{_R}   {_W}every {iv}ms{_R}")
        print(f"  {_X}{'─' * 52}{_R}")
        print()
        print(
            f"  {_D}{'step':>6}  {'energy':>10}  {'time':>8}  "
            f"{'avg W':>7}  {'peak W':>7}  {'power'}{_R}"
        )
        print(f"  {_X}{'─' * 52}{_R}")

    def step(self, s: StepResult):
        self._n += 1
        self._ej += s.energy_j

        if s.energy_j >= 1000:
            e = f"{s.energy_j / 1000:.2f} kJ"
        elif s.energy_j >= 1:
            e = f"{s.energy_j:.2f} J"
        else:
            e = f"{s.energy_j * 1000:.1f} mJ"

        b = _bar(s.avg_power_w, max(s.peak_power_w * 1.1, 300))

        print(
            f"  {_W}{s.step:>6}{_R}  "
            f"{_G}{e:>10}{_R}  "
            f"{_X}{s.duration_s:>7.3f}s{_R}  "
            f"{_C}{s.avg_power_w:>6.1f}W{_R}  "
            f"{_Y}{s.peak_power_w:>6.1f}W{_R}  "
            f"{b}"
        )
        sys.stdout.flush()

    def summary(self, s: SessionResult):
        print()
        print(f"  {_X}{'─' * 52}{_R}")
        print(f"  {_G}{_B}⚡ session summary{_R}")
        print(f"  {_X}{'─' * 52}{_R}")
        print()

        if s.total_energy_j >= 3_600_000:
            e = f"{s.energy_kwh:.4f} kWh"
        elif s.total_energy_j >= 3600:
            e = f"{s.energy_wh:.2f} Wh"
        else:
            e = f"{s.total_energy_j:.2f} J"

        print(f"  {_D}gpu{_R}            {_W}{s.gpu_name}{_R}")
        print(f"  {_D}total energy{_R}    {_G}{_B}{e}{_R}")
        print(f"  {_D}total time{_R}      {_W}{s.total_duration_s:.2f}s{_R}")
        print(f"  {_D}steps{_R}           {_W}{s.total_steps}{_R}")
        print(f"  {_D}energy/step{_R}     {_C}{s.j_per_step:.2f} J{_R}")
        print(f"  {_D}avg power{_R}       {_C}{s.avg_power_w:.1f}W{_R}")
        print(f"  {_D}peak power{_R}      {_Y}{s.peak_power_w:.1f}W{_R}")
        print(f"  {_D}samples{_R}         {_X}{s.total_samples}{_R}")
        print()

        cost = s.energy_kwh * 0.12
        if cost > 0.0001:
            print(f"  {_D}est. cost{_R}       {_W}${cost:.6f} @ $0.12/kWh{_R}")
            print()
