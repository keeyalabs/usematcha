"""
matcha.cli — Command-line interface.

Copyright (c) 2025 Keeya Labs. All rights reserved.

Usage:
    matcha run torchrun --standalone --nproc_per_node=1 train_gpt.py
    matcha wrap torchrun --standalone --nproc_per_node=1 train_gpt.py
    matcha monitor
"""

import argparse
import subprocess
import sys
import re
import time
import signal
import os
from typing import Optional

from ._engine import _Collector
from ._output import Printer
from . import __version__

_PATTERNS = [
    re.compile(r"step\s+(\d+)", re.IGNORECASE),
    re.compile(r"step:\s*(\d+)", re.IGNORECASE),
    re.compile(r"iter\s+(\d+)", re.IGNORECASE),
    re.compile(r"iteration\s+(\d+)", re.IGNORECASE),
    re.compile(r"\[(\d+)/\d+\]"),
    re.compile(r"training\s+step\s+(\d+)", re.IGNORECASE),
]


def _detect(line: str) -> Optional[int]:
    if "warmup" in line.lower():
        return None
    for p in _PATTERNS:
        m = p.search(line)
        if m:
            return int(m.group(1))
    return None


def _parse_gpus(val: str):
    """Parse GPU argument: 'all', single int, or comma-separated list."""
    if val == "all":
        return -1
    if "," in val:
        return [int(x.strip()) for x in val.split(",")]
    return int(val)


def _make_collector(args):
    """Create a _Collector from parsed args."""
    gpus = _parse_gpus(args.gpus)
    return _Collector(_gpus=gpus, _iv=args.interval)


def _run(args):
    """Run a command, append energy summary at the end. Zero overhead — no stdout piping."""
    c = _make_collector(args)
    c.start()

    proc = subprocess.Popen(args.command)

    orig = signal.signal(signal.SIGINT, lambda s, f: proc.send_signal(signal.SIGINT))

    try:
        proc.wait()
    finally:
        signal.signal(signal.SIGINT, orig)
        s = c.stop()

        print(
            f"matcha_energy gpus:{s.gpu_name} total:{s.total_energy_j:.0f}J ({s.energy_wh:.2f}Wh) "
            f"duration:{s.total_duration_s:.1f}s avg_power:{s.avg_power_w:.0f}W "
            f"peak_power:{s.peak_power_w:.0f}W samples:{s.total_samples}"
        )

    return proc.returncode


def _wrap(args):
    """Run a command, append energy data to each step line."""
    c = _make_collector(args)
    c.start()

    last: Optional[int] = None
    last_line: Optional[str] = None
    active = False

    proc = subprocess.Popen(
        args.command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )

    orig = signal.signal(signal.SIGINT, lambda s, f: proc.send_signal(signal.SIGINT))

    def _fmt_energy(e, step_gap: int) -> str:
        per_step = e.energy_j / max(step_gap, 1)
        return (
            f"energy:{per_step:.1f}J/step "
            f"avg_power:{e.avg_power_w:.0f}W peak_power:{e.peak_power_w:.0f}W"
        )

    try:
        for line in proc.stdout:
            line = line.rstrip("\n")
            det = _detect(line)

            if det is not None:
                if active and last is not None:
                    e = c.mark_end(last)
                    step_gap = max(det - last, 1)
                    print(f"{last_line} {_fmt_energy(e, step_gap)}")
                elif last_line is not None:
                    print(last_line)
                c.mark_start()
                active = True
                last = det
                last_line = line
            else:
                print(line)
                sys.stdout.flush()

        # Final step
        if active and last is not None:
            e = c.mark_end(last)
            print(f"{last_line} {_fmt_energy(e, 1)}")

        proc.wait()
    finally:
        signal.signal(signal.SIGINT, orig)
        s = c.stop()

        print(
            f"matcha_energy gpus:{s.gpu_name} total:{s.total_energy_j:.0f}J ({s.energy_wh:.2f}Wh) "
            f"duration:{s.total_duration_s:.1f}s avg_power:{s.avg_power_w:.0f}W "
            f"peak_power:{s.peak_power_w:.0f}W samples:{s.total_samples}"
        )

    return proc.returncode


def _monitor(args):
    """Monitor GPU power continuously without running a command."""
    c = _make_collector(args)
    c.start()

    pr = Printer()
    pr.header(c.gpu_name, c.power_limit_w, args.interval)

    print(f"  \033[2mmonitoring {c.gpu_name} — press Ctrl+C to stop\033[0m\n")

    try:
        step = 0
        c.mark_start()
        while True:
            time.sleep(args.window)
            pr.step(c.mark_end(step))
            step += 1
            c.mark_start()
    except KeyboardInterrupt:
        pass
    finally:
        pr.summary(c.stop())


def main():
    parser = argparse.ArgumentParser(
        prog="matcha",
        description="⚡ matcha — GPU energy metering for AI training",
    )
    parser.add_argument("--version", action="version", version=f"matcha {__version__}")
    sub = parser.add_subparsers(dest="cmd")

    rp = sub.add_parser("run", help="Run a command and report total GPU energy at the end")
    rp.add_argument("--gpus", type=str, default="all", help="GPU indices: all, 0, or 0,1,2 (default: all)")
    rp.add_argument("--interval", type=int, default=100, help="Sampling interval ms")
    rp.add_argument("command", nargs=argparse.REMAINDER)

    wp = sub.add_parser("wrap", help="Run a training script with per-step energy reporting")
    wp.add_argument("--gpus", type=str, default="all", help="GPU indices: all, 0, or 0,1,2 (default: all)")
    wp.add_argument("--interval", type=int, default=100, help="Sampling interval ms")
    wp.add_argument("command", nargs=argparse.REMAINDER)

    mp = sub.add_parser("monitor", help="Monitor GPU power continuously")
    mp.add_argument("--gpus", type=str, default="all", help="GPU indices: all, 0, or 0,1,2 (default: all)")
    mp.add_argument("--interval", type=int, default=100)
    mp.add_argument("--window", type=float, default=1.0, help="Report window seconds")

    args = parser.parse_args()

    if args.cmd == "run":
        if not args.command:
            rp.error("Usage: matcha run torchrun train_gpt.py")
        if args.command[0] == "--":
            args.command = args.command[1:]
        sys.exit(_run(args))
    elif args.cmd == "wrap":
        if not args.command:
            wp.error("Usage: matcha wrap python train.py")
        if args.command[0] == "--":
            args.command = args.command[1:]
        sys.exit(_wrap(args))
    elif args.cmd == "monitor":
        _monitor(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
