# SPDX-License-Identifier: Apache-2.0
"""matcha.cli — command-line interface."""

import argparse
import os
import re
import signal
import subprocess
import sys
import uuid
from typing import Dict, List, Optional

from ._engine import PowerSampler
from .commands import monitor as _monitor
from .commands import diff as _diff
from .commands.stdout_metrics import extract_metrics
from .exporters.jsonl import (
    JsonlEmitter,
    session_start_record,
    step_record,
    session_end_record,
)
from .exporters.prometheus import PromServer
from .exporters.otlp import OtlpExporter
from . import __version__

_PATTERNS = [
    re.compile(r"step\s+(\d+)", re.IGNORECASE),
    re.compile(r"step:\s*(\d+)", re.IGNORECASE),
    re.compile(r"iter\s+(\d+)", re.IGNORECASE),
    re.compile(r"iteration\s+(\d+)", re.IGNORECASE),
    re.compile(r"\[(\d+)/\d+\]"),
    re.compile(r"training\s+step\s+(\d+)", re.IGNORECASE),
    # Bare `N/total` at line-start followed by train_loss/val_loss — used by
    # parameter-golf submissions and modded-nanogpt forks that drop the
    # `step:` prefix. Anchored + key-suffix so we don't match TTT progress
    # lines (`tttg: c1/131`, `ttp: b782/782`) or in-line ratios.
    re.compile(r"^\s*(\d+)/\d+\s+(?:train_loss|val_loss)", re.IGNORECASE),
    # TTT per-chunk SGD progress — `tttg: c1/131 lr:0.001 t:0.3s`. Each
    # chunk is one unit of eval-time work; this lets `wrap` slice TTT eval
    # energy at chunk granularity. Numbers reset across phases (phase 1:
    # c1-c131, phase 2: c1-c219, ...) — matcha's step_gap clamp handles
    # the 3 reset boundaries cleanly enough; per-phase analysis can split
    # on cumulative timestamps in the JSONL.
    re.compile(r"^tttg:\s*c(\d+)/\d+", re.IGNORECASE),
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
    if val == "all":
        return -1
    if "," in val:
        return [int(x.strip()) for x in val.split(",")]
    return int(val)


def _parse_labels(pairs: Optional[List[str]]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not pairs:
        return out
    for p in pairs:
        if "=" not in p:
            raise SystemExit(f"matcha: --label expects KEY=VALUE, got: {p!r}")
        k, v = p.split("=", 1)
        k = k.strip()
        if not k:
            raise SystemExit(f"matcha: --label key cannot be empty: {p!r}")
        out[k] = v
    return out


def _make_sampler(args) -> PowerSampler:
    return PowerSampler(gpu_indices=_parse_gpus(args.gpus), interval_ms=args.interval)


def _resolve_run_id(args) -> str:
    return args.run_id or os.environ.get("MATCHA_RUN_ID") or uuid.uuid4().hex[:12]


def _make_emitter(args, default_stream) -> Optional[JsonlEmitter]:
    if not (args.json or args.output):
        return None
    return JsonlEmitter(args.output, default_stream=default_stream)


def _human_summary(s) -> str:
    return (
        f"matcha_energy gpus:{s.gpu_name} total:{s.total_energy_j:.0f}J ({s.energy_wh:.2f}Wh) "
        f"duration:{s.total_duration_s:.1f}s avg_power:{s.avg_power_w:.0f}W "
        f"peak_power:{s.peak_power_w:.0f}W samples:{s.total_samples}"
    )


def _run(args):
    """Run a command, report total GPU energy at the end. No stdout piping."""
    # JSON-to-stdout without --output would not conflict here (child owns its own stdout),
    # but keep the UX consistent with `wrap`: require --output when --json is set.
    if args.json and not args.output:
        # `run` can safely emit to stdout since the child process has its own stdout —
        # but we still print the human summary afterwards, which would mix. Send to stderr.
        emitter = JsonlEmitter(None, default_stream=sys.stderr)
    else:
        emitter = _make_emitter(args, default_stream=sys.stdout)

    labels = _parse_labels(args.label)
    run_id = _resolve_run_id(args)

    sampler = _make_sampler(args)
    sampler.start()

    prom = None
    if args.prometheus:
        prom = PromServer(sampler, run_id, labels, bind=args.prometheus)
        url = prom.start()
        print(f"matcha: prometheus endpoint at {url}", file=sys.stderr)

    otlp = None
    if args.otlp:
        otlp = OtlpExporter(sampler, run_id, labels, args.otlp,
                            headers=args.otlp_header, interval_ms=args.otlp_interval)
        url = otlp.start()
        print(f"matcha: otlp exporter pushing to {url}", file=sys.stderr)

    if emitter:
        emitter.emit(
            session_start_record(run_id, sampler, args.command, labels, args.interval)
        )

    proc = subprocess.Popen(args.command)
    orig = signal.signal(signal.SIGINT, lambda s, f: proc.send_signal(signal.SIGINT))

    try:
        proc.wait()
    finally:
        signal.signal(signal.SIGINT, orig)
        if prom:
            prom.stop()
        if otlp:
            otlp.stop()
        s = sampler.stop()

        if emitter:
            emitter.emit(session_end_record(run_id, s, total_steps=0))
            emitter.close()

        print(_human_summary(s))

    return proc.returncode


def _wrap(args):
    """Run a command; append per-step energy to stdout or emit structured records."""
    if args.json and not args.output:
        raise SystemExit(
            "matcha: `wrap --json` needs --output PATH so structured records don't "
            "mix with the training process stdout. Example: --output run.jsonl"
        )

    emitter = _make_emitter(args, default_stream=None)
    labels = _parse_labels(args.label)
    run_id = _resolve_run_id(args)

    sampler = _make_sampler(args)
    sampler.start()

    prom = None
    if args.prometheus:
        prom = PromServer(sampler, run_id, labels, bind=args.prometheus)
        url = prom.start()
        print(f"matcha: prometheus endpoint at {url}", file=sys.stderr)

    otlp = None
    if args.otlp:
        otlp = OtlpExporter(sampler, run_id, labels, args.otlp,
                            headers=args.otlp_header, interval_ms=args.otlp_interval)
        url = otlp.start()
        print(f"matcha: otlp exporter pushing to {url}", file=sys.stderr)

    if emitter:
        emitter.emit(
            session_start_record(run_id, sampler, args.command, labels, args.interval)
        )

    # Attribution invariant: when a line `step:N/...` arrives, the work that
    # produced it happened in the window since we last called begin_step.
    # That window IS step N's work — label it N, print the energy on line N.
    last: Optional[int] = None
    active = False
    total_steps = 0

    proc = subprocess.Popen(
        args.command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    orig = signal.signal(signal.SIGINT, lambda s, f: proc.send_signal(signal.SIGINT))

    def _fmt_inline(e, step_gap: int) -> str:
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
                metrics = extract_metrics(line)
                if metrics:
                    sampler.update_train_metrics(metrics)
                    if otlp:
                        for k in metrics:
                            otlp.note_key(k)

                if active and last is not None:
                    # Window since last begin_step = work that produced this
                    # step-N line. Attribute to det, print on the current line.
                    e = sampler.end_step(det)
                    step_gap = max(det - last, 1)
                    if emitter:
                        emitter.emit(step_record(run_id, e, step_gap,
                                                 train_metrics=metrics or None))
                    else:
                        print(f"{line} {_fmt_inline(e, step_gap)}")
                    total_steps += 1
                else:
                    # First step line ever — no measurement window yet.
                    if not emitter:
                        print(line)

                sampler.begin_step()
                active = True
                last = det
            else:
                if not emitter:
                    print(line)
                    sys.stdout.flush()

        # End of stream: we have an open begin_step with no matching end line
        # (training exited or crashed mid-step). That partial window is not a
        # complete step — drop it rather than mislabel it. Session totals in
        # session_end still include this time, so no energy is lost.
        proc.wait()
    finally:
        signal.signal(signal.SIGINT, orig)
        if prom:
            prom.stop()
        if otlp:
            otlp.stop()
        s = sampler.stop()

        if emitter:
            emitter.emit(session_end_record(run_id, s, total_steps=total_steps))
            emitter.close()

        print(_human_summary(s))

    return proc.returncode


def _monitor_cmd(args):
    """Live per-GPU monitor (nvidia-smi replacement)."""
    gpus = _parse_gpus(args.gpus)
    if gpus == -1:
        indices = None
    elif isinstance(gpus, int):
        indices = [gpus]
    else:
        indices = list(gpus)
    return _monitor.run(indices, args.interval)


def _add_common_flags(p):
    p.add_argument("--gpus", type=str, default="all",
                   help="GPU indices: all, 0, or 0,1,2 (default: all)")
    p.add_argument("--interval", type=int, default=100,
                   help="Peak-power poll interval ms (default: 100). "
                        "Energy uses the hardware counter and is independent of this.")
    p.add_argument("--json", action="store_true",
                   help="Emit structured JSONL records")
    p.add_argument("--output", type=str, default=None,
                   help="Write JSONL records to this file (implies --json)")
    p.add_argument("--label", action="append", default=None, metavar="KEY=VALUE",
                   help="Attach a label to the run (repeatable)")
    p.add_argument("--run-id", dest="run_id", type=str, default=None,
                   help="Stable run ID (or set MATCHA_RUN_ID)")
    p.add_argument("--prometheus", type=str, default=None, metavar="[HOST]:PORT",
                   help="Expose a Prometheus /metrics endpoint (e.g. :9400)")
    p.add_argument("--otlp", type=str, default=None, metavar="URL",
                   help="Push metrics to an OTLP/HTTP collector (e.g. http://collector:4318)")
    p.add_argument("--otlp-header", action="append", default=None, metavar="KEY=VALUE",
                   help="Header for OTLP export (repeatable, e.g. api-key=...)")
    p.add_argument("--otlp-interval", type=int, default=10000, metavar="MS",
                   help="OTLP export interval in ms (default: 10000)")


def main():
    parser = argparse.ArgumentParser(
        prog="matcha",
        description="matcha — GPU energy metering for AI training",
    )
    parser.add_argument("--version", action="version", version=f"matcha {__version__}")
    sub = parser.add_subparsers(dest="cmd")

    rp = sub.add_parser("run", help="Run a command and report total GPU energy at the end")
    _add_common_flags(rp)
    rp.add_argument("command", nargs=argparse.REMAINDER)

    wp = sub.add_parser("wrap", help="Run a training script with per-step energy reporting")
    _add_common_flags(wp)
    wp.add_argument("command", nargs=argparse.REMAINDER)

    dp = sub.add_parser("diff", help="Compare matcha JSONL runs (2 = pairwise, 3+ = sweep table)")
    dp.add_argument("runs", nargs="+",
                    help="JSONL files from `matcha --output` (2 for pairwise diff, 3+ for sweep)")

    mp = sub.add_parser("monitor", help="Live per-GPU power monitor")
    mp.add_argument("--gpus", type=str, default="all",
                    help="GPU indices: all, 0, or 0,1,2 (default: all)")
    mp.add_argument("--interval", type=int, default=500,
                    help="Refresh interval ms (default: 500)")

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
    elif args.cmd == "diff":
        sys.exit(_diff.run(args.runs))
    elif args.cmd == "monitor":
        sys.exit(_monitor_cmd(args))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
