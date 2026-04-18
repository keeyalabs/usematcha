# SPDX-License-Identifier: Apache-2.0
"""matcha._diff — compare two runs recorded as JSONL."""

import json
import re
import statistics
from typing import Any, Dict, List, Optional, Tuple


_GPU_MODEL_RE = re.compile(
    r"\b(H200|H100|B200|B100|A100|A40|A30|A10|V100|P100|P40|T4|L40S?|L4|"
    r"RTX\s*\d{4}\w*|GH200|GB200)\b",
    re.IGNORECASE,
)


def _short_gpu_name(name: str) -> str:
    m = _GPU_MODEL_RE.search(name or "")
    if m:
        return m.group(1).upper().replace(" ", "")
    return (name or "?")[:12]


def _fmt_energy(v: Optional[float]) -> str:
    """Auto-scale joules: 1234 J, 12.3 kJ, 1.23 MJ, 1.23 GJ."""
    if v is None:
        return "—"
    av = abs(v)
    if av < 10_000:
        return f"{v:.0f} J"
    if av < 10_000_000:
        return f"{v / 1_000:.1f} kJ"
    if av < 10_000_000_000:
        return f"{v / 1_000_000:.2f} MJ"
    return f"{v / 1_000_000_000:.2f} GJ"


def _load(path: str) -> Dict[str, Any]:
    start: Optional[Dict[str, Any]] = None
    end: Optional[Dict[str, Any]] = None
    steps: List[Dict[str, Any]] = []
    with open(path, "r") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            rec = json.loads(raw)
            t = rec.get("type")
            if t == "session_start":
                start = rec
            elif t == "session_end":
                end = rec
            elif t == "step":
                steps.append(rec)
    if start is None or end is None:
        raise SystemExit(
            f"matcha: {path} is missing session_start or session_end records — "
            f"is this a complete matcha JSONL?"
        )
    return {"start": start, "end": end, "steps": steps}


def _train_metric_medians(steps: List[Dict[str, Any]]) -> Dict[str, float]:
    buckets: Dict[str, List[float]] = {}
    for s in steps:
        for k, v in (s.get("train_metrics") or {}).items():
            try:
                buckets.setdefault(k, []).append(float(v))
            except (TypeError, ValueError):
                continue
    return {k: statistics.median(vs) for k, vs in buckets.items() if vs}


def _max_gpu_deviation(steps: List[Dict[str, Any]]) -> Optional[float]:
    """Across all steps, the largest |(gpu_energy - median) / median| observed."""
    worst = 0.0
    seen = False
    for s in steps:
        gpus = s.get("gpus") or []
        if len(gpus) < 2:
            continue
        energies = sorted(g["energy_j"] for g in gpus)
        mid = len(energies) // 2
        med = (energies[mid] if len(energies) % 2
               else 0.5 * (energies[mid - 1] + energies[mid]))
        if med <= 0:
            continue
        for g in gpus:
            dev = abs((g["energy_j"] - med) / med)
            if dev > worst:
                worst = dev
        seen = True
    return worst if seen else None


def _gpu_summary(start: Dict[str, Any]) -> str:
    gpus = start.get("gpus") or []
    if not gpus:
        return "unknown"
    names = [g.get("name", "?") for g in gpus]
    uniq = set(names)
    if len(uniq) == 1:
        return f"{len(gpus)}x {_short_gpu_name(names[0])}"
    return ", ".join(_short_gpu_name(n) for n in uniq)


def _pct(a: Optional[float], b: Optional[float]) -> str:
    if a is None or b is None or a == 0:
        return ""
    return f"{((b - a) / a * 100):+.1f}%"


def _row(label: str, a_s: str, b_s: str, d_s: str = "", p_s: str = "",
         col_w=(22, 26, 26, 14, 9)) -> str:
    def cell(s, w):
        s = str(s)
        return (s[: w - 1] + " ") if len(s) >= w else s.ljust(w)
    return cell(label, col_w[0]) + cell(a_s, col_w[1]) + cell(b_s, col_w[2]) + cell(d_s, col_w[3]) + cell(p_s, col_w[4])


def _fmt(v: Optional[float], unit: str = "", digits: int = 1) -> str:
    if v is None:
        return "—"
    return f"{v:.{digits}f}{unit}"


def _fmt_delta(a: Optional[float], b: Optional[float], unit: str, digits: int) -> str:
    if a is None or b is None:
        return ""
    d = b - a
    sign = "+" if d >= 0 else "-"
    return f"{sign}{abs(d):.{digits}f}{unit}"


def render(a_path: str, b_path: str) -> str:
    a = _load(a_path)
    b = _load(b_path)

    sa, sb = a["start"], b["start"]
    ea, eb = a["end"], b["end"]

    lines: List[str] = []
    lines.append(_row("", "baseline", "candidate", "delta", "pct"))
    lines.append("-" * 97)

    def meta(label, va, vb):
        lines.append(_row(label, str(va), str(vb)))

    meta("run_id", sa.get("run_id", "?"), sb.get("run_id", "?"))
    meta("hostname", sa.get("hostname", "?"), sb.get("hostname", "?"))
    meta("gpus", _gpu_summary(sa), _gpu_summary(sb))
    meta("energy_source", sa.get("energy_source", "?"), sb.get("energy_source", "?"))
    lines.append("")

    def metric(label, fa, fb, unit, digits=1):
        lines.append(_row(label,
                          _fmt(fa, unit, digits),
                          _fmt(fb, unit, digits),
                          _fmt_delta(fa, fb, unit, digits),
                          _pct(fa, fb)))

    metric("duration", ea.get("duration_s"), eb.get("duration_s"), "s", 1)
    metric("total energy", ea.get("total_energy_j"), eb.get("total_energy_j"), " J", 0)
    metric("  (Wh)", ea.get("energy_wh"), eb.get("energy_wh"), " Wh", 3)
    metric("avg power", ea.get("avg_power_w"), eb.get("avg_power_w"), " W", 0)
    metric("peak power", ea.get("peak_power_w"), eb.get("peak_power_w"), " W", 0)
    metric("total steps",
           float(ea.get("total_steps") or 0),
           float(eb.get("total_steps") or 0), "", 0)
    metric("energy/step", ea.get("energy_per_step_j"), eb.get("energy_per_step_j"), " J", 1)

    tma = _train_metric_medians(a["steps"])
    tmb = _train_metric_medians(b["steps"])
    keys = sorted(set(tma) | set(tmb))
    if keys:
        lines.append("")
        lines.append("train metrics (median across steps):")
        for k in keys:
            metric(f"  {k}", tma.get(k), tmb.get(k), "", 3)

    da = _max_gpu_deviation(a["steps"])
    db = _max_gpu_deviation(b["steps"])
    if da is not None or db is not None:
        lines.append("")
        lines.append("straggler signal (max |per-gpu deviation from median|):")
        metric("  max deviation",
               da * 100 if da is not None else None,
               db * 100 if db is not None else None,
               "%", 1)

    return "\n".join(lines) + "\n"


def _label_for(path: str, start: Dict[str, Any]) -> str:
    """Column header for sweep view: prefer labels.config, else filename stem."""
    labels = start.get("labels") or {}
    for key in ("config", "name", "variant"):
        if key in labels:
            return str(labels[key])
    import os
    base = os.path.basename(path)
    return base[: -len(".jsonl")] if base.endswith(".jsonl") else base


def _best_idx(values: List[Optional[float]], lower_is_better: bool = True) -> Optional[int]:
    real = [(i, v) for i, v in enumerate(values) if v is not None]
    if not real:
        return None
    vs = {v for _, v in real}
    if len(vs) == 1:
        return None  # all tied — no "best"
    fn = min if lower_is_better else max
    return fn(real, key=lambda p: p[1])[0]


def render_sweep(paths: List[str]) -> str:
    runs = [_load(p) for p in paths]
    headers = [_label_for(p, r["start"]) for p, r in zip(paths, runs)]
    n = len(runs)

    # Build the table as a list of rows, where each row is either:
    #   ("section", "heading text")  — heading-only row
    #   ("blank", None)               — blank separator
    #   ("data", label, cells)        — label + N cells (strings, * already appended)
    rows: List[Tuple[str, Any, Any]] = []

    def mark_best(values: List[Optional[float]], strs: List[str],
                  lower_is_better: bool) -> List[str]:
        best = _best_idx(values, lower_is_better) if n > 1 else None
        return [s + ("*" if i == best and v is not None else "")
                for i, (v, s) in enumerate(zip(values, strs))]

    def metric(label: str, values: List[Optional[float]],
               unit: str, digits: int, lower_is_better: bool = True,
               formatter=None):
        if formatter is None:
            strs = [_fmt(v, unit, digits) for v in values]
        else:
            strs = [formatter(v) for v in values]
        rows.append(("data", label, mark_best(values, strs, lower_is_better)))

    def static_row(label: str, cells: List[str]):
        rows.append(("data", label, cells))

    rows.append(("data", "", headers))
    rows.append(("rule", None, None))

    static_row("gpus", [_gpu_summary(r["start"]) for r in runs])
    static_row("hostname", [r["start"].get("hostname", "?") for r in runs])
    rows.append(("blank", None, None))

    def pick(key): return [r["end"].get(key) for r in runs]
    metric("duration (s)", pick("duration_s"), "", 1)
    metric("total energy", pick("total_energy_j"), "", 0, formatter=_fmt_energy)
    metric("total energy (Wh)", pick("energy_wh"), "", 2)
    metric("avg power (W)", pick("avg_power_w"), "", 0)
    metric("peak power (W)", pick("peak_power_w"), "", 0)
    metric("total steps", pick("total_steps"), "", 0, lower_is_better=False)
    metric("energy/step (J)", pick("energy_per_step_j"), "", 1)

    tms = [_train_metric_medians(r["steps"]) for r in runs]
    keys = sorted({k for tm in tms for k in tm})
    if keys:
        rows.append(("blank", None, None))
        rows.append(("section", "train metrics (median):", None))
        for k in keys:
            vals = [tm.get(k) for tm in tms]
            lib = any(tag in k.lower() for tag in ("loss", "time", "ms", "latency"))
            hib = any(tag in k.lower() for tag in ("acc", "mfu", "throughput", "tokens"))
            if lib or hib:
                metric(f"  {k}", vals, "", 3, lower_is_better=lib)
            else:
                static_row(f"  {k}", [_fmt(v, "", 3) for v in vals])

    devs = [_max_gpu_deviation(r["steps"]) for r in runs]
    if any(d is not None for d in devs):
        rows.append(("blank", None, None))
        rows.append(("section", "straggler signal:", None))
        vals = [d * 100 if d is not None else None for d in devs]
        metric("  max deviation (%)", vals, "", 1)

    # ---- Pass 2: compute column widths from actual rendered content ----
    label_w = max(
        (len(r[1]) for r in rows if r[0] == "data" and r[1]),
        default=0,
    )
    label_w = max(label_w, 16) + 2  # breathing room

    col_ws = [len(h) for h in headers]
    for kind, label, cells in rows:
        if kind != "data" or cells is None:
            continue
        for i, c in enumerate(cells):
            if len(c) > col_ws[i]:
                col_ws[i] = len(c)
    col_ws = [w + 2 for w in col_ws]  # breathing room

    total_w = label_w + sum(col_ws)

    out: List[str] = []
    for kind, label, cells in rows:
        if kind == "rule":
            out.append("-" * total_w)
        elif kind == "blank":
            out.append("")
        elif kind == "section":
            out.append(label)
        else:
            line = (label or "").ljust(label_w)
            for c, w in zip(cells, col_ws):
                line += str(c).ljust(w)
            out.append(line.rstrip())

    if n > 1:
        out.append("")
        out.append("* = best in row")

    return "\n".join(out) + "\n"


def run(paths: List[str]) -> int:
    if len(paths) < 2:
        raise SystemExit("matcha diff: needs at least 2 JSONL files")
    if len(paths) == 2:
        print(render(paths[0], paths[1]), end="")
    else:
        print(render_sweep(paths), end="")
    return 0
