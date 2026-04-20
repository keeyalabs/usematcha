# Contributing to matcha

Thanks for helping make matcha better. This is a small, focused project — the fastest path from "I have an idea" to "it's merged" is a tight PR that fits the roadmap in [ARCHITECTURE.md](ARCHITECTURE.md).

## Dev setup

```bash
git clone https://github.com/keeyalabs/usematcha.git
cd usematcha
python3 -m venv .venv && source .venv/bin/activate
pip install -e '.[otlp,hf]' ruff mypy pytest
```

The editable install + both extras gives you the full surface the CI runs against.

## Running tests

```bash
pytest                 # full suite, no NVIDIA GPU required
pytest -k callbacks    # subset by keyword
pytest -x -q           # fail fast, quiet
```

Tests run fully off-GPU: `pynvml` and `transformers` are faked where needed, so the same suite executes in CI on Linux runners without hardware. End-to-end validation on real H100 hardware happens out-of-band before each release — see [ARCHITECTURE.md §Release validation](ARCHITECTURE.md#release-validation).

## Code style

- `ruff check .` — import order, unused symbols, bugbear.
- `ruff format .` — formatting.
- `mypy matcha` — type checks.
- Line length: **100**.
- Every source file opens with `# SPDX-License-Identifier: Apache-2.0`.
- Internal modules are `_prefixed` (e.g. `_engine.py`); public surface lives in `matcha/__init__.py`, `matcha/_api.py`, and `matcha/callbacks/`.

CI runs all three checks on every PR. A PR that doesn't pass locally won't pass there either — save the round trip.

## Commits

- Small, focused commits. One concern per commit, one concern per PR.
- Subject line: imperative mood, no trailing period.  
  Good: `Add Ray Train callback adapter`  
  Avoid: `added ray train callback.`
- Body wraps at ~72 chars. Explain *why*, not *what* — the diff already shows what.
- Reference issues in the body (`Fixes #42`).

## PRs

- Update `CHANGELOG.md` under `## [unreleased]` for any user-visible change.
- If you're adding a public symbol, add a test that locks the surface (name, signature) in `tests/test_public_api.py`.
- If you're adding a framework callback, add:
  1. A file in `matcha/callbacks/<framework>.py` with its own lazy-import gate in `matcha/callbacks/__init__.py`.
  2. A new optional extra in `pyproject.toml` — e.g. `[project.optional-dependencies] lightning = ["pytorch-lightning>=2.0"]`.
  3. Off-hardware tests that fake the framework and drive every hook.

## Scope

matcha is a measurement tool. It deliberately does **not**:

- Modify training code, gradients, batch schedules, or optimizer state.
- Enforce training policies (warnings, limits, alerts) — those belong in your observability stack.
- Aggregate across hosts — Prometheus, OTLP, and your log pipeline already do this well.
- Support non-NVIDIA GPUs — different vendor APIs are different projects. We'd rather do one thing well.

Proposals that expand scope in any of those directions are unlikely to be accepted without a strong case. Proposals that **tighten accuracy, add framework adapters, reduce overhead, or improve ergonomics** are very welcome.

## Security

Please **do not** open a public issue for suspected vulnerabilities. See [SECURITY.md](SECURITY.md) for the reporting channel.

## License

By contributing, you agree that your contribution is licensed under the Apache License 2.0 (see [LICENSE](LICENSE)), and that you have the right to license it as such.
