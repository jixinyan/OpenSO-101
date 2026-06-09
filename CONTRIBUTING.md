# Contributing to OpenSO-101

Thanks for your interest in improving OpenSO-101! Contributions of all kinds —
bug reports, docs, new tasks, RL/IL improvements — are welcome.

## Getting set up

Follow [`docs/guides/install.md`](docs/guides/install.md) for a full
environment (Isaac Sim + Isaac Lab + LeRobot). In short:

```bash
git clone https://github.com/jixinyan/OpenSO-101.git
cd OpenSO-101
conda env create -f environment.yml && conda activate openso101
bash scripts/install.sh
bash scripts/fetch_so101_usd.sh        # third-party SO-101 USD mesh (not in git)
openso101 envs list                    # smoke check
```

## Workflow

1. Fork the repo and create a feature branch:
   `git checkout -b feat/short-description`
2. Make your change. Keep PRs **scoped to one concern** — small, reviewable diffs.
3. Run the relevant tests (see below) and make sure they pass.
4. Commit with a [Conventional Commits](https://www.conventionalcommits.org/)
   prefix (`feat:`, `fix:`, `docs:`, `test:`, `chore:`).
5. Open a Pull Request against `main` describing what changed and why.

For feature ideas or bugs, please open an issue first (tag `enhancement` or
`bug`) so we can discuss the approach.

## Tests

The suite lives in [`tests/`](tests/). Most tests need a CUDA GPU + Isaac Sim:
`conftest.py` boots a headless `SimulationApp` once per session, so a bare
`pytest tests/` only works on a GPU machine.

A **CPU-only subset** has no Isaac Sim dependency and runs anywhere — this is
exactly what CI runs:

```bash
pytest tests/test_shaping_rewards.py tests/test_cli_rl.py
```

Pure-logic helpers (e.g. `openso101.shaping`) should stay importable without
Isaac Lab so their tests can run on CPU. If you add such a helper, keep it out
of the `openso101.tasks` package (which eagerly registers gym envs and pulls in
`isaaclab`).

## Conventions

- **License headers.** New Python files start with:
  ```python
  # Copyright (c) 2026, Jixin Yan
  # SPDX-License-Identifier: MIT
  ```
- **Adding a task.** See
  [`docs/guides/add_a_task.md`](docs/guides/add_a_task.md) — subclass
  `OpenSO101EnvCfg` and register it with the `@register_task` decorator.
- **Style.** Match the surrounding code; CI runs `ruff check` (advisory) on
  `src/` and `tests/`.

## License

By contributing, you agree that your contributions are licensed under the
project's [MIT License](LICENSE). Note that the bundled SO-ARM101 USD mesh is a
third-party BSD-3-Clause asset (see
[`LICENSE-BSD-3-CLAUSE`](LICENSE-BSD-3-CLAUSE)).
