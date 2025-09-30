# Repository Guidelines

## Project Structure & Module Organization
- Library code and CLI live in `src/tushare_a_fundamentals/` (e.g., `cli.py`, `commands/download.py`, `downloader.py`).
- Configuration templates sit at the repo root (`config.example.yaml`, `.env.example`, `.envrc.example`).
- Tests are split into `tests/unit/` and `tests/integration/`; use markers defined in `pyproject.toml`.
- Tools and scripts reside in `tools/`; data artifacts default to `out/` (legacy) or `data/` (multi-dataset mode).

## Build, Test, and Development Commands
- `uv sync` — install project and dev dependencies into `.venv`.
- `pytest -m unit` / `pytest -m integration` — run targeted suites; use plain `pytest` before merging.
- `ruff check .` and `ruff format .` — lint and auto-format the codebase.
- `funda download --help` or `python -m tushare_a_fundamentals.cli download --help` — inspect CLI options.

## Coding Style & Naming Conventions
- Python 3.10+ with 4-space indentation; keep lines ≤88 chars (Black/Ruff defaults).
- Use snake_case for modules, functions, and variables; PascalCase for classes; UPPER_SNAKE_CASE for constants.
- Add type hints for public functions; prefer small pure functions. CLI/user messages should be in Chinese, code identifiers in English.
- Run Ruff before committing to ensure formatting parity.

## Testing Guidelines
- Pytest is the test runner; name files `test_*.py` and apply `unit`/`integration` markers as appropriate.
- Avoid network calls in unit tests—mock TuShare APIs; integration tests may hit the service when required.
- Ensure new features include regression coverage; target existing helpers such as `DummyPro` for downloader tests.

## Commit & Pull Request Guidelines
- Write commits in imperative mood (optional scopes like `feat:`, `fix:`, `tests:`). Group related changes; avoid mixed commits.
- Pull requests should summarize behavior, list linked issues, include repro steps, and capture relevant CLI outputs (e.g., `funda download` logs).
- Before requesting review, run linting and the appropriate pytest markers; document any skipped tests or known limitations.

## Security & Configuration Tips
- Never commit real TuShare tokens; use `.env`/`.envrc` with `TUSHARE_TOKEN` and let `direnv` manage loading.
- Keep `config.yml` untracked (see `.gitignore`); copy from `config.example.yaml` and adjust locally.
- Use multi-dataset state stored in `data/_state/state.json`; deleting it forces a full refresh.
