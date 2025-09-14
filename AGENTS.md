# Repository Guidelines

## Project Structure & Module Organization
- `src/tushare_a_fundamentals/`: Library code and CLI (`cli.py`, entry script `funda`).
- `tests/unit/`, `tests/integration/`: Pytest suites with markers in `pyproject.toml`.
- `tools/`: Helper scripts (e.g., `tools/check_api_availability.py`).
- Configs: `config.example.yaml`, `configs/datasets.example.yaml` (copy to local `config.yml` and `configs/datasets.yaml`).
- Root: `pyproject.toml`, `.env.example`, `.envrc.example`, `README.md`.

## Build, Test, and Development Commands
- Setup: `uv sync` (recommended) or `pip install -e .` (+ `pytest ruff black pytest-cov`).
- Prepare configs: `cp config.example.yaml config.yml`; `cp configs/datasets.example.yaml configs/datasets.yaml`.
- Lint/format: `ruff check .`; `black .` (check: `black --check .`).
- Tests: `pytest` (or `pytest -m unit`, `pytest -m integration`).
- Run CLI: `funda download --help` or `python -m tushare_a_fundamentals.cli download --help`.

## Coding Style & Naming Conventions
- Python 3.10+; 4-space indent; max line length 88 (Black, Ruff).
- Naming: `snake_case` (modules/functions/vars), `UPPER_SNAKE` (constants), `PascalCase` (classes).
- Prefer type hints, small pure functions. User-facing messages in Chinese; code identifiers in English.
- When adding CLI options or outputs, update `config.example.yaml` and `README.md`.

## Testing Guidelines
- Framework: Pytest with `unit` and `integration` markers.
- Test files in `tests/` named `test_*.py`; avoid network in unit tests, mock TuShare.
- Run `pytest -m unit` for quick checks; ensure full `pytest` passes before PRs.

## Commit & Pull Request Guidelines
- Commits: imperative mood; optional scope prefixes (`feat:`, `fix:`, `docs:`, `refactor:`, `tests:`).
- PRs: include description, linked issues, repro steps, and sample CLI commands/outputs; add tests and doc updates.

## Security & Configuration Tips
- Never commit secrets. Set `TUSHARE_TOKEN` in `.env` (copy `.env.example`).
- Use `direnv`: `cp .envrc.example .envrc && direnv allow` to auto-load `.env` and venv.
- Keep `config.yml`, dataset paths, and generated outputs out of VCS (see `.gitignore`).

## CLI Behavior Notes
- Unified download: `funda download` defaults to incremental “补全”；use `--force` to overwrite.
- Time window: `--since/--until` > `--quarters` > `--years` (default 10). Example: `funda download --since 2010-01-01`.
