# Repository Guidelines

## Project Structure & Module Organization
- `src/tushare_a_fundamentals/`: Library code. CLI entrypoint is `cli.py` (`funda`).
- `tests/unit/` and `tests/integration/`: Pytest suites; see markers in `pyproject.toml`.
- `tools/`: Helper scripts (e.g., `check_api_availability.py`).
- `docs/`: Project docs.
- Root configs: `pyproject.toml`, `config.example.yml`, `.env.example`, `.envrc.example`.

## Build, Test, and Development Commands
- Setup (recommended): `uv sync` (installs deps + dev tools). Alternative: `pip install -e .` then install `pytest ruff black pytest-cov`.
- Lint: `ruff check .`
- Format: `black .` (check only: `black --check .`)
- Run tests: `pytest` (coverage enabled). Subsets: `pytest -m unit`, `pytest -m integration`.
- Run CLI: `funda --help` or `python -m tushare_a_fundamentals.cli`.

## Coding Style & Naming Conventions
- Python 3.10+; 4-space indentation; line length 88 (`black`, `ruff`).
- Names: `snake_case` for modules/functions/vars, `UPPER_SNAKE` for constants, `PascalCase` for classes.
- Prefer type hints and small, pure functions. Keep user-facing messages in Chinese, code identifiers in English.
- Update `config.example.yml` and README when adding CLI options or output files.

## Testing Guidelines
- Framework: `pytest` with markers: `unit`, `integration`.
- Location/naming: place tests under `tests/`, files named `test_*.py`.
- Aim to cover new logic; avoid network in unit tests. Use fixtures/mocks for TuShare calls.
- Run `pytest -m unit` locally for quick feedback; ensure `pytest` passes before opening PRs.

## Commit & Pull Request Guidelines
- Commits: imperative mood, concise subject. Optional scope: `feat:`, `fix:`, `docs:`, `refactor:`, `tests:`.
- PRs: clear description, linked issues, reproduction steps, and sample CLI commands/outputs. Include tests and doc updates.

## Security & Configuration Tips
- Do not commit secrets. Set `TUSHARE_TOKEN` in `.env` (copy from `.env.example`). `direnv` will auto-load `.env` via `.envrc.example`.
- Keep `config.yml` and generated outputs out of version control (see `.gitignore`).

## Agent-Specific Instructions
- Keep patches minimal and focused; avoid broad refactors.
- When changing behavior or file paths, update tests and docs in the same patch.
- Use consistent CLI naming and output patterns described in README.
