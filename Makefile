.PHONY: install run debug clean lint lint-strict

install:
	uv sync

run:
	uv run python -m src

debug:
	uv run python -m pdb -m src

clean:
	uv run python -c "import pathlib, shutil; [shutil.rmtree(p, ignore_errors=True) for p in list(pathlib.Path('.').rglob('*')) if p.is_dir() and p.name in ('__pycache__', '.mypy_cache', '.pytest_cache')]"

FLAKE8_EXCLUDE = .venv,.git,__pycache__,.mypy_cache,.pytest_cache,data,moulinette,openspec,build,dist

lint:
	uv run flake8 . --exclude=$(FLAKE8_EXCLUDE)
	uv run mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	uv run flake8 . --exclude=$(FLAKE8_EXCLUDE)
	uv run mypy . --strict
