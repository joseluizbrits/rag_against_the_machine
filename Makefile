.PHONY: install run debug clean clean-win lint lint-strict

install:
	uv sync

run:
	.venv/Scripts/python -m src

debug:
	.venv/Scripts/python -m pdb -m src

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

clean-win:
	powershell -Command "Get-ChildItem -Recurse -Filter '__pycache__' | Remove-Item -Recurse -Force; Get-ChildItem -Recurse -Filter '.mypy_cache' | Remove-Item -Recurse -Force; Get-ChildItem -Recurse -Filter '*.pyc' -ErrorAction SilentlyContinue | Remove-Item -Force"

lint:
	.venv/Scripts/flake8 src
	.venv/Scripts/mypy src --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	.venv/Scripts/flake8 src
	.venv/Scripts/mypy src --strict
