.PHONY: install validate test lint format dry-run clean

install:
	pip install -e ".[dev,nlp,notebooks]"

validate:
	python scripts/validate_config.py --config configs/base.yaml

test:
	pytest

lint:
	ruff check src tests scripts
	mypy src

format:
	ruff format src tests scripts
	ruff check --fix src tests scripts

dry-run:
	python scripts/run_pipeline.py --config configs/base.yaml --dry-run

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache build dist *.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
