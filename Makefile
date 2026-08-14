.PHONY: fmt test

fmt:
	python -m ruff format .
	python -m ruff check .

test:
	python -m pytest -q
