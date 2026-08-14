.PHONY: fmt test docs

fmt:
	python -m ruff format .
	python -m ruff check .

test:
	python -m pytest -q

docs:
	python -m mkdocs serve
