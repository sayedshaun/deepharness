.PHONY: fmt test

fmt:
	ruff format .
	ruff check .

test:
	pytest -q
