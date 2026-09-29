.PHONY: check format test
check:
	uv run --locked ruff check .
	uv run --locked ruff format --check .
	uv run --locked python -W error -m unittest discover -s tests -v
test:
	uv run --locked python -W error -m unittest discover -s tests -v
format:
	uv run --locked ruff check --fix .
	uv run --locked ruff format .
