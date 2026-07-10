all:
	install train tune evaluate predict

install:
	uv sync --all-extras

train:
	uv run python -m src.stroke_risk.models.train

tune:
	uv run python -m src.stroke_risk.models.tune

evaluate:
	uv run python -m src.stroke_risk.models.evaluate

inference:
	uv run python -m src.stroke_risk.serving.inference

test:
	uv run pytest tests/

lint:
	uv run ruff check src tests