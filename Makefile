.PHONY: install lint format test test-models evaluate ui api docker-build docker-run

TORCH_CPU = --extra-index-url https://download.pytorch.org/whl/cpu

install:
	pip install -e ".[models,ui,api,browser,dev]" $(TORCH_CPU)
	python -m playwright install chromium
	pre-commit install || true

lint:
	ruff check .
	ruff format --check .

format:
	ruff check --fix .
	ruff format .

test: 
	pytest --cov=text_classifier --cov-report=term-missing

test-models: 
	pytest -m models

evaluate: 
	text-classifier evaluate

ui:
	streamlit run text_classifier/app.py

api:
	text-classifier serve

docker-build:
	docker build -t text-classifier-api .

docker-run: docker-build
	docker run --rm -p 8000:8000 -v hf-cache:/models text-classifier-api
