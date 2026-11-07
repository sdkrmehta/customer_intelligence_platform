# ============================================================
# Makefile for Retail Customer Purchase Prediction System
# ============================================================
.PHONY: help install test test-cov train train-sample app lint format docker-build docker-run clean

PYTHON ?= python

help:
	@echo "Available commands:"
	@echo "  make install       - Install package and core dependencies in editable mode"
	@echo "  make test          - Run full pytest test suite"
	@echo "  make test-cov      - Run tests with test coverage report"
	@echo "  make train         - Train default XGBoost model on full dataset"
	@echo "  make train-sample  - Fast training run on 5% data sample"
	@echo "  make app           - Launch interactive Streamlit dashboard"
	@echo "  make lint          - Check code quality with ruff"
	@echo "  make format        - Automatically format code with ruff"
	@echo "  make docker-build  - Build production Docker container image"
	@echo "  make docker-run    - Run application inside Docker container on port 8501"
	@echo "  make clean         - Remove cache files, build artifacts, and temp files"

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e ".[dev,notebooks]"

test:
	$(PYTHON) -m pytest tests/ -v

test-cov:
	$(PYTHON) -m pytest tests/ --cov=src --cov-report=term-missing

train:
	$(PYTHON) train_model.py --model xgboost

train-sample:
	$(PYTHON) train_model.py --model xgboost --sample 0.05

app:
	streamlit run app/streamlit_app.py

lint:
	$(PYTHON) -m ruff check src tests app

format:
	$(PYTHON) -m ruff format src tests app

docker-build:
	docker build -t retail-purchase-predictor:latest .

docker-run:
	docker run -p 8501:8501 -v $$(pwd)/models:/app/models retail-purchase-predictor:latest

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	rm -rf build dist *.egg-info .coverage htmlcov
