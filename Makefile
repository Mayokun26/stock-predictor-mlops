# MLOps Pipeline Makefile
# Provides convenient commands for development and CI/CD

.PHONY: help install test train deploy clean lint format setup-dev

# Default target
help:
	@echo "MLOps Pipeline Commands:"
	@echo "========================"
	@echo "Development:"
	@echo "  make install     - Install dependencies"
	@echo "  make setup-dev   - Set up development environment"
	@echo "  make lint        - Run linting checks"
	@echo "  make format      - Format code"
	@echo ""
	@echo "Testing:"
	@echo "  make test        - Run all tests"
	@echo "  make test-unit   - Run unit tests only"
	@echo "  make test-api    - Run API tests only"
	@echo "  make test-mlflow - Run MLflow tests only"
	@echo ""
	@echo "MLOps Pipeline:"
	@echo "  make train       - Run model training pipeline"
	@echo "  make evaluate    - Run model evaluation"
	@echo "  make deploy      - Deploy models (local)"
	@echo ""
	@echo "Services:"
	@echo "  make start-services  - Start all services (MLflow, API, DB)"
	@echo "  make stop-services   - Stop all services"
	@echo "  make restart-services - Restart all services"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean       - Clean temporary files"
	@echo "  make clean-all   - Clean everything (including models)"

# Installation
install:
	@echo "📦 Installing dependencies..."
	pip install --upgrade pip
	pip install -r requirements.txt

setup-dev: install
	@echo "🔧 Setting up development environment..."
	mkdir -p logs mlruns/artifacts data/raw data/processed
	pip install pre-commit
	pre-commit install
	@echo "✅ Development environment ready!"

# Testing
test:
	@echo "🧪 Running all tests..."
	python scripts/run_tests.py --type all

test-unit:
	@echo "🧪 Running unit tests..."
	python scripts/run_tests.py --type unit

test-api:
	@echo "🧪 Running API tests..."
	python scripts/run_tests.py --type api

test-mlflow:
	@echo "🧪 Running MLflow tests..."
	python scripts/run_tests.py --type mlflow

# MLOps Pipeline
train:
	@echo "🤖 Starting model training pipeline..."
	python scripts/train_pipeline.py

evaluate:
	@echo "📊 Running model evaluation..."
	python comprehensive_model_evaluation.py

# Services management
start-services:
	@echo "🚀 Starting all services..."
	@echo "Starting MLflow server..."
	@/home/user/.local/bin/mlflow server --host 0.0.0.0 --port 5001 --backend-store-uri sqlite:///mlruns/mlflow.db --default-artifact-root ./mlruns/artifacts &
	@sleep 5
	@echo "Starting API server..."
	@cd src/api && python production_api.py &
	@sleep 3
	@echo "✅ All services started!"

stop-services:
	@echo "🛑 Stopping all services..."
	@pkill -f "mlflow server" || true
	@pkill -f "production_api.py" || true
	@echo "✅ All services stopped!"

restart-services: stop-services
	@sleep 2
	@make start-services

# Cleanup
clean:
	@echo "🧹 Cleaning temporary files..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -f *.log
	@echo "✅ Cleanup completed!"

clean-all: clean
	@echo "🧹 Deep cleaning (including models and data)..."
	rm -rf mlruns/
	rm -rf logs/
	rm -f *.db
	rm -f pipeline_execution_report.txt
	rm -f model_evaluation_report.txt
	@echo "⚠️  All models and logs have been removed!"

# Health checks
health-check:
	@echo "🔍 Running health checks..."
	@curl -s http://localhost:8000/health | python -m json.tool || echo "❌ API not responding"
	@curl -s http://localhost:5001/health || echo "❌ MLflow not responding"
