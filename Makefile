.PHONY: help setup install clean test run-api run-pipeline docker-build docker-run deploy monitor

# Default target
help:
	@echo "MLOps Stock Prediction System"
	@echo "=============================="
	@echo ""
	@echo "Available commands:"
	@echo "  setup          - Initial project setup"
	@echo "  install        - Install Python dependencies"
	@echo "  clean          - Clean up generated files"
	@echo "  test           - Run tests"
	@echo "  run-api        - Start the FastAPI server"
	@echo "  run-pipeline   - Run the complete MLOps pipeline"
	@echo "  collect-data   - Collect stock market data"
	@echo "  train-models   - Train ML models"
	@echo "  docker-build   - Build Docker containers"
	@echo "  docker-run     - Run with Docker Compose"
	@echo "  docker-stop    - Stop Docker containers"
	@echo "  deploy         - Deploy to production"
	@echo "  monitor        - Run monitoring checks"
	@echo "  monitor-live   - Start continuous monitoring"

# Setup and installation
setup: install
	@echo "🚀 Setting up MLOps environment..."
	python setup_database.py
	@echo "✅ Setup complete!"

install:
	@echo "📦 Installing dependencies..."
	pip install -r requirements.txt
	@echo "✅ Dependencies installed!"

# Data and training
collect-data:
	@echo "📊 Collecting stock market data..."
	python collect_data.py

train-models:
	@echo "🤖 Training ML models..."
	python train_with_mlflow.py

run-pipeline:
	@echo "🔄 Running complete MLOps pipeline..."
	python pipeline.py

# API and services
run-api:
	@echo "🌐 Starting API server..."
	uvicorn api:app --host 0.0.0.0 --port 8000 --reload

run-mlflow:
	@echo "📈 Starting MLflow server..."
	mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri file:///mlruns

# Docker operations
docker-build:
	@echo "🐳 Building Docker containers..."
	docker-compose build

docker-run:
	@echo "🐳 Running with Docker Compose..."
	docker-compose up -d
	@echo "✅ Services started!"
	@echo "API: http://localhost:8000"
	@echo "MLflow: http://localhost:5000"

docker-stop:
	@echo "🛑 Stopping Docker containers..."
	docker-compose down

docker-logs:
	@echo "📝 Showing Docker logs..."
	docker-compose logs -f

# Testing
test:
	@echo "🧪 Running tests..."
	python -m pytest tests/ -v

test-api:
	@echo "🧪 Testing API endpoints..."
	python -c "import requests; r=requests.get('http://localhost:8000/health'); print('✅ API healthy' if r.status_code==200 else '❌ API failed')"

# Monitoring
monitor:
	@echo "📊 Running monitoring checks..."
	python monitoring.py report

monitor-live:
	@echo "📊 Starting continuous monitoring..."
	python monitoring.py continuous

# Deployment
deploy-local: docker-run
	@echo "🚀 Local deployment complete!"
	@echo "Waiting for services to start..."
	@sleep 10
	@make test-api
	@make monitor

deploy-aws:
	@echo "🚀 Deploying to AWS..."
	@echo "⚠️  AWS deployment not implemented yet"
	@echo "Would run: terraform apply, docker push, kubectl apply"

# Cleanup
clean:
	@echo "🧹 Cleaning up..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name ".pytest_cache" -delete
	rm -rf mlruns/
	rm -rf monitoring/
	rm -f stocks.db
	@echo "✅ Cleanup complete!"

# Development helpers
dev-setup: setup
	@echo "👩‍💻 Setting up development environment..."
	pip install pytest pytest-asyncio black flake8
	python collect_data.py
	python train_with_mlflow.py
	@echo "✅ Development environment ready!"

lint:
	@echo "🔍 Running code linting..."
	flake8 --max-line-length=100 --ignore=E501,W503 *.py src/

format:
	@echo "💅 Formatting code..."
	black --line-length=100 *.py src/

# Demo and presentation
demo: docker-run
	@echo "🎬 Starting demo environment..."
	@sleep 10
	@echo ""
	@echo "Demo URLs:"
	@echo "=========="
	@echo "API Documentation: http://localhost:8000/docs"
	@echo "Health Check: http://localhost:8000/health"
	@echo "MLflow UI: http://localhost:5000"
	@echo ""
	@echo "Example API call:"
	@echo 'curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d '"'"'{"symbol": "AAPL", "news_headlines": ["Apple reports strong earnings"]}'"'"''

status:
	@echo "📊 System Status"
	@echo "==============="
	@python -c "import sqlite3; conn=sqlite3.connect('stocks.db'); print(f'Database: {conn.execute(\"SELECT COUNT(*) FROM stock_info\").fetchone()[0]} stocks'); conn.close()" 2>/dev/null || echo "Database: Not initialized"
	@docker-compose ps 2>/dev/null || echo "Docker: Not running"
	@curl -s http://localhost:8000/health >/dev/null && echo "API: Running" || echo "API: Not running"
	@curl -s http://localhost:5000 >/dev/null && echo "MLflow: Running" || echo "MLflow: Not running"