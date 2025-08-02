# Stock Market MLOps Predictor

A production-ready MLOps system for predicting stock prices using machine learning, news sentiment analysis, and automated deployment pipelines.

## 🚀 Quick Start

```bash
# 1. Setup the system
make setup

# 2. Run the complete pipeline
make run-pipeline

# 3. Start the API
make run-api

# 4. Test a prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"symbol": "AAPL", "news_headlines": ["Apple reports strong earnings"]}'
```

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Sources  │────│  Data Pipeline  │────│  Feature Store  │
│                 │    │                 │    │                 │
│ • Yahoo Finance │    │ • Data cleaning │    │ • SQLite DB     │
│ • News APIs     │    │ • Feature eng   │    │ • Price history │
│ • Market Data   │    │ • Validation    │    │ • Technical ind │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Model Training  │────│ Model Registry  │────│ Model Serving   │
│                 │    │                 │    │                 │
│ • MLflow        │    │ • Version       │    │ • FastAPI       │
│ • Experiments   │    │   control       │    │ • Docker        │
│ • Validation    │    │ • Artifacts     │    │ • Health checks │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Monitoring    │    │   Deployment    │    │   Frontend      │
│                 │    │                 │    │                 │
│ • Prometheus    │    │ • Docker        │    │ • API docs      │
│ • Health checks │    │ • Kubernetes    │    │ • Swagger UI    │
│ • Alerting      │    │ • CI/CD ready   │    │ • Monitoring    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 📦 Installation

### Prerequisites
- Python 3.9+
- Docker (optional)
- Make (optional, for convenience commands)

### Local Setup
```bash
# Clone and setup
git clone <repo-url>
cd betting
make setup

# Or manual setup
pip install -r requirements.txt
python setup_database.py
```

### Docker Setup
```bash
# Build and run with Docker
make docker-build
make docker-run

# Services will be available at:
# API: http://localhost:8000
# MLflow: http://localhost:5000
```

## 🔄 Usage

### 1. Data Collection
```bash
# Collect stock market data
make collect-data

# Or run directly
python collect_data.py
```

### 2. Model Training
```bash
# Train ML models with MLflow tracking
make train-models

# Or run directly
python train_with_mlflow.py
```

### 3. Complete Pipeline
```bash
# Run end-to-end pipeline
make run-pipeline

# This will:
# - Setup database
# - Collect fresh data
# - Train models
# - Validate results
```

### 4. API Server
```bash
# Start the prediction API
make run-api

# Test the API
curl http://localhost:8000/health
```

## 🔍 API Endpoints

### Health Check
```bash
GET /health
```
Returns system health status including database and model status.

### Stock Prediction
```bash
POST /predict
Content-Type: application/json

{
  "symbol": "AAPL",
  "news_headlines": [
    "Apple reports strong quarterly earnings",
    "New iPhone sales exceed expectations"
  ]
}
```

Returns:
```json
{
  "symbol": "AAPL",
  "current_price": 150.25,
  "predicted_price": 152.80,
  "price_change": 2.55,
  "sentiment_score": 0.245,
  "confidence": 0.847,
  "timestamp": "2024-01-15T10:30:00"
}
```

### Available Stocks
```bash
GET /stocks
```
Returns list of available stocks for prediction.

## 📊 Monitoring

### Health Monitoring
```bash
# Run monitoring checks
make monitor

# Continuous monitoring
make monitor-live
```

### Metrics
- **API Performance**: Response times, error rates
- **Model Performance**: RMSE, R² scores, confidence
- **Data Quality**: Freshness, completeness
- **System Health**: Database status, model availability

### Prometheus Metrics
Metrics are exposed at `http://localhost:8001/metrics` when monitoring is running.

## 🧪 Testing

```bash
# Run all tests
make test

# Test API endpoints
make test-api

# Check system status
make status
```

## 🚀 Deployment

### Local Deployment
```bash
# Deploy locally with Docker
make deploy-local
```

### Production Deployment
```bash
# AWS deployment (requires setup)
make deploy-aws

# Or use Kubernetes manifests
kubectl apply -f k8s/
```

## 📁 Project Structure

```
betting/
├── api.py                 # FastAPI application
├── pipeline.py           # MLOps pipeline orchestrator
├── monitoring.py         # System monitoring
├── collect_data.py       # Data collection
├── train_with_mlflow.py  # Model training
├── requirements.txt      # Python dependencies
├── Dockerfile           # Container definition
├── docker-compose.yml   # Multi-service setup
├── Makefile            # Automation commands
├── src/
│   ├── data/           # Data ingestion modules
│   ├── database/       # Database models
│   ├── models/         # ML models
│   └── api/           # API components
├── tests/             # Test suite
├── mlruns/           # MLflow experiments
├── monitoring/       # Monitoring reports
└── models/           # Trained model artifacts
```

## 🔧 Configuration

### Environment Variables
```bash
# Data sources
ALPHA_VANTAGE_API_KEY=your_key
ALPACA_API_KEY=your_key
ALPACA_SECRET_KEY=your_secret

# MLflow
MLFLOW_TRACKING_URI=file:///app/mlruns

# API
API_HOST=0.0.0.0
API_PORT=8000
```

### Database Configuration
The system uses SQLite by default. For production, configure PostgreSQL:

```python
# src/config.py
DATABASE_URL = "postgresql://user:pass@localhost/mlops"
```

## 📈 MLflow Integration

### Experiment Tracking
- All model training is tracked in MLflow
- Experiments are organized by date and model type
- Metrics, parameters, and artifacts are logged automatically

### Model Registry
- Models are versioned and stored in MLflow
- Production models can be tagged for deployment
- Model lineage and performance history is tracked

### Access MLflow UI
```bash
# Start MLflow server
make run-mlflow

# Access at http://localhost:5000
```

## 🎯 Features

### Core Functionality
- ✅ **Real-time Predictions**: REST API for stock price predictions
- ✅ **Sentiment Analysis**: News headline sentiment using FinBERT
- ✅ **Technical Analysis**: RSI, moving averages, volatility indicators
- ✅ **Multiple Models**: Random Forest, Linear Regression with comparison
- ✅ **Data Pipeline**: Automated data collection and processing

### MLOps Features
- ✅ **Experiment Tracking**: MLflow integration for all experiments
- ✅ **Model Versioning**: Automatic model versioning and storage
- ✅ **Containerization**: Docker containers for all components
- ✅ **Health Monitoring**: Comprehensive system health checks
- ✅ **API Documentation**: Auto-generated OpenAPI docs
- ✅ **Prometheus Metrics**: System and model performance metrics

### Production Ready
- ✅ **Scalable Architecture**: Microservices with Docker Compose
- ✅ **Error Handling**: Comprehensive error handling and logging
- ✅ **Data Validation**: Input validation and data quality checks
- ✅ **Security**: Non-root containers, input sanitization
- ✅ **Documentation**: Complete API and system documentation

## 🎓 Learning Outcomes

This project demonstrates key MLOps competencies:

1. **Data Engineering**: Automated data pipelines, quality checks
2. **Model Development**: Experiment tracking, model comparison
3. **Model Serving**: Production API deployment, monitoring
4. **DevOps Integration**: Containerization, CI/CD readiness
5. **Monitoring & Observability**: Health checks, metrics, alerting

## 🚨 Troubleshooting

### Common Issues

**API not starting**
```bash
# Check if port is in use
lsof -i :8000

# Check logs
make docker-logs
```

**Database issues**
```bash
# Reinitialize database
rm stocks.db
python setup_database.py
```

**Model training fails**
```bash
# Check data availability
python -c "import sqlite3; conn=sqlite3.connect('stocks.db'); print(conn.execute('SELECT COUNT(*) FROM stock_prices').fetchone())"

# Collect fresh data
make collect-data
```

**Docker issues**
```bash
# Rebuild containers
make docker-stop
make docker-build
make docker-run
```

## 📝 License

MIT License - see LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `make test`
5. Submit a pull request

## 📞 Support

For issues and questions:
- Check the troubleshooting section
- Review logs: `make docker-logs`
- Run health checks: `make monitor`
- Check system status: `make status`