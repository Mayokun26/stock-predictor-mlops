# 🚀 MLOps Stock Prediction System - Quick Start Guide

## Prerequisites
- Python 3.9+
- Docker (for PostgreSQL/Redis)
- OpenAI API key (required)

## 1. Initial Setup (5 minutes)

### Clone and Setup Environment
```bash
cd betting/
python3 setup.py  # Interactive setup wizard
```

**The setup wizard will:**
- Create `.env` file from template
- Prompt for your OpenAI API key
- Check dependencies and install missing ones
- Verify local services (PostgreSQL, Redis)
- Set up Terraform configuration

### Manual Setup (Alternative)
```bash
# Copy environment template
cp .env.example .env

# Edit .env file with your credentials (minimum required):
OPENAI_API_KEY="sk-proj-your-openai-key-here"
NEWS_API_KEY="your-news-api-key-here"  # Get free at newsapi.org
```

## 2. Start Local Services

### Using Docker (Recommended)
```bash
# Start PostgreSQL
docker run -d --name postgres -p 5432:5432 \
  -e POSTGRES_PASSWORD=postgres_password postgres:13

# Start Redis
docker run -d --name redis -p 6379:6379 redis:7
```

### Using Docker Compose (Even Easier)
```bash
docker-compose up -d postgres redis
```

## 3. Run the Application

```bash
# Start the API server
python3 src/api/production_api.py

# Or with uvicorn directly
uvicorn src.api.production_api:app --host 0.0.0.0 --port 8000 --reload
```

## 4. Test the System

### Basic Health Check
```bash
curl http://localhost:8000/health
```

### Test LLM Integration (Requires OpenAI Key)
```bash
curl -X POST "http://localhost:8000/predict/explain" \
  -H "Content-Type: application/json" \
  -d '{"symbol":"AAPL","include_risk_assessment":true}'
```

### Test Model Comparison
```bash
curl -X POST "http://localhost:8000/llm/compare-models" \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Analyze AAPL stock performance","task_type":"financial_analysis"}'
```

### Browse API Documentation
Open http://localhost:8000/docs in your browser for interactive API documentation.

## 5. Key Endpoints to Demo

| Endpoint | Purpose | Demo Value |
|----------|---------|------------|
| `GET /health` | System health | Shows all services connected |
| `POST /predict` | Stock prediction | Core ML functionality |
| `POST /predict/explain` | AI explanations | LLM integration |
| `POST /llm/compare-models` | Model comparison | Advanced ML engineering |
| `GET /llm/hybrid-stats` | Model statistics | System monitoring |
| `GET /metrics` | Prometheus metrics | Production observability |
| `POST /rag/search` | Semantic search | Vector database integration |
| `GET /backtest/AAPL` | Financial backtesting | Domain expertise |

## 6. AWS Deployment (Optional)

### Setup Infrastructure
```bash
cd infrastructure/
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars with your AWS credentials and settings

terraform init
terraform plan
terraform apply
```

### Deploy Application
```bash
# Build and push Docker image
docker build -t betting-mlops .
# Push to ECR (commands provided by terraform output)

# Deploy to ECS
aws ecs update-service --cluster betting-prod --service betting-api --force-new-deployment
```

## 7. Troubleshooting

### Common Issues

**"LLM service not available"**
- Check OPENAI_API_KEY in .env file
- Verify API key has credits
- Test: `curl -H "Authorization: Bearer YOUR_KEY" https://api.openai.com/v1/models`

**"Database connection failed"**
- Start PostgreSQL: `docker start postgres`
- Check connection: `psql -h localhost -U postgres -d postgres`

**"Redis connection failed"**
- Start Redis: `docker start redis`
- Check connection: `redis-cli ping`

**"Import errors"**
- Install dependencies: `pip install -r requirements.txt`
- Check Python version: `python3 --version` (needs 3.9+)

### Performance Issues
- Increase Docker memory allocation (4GB+ recommended)
- For production: Use AWS RDS/ElastiCache instead of local services

## 8. Configuration Reference

### Environment Variables Priority
```
1. OPENAI_API_KEY - Required for LLM features
2. NEWS_API_KEY - Required for real news data
3. DATABASE_URL - Auto-configured for local/AWS
4. REDIS_URL - Auto-configured for local/AWS
```

### Feature Flags
```bash
ENABLE_RAG=true              # Vector search functionality  
ENABLE_HYBRID_LLM=true       # Multi-model system
ENABLE_MONITORING=true       # Prometheus metrics
ENABLE_CACHING=true          # Redis caching
```

### Cost Controls
```bash
MAX_TOKENS_PER_REQUEST=2000  # Limit LLM token usage
MONTHLY_BUDGET_USD=100       # Alert threshold
```

## 9. Development Workflow

### Local Development
```bash
# Hot reload development server
uvicorn src.api.production_api:app --reload

# Run tests
python3 -m pytest tests/

# Check code quality
black src/
flake8 src/
```

### Adding New Features
1. Modify code in `src/`
2. Update environment variables in `.env.example` if needed
3. Add tests in `tests/`
4. Update API documentation
5. Test locally before deploying

## 10. Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   FastAPI       │────│  Hybrid LLM     │────│   PostgreSQL    │
│   21 Endpoints  │    │   Engine        │    │   Database      │
│                 │    │ 5 Model Types   │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │              ┌─────────────────┐              │
         │──────────────│      Redis      │──────────────│
         │              │  Feature Store  │              │
         │              │   + Caching     │              │
         │              └─────────────────┘              │
         │                                               │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Monitoring    │    │      RAG        │    │   External      │
│   Prometheus    │    │ Vector Search   │    │     APIs        │
│   + Grafana     │    │  Embeddings     │    │ yfinance/News   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Ready to Deploy!

Your MLOps system includes:
- ✅ 21 production API endpoints
- ✅ Multi-model LLM architecture  
- ✅ RAG vector search system
- ✅ Financial domain expertise
- ✅ Production monitoring
- ✅ AWS deployment ready

**Interview-ready features:** Model comparison, evaluation frameworks, cost optimization, production monitoring, domain expertise.