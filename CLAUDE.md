# CLAUDE.md

This file provides guidance to Claude Code when working with the Stock Predictor MLOps application.

## Git Workflow - MANDATORY FOR ALL CHANGES

**🚨 CRITICAL: COMMIT AND PUSH FREQUENTLY 🚨**

Claude Code MUST follow this workflow for every change:

```bash
# After ANY meaningful change (no matter how small):
git add .
git commit -m "descriptive message"
git push origin main

# Commit triggers (ALWAYS commit after):
# - Adding/modifying any model files
# - Updating feature engineering
# - Fixing any bug
# - Adding new monitoring
# - Configuration changes
# - Documentation updates
# - Data pipeline changes
# - Infrastructure updates
```

**Commit Message Examples:**
- `feat: add drift detection for stock prices`
- `fix: resolve streaming pipeline memory leak`
- `model: update ensemble weights for better accuracy`
- `infra: add Docker container for model serving`
- `data: implement new technical indicators`
- `monitor: add MLflow experiment tracking`

**Why This Matters:**
- ML experiments need proper versioning
- Model performance changes must be tracked
- Data pipeline failures require quick rollback
- Infrastructure changes need documented history
- Prevents lost experimental results

**NO EXCEPTIONS** - Even parameter tweaks should be committed and pushed immediately.

## Project Overview

This is a production-grade MLOps system for stock market prediction, designed to demonstrate enterprise-level machine learning engineering capabilities.

## Architecture

### Core MLOps Pipeline
- **Feature Store**: Real-time technical indicator computation
- **Model Training**: Ensemble methods with automated retraining
- **Drift Detection**: Statistical monitoring for data/concept drift
- **Model Serving**: REST API with A/B testing capabilities
- **Monitoring**: MLflow + custom metrics dashboards
- **Infrastructure**: Docker containers + cloud deployment

### Key Components
- `src/features/feature_store.py` - Feature engineering pipeline
- `src/models/ensemble_models.py` - ML model implementations
- `src/streaming/` - Real-time data processing
- `src/monitoring/` - Model performance monitoring
- `src/api/production_api.py` - Model serving API
- `infrastructure/` - Deployment configurations

## Development Commands

### Local Development
```bash
# Environment setup
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
pip install -r requirements.txt

# Feature engineering
python src/features/feature_store.py

# Model training
python src/models/train_ensemble.py

# API testing
python src/api/production_api.py
```

### Production Deployment
```bash
# Docker build
docker build -t stock-predictor .
docker run -p 8000:8000 stock-predictor

# Streaming pipeline
python src/streaming/kafka_consumer.py

# Monitoring dashboard
python src/monitoring/drift_monitor.py
```

## Claude Code Instructions

1. **Always commit and push immediately** after any change
2. **Follow MLOps best practices** - version everything, track experiments
3. **Test thoroughly** - ML systems fail silently, tests catch issues
4. **Monitor continuously** - add metrics for any new functionality
5. **Document experiments** - include reasoning in commit messages
6. **Use type hints** - ML code gets complex, typing helps maintenance
7. **Handle failures gracefully** - external APIs and models can fail
8. **Log extensively** - ML debugging requires detailed logs

This project demonstrates production-grade MLOps capabilities for professional development.