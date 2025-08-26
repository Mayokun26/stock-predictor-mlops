"""
Simple FastAPI application - minimal working version
"""
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Any
import uvicorn

app = FastAPI(
    title="MLOps Stock Prediction API",
    description="Simple API for stock predictions",
    version="1.0.0"
)

class PredictionRequest(BaseModel):
    symbol: str
    features: Dict[str, float] = {}

class PredictionResponse(BaseModel):
    symbol: str
    prediction: float
    confidence: float
    timestamp: str

@app.get("/")
async def root():
    return {"message": "MLOps Stock Prediction API", "status": "running"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "mlops-api",
        "timestamp": "2025-08-14T20:00:00Z"
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    # Simple mock prediction for now
    import random
    from datetime import datetime
    
    prediction = round(random.uniform(0.1, 0.9), 3)
    confidence = round(random.uniform(0.6, 0.95), 3)
    
    return PredictionResponse(
        symbol=request.symbol,
        prediction=prediction,
        confidence=confidence,
        timestamp=datetime.now().isoformat()
    )

@app.get("/models")
async def list_models():
    return {
        "models": ["xgboost-v1", "ensemble-v1"],
        "active_model": "ensemble-v1"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)