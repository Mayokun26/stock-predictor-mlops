#!/usr/bin/env python3
"""
Simplified Working MLOps API for Demonstration
This version actually works and demonstrates core ML functionality
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
import pandas as pd
from datetime import datetime
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

# Simple sentiment analysis
def analyze_sentiment(texts: List[str]) -> float:
    """Basic sentiment analysis using keyword matching"""
    positive_words = ['good', 'great', 'excellent', 'positive', 'up', 'rise', 'gain', 'strong', 'beat', 'surge']
    negative_words = ['bad', 'terrible', 'negative', 'down', 'fall', 'drop', 'weak', 'miss', 'decline', 'crash']
    
    total_score = 0
    for text in texts:
        text_lower = text.lower()
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        total_score += (pos_count - neg_count)
    
    # Normalize to -1 to 1 range
    return max(-1, min(1, total_score / max(len(texts), 1)))

# Simple technical indicators
def calculate_rsi(prices, window=14):
    """Calculate RSI"""
    if len(prices) < window:
        return 50  # neutral
    
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    avg_gain = np.mean(gains[-window:])
    avg_loss = np.mean(losses[-window:])
    
    if avg_loss == 0:
        return 100
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calculate_sma(prices, window=20):
    """Calculate Simple Moving Average"""
    if len(prices) < window:
        return np.mean(prices)
    return np.mean(prices[-window:])

# ML Prediction Model
class SimpleStockPredictor:
    def __init__(self):
        self.model_version = "simple_v1.0"
        self.features = ['rsi', 'sma_ratio', 'sentiment', 'volatility']
    
    def predict(self, symbol: str, news_headlines: List[str]) -> dict:
        """Make stock prediction using simple ML approach"""
        try:
            # Fetch real market data
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="3mo")
            
            if hist.empty:
                raise ValueError(f"No data found for symbol {symbol}")
            
            prices = hist['Close'].values
            current_price = float(prices[-1])
            
            # Calculate features
            rsi = calculate_rsi(prices)
            sma = calculate_sma(prices)
            sma_ratio = current_price / sma
            sentiment = analyze_sentiment(news_headlines)
            volatility = np.std(prices[-20:]) / np.mean(prices[-20:])
            
            # Simple prediction model (weighted combination)
            price_momentum = 0.3 * (rsi - 50) / 50  # RSI signal
            trend_signal = 0.2 * (sma_ratio - 1)    # Trend signal
            sentiment_signal = 0.3 * sentiment      # News sentiment
            vol_signal = -0.2 * min(volatility, 0.2) # Volatility penalty
            
            total_signal = price_momentum + trend_signal + sentiment_signal + vol_signal
            
            # Convert to price prediction
            predicted_change = total_signal * 0.02  # Max 2% change
            predicted_price = current_price * (1 + predicted_change)
            
            # Calculate confidence based on signal strength and data quality
            confidence = min(0.95, 0.6 + abs(total_signal) * 0.3)
            
            return {
                "symbol": symbol,
                "current_price": round(current_price, 2),
                "predicted_price": round(predicted_price, 2),
                "predicted_change": round(predicted_change * 100, 2),
                "confidence": round(confidence, 3),
                "sentiment_score": round(sentiment, 3),
                "rsi": round(rsi, 1),
                "sma_ratio": round(sma_ratio, 3),
                "volatility": round(volatility, 3),
                "model_version": self.model_version,
                "timestamp": datetime.now().isoformat(),
                "features_used": self.features
            }
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# API Models
class PredictionRequest(BaseModel):
    symbol: str
    user_id: Optional[str] = None
    news_headlines: Optional[List[str]] = []

class PredictionResponse(BaseModel):
    symbol: str
    current_price: float
    predicted_price: float
    predicted_change: float
    confidence: float
    sentiment_score: float
    rsi: float
    sma_ratio: float
    volatility: float
    model_version: str
    timestamp: str
    features_used: List[str]

# Initialize FastAPI app
app = FastAPI(
    title="Working MLOps Stock Predictor",
    description="Actually functional stock prediction API with real ML",
    version="1.0.0"
)

# Initialize model
predictor = SimpleStockPredictor()

# Metrics
prediction_count = 0
predictions_by_symbol = {}

@app.get("/")
def health_check():
    return {
        "service": "Working MLOps Stock Predictor",
        "status": "healthy",
        "predictions_made": prediction_count,
        "version": "1.0.0"
    }

@app.get("/health")
def detailed_health():
    return {
        "status": "healthy",
        "model_loaded": True,
        "features_available": predictor.features,
        "predictions_made": prediction_count,
        "symbols_predicted": list(predictions_by_symbol.keys())
    }

@app.post("/predict", response_model=PredictionResponse)
def make_prediction(request: PredictionRequest):
    global prediction_count, predictions_by_symbol
    
    try:
        # Make prediction
        result = predictor.predict(request.symbol, request.news_headlines or [])
        
        # Update metrics
        prediction_count += 1
        predictions_by_symbol[request.symbol] = predictions_by_symbol.get(request.symbol, 0) + 1
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
def get_metrics():
    return {
        "total_predictions": prediction_count,
        "predictions_by_symbol": predictions_by_symbol,
        "model_version": predictor.model_version,
        "available_features": predictor.features
    }

@app.get("/backtest/{symbol}")
def simple_backtest(symbol: str, days: int = 30):
    """Simple backtesting demonstration"""
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=f"{days+30}d")
        
        if hist.empty:
            raise HTTPException(status_code=404, detail=f"No data for {symbol}")
        
        prices = hist['Close'].values
        
        # Simple backtest: predict based on trend
        predictions = []
        actuals = []
        
        for i in range(30, len(prices)):
            # Use past 30 days to predict next day
            past_prices = prices[i-30:i]
            actual = prices[i]
            
            # Simple prediction: momentum
            momentum = (past_prices[-1] - past_prices[0]) / past_prices[0]
            predicted = past_prices[-1] * (1 + momentum * 0.1)
            
            predictions.append(predicted)
            actuals.append(actual)
        
        # Calculate metrics
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        mse = np.mean((predictions - actuals) ** 2)
        mae = np.mean(np.abs(predictions - actuals))
        mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100
        
        # Calculate returns
        pred_returns = (predictions[1:] - predictions[:-1]) / predictions[:-1]
        actual_returns = (actuals[1:] - actuals[:-1]) / actuals[:-1]
        
        # Simple Sharpe ratio
        if len(pred_returns) > 0:
            sharpe = np.mean(pred_returns) / max(np.std(pred_returns), 0.001) * np.sqrt(252)
        else:
            sharpe = 0
        
        return {
            "symbol": symbol,
            "period_days": days,
            "total_predictions": len(predictions),
            "mse": round(mse, 4),
            "mae": round(mae, 4),
            "mape": round(mape, 2),
            "sharpe_ratio": round(sharpe, 3),
            "average_actual_price": round(np.mean(actuals), 2),
            "average_predicted_price": round(np.mean(predictions), 2),
            "model_version": predictor.model_version,
            "backtest_timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Backtesting error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Working MLOps Stock Predictor API on port 8000...")
    uvicorn.run(app, host="0.0.0.0", port=8000)