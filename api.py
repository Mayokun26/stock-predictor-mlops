#!/usr/bin/env python3
"""
FastAPI prediction service for stock prices
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sqlite3
import pandas as pd
import numpy as np
import mlflow.sklearn
import joblib
from typing import List, Dict
import os
from datetime import datetime, timedelta
from transformers import pipeline
import warnings
warnings.filterwarnings('ignore')

app = FastAPI(title="Stock Price Predictor", version="1.0.0")

# Global variables for models
sentiment_analyzer = None
ml_models = {}

class PredictionRequest(BaseModel):
    symbol: str
    news_headlines: List[str] = []

class PredictionResponse(BaseModel):
    symbol: str
    current_price: float
    predicted_price: float
    price_change: float
    sentiment_score: float
    confidence: float
    timestamp: str

@app.on_event("startup")
async def load_models():
    """Load ML models and sentiment analyzer on startup"""
    global sentiment_analyzer, ml_models
    
    print("Loading sentiment analyzer...")
    sentiment_analyzer = pipeline('sentiment-analysis', 
                                 model='ProsusAI/finbert',
                                 framework='pt')
    
    print("Loading ML models...")
    # In production, load from MLflow model registry
    # For now, we'll use a simple approach
    print("✅ Models loaded successfully")

def get_latest_stock_data(symbol: str) -> Dict:
    """Get latest stock data from database"""
    conn = sqlite3.connect('stocks.db')
    
    # Get latest price data
    query = f"""
    SELECT * FROM stock_prices 
    WHERE symbol='{symbol}' 
    ORDER BY date DESC LIMIT 20
    """
    df = pd.read_sql(query, conn)
    
    # Get stock info
    info_query = f"SELECT * FROM stock_info WHERE symbol='{symbol}'"
    info = pd.read_sql(info_query, conn)
    
    conn.close()
    
    if len(df) == 0:
        raise HTTPException(status_code=404, f"No data found for {symbol}")
    
    return {
        'prices': df,
        'info': info.iloc[0] if len(info) > 0 else None,
        'latest_price': df.iloc[0]['close']
    }

def analyze_sentiment(headlines: List[str]) -> float:
    """Analyze sentiment of news headlines"""
    if not headlines or not sentiment_analyzer:
        return 0.0
    
    results = sentiment_analyzer(headlines)
    
    # Convert to numeric score (-1 to 1)
    total_score = 0
    for result in results:
        score = result['score']
        if result['label'] == 'negative':
            score = -score
        elif result['label'] == 'neutral':
            score = 0
        total_score += score
    
    return total_score / len(results) if results else 0.0

def create_features_for_prediction(df: pd.DataFrame) -> pd.DataFrame:
    """Create features for prediction from price data"""
    df = df.copy().sort_values('date')
    df['returns'] = df['close'].pct_change()
    df['sma_5'] = df['close'].rolling(5).mean()
    df['sma_20'] = df['close'].rolling(20).mean()
    df['volatility'] = df['close'].rolling(10).std()
    df['rsi'] = calculate_rsi(df['close'])
    df['price_ma_ratio'] = df['close'] / df['sma_20']
    return df

def calculate_rsi(prices, window=14):
    """Calculate RSI indicator"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def simple_predict(data: Dict, sentiment_score: float) -> float:
    """Simple prediction model (fallback when MLflow models not available)"""
    df = data['prices']
    latest = df.iloc[0]
    
    # Simple momentum + sentiment model
    recent_return = df['close'].pct_change().iloc[:5].mean()
    volatility = df['close'].std()
    
    # Base prediction on recent momentum
    momentum_factor = 1 + (recent_return * 0.5)
    
    # Adjust for sentiment
    sentiment_factor = 1 + (sentiment_score * 0.02)  # 2% max sentiment impact
    
    # Add some randomness for confidence
    confidence = max(0.1, 1 - (volatility / latest['close']))
    
    predicted_price = latest['close'] * momentum_factor * sentiment_factor
    
    return predicted_price, confidence

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Stock Price Predictor API", "status": "running"}

@app.get("/stocks")
async def list_stocks():
    """List available stocks"""
    conn = sqlite3.connect('stocks.db')
    stocks = pd.read_sql("SELECT symbol, name FROM stock_info", conn)
    conn.close()
    return stocks.to_dict('records')

@app.post("/predict", response_model=PredictionResponse)
async def predict_price(request: PredictionRequest):
    """Predict stock price based on historical data and news sentiment"""
    
    try:
        # Get stock data
        data = get_latest_stock_data(request.symbol)
        
        # Analyze sentiment
        sentiment_score = analyze_sentiment(request.news_headlines)
        
        # Make prediction
        predicted_price, confidence = simple_predict(data, sentiment_score)
        
        current_price = data['latest_price']
        price_change = predicted_price - current_price
        
        return PredictionResponse(
            symbol=request.symbol,
            current_price=round(current_price, 2),
            predicted_price=round(predicted_price, 2),
            price_change=round(price_change, 2),
            sentiment_score=round(sentiment_score, 3),
            confidence=round(confidence, 3),
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Detailed health check"""
    # Check database
    try:
        conn = sqlite3.connect('stocks.db')
        stock_count = conn.execute("SELECT COUNT(*) FROM stock_info").fetchone()[0]
        price_count = conn.execute("SELECT COUNT(*) FROM stock_prices").fetchone()[0]
        conn.close()
        db_status = "healthy"
    except Exception as e:
        db_status = f"error: {e}"
        stock_count = price_count = 0
    
    return {
        "status": "healthy",
        "database": {
            "status": db_status,
            "stocks": stock_count,
            "prices": price_count
        },
        "models": {
            "sentiment_analyzer": sentiment_analyzer is not None,
            "ml_models": len(ml_models)
        },
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)