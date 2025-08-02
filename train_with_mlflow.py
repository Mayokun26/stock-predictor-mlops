#!/usr/bin/env python3
"""
Stock prediction training with MLflow experiment tracking
"""
import sqlite3
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import mlflow.sklearn
from datetime import datetime
import joblib
import os

def load_stock_data(symbol='AAPL'):
    """Load stock data from SQLite"""
    conn = sqlite3.connect('stocks.db')
    df = pd.read_sql(f'SELECT * FROM stock_prices WHERE symbol="{symbol}" ORDER BY date', conn)
    conn.close()
    return df

def create_features(df):
    """Create ML features from price data"""
    df = df.copy()
    df['target'] = df['close'].shift(-1)  # Next day's close
    df['returns'] = df['close'].pct_change()
    df['sma_5'] = df['close'].rolling(5).mean()
    df['sma_20'] = df['close'].rolling(20).mean()
    df['volatility'] = df['close'].rolling(10).std()
    df['rsi'] = calculate_rsi(df['close'])
    df['price_ma_ratio'] = df['close'] / df['sma_20']
    return df.dropna()

def calculate_rsi(prices, window=14):
    """Calculate RSI indicator"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def train_model(symbol='AAPL', model_type='rf'):
    """Train stock prediction model with MLflow tracking"""
    
    # Set MLflow experiment
    mlflow.set_experiment("stock_prediction")
    
    with mlflow.start_run(run_name=f"{symbol}_{model_type}_{datetime.now().strftime('%Y%m%d_%H%M')}"):
        
        # Load and prepare data
        print(f"Training {model_type} model for {symbol}...")
        df = load_stock_data(symbol)
        df = create_features(df)
        
        # Log data info
        mlflow.log_param("symbol", symbol)
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("data_points", len(df))
        mlflow.log_param("date_range", f"{df['date'].min()} to {df['date'].max()}")
        
        # Features and target
        feature_cols = ['open', 'high', 'low', 'volume', 'returns', 
                       'sma_5', 'sma_20', 'volatility', 'rsi', 'price_ma_ratio']
        X = df[feature_cols]
        y = df['target']
        
        # Train/test split (80/20)
        split_idx = int(len(df) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))
        
        # Initialize model
        if model_type == 'rf':
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42
            )
            mlflow.log_param("n_estimators", 100)
            mlflow.log_param("max_depth", 10)
        else:  # linear
            model = LinearRegression()
        
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        # Calculate metrics
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        train_mae = mean_absolute_error(y_train, train_pred)
        test_mae = mean_absolute_error(y_test, test_pred)
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
        
        # Log metrics
        mlflow.log_metric("train_rmse", train_rmse)
        mlflow.log_metric("test_rmse", test_rmse)
        mlflow.log_metric("train_mae", train_mae)
        mlflow.log_metric("test_mae", test_mae)
        mlflow.log_metric("train_r2", train_r2)
        mlflow.log_metric("test_r2", test_r2)
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        # Save feature names for serving
        mlflow.log_param("features", feature_cols)
        
        print(f"Model trained successfully!")
        print(f"  Test RMSE: ${test_rmse:.2f}")
        print(f"  Test MAE: ${test_mae:.2f}")
        print(f"  Test R²: {test_r2:.3f}")
        
        return model, test_rmse

def main():
    """Train models for multiple stocks and algorithms"""
    
    # Create models directory
    os.makedirs("models", exist_ok=True)
    
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA']
    models = ['rf', 'linear']
    
    results = []
    
    for symbol in symbols:
        for model_type in models:
            try:
                model, rmse = train_model(symbol, model_type)
                results.append({
                    'symbol': symbol,
                    'model': model_type,
                    'rmse': rmse
                })
                print(f"✅ {symbol} {model_type}: ${rmse:.2f} RMSE")
            except Exception as e:
                print(f"❌ {symbol} {model_type}: {e}")
    
    # Show best models
    print("\n=== Best Models by Stock ===")
    df_results = pd.DataFrame(results)
    for symbol in df_results['symbol'].unique():
        best = df_results[df_results['symbol']==symbol].nsmallest(1, 'rmse')
        print(f"{symbol}: {best.iloc[0]['model']} (${best.iloc[0]['rmse']:.2f} RMSE)")

if __name__ == "__main__":
    main()