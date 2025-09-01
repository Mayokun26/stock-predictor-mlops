#!/usr/bin/env python3
"""
Complete Model Training Script with MLflow Integration
Trains a real model and saves it to MLflow Registry
"""

import os
import sys
import asyncio
import logging
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import mlflow.sklearn
import joblib

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MLflow configuration
MLFLOW_TRACKING_URI = "http://localhost:5001"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

def create_advanced_features(data):
    """Create comprehensive features for stock prediction"""
    df = data.copy()
    features = {}
    
    # Basic price features
    features['current_price'] = df['Close'].iloc[-1]
    features['volume'] = df['Volume'].iloc[-1]
    features['high_52w'] = df['High'].max()
    features['low_52w'] = df['Low'].min()
    
    # Moving averages
    for window in [5, 10, 20, 50]:
        if len(df) >= window:
            sma = df['Close'].rolling(window=window).mean()
            features[f'sma_{window}'] = sma.iloc[-1]
            features[f'sma_{window}_ratio'] = features['current_price'] / sma.iloc[-1]
    
    # Technical indicators
    # RSI
    if len(df) >= 14:
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        features['rsi'] = rsi.iloc[-1] if not np.isnan(rsi.iloc[-1]) else 50.0
    else:
        features['rsi'] = 50.0
        
    # MACD
    if len(df) >= 26:
        ema12 = df['Close'].ewm(span=12).mean()
        ema26 = df['Close'].ewm(span=26).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9).mean()
        features['macd'] = macd.iloc[-1]
        features['macd_signal'] = signal.iloc[-1]
        features['macd_histogram'] = features['macd'] - features['macd_signal']
    else:
        features['macd'] = 0.0
        features['macd_signal'] = 0.0
        features['macd_histogram'] = 0.0
    
    # Bollinger Bands
    if len(df) >= 20:
        sma20 = df['Close'].rolling(window=20).mean()
        std20 = df['Close'].rolling(window=20).std()
        upper_band = sma20 + (std20 * 2)
        lower_band = sma20 - (std20 * 2)
        features['bollinger_upper'] = upper_band.iloc[-1]
        features['bollinger_lower'] = lower_band.iloc[-1]
        features['bollinger_width'] = features['bollinger_upper'] - features['bollinger_lower']
        features['bollinger_position'] = (features['current_price'] - features['bollinger_lower']) / features['bollinger_width'] if features['bollinger_width'] > 0 else 0.5
    else:
        features['bollinger_upper'] = features['current_price'] * 1.02
        features['bollinger_lower'] = features['current_price'] * 0.98
        features['bollinger_width'] = features['bollinger_upper'] - features['bollinger_lower']
        features['bollinger_position'] = 0.5
    
    # Volume indicators
    if len(df) >= 10:
        vol_sma = df['Volume'].rolling(10).mean()
        features['volume_sma_10'] = vol_sma.iloc[-1]
        features['volume_ratio'] = features['volume'] / vol_sma.iloc[-1] if vol_sma.iloc[-1] > 0 else 1.0
    else:
        features['volume_sma_10'] = features['volume']
        features['volume_ratio'] = 1.0
    
    # Volatility measures
    for window in [10, 20, 30]:
        if len(df) >= window:
            returns = df['Close'].pct_change()
            volatility = returns.rolling(window).std() * np.sqrt(252)
            features[f'volatility_{window}d'] = volatility.iloc[-1] if not np.isnan(volatility.iloc[-1]) else 0.2
        else:
            features[f'volatility_{window}d'] = 0.2
    
    # Momentum indicators
    for window in [1, 5, 10, 20]:
        if len(df) >= window + 1:
            ret = (df['Close'].iloc[-1] / df['Close'].iloc[-window-1] - 1)
            features[f'return_{window}d'] = ret
        else:
            features[f'return_{window}d'] = 0.0
    
    # Advanced Technical Indicators
    
    # Williams %R
    if len(df) >= 14:
        high_14 = df['High'].rolling(window=14).max()
        low_14 = df['Low'].rolling(window=14).min()
        williams_r = -100 * (high_14.iloc[-1] - df['Close'].iloc[-1]) / (high_14.iloc[-1] - low_14.iloc[-1])
        features['williams_r'] = williams_r if not np.isnan(williams_r) else -50.0
    else:
        features['williams_r'] = -50.0
    
    # Commodity Channel Index (CCI)
    if len(df) >= 20:
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        sma_tp = typical_price.rolling(window=20).mean()
        mad = typical_price.rolling(window=20).apply(lambda x: np.mean(np.abs(x - x.mean())))
        cci = (typical_price.iloc[-1] - sma_tp.iloc[-1]) / (0.015 * mad.iloc[-1])
        features['cci'] = cci if not np.isnan(cci) else 0.0
    else:
        features['cci'] = 0.0
    
    # On-Balance Volume (OBV)
    if len(df) >= 2:
        obv = [0]
        for i in range(1, len(df)):
            if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
                obv.append(obv[-1] + df['Volume'].iloc[i])
            elif df['Close'].iloc[i] < df['Close'].iloc[i-1]:
                obv.append(obv[-1] - df['Volume'].iloc[i])
            else:
                obv.append(obv[-1])
        features['obv'] = obv[-1]
        features['obv_ratio'] = obv[-1] / df['Volume'].mean() if df['Volume'].mean() > 0 else 0.0
    else:
        features['obv'] = 0.0
        features['obv_ratio'] = 0.0
    
    # Volume Weighted Average Price (VWAP)
    if len(df) >= 20:
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        vwap_20 = (typical_price * df['Volume']).rolling(window=20).sum() / df['Volume'].rolling(window=20).sum()
        features['vwap'] = vwap_20.iloc[-1] if not np.isnan(vwap_20.iloc[-1]) else df['Close'].iloc[-1]
        features['vwap_ratio'] = features['current_price'] / features['vwap']
    else:
        features['vwap'] = df['Close'].iloc[-1]
        features['vwap_ratio'] = 1.0
    
    # Stochastic Oscillator
    if len(df) >= 14:
        low_14 = df['Low'].rolling(window=14).min()
        high_14 = df['High'].rolling(window=14).max()
        k_percent = 100 * ((df['Close'] - low_14) / (high_14 - low_14))
        d_percent = k_percent.rolling(window=3).mean()
        features['stoch_k'] = k_percent.iloc[-1] if not np.isnan(k_percent.iloc[-1]) else 50.0
        features['stoch_d'] = d_percent.iloc[-1] if not np.isnan(d_percent.iloc[-1]) else 50.0
    else:
        features['stoch_k'] = 50.0
        features['stoch_d'] = 50.0
    
    # Average True Range (ATR)
    if len(df) >= 14:
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        atr = true_range.rolling(window=14).mean()
        features['atr'] = atr.iloc[-1] if not np.isnan(atr.iloc[-1]) else 0.0
        features['atr_ratio'] = features['atr'] / features['current_price'] if features['current_price'] > 0 else 0.0
    else:
        features['atr'] = 0.0
        features['atr_ratio'] = 0.0
    
    return features

def prepare_training_data(symbol, period="2y"):
    """Prepare comprehensive training dataset"""
    logger.info(f"Preparing training data for {symbol}")
    
    # Get historical data
    ticker = yf.Ticker(symbol)
    hist = ticker.history(period=period)
    
    if len(hist) < 100:
        raise ValueError(f"Insufficient data for {symbol}: {len(hist)} days")
    
    feature_list = []
    target_list = []
    
    # Calculate features for each time step
    for i in range(60, len(hist) - 1):  # Start from day 60 for stable indicators
        try:
            # Get subset for feature calculation
            subset = hist.iloc[:i+1]
            
            # Calculate features
            features = create_advanced_features(subset)
            
            # Target: next day return
            current_price = hist['Close'].iloc[i]
            next_price = hist['Close'].iloc[i + 1]
            target = (next_price / current_price - 1)  # Return
            
            # Convert to array
            feature_array = np.array(list(features.values()))
            
            # Check for invalid values
            if not np.any(np.isnan(feature_array)) and not np.any(np.isinf(feature_array)) and not np.isnan(target) and not np.isinf(target):
                feature_list.append(feature_array)
                target_list.append(target)
                
        except Exception as e:
            logger.warning(f"Error calculating features for day {i}: {e}")
            continue
    
    if len(feature_list) < 50:
        raise ValueError(f"Insufficient valid samples for {symbol}: {len(feature_list)}")
    
    X = np.array(feature_list)
    y = np.array(target_list)
    
    logger.info(f"Prepared {len(X)} training samples with {X.shape[1]} features")
    
    # Feature names for tracking
    feature_names = list(create_advanced_features(hist).keys())
    
    return X, y, feature_names

def hyperparameter_optimization(X_train, y_train, symbol):
    """
    Advanced hyperparameter optimization using Grid Search, Random Search, and Bayesian Optimization
    """
    from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
    from sklearn.ensemble import RandomForestRegressor
    import numpy as np
    
    logger.info(f"🔍 Hyperparameter optimization for {symbol}")
    
    # Strategy 1: Grid Search (exhaustive but limited scope)
    logger.info("🔹 Grid Search optimization...")
    grid_param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 15, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    
    grid_search = GridSearchCV(
        RandomForestRegressor(random_state=42, n_jobs=-1),
        grid_param_grid,
        cv=3,
        scoring='r2',
        verbose=0,
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    grid_best_score = grid_search.best_score_
    grid_best_params = grid_search.best_params_
    
    mlflow.log_param("grid_search_best_score", grid_best_score)
    mlflow.log_params({f"grid_{k}": v for k, v in grid_best_params.items()})
    
    # Strategy 2: Random Search (broader exploration)
    logger.info("🔹 Random Search optimization...")
    random_param_grid = {
        'n_estimators': [50, 100, 200, 300, 500],
        'max_depth': [5, 10, 15, 20, 25, None],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [1, 2, 4, 8],
        'max_features': ['sqrt', 'log2', None],
        'bootstrap': [True, False]
    }
    
    random_search = RandomizedSearchCV(
        RandomForestRegressor(random_state=42, n_jobs=-1),
        random_param_grid,
        n_iter=50,  # 50 random combinations
        cv=3,
        scoring='r2',
        verbose=0,
        n_jobs=-1,
        random_state=42
    )
    random_search.fit(X_train, y_train)
    random_best_score = random_search.best_score_
    random_best_params = random_search.best_params_
    
    mlflow.log_param("random_search_best_score", random_best_score)
    mlflow.log_params({f"random_{k}": v for k, v in random_best_params.items()})
    
    # Strategy 3: Bayesian Optimization (intelligent search)
    logger.info("🔹 Bayesian optimization...")
    try:
        from skopt import BayesSearchCV
        from skopt.space import Real, Integer
        
        bayesian_param_space = {
            'n_estimators': Integer(50, 500),
            'max_depth': Integer(5, 30),
            'min_samples_split': Integer(2, 20),
            'min_samples_leaf': Integer(1, 10),
            'max_features': Real(0.1, 1.0)
        }
        
        bayesian_search = BayesSearchCV(
            RandomForestRegressor(random_state=42, n_jobs=-1),
            bayesian_param_space,
            n_iter=30,  # 30 intelligent iterations
            cv=3,
            scoring='r2',
            verbose=0,
            n_jobs=-1,
            random_state=42
        )
        bayesian_search.fit(X_train, y_train)
        bayesian_best_score = bayesian_search.best_score_
        bayesian_best_params = bayesian_search.best_params_
        
        mlflow.log_param("bayesian_search_best_score", bayesian_best_score)
        mlflow.log_params({f"bayesian_{k}": v for k, v in bayesian_best_params.items()})
        
        # Compare all three approaches
        strategies = [
            ("Grid Search", grid_search.best_estimator_, grid_best_score, grid_best_params),
            ("Random Search", random_search.best_estimator_, random_best_score, random_best_params),
            ("Bayesian Search", bayesian_search.best_estimator_, bayesian_best_score, bayesian_best_params)
        ]
        
    except ImportError:
        logger.warning("scikit-optimize not available, skipping Bayesian optimization")
        strategies = [
            ("Grid Search", grid_search.best_estimator_, grid_best_score, grid_best_params),
            ("Random Search", random_search.best_estimator_, random_best_score, random_best_params)
        ]
    
    # Select the best performing strategy
    best_strategy = max(strategies, key=lambda x: x[2])
    strategy_name, best_model, best_score, best_params = best_strategy
    
    logger.info(f"🏆 Best strategy: {strategy_name} with R² = {best_score:.4f}")
    logger.info(f"🎯 Best parameters: {best_params}")
    
    # Log the winning strategy
    mlflow.log_param("best_optimization_strategy", strategy_name)
    mlflow.log_param("best_hyperparameter_score", best_score)
    mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
    
    return best_model, best_params, best_score

async def train_and_register_model(symbol="AAPL"):
    """Train model and register in MLflow"""
    logger.info(f"Starting model training for {symbol}")
    
    # Create experiment
    experiment_name = f"stock_prediction_{symbol.lower()}"
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(
                experiment_name,
                f"Stock price prediction model for {symbol}"
            )
        else:
            experiment_id = experiment.experiment_id
    except Exception as e:
        logger.warning(f"Could not create experiment: {e}")
        experiment_id = "0"
    
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_name=f"{symbol}_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        
        # Log parameters
        mlflow.log_param("symbol", symbol)
        mlflow.log_param("model_type", "RandomForestRegressor")
        mlflow.log_param("feature_engineering", "advanced_technical_indicators")
        mlflow.log_param("training_period", "2y")
        
        # Prepare data
        X, y, feature_names = prepare_training_data(symbol)
        
        # Train/test split (80/20)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False  # No shuffle for time series
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Hyperparameter tuning with multiple strategies
        logger.info("Starting hyperparameter optimization...")
        
        best_model, best_params, best_score = hyperparameter_optimization(
            X_train_scaled, y_train, symbol
        )
        
        # Use the best model found
        model = best_model
        
        # Evaluate model
        train_score = model.score(X_train_scaled, y_train)
        test_score = model.score(X_test_scaled, y_test)
        
        y_train_pred = model.predict(X_train_scaled)
        y_test_pred = model.predict(X_test_scaled)
        
        train_mse = mean_squared_error(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        train_rmse = np.sqrt(train_mse)
        test_rmse = np.sqrt(test_mse)
        
        # Log metrics
        mlflow.log_metric("train_r2", train_score)
        mlflow.log_metric("test_r2", test_score)
        mlflow.log_metric("train_mse", train_mse)
        mlflow.log_metric("test_mse", test_mse)
        mlflow.log_metric("train_mae", train_mae)
        mlflow.log_metric("test_mae", test_mae)
        mlflow.log_metric("train_rmse", train_rmse)
        mlflow.log_metric("test_rmse", test_rmse)
        mlflow.log_metric("feature_count", X.shape[1])
        mlflow.log_metric("train_samples", len(X_train))
        mlflow.log_metric("test_samples", len(X_test))
        
        # Log model parameters from hyperparameter tuning
        for param, value in best_params.items():
            mlflow.log_param(f"model_{param}", value)
        
        # Register model in MLflow registry
        model_name = f"{symbol}_predictor"
        
        # Log and register the main model
        mlflow.sklearn.log_model(
            model,
            "model",
            registered_model_name=model_name,
            signature=mlflow.models.signature.infer_signature(X_train_scaled, y_train)
        )
        
        # Log and register the scaler
        scaler_name = f"{symbol}_predictor_scaler"
        mlflow.sklearn.log_model(
            scaler,
            "scaler",
            registered_model_name=scaler_name
        )
        
        # Transition to staging if performance is acceptable
        client = mlflow.MlflowClient()
        
        if test_score > 0.1:  # Reasonable threshold for financial data
            # Get the latest version
            latest_versions = client.get_latest_versions(model_name, stages=["None"])
            if latest_versions:
                latest_version = latest_versions[0]
                client.transition_model_version_stage(
                    name=model_name,
                    version=latest_version.version,
                    stage="Staging"
                )
                logger.info(f"Promoted {model_name} v{latest_version.version} to Staging")
                
                # Also transition scaler
                try:
                    scaler_versions = client.get_latest_versions(scaler_name, stages=["None"])
                    if scaler_versions:
                        scaler_version = scaler_versions[0]
                        client.transition_model_version_stage(
                            name=scaler_name,
                            version=scaler_version.version,
                            stage="Staging"
                        )
                        logger.info(f"Promoted {scaler_name} v{scaler_version.version} to Staging")
                except Exception as e:
                    logger.warning(f"Could not promote scaler: {e}")
        
        logger.info(f"Model training completed for {symbol}")
        logger.info(f"Performance: Train R²={train_score:.4f}, Test R²={test_score:.4f}")
        logger.info(f"Error metrics: Train MAE={train_mae:.6f}, Test MAE={test_mae:.6f}")
        
        return {
            'model_name': model_name,
            'train_score': train_score,
            'test_score': test_score,
            'train_mae': train_mae,
            'test_mae': test_mae
        }

if __name__ == "__main__":
    # Train models for a few different symbols
    symbols = ["AAPL", "MSFT", "TSLA"]
    
    for symbol in symbols:
        try:
            print(f"\n{'='*50}")
            print(f"Training model for {symbol}")
            print(f"{'='*50}")
            
            result = asyncio.run(train_and_register_model(symbol))
            
            print(f"✅ {symbol} model trained successfully!")
            print(f"   Test R²: {result['test_score']:.4f}")
            print(f"   Test MAE: {result['test_mae']:.6f}")
            
        except Exception as e:
            print(f"❌ Failed to train model for {symbol}: {e}")
            continue
    
    print(f"\n{'='*50}")
    print("Model training completed!")
    print("Check MLflow UI at http://localhost:5001")
    print(f"{'='*50}")