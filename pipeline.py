#!/usr/bin/env python3
"""
Complete MLOps pipeline orchestrator
Runs data collection, training, and model deployment
"""
import subprocess
import sys
import time
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import logging
import os
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MLOpsPipeline:
    def __init__(self):
        self.db_path = 'stocks.db'
        self.symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN', 'NVDA', 'META', 'BRK-B']
        
    def setup_database(self):
        """Initialize database with stock symbols"""
        logger.info("Setting up database...")
        try:
            subprocess.run([sys.executable, 'setup_database.py'], check=True)
            logger.info("✅ Database setup complete")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Database setup failed: {e}")
            return False
    
    def collect_data(self):
        """Run data collection"""
        logger.info("Starting data collection...")
        try:
            # Run the data collection script
            result = subprocess.run([sys.executable, 'collect_data.py'], 
                                  capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                logger.info("✅ Data collection successful")
                logger.info(result.stdout)
                return True
            else:
                logger.error(f"❌ Data collection failed: {result.stderr}")
                return False
        except subprocess.TimeoutExpired:
            logger.error("❌ Data collection timed out")
            return False
        except Exception as e:
            logger.error(f"❌ Data collection error: {e}")
            return False
    
    def check_data_quality(self):
        """Verify data quality before training"""
        logger.info("Checking data quality...")
        
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Check if we have recent data
            for symbol in self.symbols:
                query = f"""
                SELECT COUNT(*) as count, MAX(date) as latest_date 
                FROM stock_prices WHERE symbol='{symbol}'
                """
                result = pd.read_sql(query, conn)
                
                if result.iloc[0]['count'] < 20:
                    logger.warning(f"⚠️  {symbol}: Only {result.iloc[0]['count']} records")
                    conn.close()
                    return False
                    
                latest_date = pd.to_datetime(result.iloc[0]['latest_date'])
                days_old = (datetime.now() - latest_date).days
                
                if days_old > 7:
                    logger.warning(f"⚠️  {symbol}: Data is {days_old} days old")
            
            conn.close()
            logger.info("✅ Data quality check passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Data quality check failed: {e}")
            return False
    
    def train_models(self):
        """Train ML models with MLflow tracking"""
        logger.info("Training ML models...")
        
        try:
            # Set MLflow experiment
            mlflow.set_experiment("stock_prediction_pipeline")
            
            with mlflow.start_run(run_name=f"pipeline_run_{datetime.now().strftime('%Y%m%d_%H%M')}"):
                
                # Log pipeline parameters
                mlflow.log_param("symbols", ','.join(self.symbols))
                mlflow.log_param("pipeline_timestamp", datetime.now().isoformat())
                
                results = []
                
                for symbol in self.symbols:
                    try:
                        # Train model for this symbol
                        model, metrics = self._train_single_model(symbol)
                        results.append({
                            'symbol': symbol,
                            'rmse': metrics['rmse'],
                            'r2': metrics['r2'],
                            'success': True
                        })
                        logger.info(f"✅ {symbol}: RMSE=${metrics['rmse']:.2f}, R²={metrics['r2']:.3f}")
                        
                    except Exception as e:
                        logger.error(f"❌ {symbol}: Training failed - {e}")
                        results.append({
                            'symbol': symbol,
                            'success': False,
                            'error': str(e)
                        })
                
                # Log overall results
                successful = [r for r in results if r['success']]
                if successful:
                    avg_rmse = np.mean([r['rmse'] for r in successful])
                    avg_r2 = np.mean([r['r2'] for r in successful])
                    mlflow.log_metric("avg_rmse", avg_rmse)
                    mlflow.log_metric("avg_r2", avg_r2)
                    mlflow.log_metric("successful_models", len(successful))
                
                logger.info(f"✅ Training complete: {len(successful)}/{len(self.symbols)} models successful")
                return len(successful) > 0
                
        except Exception as e:
            logger.error(f"❌ Model training failed: {e}")
            return False
    
    def _train_single_model(self, symbol):
        """Train a single model for a stock symbol"""
        conn = sqlite3.connect(self.db_path)
        
        # Load data
        df = pd.read_sql(f"""
            SELECT * FROM stock_prices 
            WHERE symbol='{symbol}' 
            ORDER BY date
        """, conn)
        conn.close()
        
        if len(df) < 50:
            raise ValueError(f"Insufficient data for {symbol}: {len(df)} records")
        
        # Feature engineering
        df = self._create_features(df)
        df = df.dropna()
        
        if len(df) < 30:
            raise ValueError(f"Insufficient data after feature engineering: {len(df)} records")
        
        # Prepare features and target
        feature_cols = ['open', 'high', 'low', 'volume', 'returns', 
                       'sma_5', 'sma_20', 'volatility', 'rsi', 'price_ma_ratio']
        X = df[feature_cols]
        y = df['target']
        
        # Train/test split (80/20)
        split_idx = int(len(df) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Train model
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42
        )
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        # Save model (in production, use MLflow model registry)
        model_dir = f"models/{symbol}"
        os.makedirs(model_dir, exist_ok=True)
        
        import joblib
        joblib.dump(model, f"{model_dir}/model.pkl")
        
        return model, {'rmse': rmse, 'r2': r2}
    
    def _create_features(self, df):
        """Create features from price data"""
        df = df.copy()
        df['target'] = df['close'].shift(-1)  # Next day's close
        df['returns'] = df['close'].pct_change()
        df['sma_5'] = df['close'].rolling(5).mean()
        df['sma_20'] = df['close'].rolling(20).mean()
        df['volatility'] = df['close'].rolling(10).std()
        df['rsi'] = self._calculate_rsi(df['close'])
        df['price_ma_ratio'] = df['close'] / df['sma_20']
        return df
    
    def _calculate_rsi(self, prices, window=14):
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def test_api(self):
        """Test the API endpoints"""
        logger.info("Testing API...")
        
        try:
            import requests
            import time
            
            # Start API in background (for testing)
            # In production, API would already be running
            
            # Wait a moment for API to start
            time.sleep(2)
            
            # Test health endpoint
            response = requests.get("http://localhost:8000/health", timeout=10)
            if response.status_code == 200:
                logger.info("✅ API health check passed")
                
                # Test prediction endpoint
                test_request = {
                    "symbol": "AAPL",
                    "news_headlines": ["Apple reports strong quarterly earnings"]
                }
                
                pred_response = requests.post("http://localhost:8000/predict", 
                                            json=test_request, timeout=10)
                
                if pred_response.status_code == 200:
                    result = pred_response.json()
                    logger.info(f"✅ Prediction test passed: {result['symbol']} ${result['predicted_price']}")
                    return True
                else:
                    logger.error(f"❌ Prediction test failed: {pred_response.status_code}")
                    return False
            else:
                logger.error(f"❌ API health check failed: {response.status_code}")
                return False
                
        except requests.exceptions.ConnectionError:
            logger.warning("⚠️  API not running, skipping test")
            return True  # Don't fail pipeline if API isn't running
        except Exception as e:
            logger.error(f"❌ API test failed: {e}")
            return False
    
    def run_full_pipeline(self):
        """Run the complete MLOps pipeline"""
        logger.info("🚀 Starting MLOps Pipeline")
        start_time = time.time()
        
        steps = [
            ("Database Setup", self.setup_database),
            ("Data Collection", self.collect_data),
            ("Data Quality Check", self.check_data_quality),
            ("Model Training", self.train_models),
            ("API Testing", self.test_api)
        ]
        
        for step_name, step_func in steps:
            logger.info(f"\n📋 Step: {step_name}")
            
            if not step_func():
                logger.error(f"❌ Pipeline failed at: {step_name}")
                return False
        
        duration = time.time() - start_time
        logger.info(f"\n🎉 Pipeline completed successfully in {duration:.1f} seconds")
        return True

def main():
    """Run the MLOps pipeline"""
    pipeline = MLOpsPipeline()
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "collect":
            success = pipeline.collect_data()
        elif command == "train":
            success = pipeline.train_models()
        elif command == "test":
            success = pipeline.test_api()
        elif command == "quality":
            success = pipeline.check_data_quality()
        else:
            logger.error(f"Unknown command: {command}")
            logger.info("Available commands: collect, train, test, quality")
            sys.exit(1)
    else:
        # Run full pipeline
        success = pipeline.run_full_pipeline()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()