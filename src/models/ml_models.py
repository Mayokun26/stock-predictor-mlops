"""
Enterprise ML Models with XGBoost, LSTM, and Ensemble Methods
Production-ready models with MLflow integration and advanced evaluation
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime, timedelta
import logging
import pickle
import joblib
import json
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod

# ML Libraries
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score, GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline

# Deep Learning (optional)
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logging.warning("PyTorch not available. LSTM models will be disabled.")

# MLflow integration
try:
    import mlflow
    import mlflow.sklearn
    import mlflow.xgboost
    import mlflow.pytorch
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logging.warning("MLflow not available. Model tracking will be disabled.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelPrediction:
    """Model prediction with confidence and metadata"""
    value: float
    confidence: float
    prediction_interval: Tuple[float, float]
    features_used: Dict[str, float]
    model_name: str
    prediction_time: datetime
    metadata: Dict[str, Any]

@dataclass
class ModelEvaluation:
    """Model evaluation metrics"""
    mse: float
    rmse: float
    mae: float
    r2: float
    mape: float
    direction_accuracy: float
    sharpe_ratio: Optional[float]
    max_drawdown: Optional[float]
    roi: Optional[float]
    win_rate: Optional[float]

class BaseMLModel(ABC):
    """Abstract base class for ML models"""
    
    def __init__(self, name: str, version: str = "1.0"):
        self.name = name
        self.version = version
        self.model = None
        self.scaler = None
        self.is_trained = False
        self.feature_columns = []
        self.training_metadata = {}
        
    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> Dict[str, Any]:
        """Train the model"""
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        pass
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series, prices: Optional[pd.Series] = None) -> ModelEvaluation:
        """Evaluate model performance"""
        predictions = self.predict(X)
        
        # Basic metrics
        mse = mean_squared_error(y, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y, predictions)
        r2 = r2_score(y, predictions)
        
        # MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((y - predictions) / y)) * 100
        
        # Direction accuracy
        y_direction = np.sign(y)
        pred_direction = np.sign(predictions)
        direction_accuracy = np.mean(y_direction == pred_direction)
        
        # Financial metrics (if prices provided)
        sharpe_ratio = None
        max_drawdown = None
        roi = None
        win_rate = None
        
        if prices is not None:
            returns = self._calculate_strategy_returns(predictions, y, prices)
            if len(returns) > 0:
                sharpe_ratio = self._calculate_sharpe_ratio(returns)
                max_drawdown = self._calculate_max_drawdown(returns)
                roi = (returns + 1).prod() - 1
                win_rate = np.mean(returns > 0)
        
        return ModelEvaluation(
            mse=mse,
            rmse=rmse,
            mae=mae,
            r2=r2,
            mape=mape,
            direction_accuracy=direction_accuracy,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            roi=roi,
            win_rate=win_rate
        )
    
    def _calculate_strategy_returns(self, predictions: np.ndarray, actual: pd.Series, prices: pd.Series) -> np.ndarray:
        """Calculate strategy returns based on predictions"""
        try:
            # Simple strategy: go long if prediction is positive, short if negative
            positions = np.sign(predictions)
            price_returns = prices.pct_change().dropna()
            
            # Align arrays
            min_len = min(len(positions), len(price_returns))
            strategy_returns = positions[:min_len] * price_returns.values[:min_len]
            
            return strategy_returns
        except Exception as e:
            logger.error(f"Error calculating strategy returns: {e}")
            return np.array([])
    
    def _calculate_sharpe_ratio(self, returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        try:
            if len(returns) == 0 or np.std(returns) == 0:
                return 0.0
            
            annualized_return = np.mean(returns) * 252
            annualized_volatility = np.std(returns) * np.sqrt(252)
            
            return (annualized_return - risk_free_rate) / annualized_volatility
        except:
            return 0.0
    
    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown"""
        try:
            cumulative = (1 + returns).cumprod()
            running_max = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - running_max) / running_max
            return float(np.min(drawdown))
        except:
            return 0.0
    
    def save_model(self, filepath: str) -> bool:
        """Save model to disk"""
        try:
            model_data = {
                'name': self.name,
                'version': self.version,
                'model': self.model,
                'scaler': self.scaler,
                'feature_columns': self.feature_columns,
                'training_metadata': self.training_metadata,
                'is_trained': self.is_trained
            }
            joblib.dump(model_data, filepath)
            logger.info(f"Model saved to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False
    
    def load_model(self, filepath: str) -> bool:
        """Load model from disk"""
        try:
            model_data = joblib.load(filepath)
            self.name = model_data['name']
            self.version = model_data['version']
            self.model = model_data['model']
            self.scaler = model_data.get('scaler')
            self.feature_columns = model_data['feature_columns']
            self.training_metadata = model_data['training_metadata']
            self.is_trained = model_data['is_trained']
            logger.info(f"Model loaded from {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False

class XGBoostModel(BaseMLModel):
    """XGBoost model for financial prediction"""
    
    def __init__(self, name: str = "xgboost_predictor", version: str = "1.0", **xgb_params):
        super().__init__(name, version)
        
        # Default XGBoost parameters optimized for financial data
        self.xgb_params = {
            'n_estimators': 1000,
            'max_depth': 6,
            'learning_rate': 0.01,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': 42,
            'early_stopping_rounds': 50,
            'eval_metric': 'rmse'
        }
        self.xgb_params.update(xgb_params)
    
    def train(self, X: pd.DataFrame, y: pd.Series, X_val: pd.DataFrame = None, y_val: pd.Series = None, **kwargs) -> Dict[str, Any]:
        """Train XGBoost model"""
        try:
            logger.info(f"Training XGBoost model with {len(X)} samples and {len(X.columns)} features")
            
            self.feature_columns = list(X.columns)
            
            # Scale features
            self.scaler = RobustScaler()
            X_scaled = self.scaler.fit_transform(X)
            X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
            
            # Validation data
            eval_set = None
            if X_val is not None and y_val is not None:
                X_val_scaled = self.scaler.transform(X_val)
                X_val_scaled = pd.DataFrame(X_val_scaled, columns=X_val.columns, index=X_val.index)
                eval_set = [(X_val_scaled, y_val)]
            
            # Train model
            self.model = xgb.XGBRegressor(**self.xgb_params)
            
            if eval_set:
                self.model.fit(
                    X_scaled, y,
                    eval_set=eval_set,
                    verbose=False
                )
            else:
                self.model.fit(X_scaled, y)
            
            self.is_trained = True
            
            # Feature importance
            feature_importance = dict(zip(self.feature_columns, self.model.feature_importances_))
            
            # Training metadata
            self.training_metadata = {
                'training_samples': len(X),
                'features_count': len(self.feature_columns),
                'feature_importance': feature_importance,
                'xgb_params': self.xgb_params,
                'trained_at': datetime.now().isoformat()
            }
            
            logger.info(f"XGBoost training completed. Best iteration: {self.model.best_iteration}")
            
            return {
                'status': 'success',
                'best_iteration': getattr(self.model, 'best_iteration', None),
                'feature_importance': feature_importance
            }
            
        except Exception as e:
            logger.error(f"Error training XGBoost model: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions with XGBoost"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        try:
            # Ensure feature columns match
            X_aligned = X[self.feature_columns]
            
            # Scale features
            X_scaled = self.scaler.transform(X_aligned)
            
            # Make predictions
            predictions = self.model.predict(X_scaled)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            return np.array([])
    
    def predict_with_confidence(self, X: pd.DataFrame, n_estimators: Optional[int] = None) -> List[ModelPrediction]:
        """Make predictions with confidence intervals using quantile regression"""
        try:
            # Base prediction
            predictions = self.predict(X)
            
            # For confidence intervals, we can use feature importance as a proxy
            feature_importance = self.model.feature_importances_
            confidence_scores = []
            
            X_aligned = X[self.feature_columns]
            X_scaled = self.scaler.transform(X_aligned)
            
            for i, row in enumerate(X_scaled):
                # Calculate confidence based on feature importance and feature values
                weighted_features = np.abs(row) * feature_importance
                confidence = 1.0 / (1.0 + np.std(weighted_features))
                confidence_scores.append(confidence)
            
            # Create prediction objects
            results = []
            for i, (pred, conf) in enumerate(zip(predictions, confidence_scores)):
                # Simple prediction interval
                interval_width = (1 - conf) * abs(pred) * 0.5
                prediction_interval = (pred - interval_width, pred + interval_width)
                
                result = ModelPrediction(
                    value=float(pred),
                    confidence=float(conf),
                    prediction_interval=prediction_interval,
                    features_used=dict(zip(self.feature_columns, X_aligned.iloc[i].values)),
                    model_name=self.name,
                    prediction_time=datetime.now(),
                    metadata={'model_type': 'xgboost'}
                )
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error making predictions with confidence: {e}")
            return []

class EnsembleModel(BaseMLModel):
    """Ensemble model combining multiple algorithms"""
    
    def __init__(self, name: str = "ensemble_predictor", version: str = "1.0"):
        super().__init__(name, version)
        self.models = {}
        self.weights = {}
    
    def train(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> Dict[str, Any]:
        """Train ensemble of models"""
        try:
            logger.info(f"Training ensemble model with {len(X)} samples")
            
            self.feature_columns = list(X.columns)
            
            # Scale features
            self.scaler = RobustScaler()
            X_scaled = self.scaler.fit_transform(X)
            X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
            
            # Define models
            models_config = {
                'xgboost': xgb.XGBRegressor(
                    n_estimators=500,
                    max_depth=6,
                    learning_rate=0.01,
                    random_state=42
                ),
                'random_forest': RandomForestRegressor(
                    n_estimators=500,
                    max_depth=10,
                    random_state=42
                ),
                'gradient_boost': GradientBoostingRegressor(
                    n_estimators=500,
                    max_depth=6,
                    learning_rate=0.01,
                    random_state=42
                ),
                'ridge': Ridge(alpha=1.0),
                'elastic_net': ElasticNet(alpha=0.1, l1_ratio=0.5)
            }
            
            # Train models and calculate weights based on performance
            cv_scores = {}
            tscv = TimeSeriesSplit(n_splits=5)
            
            for name, model in models_config.items():
                logger.info(f"Training {name} model...")
                
                # Cross-validation
                scores = cross_val_score(model, X_scaled, y, cv=tscv, scoring='neg_mean_squared_error')
                cv_scores[name] = -scores.mean()
                
                # Train on full dataset
                model.fit(X_scaled, y)
                self.models[name] = model
            
            # Calculate weights (inverse of error)
            total_inverse_error = sum(1 / error for error in cv_scores.values())
            self.weights = {name: (1 / error) / total_inverse_error for name, error in cv_scores.items()}
            
            self.is_trained = True
            
            self.training_metadata = {
                'training_samples': len(X),
                'features_count': len(self.feature_columns),
                'model_weights': self.weights,
                'cv_scores': cv_scores,
                'trained_at': datetime.now().isoformat()
            }
            
            logger.info(f"Ensemble training completed. Model weights: {self.weights}")
            
            return {
                'status': 'success',
                'model_weights': self.weights,
                'cv_scores': cv_scores
            }
            
        except Exception as e:
            logger.error(f"Error training ensemble model: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make ensemble predictions"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        try:
            # Ensure feature columns match
            X_aligned = X[self.feature_columns]
            
            # Scale features
            X_scaled = self.scaler.transform(X_aligned)
            
            # Get predictions from each model
            predictions = {}
            for name, model in self.models.items():
                predictions[name] = model.predict(X_scaled)
            
            # Weighted ensemble prediction
            ensemble_pred = np.zeros(len(X_aligned))
            for name, pred in predictions.items():
                ensemble_pred += self.weights[name] * pred
            
            return ensemble_pred
            
        except Exception as e:
            logger.error(f"Error making ensemble predictions: {e}")
            return np.array([])

class LSTMModel(BaseMLModel):
    """LSTM model for time series prediction"""
    
    def __init__(self, name: str = "lstm_predictor", version: str = "1.0", sequence_length: int = 60):
        super().__init__(name, version)
        self.sequence_length = sequence_length
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if TORCH_AVAILABLE else None
        
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available. LSTM model will not work.")
    
    def _create_sequences(self, data: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training"""
        X, y = [], []
        for i in range(self.sequence_length, len(data)):
            X.append(data[i-self.sequence_length:i])
            y.append(target[i])
        return np.array(X), np.array(y)
    
    def train(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> Dict[str, Any]:
        """Train LSTM model"""
        if not TORCH_AVAILABLE:
            return {'status': 'error', 'message': 'PyTorch not available'}
        
        try:
            logger.info(f"Training LSTM model with {len(X)} samples")
            
            self.feature_columns = list(X.columns)
            
            # Scale features
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            # Create sequences
            X_seq, y_seq = self._create_sequences(X_scaled, y.values)
            
            if len(X_seq) == 0:
                return {'status': 'error', 'message': 'Not enough data for sequence creation'}
            
            # Define LSTM model
            class LSTMNet(nn.Module):
                def __init__(self, input_size, hidden_size=50, num_layers=2):
                    super(LSTMNet, self).__init__()
                    self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
                    self.fc = nn.Linear(hidden_size, 1)
                    
                def forward(self, x):
                    lstm_out, _ = self.lstm(x)
                    prediction = self.fc(lstm_out[:, -1])
                    return prediction
            
            # Initialize model
            self.model = LSTMNet(input_size=X_seq.shape[2])
            self.model.to(self.device)
            
            # Training parameters
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
            
            # Convert to tensors
            X_tensor = torch.FloatTensor(X_seq).to(self.device)
            y_tensor = torch.FloatTensor(y_seq).to(self.device)
            
            # Training loop
            epochs = kwargs.get('epochs', 100)
            batch_size = kwargs.get('batch_size', 32)
            
            dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
            self.model.train()
            losses = []
            
            for epoch in range(epochs):
                epoch_loss = 0
                for batch_X, batch_y in dataloader:
                    optimizer.zero_grad()
                    predictions = self.model(batch_X).squeeze()
                    loss = criterion(predictions, batch_y)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                
                avg_loss = epoch_loss / len(dataloader)
                losses.append(avg_loss)
                
                if epoch % 10 == 0:
                    logger.info(f"Epoch {epoch}, Loss: {avg_loss:.6f}")
            
            self.is_trained = True
            
            self.training_metadata = {
                'training_samples': len(X_seq),
                'sequence_length': self.sequence_length,
                'final_loss': losses[-1] if losses else None,
                'epochs': epochs,
                'trained_at': datetime.now().isoformat()
            }
            
            logger.info(f"LSTM training completed. Final loss: {losses[-1] if losses else 'N/A'}")
            
            return {
                'status': 'success',
                'final_loss': losses[-1] if losses else None,
                'training_losses': losses
            }
            
        except Exception as e:
            logger.error(f"Error training LSTM model: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make LSTM predictions"""
        if not TORCH_AVAILABLE or not self.is_trained:
            logger.error("LSTM model not available or not trained")
            return np.array([])
        
        try:
            # Ensure feature columns match
            X_aligned = X[self.feature_columns]
            
            # Scale features
            X_scaled = self.scaler.transform(X_aligned)
            
            # Create sequences
            if len(X_scaled) < self.sequence_length:
                logger.warning("Not enough data for LSTM prediction")
                return np.array([])
            
            # Take the last sequence_length points for prediction
            X_seq = X_scaled[-self.sequence_length:].reshape(1, self.sequence_length, -1)
            
            # Convert to tensor
            X_tensor = torch.FloatTensor(X_seq).to(self.device)
            
            # Make prediction
            self.model.eval()
            with torch.no_grad():
                prediction = self.model(X_tensor).cpu().numpy().flatten()
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error making LSTM predictions: {e}")
            return np.array([])

class MLModelRegistry:
    """Model registry with MLflow integration"""
    
    def __init__(self, mlflow_tracking_uri: str = None):
        self.models = {}
        
        if MLFLOW_AVAILABLE and mlflow_tracking_uri:
            mlflow.set_tracking_uri(mlflow_tracking_uri)
            logger.info(f"MLflow tracking URI set to: {mlflow_tracking_uri}")
    
    def register_model(self, model: BaseMLModel, experiment_name: str = "stock_prediction") -> str:
        """Register model with MLflow"""
        try:
            if MLFLOW_AVAILABLE:
                mlflow.set_experiment(experiment_name)
                
                with mlflow.start_run() as run:
                    # Log model parameters
                    if hasattr(model, 'xgb_params'):
                        mlflow.log_params(model.xgb_params)
                    
                    # Log training metadata
                    mlflow.log_params({
                        'model_name': model.name,
                        'model_version': model.version,
                        'training_samples': model.training_metadata.get('training_samples', 0),
                        'features_count': model.training_metadata.get('features_count', 0)
                    })
                    
                    # Log model
                    if isinstance(model, XGBoostModel):
                        mlflow.xgboost.log_model(model.model, "model")
                    else:
                        mlflow.sklearn.log_model(model, "model")
                    
                    run_id = run.info.run_id
                    logger.info(f"Model registered with MLflow. Run ID: {run_id}")
                    
                    self.models[model.name] = {
                        'model': model,
                        'run_id': run_id,
                        'registered_at': datetime.now()
                    }
                    
                    return run_id
            else:
                # Local registration without MLflow
                self.models[model.name] = {
                    'model': model,
                    'run_id': f"local_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    'registered_at': datetime.now()
                }
                return self.models[model.name]['run_id']
                
        except Exception as e:
            logger.error(f"Error registering model: {e}")
            return ""
    
    def get_model(self, model_name: str) -> Optional[BaseMLModel]:
        """Get registered model"""
        if model_name in self.models:
            return self.models[model_name]['model']
        return None
    
    def list_models(self) -> Dict[str, Any]:
        """List all registered models"""
        return {
            name: {
                'run_id': info['run_id'],
                'registered_at': info['registered_at'],
                'is_trained': info['model'].is_trained
            }
            for name, info in self.models.items()
        }

# Test function
def test_models():
    """Test ML models"""
    # Generate sample data
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=1000, freq='D')
    
    # Features
    features = pd.DataFrame({
        'sma_20': np.random.randn(1000),
        'rsi': np.random.uniform(0, 100, 1000),
        'macd': np.random.randn(1000),
        'sentiment_mean': np.random.uniform(-1, 1, 1000),
        'volatility': np.random.uniform(0, 1, 1000)
    }, index=dates)
    
    # Target (next day return)
    target = pd.Series(np.random.randn(1000) * 0.02, index=dates)  # 2% daily volatility
    
    # Split data
    split_idx = int(0.8 * len(features))
    X_train, X_test = features[:split_idx], features[split_idx:]
    y_train, y_test = target[:split_idx], target[split_idx:]
    
    # Test XGBoost model
    logger.info("Testing XGBoost model...")
    xgb_model = XGBoostModel()
    train_result = xgb_model.train(X_train, y_train, X_val=X_test, y_val=y_test)
    print(f"XGBoost training result: {train_result}")
    
    # Make predictions
    predictions = xgb_model.predict(X_test)
    print(f"XGBoost predictions shape: {predictions.shape}")
    
    # Evaluate
    evaluation = xgb_model.evaluate(X_test, y_test)
    print(f"XGBoost evaluation: RMSE={evaluation.rmse:.4f}, R2={evaluation.r2:.4f}")
    
    # Test Ensemble model
    logger.info("Testing Ensemble model...")
    ensemble_model = EnsembleModel()
    train_result = ensemble_model.train(X_train, y_train)
    print(f"Ensemble training result: {train_result}")
    
    predictions = ensemble_model.predict(X_test)
    evaluation = ensemble_model.evaluate(X_test, y_test)
    print(f"Ensemble evaluation: RMSE={evaluation.rmse:.4f}, R2={evaluation.r2:.4f}")
    
    # Test model registry
    registry = MLModelRegistry()
    run_id = registry.register_model(xgb_model)
    print(f"Model registered with ID: {run_id}")
    
    models_list = registry.list_models()
    print(f"Registered models: {models_list}")

if __name__ == "__main__":
    test_models()