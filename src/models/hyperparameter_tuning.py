#!/usr/bin/env python3
"""
Advanced Hyperparameter Tuning for MLOps Stock Prediction
Implements Grid Search, Random Search, and Bayesian optimization
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import mlflow
import mlflow.sklearn
from datetime import datetime
import logging
import asyncio
import json
from typing import Dict, List, Tuple, Any, Optional
import optuna
from optuna.integration.mlflow import MLflowCallback
import joblib
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class AdvancedHyperparameterTuner:
    """
    Enterprise-grade hyperparameter tuning with multiple optimization strategies
    """
    
    def __init__(self, mlflow_tracking_uri: str = "http://localhost:5001"):
        self.mlflow_tracking_uri = mlflow_tracking_uri
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        
        # Model configurations with parameter grids
        self.model_configs = {
            'random_forest': {
                'model': RandomForestRegressor(random_state=42),
                'param_grid': {
                    'n_estimators': [50, 100, 200, 300],
                    'max_depth': [10, 20, 30, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['auto', 'sqrt', 'log2'],
                    'bootstrap': [True, False]
                },
                'param_distributions': {
                    'n_estimators': [50, 100, 150, 200, 250, 300],
                    'max_depth': [5, 10, 15, 20, 25, 30, None],
                    'min_samples_split': [2, 5, 10, 15, 20],
                    'min_samples_leaf': [1, 2, 4, 8],
                    'max_features': ['auto', 'sqrt', 'log2'],
                    'bootstrap': [True, False]
                }
            },
            
            'gradient_boosting': {
                'model': GradientBoostingRegressor(random_state=42),
                'param_grid': {
                    'n_estimators': [100, 200, 300],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7, 9],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'subsample': [0.8, 0.9, 1.0]
                },
                'param_distributions': {
                    'n_estimators': [50, 100, 150, 200, 250, 300],
                    'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
                    'max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
                    'min_samples_split': [2, 5, 10, 15],
                    'min_samples_leaf': [1, 2, 4, 8],
                    'subsample': [0.7, 0.8, 0.9, 1.0]
                }
            },
            
            'ridge': {
                'model': Ridge(random_state=42),
                'param_grid': {
                    'alpha': [0.1, 1.0, 10.0, 100.0],
                    'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg']
                },
                'param_distributions': {
                    'alpha': [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0],
                    'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg']
                }
            },
            
            'svr': {
                'model': SVR(),
                'param_grid': {
                    'kernel': ['linear', 'rbf', 'poly'],
                    'C': [0.1, 1.0, 10.0],
                    'gamma': ['scale', 'auto', 0.001, 0.01],
                    'epsilon': [0.01, 0.1, 0.2]
                },
                'param_distributions': {
                    'kernel': ['linear', 'rbf', 'poly'],
                    'C': [0.01, 0.1, 1.0, 10.0, 100.0],
                    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1.0],
                    'epsilon': [0.01, 0.05, 0.1, 0.15, 0.2]
                }
            }
        }
        
        # Cross-validation strategy for time series
        self.cv_strategy = TimeSeriesSplit(n_splits=5)
        
    def create_experiment(self, symbol: str, tuning_method: str) -> str:
        """Create MLflow experiment for hyperparameter tuning"""
        experiment_name = f"hyperparameter_tuning_{symbol.lower()}_{tuning_method}"
        
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(
                    experiment_name,
                    f"Hyperparameter tuning for {symbol} using {tuning_method}"
                )
            else:
                experiment_id = experiment.experiment_id
        except Exception as e:
            logger.warning(f"Could not create experiment: {e}")
            experiment_id = "0"  # Default experiment
            
        mlflow.set_experiment(experiment_name)
        return experiment_id
        
    def evaluate_model(self, model, X_train, X_test, y_train, y_test, scaler=None) -> Dict[str, float]:
        """Comprehensive model evaluation with multiple metrics"""
        
        # Scale data if scaler provided
        if scaler:
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
        else:
            X_train_scaled = X_train
            X_test_scaled = X_test
            
        # Fit model
        model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_train_pred = model.predict(X_train_scaled)
        y_test_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        metrics = {
            'train_r2': r2_score(y_train, y_train_pred),
            'test_r2': r2_score(y_test, y_test_pred),
            'train_mse': mean_squared_error(y_train, y_train_pred),
            'test_mse': mean_squared_error(y_test, y_test_pred),
            'train_mae': mean_absolute_error(y_train, y_train_pred),
            'test_mae': mean_absolute_error(y_test, y_test_pred),
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred))
        }
        
        # Overfitting check
        metrics['overfitting'] = metrics['train_r2'] - metrics['test_r2']
        metrics['generalization_score'] = metrics['test_r2'] / max(metrics['train_r2'], 0.001)
        
        return metrics
        
    def grid_search_tuning(self, symbol: str, X_train, X_test, y_train, y_test, 
                          model_name: str = 'random_forest', n_jobs: int = -1) -> Dict[str, Any]:
        """Grid search hyperparameter tuning with comprehensive logging"""
        
        logger.info(f"Starting grid search tuning for {symbol} using {model_name}")
        experiment_id = self.create_experiment(symbol, "grid_search")
        
        model_config = self.model_configs.get(model_name)
        if not model_config:
            raise ValueError(f"Model {model_name} not supported")
            
        with mlflow.start_run(run_name=f"{symbol}_{model_name}_gridsearch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            
            # Log experiment parameters
            mlflow.log_param("symbol", symbol)
            mlflow.log_param("model_type", model_name)
            mlflow.log_param("tuning_method", "grid_search")
            mlflow.log_param("cv_splits", self.cv_strategy.n_splits)
            mlflow.log_param("feature_count", X_train.shape[1])
            mlflow.log_param("train_samples", X_train.shape[0])
            mlflow.log_param("test_samples", X_test.shape[0])
            
            # Grid search with time series cross-validation
            grid_search = GridSearchCV(
                estimator=model_config['model'],
                param_grid=model_config['param_grid'],
                cv=self.cv_strategy,
                scoring='r2',
                n_jobs=n_jobs,
                verbose=1,
                return_train_score=True
            )
            
            # Fit grid search
            start_time = datetime.now()
            grid_search.fit(X_train, y_train)
            end_time = datetime.now()
            
            # Get best model
            best_model = grid_search.best_estimator_
            
            # Evaluate best model
            scaler = StandardScaler()
            metrics = self.evaluate_model(best_model, X_train, X_test, y_train, y_test, scaler)
            
            # Log results
            mlflow.log_param("best_params", str(grid_search.best_params_))
            mlflow.log_metric("best_cv_score", grid_search.best_score_)
            mlflow.log_metric("tuning_time_minutes", (end_time - start_time).total_seconds() / 60)
            
            # Log all metrics
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
                
            # Log model artifacts
            mlflow.sklearn.log_model(
                best_model, 
                "best_model",
                signature=mlflow.models.signature.infer_signature(X_train, y_train)
            )
            mlflow.sklearn.log_model(scaler, "scaler")
            
            # Log cross-validation results
            cv_results_df = pd.DataFrame(grid_search.cv_results_)
            cv_results_df.to_csv("cv_results.csv", index=False)
            mlflow.log_artifact("cv_results.csv")
            
            results = {
                'best_model': best_model,
                'best_params': grid_search.best_params_,
                'best_score': grid_search.best_score_,
                'metrics': metrics,
                'scaler': scaler,
                'cv_results': grid_search.cv_results_
            }
            
            logger.info(f"Grid search completed for {symbol}. Best R²: {grid_search.best_score_:.4f}")
            return results
            
    def random_search_tuning(self, symbol: str, X_train, X_test, y_train, y_test,
                           model_name: str = 'random_forest', n_iter: int = 50, 
                           n_jobs: int = -1) -> Dict[str, Any]:
        """Random search hyperparameter tuning for faster optimization"""
        
        logger.info(f"Starting random search tuning for {symbol} using {model_name}")
        experiment_id = self.create_experiment(symbol, "random_search")
        
        model_config = self.model_configs.get(model_name)
        if not model_config:
            raise ValueError(f"Model {model_name} not supported")
            
        with mlflow.start_run(run_name=f"{symbol}_{model_name}_randomsearch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            
            # Log experiment parameters
            mlflow.log_param("symbol", symbol)
            mlflow.log_param("model_type", model_name)
            mlflow.log_param("tuning_method", "random_search")
            mlflow.log_param("n_iter", n_iter)
            mlflow.log_param("cv_splits", self.cv_strategy.n_splits)
            mlflow.log_param("feature_count", X_train.shape[1])
            
            # Random search
            random_search = RandomizedSearchCV(
                estimator=model_config['model'],
                param_distributions=model_config['param_distributions'],
                n_iter=n_iter,
                cv=self.cv_strategy,
                scoring='r2',
                n_jobs=n_jobs,
                verbose=1,
                random_state=42,
                return_train_score=True
            )
            
            # Fit random search
            start_time = datetime.now()
            random_search.fit(X_train, y_train)
            end_time = datetime.now()
            
            # Get best model and evaluate
            best_model = random_search.best_estimator_
            scaler = StandardScaler()
            metrics = self.evaluate_model(best_model, X_train, X_test, y_train, y_test, scaler)
            
            # Log results
            mlflow.log_param("best_params", str(random_search.best_params_))
            mlflow.log_metric("best_cv_score", random_search.best_score_)
            mlflow.log_metric("tuning_time_minutes", (end_time - start_time).total_seconds() / 60)
            
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
                
            # Log artifacts
            mlflow.sklearn.log_model(best_model, "best_model")
            mlflow.sklearn.log_model(scaler, "scaler")
            
            results = {
                'best_model': best_model,
                'best_params': random_search.best_params_,
                'best_score': random_search.best_score_,
                'metrics': metrics,
                'scaler': scaler
            }
            
            logger.info(f"Random search completed for {symbol}. Best R²: {random_search.best_score_:.4f}")
            return results
            
    def bayesian_optimization_tuning(self, symbol: str, X_train, X_test, y_train, y_test,
                                   model_name: str = 'random_forest', n_trials: int = 100) -> Dict[str, Any]:
        """Bayesian optimization using Optuna for intelligent hyperparameter search"""
        
        logger.info(f"Starting Bayesian optimization for {symbol} using {model_name}")
        experiment_id = self.create_experiment(symbol, "bayesian_optimization")
        
        def objective(trial):
            """Optuna objective function"""
            
            # Define hyperparameter search space based on model type
            if model_name == 'random_forest':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300, step=50),
                    'max_depth': trial.suggest_categorical('max_depth', [10, 20, 30, None]),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                    'max_features': trial.suggest_categorical('max_features', ['auto', 'sqrt', 'log2']),
                    'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
                    'random_state': 42
                }
                model = RandomForestRegressor(**params)
                
            elif model_name == 'gradient_boosting':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300, step=50),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                    'subsample': trial.suggest_float('subsample', 0.7, 1.0),
                    'random_state': 42
                }
                model = GradientBoostingRegressor(**params)
                
            else:
                raise ValueError(f"Bayesian optimization not implemented for {model_name}")
                
            # Cross-validation score
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            
            # Use time series cross-validation
            scores = []
            for train_idx, val_idx in self.cv_strategy.split(X_train):
                X_cv_train, X_cv_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
                y_cv_train, y_cv_val = y_train[train_idx], y_train[val_idx]
                
                model.fit(X_cv_train, y_cv_train)
                y_cv_pred = model.predict(X_cv_val)
                score = r2_score(y_cv_val, y_cv_pred)
                scores.append(score)
                
            return np.mean(scores)
            
        # Run Optuna optimization
        with mlflow.start_run(run_name=f"{symbol}_{model_name}_bayesian_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            
            # Log experiment info
            mlflow.log_param("symbol", symbol)
            mlflow.log_param("model_type", model_name)
            mlflow.log_param("tuning_method", "bayesian_optimization")
            mlflow.log_param("n_trials", n_trials)
            
            # Create study
            study = optuna.create_study(
                direction='maximize',
                study_name=f"{symbol}_{model_name}_optimization"
            )
            
            # Optimize
            start_time = datetime.now()
            study.optimize(objective, n_trials=n_trials)
            end_time = datetime.now()
            
            # Get best parameters and train final model
            best_params = study.best_params
            
            if model_name == 'random_forest':
                best_model = RandomForestRegressor(**best_params, random_state=42)
            elif model_name == 'gradient_boosting':
                best_model = GradientBoostingRegressor(**best_params, random_state=42)
                
            # Final evaluation
            scaler = StandardScaler()
            metrics = self.evaluate_model(best_model, X_train, X_test, y_train, y_test, scaler)
            
            # Log results
            mlflow.log_param("best_params", str(best_params))
            mlflow.log_metric("best_cv_score", study.best_value)
            mlflow.log_metric("n_trials_completed", len(study.trials))
            mlflow.log_metric("tuning_time_minutes", (end_time - start_time).total_seconds() / 60)
            
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
                
            # Log artifacts
            mlflow.sklearn.log_model(best_model, "best_model")
            mlflow.sklearn.log_model(scaler, "scaler")
            
            # Save optimization history
            trials_df = study.trials_dataframe()
            trials_df.to_csv("optimization_trials.csv", index=False)
            mlflow.log_artifact("optimization_trials.csv")
            
            results = {
                'best_model': best_model,
                'best_params': best_params,
                'best_score': study.best_value,
                'metrics': metrics,
                'scaler': scaler,
                'study': study
            }
            
            logger.info(f"Bayesian optimization completed for {symbol}. Best R²: {study.best_value:.4f}")
            return results
            
    def compare_models(self, symbol: str, X_train, X_test, y_train, y_test,
                      models_to_compare: List[str] = None, 
                      tuning_method: str = "random_search") -> Dict[str, Any]:
        """Compare multiple models with hyperparameter tuning"""
        
        if models_to_compare is None:
            models_to_compare = ['random_forest', 'gradient_boosting', 'ridge']
            
        logger.info(f"Comparing {len(models_to_compare)} models for {symbol}")
        
        experiment_id = self.create_experiment(symbol, "model_comparison")
        
        results = {}
        
        with mlflow.start_run(run_name=f"{symbol}_model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            
            # Log comparison parameters
            mlflow.log_param("symbol", symbol)
            mlflow.log_param("comparison_method", tuning_method)
            mlflow.log_param("models_compared", str(models_to_compare))
            
            model_results = []
            
            for model_name in models_to_compare:
                logger.info(f"Tuning {model_name} for {symbol}")
                
                try:
                    if tuning_method == "grid_search":
                        result = self.grid_search_tuning(symbol, X_train, X_test, y_train, y_test, model_name)
                    elif tuning_method == "random_search":
                        result = self.random_search_tuning(symbol, X_train, X_test, y_train, y_test, model_name)
                    elif tuning_method == "bayesian_optimization":
                        result = self.bayesian_optimization_tuning(symbol, X_train, X_test, y_train, y_test, model_name)
                    else:
                        raise ValueError(f"Unknown tuning method: {tuning_method}")
                        
                    result['model_name'] = model_name
                    results[model_name] = result
                    model_results.append(result)
                    
                    # Log individual model results
                    mlflow.log_metric(f"{model_name}_test_r2", result['metrics']['test_r2'])
                    mlflow.log_metric(f"{model_name}_test_mae", result['metrics']['test_mae'])
                    
                except Exception as e:
                    logger.error(f"Failed to tune {model_name}: {e}")
                    continue
                    
            # Find best model
            if model_results:
                best_result = max(model_results, key=lambda x: x['metrics']['test_r2'])
                
                mlflow.log_param("best_model_name", best_result['model_name'])
                mlflow.log_metric("best_model_test_r2", best_result['metrics']['test_r2'])
                
                # Log best model
                mlflow.sklearn.log_model(
                    best_result['best_model'], 
                    "champion_model",
                    registered_model_name=f"{symbol}_champion_predictor"
                )
                mlflow.sklearn.log_model(best_result['scaler'], "champion_scaler")
                
                results['champion'] = best_result
                
                logger.info(f"Model comparison completed. Champion: {best_result['model_name']} (R²: {best_result['metrics']['test_r2']:.4f})")
                
        return results

# Async wrapper for production use
class AsyncHyperparameterTuner:
    """Async wrapper for hyperparameter tuning in production environments"""
    
    def __init__(self, mlflow_tracking_uri: str = "http://localhost:5001"):
        self.tuner = AdvancedHyperparameterTuner(mlflow_tracking_uri)
        
    async def tune_model_async(self, symbol: str, X_train, X_test, y_train, y_test,
                              method: str = "random_search", model_name: str = "random_forest") -> Dict[str, Any]:
        """Async wrapper for hyperparameter tuning"""
        
        loop = asyncio.get_event_loop()
        
        if method == "grid_search":
            result = await loop.run_in_executor(
                None, 
                self.tuner.grid_search_tuning,
                symbol, X_train, X_test, y_train, y_test, model_name
            )
        elif method == "random_search":
            result = await loop.run_in_executor(
                None,
                self.tuner.random_search_tuning, 
                symbol, X_train, X_test, y_train, y_test, model_name
            )
        elif method == "bayesian_optimization":
            result = await loop.run_in_executor(
                None,
                self.tuner.bayesian_optimization_tuning,
                symbol, X_train, X_test, y_train, y_test, model_name
            )
        else:
            raise ValueError(f"Unknown tuning method: {method}")
            
        return result
        
    async def compare_models_async(self, symbol: str, X_train, X_test, y_train, y_test,
                                 models_to_compare: List[str] = None,
                                 tuning_method: str = "random_search") -> Dict[str, Any]:
        """Async model comparison"""
        
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            self.tuner.compare_models,
            symbol, X_train, X_test, y_train, y_test, models_to_compare, tuning_method
        )
        
        return result

if __name__ == "__main__":
    # Example usage
    print("Advanced Hyperparameter Tuning System Initialized")
    print("Available tuning methods:")
    print("  - Grid Search: Exhaustive search over parameter grid")
    print("  - Random Search: Random sampling for faster optimization") 
    print("  - Bayesian Optimization: Intelligent search using Optuna")
    print("  - Model Comparison: Compare multiple algorithms with tuning")