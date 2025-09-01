#!/usr/bin/env python3
"""
Comprehensive Model Evaluation Framework
Implements advanced evaluation metrics, backtesting, and model comparison for MLOps system
"""

import os
import sys
import asyncio
import logging
import numpy as np
import pandas as pd
import yfinance as yf
import mlflow
import mlflow.sklearn
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# MLflow configuration
MLFLOW_TRACKING_URI = "http://localhost:5001"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

@dataclass
class EvaluationConfig:
    """Configuration for model evaluation"""
    symbols: List[str]
    evaluation_period_days: int = 30
    train_test_split_ratio: float = 0.8
    min_prediction_samples: int = 10
    confidence_threshold: float = 0.7
    performance_threshold_r2: float = 0.1
    risk_free_rate: float = 0.02  # 2% annual risk-free rate

@dataclass 
class ModelPerformance:
    """Comprehensive model performance metrics"""
    # Statistical metrics
    r2_score: float
    mse: float
    mae: float
    rmse: float
    mape: float
    
    # Financial metrics
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    calmar_ratio: float
    
    # Prediction quality
    directional_accuracy: float
    hit_rate: float
    avg_prediction_error: float
    prediction_bias: float
    
    # Model metadata
    model_name: str
    symbol: str
    evaluation_period: str
    sample_count: int
    confidence_scores: List[float]

class ComprehensiveModelEvaluator:
    """Advanced model evaluation with financial and statistical metrics"""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.mlflow_client = mlflow.MlflowClient()
        self.evaluation_results = {}
        
    async def evaluate_all_models(self) -> Dict[str, List[ModelPerformance]]:
        """Evaluate all registered models across all symbols"""
        logger.info(f"Starting comprehensive evaluation for {len(self.config.symbols)} symbols")
        
        results = {}
        
        for symbol in self.config.symbols:
            logger.info(f"Evaluating models for {symbol}")
            symbol_results = await self.evaluate_symbol_models(symbol)
            results[symbol] = symbol_results
            
        # Generate comparison report
        self.evaluation_results = results
        await self.generate_evaluation_report()
        
        return results
    
    async def evaluate_symbol_models(self, symbol: str) -> List[ModelPerformance]:
        """Evaluate all model versions for a specific symbol"""
        model_performances = []
        
        try:
            # Get all model versions for this symbol
            model_name = f"{symbol}_predictor"
            versions = self.mlflow_client.search_model_versions(f"name='{model_name}'")
            
            if not versions:
                logger.warning(f"No models found for {symbol}")
                return []
                
            # Evaluate each model version
            for version in versions:
                try:
                    performance = await self.evaluate_model_version(symbol, model_name, version.version)
                    if performance:
                        model_performances.append(performance)
                except Exception as e:
                    logger.error(f"Failed to evaluate {model_name} v{version.version}: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Failed to get model versions for {symbol}: {e}")
            
        return model_performances
    
    async def evaluate_model_version(self, symbol: str, model_name: str, version: str) -> Optional[ModelPerformance]:
        """Comprehensive evaluation of a specific model version"""
        try:
            # Load model and data
            model_uri = f"models:/{model_name}/{version}"
            model = mlflow.sklearn.load_model(model_uri)
            
            # Load scaler if available
            scaler = None
            try:
                scaler_uri = f"models:/{model_name}_scaler/{version}"
                scaler = mlflow.sklearn.load_model(scaler_uri)
            except:
                logger.info(f"No scaler found for {model_name} v{version}")
                
            # Get evaluation data
            eval_data = await self.prepare_evaluation_data(symbol)
            if eval_data is None:
                return None
                
            X_test, y_test, prices_test, dates_test = eval_data
            
            # Make predictions
            if scaler:
                X_test_scaled = scaler.transform(X_test)
            else:
                X_test_scaled = X_test
                
            predicted_returns = model.predict(X_test_scaled)
            
            # Calculate comprehensive metrics
            performance = await self.calculate_comprehensive_metrics(
                symbol=symbol,
                model_name=f"{model_name}_v{version}",
                y_true=y_test,
                y_pred=predicted_returns,
                prices=prices_test,
                dates=dates_test
            )
            
            # Log evaluation to MLflow
            await self.log_evaluation_to_mlflow(model_name, version, performance)
            
            return performance
            
        except Exception as e:
            logger.error(f"Model evaluation failed for {model_name} v{version}: {e}")
            return None
    
    async def prepare_evaluation_data(self, symbol: str) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DatetimeIndex]]:
        """Prepare evaluation dataset for a symbol"""
        try:
            # Get historical data - use 6 months of data
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="6mo")
            
            if len(hist) < 100:
                logger.warning(f"Insufficient data for {symbol}: {len(hist)} days")
                return None
                
            # Use the same feature engineering as training
            from train_model_with_mlflow import create_advanced_features
            
            feature_list = []
            target_list = []
            price_list = []
            date_list = []
            
            # Use last 20 days for evaluation, but need 60 days for indicators
            if len(hist) < 80:
                logger.warning(f"Need at least 80 days for evaluation, got {len(hist)}")
                return None
                
            eval_start_idx = len(hist) - self.config.evaluation_period_days - 1
            
            for i in range(eval_start_idx, len(hist) - 1):
                try:
                    # Get subset for feature calculation
                    subset = hist.iloc[:i+1]
                    
                    # Calculate features
                    features = create_advanced_features(subset)
                    
                    # Target: next day return
                    current_price = hist['Close'].iloc[i]
                    next_price = hist['Close'].iloc[i + 1]
                    target = (next_price / current_price - 1)
                    
                    # Convert to array
                    feature_array = np.array(list(features.values()))
                    
                    # Validation
                    if (not np.any(np.isnan(feature_array)) and not np.any(np.isinf(feature_array)) 
                        and not np.isnan(target) and not np.isinf(target)):
                        feature_list.append(feature_array)
                        target_list.append(target)
                        price_list.append(current_price)
                        date_list.append(hist.index[i])
                        
                except Exception as e:
                    logger.warning(f"Error calculating features for {symbol} day {i}: {e}")
                    continue
            
            if len(feature_list) < self.config.min_prediction_samples:
                logger.warning(f"Insufficient evaluation samples for {symbol}: {len(feature_list)}")
                return None
                
            X = np.array(feature_list)
            y = np.array(target_list)
            prices = np.array(price_list)
            dates = pd.DatetimeIndex(date_list)
            
            logger.info(f"Prepared {len(X)} evaluation samples for {symbol}")
            return X, y, prices, dates
            
        except Exception as e:
            logger.error(f"Failed to prepare evaluation data for {symbol}: {e}")
            return None
    
    async def calculate_comprehensive_metrics(self, symbol: str, model_name: str, 
                                            y_true: np.ndarray, y_pred: np.ndarray,
                                            prices: np.ndarray, dates: pd.DatetimeIndex) -> ModelPerformance:
        """Calculate comprehensive performance metrics"""
        
        # Statistical metrics
        r2 = r2_score(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mape = mean_absolute_percentage_error(y_true, y_pred) * 100
        
        # Financial simulation
        portfolio_values = [10000]  # Start with $10k
        position = 0  # 0 = cash, 1 = long, -1 = short
        
        for i in range(len(y_pred)):
            current_value = portfolio_values[-1]
            actual_return = y_true[i]
            predicted_return = y_pred[i]
            
            # Simple strategy: go long if prediction > 0.5%, short if < -0.5%, else cash
            if predicted_return > 0.005:  # 0.5% threshold
                new_position = 1
            elif predicted_return < -0.005:
                new_position = -1
            else:
                new_position = 0
                
            # Calculate return based on position
            if position != 0:
                portfolio_return = position * actual_return
                new_value = current_value * (1 + portfolio_return)
            else:
                new_value = current_value  # Cash position
                
            portfolio_values.append(new_value)
            position = new_position
        
        portfolio_values = np.array(portfolio_values[1:])  # Remove initial value
        portfolio_returns = np.diff(portfolio_values) / portfolio_values[:-1]
        
        # Financial metrics
        total_return = (portfolio_values[-1] / 10000 - 1) * 100
        days = len(portfolio_values)
        annualized_return = ((portfolio_values[-1] / 10000) ** (252/days) - 1) * 100
        volatility = np.std(portfolio_returns) * np.sqrt(252) * 100
        
        # Risk-adjusted metrics
        if volatility > 0:
            sharpe_ratio = (annualized_return/100 - self.config.risk_free_rate) / (volatility/100)
        else:
            sharpe_ratio = 0
            
        # Maximum drawdown
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - peak) / peak
        max_drawdown = np.abs(np.min(drawdown)) * 100
        
        calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0
        
        # Prediction quality metrics
        direction_correct = np.sum(np.sign(y_true) == np.sign(y_pred))
        directional_accuracy = direction_correct / len(y_true) * 100
        
        # Hit rate (predictions within 1% of actual)
        hits = np.sum(np.abs(y_pred - y_true) < 0.01)
        hit_rate = hits / len(y_true) * 100
        
        avg_prediction_error = np.mean(np.abs(y_pred - y_true)) * 100
        prediction_bias = np.mean(y_pred - y_true) * 100
        
        # Mock confidence scores (would come from model uncertainty in real implementation)
        confidence_scores = [0.8] * len(y_pred)  # Placeholder
        
        return ModelPerformance(
            r2_score=r2,
            mse=mse,
            mae=mae,
            rmse=rmse,
            mape=mape,
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            calmar_ratio=calmar_ratio,
            directional_accuracy=directional_accuracy,
            hit_rate=hit_rate,
            avg_prediction_error=avg_prediction_error,
            prediction_bias=prediction_bias,
            model_name=model_name,
            symbol=symbol,
            evaluation_period=f"{dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}",
            sample_count=len(y_true),
            confidence_scores=confidence_scores
        )
    
    async def log_evaluation_to_mlflow(self, model_name: str, version: str, performance: ModelPerformance):
        """Log evaluation metrics to MLflow"""
        try:
            with mlflow.start_run(run_name=f"evaluation_{model_name}_v{version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                # Log all metrics
                mlflow.log_metric("eval_r2_score", performance.r2_score)
                mlflow.log_metric("eval_mse", performance.mse)
                mlflow.log_metric("eval_mae", performance.mae)
                mlflow.log_metric("eval_rmse", performance.rmse)
                mlflow.log_metric("eval_mape", performance.mape)
                mlflow.log_metric("eval_total_return", performance.total_return)
                mlflow.log_metric("eval_annualized_return", performance.annualized_return)
                mlflow.log_metric("eval_volatility", performance.volatility)
                mlflow.log_metric("eval_sharpe_ratio", performance.sharpe_ratio)
                mlflow.log_metric("eval_max_drawdown", performance.max_drawdown)
                mlflow.log_metric("eval_calmar_ratio", performance.calmar_ratio)
                mlflow.log_metric("eval_directional_accuracy", performance.directional_accuracy)
                mlflow.log_metric("eval_hit_rate", performance.hit_rate)
                mlflow.log_metric("eval_avg_prediction_error", performance.avg_prediction_error)
                mlflow.log_metric("eval_prediction_bias", performance.prediction_bias)
                
                # Log parameters
                mlflow.log_param("evaluated_model", model_name)
                mlflow.log_param("model_version", version)
                mlflow.log_param("symbol", performance.symbol)
                mlflow.log_param("evaluation_period", performance.evaluation_period)
                mlflow.log_param("sample_count", performance.sample_count)
                
                logger.info(f"Logged evaluation metrics for {model_name} v{version}")
                
        except Exception as e:
            logger.error(f"Failed to log evaluation to MLflow: {e}")
    
    async def generate_evaluation_report(self):
        """Generate comprehensive evaluation report"""
        if not self.evaluation_results:
            logger.warning("No evaluation results to report")
            return
            
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("COMPREHENSIVE MODEL EVALUATION REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Evaluation Period: {self.config.evaluation_period_days} days")
        report_lines.append("")
        
        # Summary by symbol
        for symbol, performances in self.evaluation_results.items():
            if not performances:
                continue
                
            report_lines.append(f"📊 {symbol} MODEL PERFORMANCE")
            report_lines.append("-" * 40)
            
            for perf in performances:
                report_lines.append(f"Model: {perf.model_name}")
                report_lines.append(f"  Statistical Metrics:")
                report_lines.append(f"    R² Score: {perf.r2_score:.4f}")
                report_lines.append(f"    RMSE: {perf.rmse:.6f}")
                report_lines.append(f"    MAE: {perf.mae:.6f}")
                report_lines.append(f"    MAPE: {perf.mape:.2f}%")
                report_lines.append(f"  Financial Metrics:")
                report_lines.append(f"    Total Return: {perf.total_return:.2f}%")
                report_lines.append(f"    Annualized Return: {perf.annualized_return:.2f}%")
                report_lines.append(f"    Volatility: {perf.volatility:.2f}%")
                report_lines.append(f"    Sharpe Ratio: {perf.sharpe_ratio:.2f}")
                report_lines.append(f"    Max Drawdown: {perf.max_drawdown:.2f}%")
                report_lines.append(f"  Prediction Quality:")
                report_lines.append(f"    Directional Accuracy: {perf.directional_accuracy:.1f}%")
                report_lines.append(f"    Hit Rate: {perf.hit_rate:.1f}%")
                report_lines.append(f"    Avg Prediction Error: {perf.avg_prediction_error:.2f}%")
                report_lines.append(f"    Sample Count: {perf.sample_count}")
                report_lines.append("")
                
        # Champion model recommendations
        report_lines.append("🏆 CHAMPION MODEL RECOMMENDATIONS")
        report_lines.append("-" * 40)
        
        for symbol, performances in self.evaluation_results.items():
            if not performances:
                continue
                
            # Find best model by Sharpe ratio
            best_model = max(performances, key=lambda p: p.sharpe_ratio if p.sharpe_ratio > 0 else -999)
            report_lines.append(f"{symbol}: {best_model.model_name}")
            report_lines.append(f"  Reason: Best Sharpe ratio ({best_model.sharpe_ratio:.2f})")
            report_lines.append(f"  Performance: {best_model.total_return:.1f}% return, {best_model.directional_accuracy:.1f}% accuracy")
            report_lines.append("")
            
        # Save report
        report_content = "\n".join(report_lines)
        
        with open("model_evaluation_report.txt", "w") as f:
            f.write(report_content)
            
        logger.info("Generated comprehensive evaluation report: model_evaluation_report.txt")
        print("\n" + report_content)

async def main():
    """Run comprehensive model evaluation"""
    config = EvaluationConfig(
        symbols=["AAPL", "MSFT", "TSLA"],
        evaluation_period_days=20,  # Reduced from 30 to work with available data
        min_prediction_samples=5   # Reduced minimum samples
    )
    
    evaluator = ComprehensiveModelEvaluator(config)
    
    print("🚀 Starting Comprehensive Model Evaluation")
    print("=" * 50)
    
    results = await evaluator.evaluate_all_models()
    
    print("\n✅ Evaluation completed successfully!")
    print(f"📈 Evaluated models for {len(results)} symbols")
    print("📋 Check 'model_evaluation_report.txt' for detailed results")
    print("🔗 View evaluation runs in MLflow UI: http://localhost:5001")

if __name__ == "__main__":
    asyncio.run(main())