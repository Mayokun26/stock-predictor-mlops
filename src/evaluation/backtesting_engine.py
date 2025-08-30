#!/usr/bin/env python3
"""
Production Backtesting Engine for MLOps Stock Prediction System
Implements walk-forward validation with financial metrics and risk analysis
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass, asdict
import yfinance as yf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

@dataclass
class BacktestResult:
    """Comprehensive backtesting results with financial and statistical metrics"""
    
    # Time period
    start_date: str
    end_date: str
    total_days: int
    trading_days: int
    
    # Statistical Metrics
    rmse: float
    mae: float
    r2_score: float
    mape: float
    directional_accuracy: float
    
    # Financial Metrics
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    max_drawdown: float
    calmar_ratio: float
    win_rate: float
    profit_factor: float
    
    # Risk Metrics
    volatility: float
    downside_deviation: float
    sortino_ratio: float
    var_95: float
    cvar_95: float
    
    # Trade Analytics
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_trade_return: float
    best_trade: float
    worst_trade: float
    
    # Model Performance
    prediction_count: int
    avg_prediction_confidence: float
    feature_importance: Dict[str, float]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def summary(self) -> str:
        """Generate human-readable summary"""
        return f"""
Backtesting Results ({self.start_date} to {self.end_date})
========================================================
Statistical Performance:
  - RMSE: {self.rmse:.4f}
  - R² Score: {self.r2_score:.4f} 
  - Directional Accuracy: {self.directional_accuracy:.2%}

Financial Performance:
  - Total Return: {self.total_return:.2%}
  - Annualized Return: {self.annualized_return:.2%}
  - Sharpe Ratio: {self.sharpe_ratio:.2f}
  - Max Drawdown: {self.max_drawdown:.2%}
  
Risk Metrics:
  - Volatility: {self.volatility:.2%}
  - Sortino Ratio: {self.sortino_ratio:.2f}
  - VaR (95%): {self.var_95:.2%}
        """


class BacktestingEngine:
    """
    Production-grade backtesting engine with walk-forward validation
    """
    
    def __init__(self, 
                 initial_capital: float = 100000.0,
                 transaction_cost: float = 0.001,
                 min_training_days: int = 252,
                 rebalance_frequency: str = 'daily'):
        """
        Initialize backtesting engine
        
        Args:
            initial_capital: Starting capital for backtesting
            transaction_cost: Transaction cost as percentage (0.001 = 0.1%)
            min_training_days: Minimum days needed for model training
            rebalance_frequency: How often to rebalance ('daily', 'weekly', 'monthly')
        """
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.min_training_days = min_training_days
        self.rebalance_frequency = rebalance_frequency
        
        # Results storage
        self.results: List[BacktestResult] = []
        self.trade_log: List[Dict] = []
        
    def fetch_historical_data(self, 
                            symbol: str, 
                            start_date: str, 
                            end_date: str) -> pd.DataFrame:
        """
        Fetch historical stock data for backtesting
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            DataFrame with OHLCV data and additional features
        """
        try:
            # Fetch data from Yahoo Finance
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date, interval='1d')
            
            if data.empty:
                raise ValueError(f"No data found for {symbol}")
            
            # Calculate technical indicators
            data = self._add_technical_indicators(data)
            
            # Add forward returns for backtesting
            data['forward_return_1d'] = data['Close'].pct_change().shift(-1)
            data['forward_price_1d'] = data['Close'].shift(-1)
            
            # Clean data
            data = data.dropna()
            
            logger.info(f"Fetched {len(data)} days of data for {symbol}")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            raise
    
    def _add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators for feature engineering"""
        
        # Price-based indicators
        data['sma_5'] = data['Close'].rolling(5).mean()
        data['sma_20'] = data['Close'].rolling(20).mean()
        data['sma_50'] = data['Close'].rolling(50).mean()
        
        data['ema_12'] = data['Close'].ewm(span=12).mean()
        data['ema_26'] = data['Close'].ewm(span=26).mean()
        
        # Momentum indicators
        data['rsi_14'] = self._calculate_rsi(data['Close'], 14)
        data['macd'] = data['ema_12'] - data['ema_26']
        data['macd_signal'] = data['macd'].ewm(span=9).mean()
        
        # Volatility indicators
        data['volatility_20'] = data['Close'].pct_change().rolling(20).std()
        data['atr_14'] = self._calculate_atr(data, 14)
        
        # Volume indicators
        data['volume_sma_20'] = data['Volume'].rolling(20).mean()
        data['volume_ratio'] = data['Volume'] / data['volume_sma_20']
        
        # Price position indicators
        data['price_position'] = (data['Close'] - data['sma_20']) / data['sma_20']
        data['high_low_ratio'] = (data['High'] - data['Low']) / data['Close']
        
        return data
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_atr(self, data: pd.DataFrame, window: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high_low = data['High'] - data['Low']
        high_close = np.abs(data['High'] - data['Close'].shift())
        low_close = np.abs(data['Low'] - data['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        atr = true_range.rolling(window).mean()
        return atr
    
    def walk_forward_backtest(self, 
                            symbol: str,
                            start_date: str, 
                            end_date: str,
                            model_class,
                            prediction_horizon: int = 1) -> BacktestResult:
        """
        Perform walk-forward backtesting
        
        Args:
            symbol: Stock symbol to backtest
            start_date: Start date for backtesting
            end_date: End date for backtesting  
            model_class: ML model class to use for predictions
            prediction_horizon: Days ahead to predict
            
        Returns:
            BacktestResult with comprehensive metrics
        """
        logger.info(f"Starting walk-forward backtest for {symbol}")
        
        # Fetch historical data
        data = self.fetch_historical_data(symbol, start_date, end_date)
        
        # Initialize tracking variables
        portfolio_values = []
        predictions = []
        actuals = []
        trades = []
        positions = []
        
        current_capital = self.initial_capital
        current_position = 0  # 0 = cash, 1 = long, -1 = short
        
        # Feature columns (exclude target and metadata)
        feature_cols = [col for col in data.columns if col not in [
            'forward_return_1d', 'forward_price_1d', 'Open', 'High', 'Low', 'Close', 'Volume'
        ]]
        
        # Walk-forward validation
        for i in range(self.min_training_days, len(data) - prediction_horizon):
            current_date = data.index[i]
            
            # Training data (expanding window)
            train_data = data.iloc[:i]
            test_data = data.iloc[i:i+1]
            
            # Prepare features and targets
            X_train = train_data[feature_cols].dropna()
            y_train = train_data['forward_return_1d'].dropna()
            
            # Align training data
            min_len = min(len(X_train), len(y_train))
            X_train = X_train.iloc[-min_len:]
            y_train = y_train.iloc[-min_len:]
            
            if len(X_train) < 50:  # Need minimum training data
                continue
            
            # Train model
            try:
                model = model_class()
                model.fit(X_train, y_train)
                
                # Make prediction
                X_test = test_data[feature_cols]
                if X_test.isnull().any().any():
                    prediction = 0  # Conservative prediction for missing data
                else:
                    prediction = model.predict(X_test)[0]
                    
            except Exception as e:
                logger.warning(f"Model training failed at {current_date}: {e}")
                prediction = 0
            
            # Get actual return
            actual_return = data.iloc[i]['forward_return_1d']
            current_price = data.iloc[i]['Close']
            
            # Trading logic
            trade_signal = self._generate_trade_signal(prediction, actual_return)
            
            # Execute trade
            if trade_signal != current_position:
                # Record trade
                trade = {
                    'date': current_date,
                    'symbol': symbol,
                    'action': 'BUY' if trade_signal == 1 else 'SELL' if trade_signal == -1 else 'HOLD',
                    'price': current_price,
                    'position_before': current_position,
                    'position_after': trade_signal,
                    'capital_before': current_capital,
                    'prediction': prediction,
                    'actual_return': actual_return
                }
                
                # Apply transaction costs
                if current_position != trade_signal:
                    current_capital *= (1 - self.transaction_cost)
                
                current_position = trade_signal
                trade['capital_after'] = current_capital
                trades.append(trade)
            
            # Update portfolio value based on position
            if current_position == 1:  # Long position
                current_capital *= (1 + actual_return)
            elif current_position == -1:  # Short position
                current_capital *= (1 - actual_return)
            # If position == 0 (cash), capital remains unchanged
            
            # Record results
            predictions.append(prediction)
            actuals.append(actual_return)
            portfolio_values.append(current_capital)
            positions.append(current_position)
        
        # Calculate comprehensive metrics
        result = self._calculate_backtest_metrics(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            predictions=predictions,
            actuals=actuals,
            portfolio_values=portfolio_values,
            trades=trades,
            initial_capital=self.initial_capital
        )
        
        self.results.append(result)
        self.trade_log.extend(trades)
        
        logger.info(f"Backtest completed. Final capital: ${current_capital:,.2f}")
        return result
    
    def _generate_trade_signal(self, prediction: float, actual_return: float) -> int:
        """
        Generate trade signal based on prediction
        
        Args:
            prediction: Model prediction
            actual_return: Actual return (for analysis)
            
        Returns:
            Trade signal: 1 (long), -1 (short), 0 (hold cash)
        """
        threshold = 0.01  # 1% threshold for trading
        
        if prediction > threshold:
            return 1  # Long position
        elif prediction < -threshold:
            return -1  # Short position
        else:
            return 0  # Hold cash
    
    def _calculate_backtest_metrics(self, 
                                  symbol: str,
                                  start_date: str,
                                  end_date: str,
                                  predictions: List[float],
                                  actuals: List[float],
                                  portfolio_values: List[float],
                                  trades: List[Dict],
                                  initial_capital: float) -> BacktestResult:
        """Calculate comprehensive backtesting metrics"""
        
        predictions_arr = np.array(predictions)
        actuals_arr = np.array(actuals)
        portfolio_arr = np.array(portfolio_values)
        
        # Time metrics
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        total_days = (end_dt - start_dt).days
        trading_days = len(portfolio_values)
        
        # Statistical metrics
        rmse = np.sqrt(mean_squared_error(actuals_arr, predictions_arr))
        mae = mean_absolute_error(actuals_arr, predictions_arr)
        r2 = r2_score(actuals_arr, predictions_arr)
        mape = np.mean(np.abs((actuals_arr - predictions_arr) / actuals_arr)) * 100
        
        # Directional accuracy
        actual_direction = np.sign(actuals_arr)
        predicted_direction = np.sign(predictions_arr)
        directional_accuracy = np.mean(actual_direction == predicted_direction)
        
        # Financial metrics
        portfolio_returns = np.diff(portfolio_arr) / portfolio_arr[:-1]
        total_return = (portfolio_arr[-1] - initial_capital) / initial_capital
        
        # Annualized return
        years = trading_days / 252
        annualized_return = (portfolio_arr[-1] / initial_capital) ** (1/years) - 1
        
        # Risk metrics
        portfolio_volatility = np.std(portfolio_returns) * np.sqrt(252)
        
        # Sharpe ratio (assume 0% risk-free rate)
        sharpe_ratio = annualized_return / portfolio_volatility if portfolio_volatility > 0 else 0
        
        # Drawdown analysis
        cumulative = portfolio_arr / initial_capital
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Downside deviation
        negative_returns = portfolio_returns[portfolio_returns < 0]
        downside_deviation = np.std(negative_returns) * np.sqrt(252) if len(negative_returns) > 0 else 0
        
        # Sortino ratio
        sortino_ratio = annualized_return / downside_deviation if downside_deviation > 0 else 0
        
        # VaR and CVaR
        var_95 = np.percentile(portfolio_returns, 5)
        cvar_95 = np.mean(portfolio_returns[portfolio_returns <= var_95])
        
        # Trade analytics
        total_trades = len(trades)
        trade_returns = []
        
        for trade in trades:
            if 'return' in trade:
                trade_returns.append(trade['return'])
        
        if trade_returns:
            winning_trades = sum(1 for r in trade_returns if r > 0)
            losing_trades = sum(1 for r in trade_returns if r < 0)
            win_rate = winning_trades / len(trade_returns) if trade_returns else 0
            
            positive_returns = [r for r in trade_returns if r > 0]
            negative_returns = [r for r in trade_returns if r < 0]
            
            gross_profit = sum(positive_returns) if positive_returns else 0
            gross_loss = abs(sum(negative_returns)) if negative_returns else 0
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            avg_trade_return = np.mean(trade_returns)
            best_trade = max(trade_returns) if trade_returns else 0
            worst_trade = min(trade_returns) if trade_returns else 0
        else:
            winning_trades = losing_trades = 0
            win_rate = profit_factor = avg_trade_return = best_trade = worst_trade = 0
        
        # Feature importance (placeholder - would come from model)
        feature_importance = {
            'technical_indicators': 0.4,
            'momentum': 0.3,
            'volatility': 0.2,
            'volume': 0.1
        }
        
        return BacktestResult(
            start_date=start_date,
            end_date=end_date,
            total_days=total_days,
            trading_days=trading_days,
            rmse=rmse,
            mae=mae,
            r2_score=r2,
            mape=mape,
            directional_accuracy=directional_accuracy,
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            calmar_ratio=calmar_ratio,
            win_rate=win_rate,
            profit_factor=profit_factor,
            volatility=portfolio_volatility,
            downside_deviation=downside_deviation,
            sortino_ratio=sortino_ratio,
            var_95=var_95,
            cvar_95=cvar_95,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            avg_trade_return=avg_trade_return,
            best_trade=best_trade,
            worst_trade=worst_trade,
            prediction_count=len(predictions),
            avg_prediction_confidence=0.75,  # Placeholder
            feature_importance=feature_importance
        )
    
    def run_multi_symbol_backtest(self, 
                                symbols: List[str],
                                start_date: str,
                                end_date: str,
                                model_class) -> Dict[str, BacktestResult]:
        """Run backtesting across multiple symbols"""
        results = {}
        
        for symbol in symbols:
            try:
                logger.info(f"Running backtest for {symbol}")
                result = self.walk_forward_backtest(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    model_class=model_class
                )
                results[symbol] = result
                
            except Exception as e:
                logger.error(f"Backtest failed for {symbol}: {e}")
                continue
        
        return results
    
    def generate_performance_report(self, results: Dict[str, BacktestResult]) -> str:
        """Generate comprehensive performance report"""
        
        report = """
MLOps Stock Prediction Backtesting Report
==========================================

"""
        
        # Summary statistics
        total_symbols = len(results)
        avg_sharpe = np.mean([r.sharpe_ratio for r in results.values()])
        avg_return = np.mean([r.annualized_return for r in results.values()])
        avg_accuracy = np.mean([r.directional_accuracy for r in results.values()])
        
        report += f"""
Portfolio Summary:
  - Total Symbols Tested: {total_symbols}
  - Average Sharpe Ratio: {avg_sharpe:.2f}
  - Average Annualized Return: {avg_return:.2%}
  - Average Directional Accuracy: {avg_accuracy:.2%}

Individual Results:
"""
        
        for symbol, result in results.items():
            report += f"""
{symbol}:
  - Return: {result.annualized_return:.2%} (Sharpe: {result.sharpe_ratio:.2f})
  - Max Drawdown: {result.max_drawdown:.2%}
  - Directional Accuracy: {result.directional_accuracy:.2%}
  - Total Trades: {result.total_trades}
"""
        
        return report


# Simple model classes for backtesting
class SimpleMovingAverageModel:
    """Simple moving average crossover model for backtesting"""
    
    def fit(self, X, y):
        # No actual training needed for MA model
        pass
    
    def predict(self, X):
        # Simple momentum prediction based on MA
        if 'sma_5' in X.columns and 'sma_20' in X.columns:
            ma_signal = (X['sma_5'] - X['sma_20']) / X['sma_20']
            return ma_signal.values
        return np.array([0])


class SimpleLinearModel:
    """Simple linear regression model for backtesting"""
    
    def __init__(self):
        from sklearn.linear_model import LinearRegression
        self.model = LinearRegression()
    
    def fit(self, X, y):
        # Handle NaN values
        X_clean = X.fillna(0)
        self.model.fit(X_clean, y)
    
    def predict(self, X):
        X_clean = X.fillna(0)
        return self.model.predict(X_clean)


if __name__ == "__main__":
    # Example usage
    engine = BacktestingEngine(initial_capital=100000)
    
    # Test with Apple stock
    result = engine.walk_forward_backtest(
        symbol="AAPL",
        start_date="2020-01-01", 
        end_date="2024-01-01",
        model_class=SimpleLinearModel
    )
    
    print(result.summary())