#!/usr/bin/env python3
"""
Production Model Evaluation Framework
Comprehensive evaluation with business metrics, statistical tests, and automated decisions
"""
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from dataclasses import dataclass, asdict
from enum import Enum

import psycopg2
from psycopg2.extras import RealDictCursor
from scipy import stats
from scipy.stats import jarque_bera, shapiro, kstest
import mlflow
import mlflow.sklearn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

class EvaluationStatus(Enum):
    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"
    PENDING = "pending"

class ModelPerformanceTier(Enum):
    CHAMPION = "champion"
    CHALLENGER_WINNING = "challenger_winning"
    CHALLENGER_ACCEPTABLE = "challenger_acceptable"
    CHALLENGER_FAILING = "challenger_failing"
    DEPRECATED = "deprecated"

@dataclass
class EvaluationThresholds:
    """Production evaluation thresholds for model performance"""
    # Statistical Metrics
    min_rmse_threshold: float = 10.0  # Maximum acceptable RMSE
    min_mae_threshold: float = 5.0    # Maximum acceptable MAE
    min_r2_score: float = 0.3         # Minimum R² score
    min_mape_threshold: float = 0.15  # Maximum acceptable MAPE (15%)
    
    # Business Metrics
    min_roi_threshold: float = 0.02   # Minimum ROI (2%)
    max_drawdown_threshold: float = 0.1  # Maximum drawdown (10%)
    min_sharpe_ratio: float = 0.5     # Minimum Sharpe ratio
    min_accuracy_1pct: float = 0.3    # 30% predictions within 1%
    min_accuracy_5pct: float = 0.6    # 60% predictions within 5%
    
    # Operational Metrics
    max_latency_p95_ms: float = 1000  # 95th percentile latency
    max_latency_p99_ms: float = 2000  # 99th percentile latency
    min_availability: float = 0.995   # 99.5% availability
    
    # Data Quality
    min_data_freshness_hours: float = 24  # Data can't be older than 24h
    min_sample_size: int = 100           # Minimum sample size for evaluation
    max_missing_data_rate: float = 0.05  # Maximum 5% missing data

@dataclass
class EvaluationResult:
    """Complete evaluation result for a model"""
    model_version: str
    evaluation_id: str
    timestamp: datetime
    
    # Statistical Performance
    statistical_metrics: Dict[str, float]
    statistical_tests: Dict[str, Any]
    
    # Business Performance
    business_metrics: Dict[str, float]
    roi_analysis: Dict[str, float]
    
    # Operational Performance
    operational_metrics: Dict[str, float]
    
    # Data Quality Assessment
    data_quality: Dict[str, Any]
    
    # Overall Assessment
    overall_score: float
    performance_tier: ModelPerformanceTier
    status: EvaluationStatus
    issues: List[str]
    recommendations: List[str]
    
    # Comparison with Champion
    vs_champion: Optional[Dict[str, Any]]

class ModelEvaluator:
    """
    Enterprise-grade model evaluation system
    
    Features:
    - Comprehensive metric calculation (statistical, business, operational)
    - Automated performance tier classification
    - Champion/challenger comparison with statistical significance
    - Data quality assessment and drift detection
    - Business impact analysis with ROI calculations
    - Automated recommendations and decision support
    """
    
    def __init__(self, 
                 feature_store=None,
                 postgres_conn: psycopg2.connect = None,
                 thresholds: EvaluationThresholds = None):
        self.feature_store = feature_store
        self.postgres = postgres_conn
        self.thresholds = thresholds or EvaluationThresholds()
        self.champion_model_cache = {}
        
    async def initialize(self):
        """Initialize evaluation system tables and baseline data"""
        if not self.postgres:
            logger.warning("PostgreSQL connection not provided - limited functionality")
            return
            
        try:
            with self.postgres.cursor() as cur:
                # Model evaluations table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS model_evaluations (
                        id VARCHAR(100) PRIMARY KEY,
                        model_version VARCHAR(50) NOT NULL,
                        evaluation_timestamp TIMESTAMP NOT NULL,
                        statistical_metrics JSONB NOT NULL,
                        business_metrics JSONB NOT NULL,
                        operational_metrics JSONB NOT NULL,
                        data_quality JSONB NOT NULL,
                        overall_score FLOAT NOT NULL,
                        performance_tier VARCHAR(50) NOT NULL,
                        status VARCHAR(20) NOT NULL,
                        issues TEXT[],
                        recommendations TEXT[],
                        vs_champion JSONB,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Model performance history
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS model_performance_history (
                        id SERIAL PRIMARY KEY,
                        model_version VARCHAR(50) NOT NULL,
                        metric_name VARCHAR(100) NOT NULL,
                        metric_value FLOAT NOT NULL,
                        measurement_timestamp TIMESTAMP NOT NULL,
                        evaluation_period_hours INTEGER DEFAULT 24,
                        INDEX ON (model_version, metric_name, measurement_timestamp)
                    )
                """)
                
                # Business impact tracking
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS model_business_impact (
                        id SERIAL PRIMARY KEY,
                        model_version VARCHAR(50) NOT NULL,
                        trading_signal_date DATE NOT NULL,
                        predicted_return FLOAT,
                        actual_return FLOAT,
                        position_size FLOAT,
                        transaction_cost FLOAT,
                        net_pnl FLOAT,
                        cumulative_pnl FLOAT,
                        sharpe_ratio FLOAT,
                        max_drawdown FLOAT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Evaluation thresholds (configurable)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS evaluation_thresholds (
                        id SERIAL PRIMARY KEY,
                        threshold_name VARCHAR(100) NOT NULL,
                        threshold_value FLOAT NOT NULL,
                        threshold_type VARCHAR(50) NOT NULL,
                        description TEXT,
                        active BOOLEAN DEFAULT TRUE,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
            self.postgres.commit()
            await self._initialize_default_thresholds()
            logger.info("✅ Model Evaluator initialized")
            
        except Exception as e:
            logger.error(f"❌ Model Evaluator initialization failed: {e}")
            raise
    
    async def _initialize_default_thresholds(self):
        """Initialize default evaluation thresholds"""
        default_thresholds = [
            ('min_rmse_threshold', 10.0, 'statistical', 'Maximum acceptable RMSE'),
            ('min_mae_threshold', 5.0, 'statistical', 'Maximum acceptable MAE'),
            ('min_r2_score', 0.3, 'statistical', 'Minimum R² score'),
            ('min_roi_threshold', 0.02, 'business', 'Minimum ROI (2%)'),
            ('max_drawdown_threshold', 0.1, 'business', 'Maximum drawdown (10%)'),
            ('max_latency_p95_ms', 1000, 'operational', '95th percentile latency limit'),
        ]
        
        with self.postgres.cursor() as cur:
            for name, value, threshold_type, description in default_thresholds:
                cur.execute("""
                    INSERT INTO evaluation_thresholds (threshold_name, threshold_value, threshold_type, description)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT DO NOTHING
                """, (name, value, threshold_type, description))
        
        self.postgres.commit()
    
    async def evaluate_model(self,
                           model_version: str,
                           evaluation_period_hours: int = 24,
                           include_champion_comparison: bool = True) -> EvaluationResult:
        """
        Comprehensive model evaluation with all metrics and assessments
        """
        try:
            evaluation_id = f"eval_{model_version}_{int(datetime.now().timestamp())}"
            logger.info(f"Starting evaluation {evaluation_id} for model {model_version}")
            
            # Get evaluation data from recent predictions
            evaluation_data = await self._get_evaluation_data(model_version, evaluation_period_hours)
            
            if evaluation_data is None or len(evaluation_data) == 0:
                return EvaluationResult(
                    model_version=model_version,
                    evaluation_id=evaluation_id,
                    timestamp=datetime.now(),
                    statistical_metrics={},
                    statistical_tests={},
                    business_metrics={},
                    roi_analysis={},
                    operational_metrics={},
                    data_quality={'error': 'No evaluation data available'},
                    overall_score=0.0,
                    performance_tier=ModelPerformanceTier.CHALLENGER_FAILING,
                    status=EvaluationStatus.FAIL,
                    issues=["No evaluation data available"],
                    recommendations=["Ensure model is making predictions and actual values are being recorded"],
                    vs_champion=None
                )
            
            # 1. Statistical Performance Evaluation
            statistical_metrics = await self._calculate_statistical_metrics(evaluation_data)
            statistical_tests = await self._perform_statistical_tests(evaluation_data)
            
            # 2. Business Performance Evaluation
            business_metrics = await self._calculate_business_metrics(evaluation_data, model_version)
            roi_analysis = await self._calculate_roi_analysis(evaluation_data, model_version)
            
            # 3. Operational Performance Evaluation
            operational_metrics = await self._calculate_operational_metrics(evaluation_data, evaluation_period_hours)
            
            # 4. Data Quality Assessment
            data_quality = await self._assess_data_quality(evaluation_data)
            
            # 5. Overall Scoring and Classification
            overall_score = self._calculate_overall_score(
                statistical_metrics, business_metrics, operational_metrics, data_quality
            )
            
            performance_tier = self._classify_performance_tier(
                overall_score, statistical_metrics, business_metrics
            )
            
            # 6. Issues and Recommendations
            issues = self._identify_issues(statistical_metrics, business_metrics, operational_metrics, data_quality)
            recommendations = self._generate_recommendations(issues, statistical_metrics, business_metrics)
            
            # 7. Champion Comparison (if requested and available)
            vs_champion = None
            if include_champion_comparison:
                vs_champion = await self._compare_with_champion(model_version, statistical_metrics, business_metrics)
            
            # 8. Determine Overall Status
            status = self._determine_evaluation_status(overall_score, issues, performance_tier)
            
            # Create evaluation result
            result = EvaluationResult(
                model_version=model_version,
                evaluation_id=evaluation_id,
                timestamp=datetime.now(),
                statistical_metrics=statistical_metrics,
                statistical_tests=statistical_tests,
                business_metrics=business_metrics,
                roi_analysis=roi_analysis,
                operational_metrics=operational_metrics,
                data_quality=data_quality,
                overall_score=overall_score,
                performance_tier=performance_tier,
                status=status,
                issues=issues,
                recommendations=recommendations,
                vs_champion=vs_champion
            )
            
            # Store evaluation result
            await self._store_evaluation_result(result)
            
            logger.info(f"✅ Evaluation {evaluation_id} completed - Score: {overall_score:.3f}, Status: {status.value}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Model evaluation failed for {model_version}: {e}")
            return EvaluationResult(
                model_version=model_version,
                evaluation_id=f"eval_error_{int(datetime.now().timestamp())}",
                timestamp=datetime.now(),
                statistical_metrics={},
                statistical_tests={},
                business_metrics={},
                roi_analysis={},
                operational_metrics={},
                data_quality={'error': str(e)},
                overall_score=0.0,
                performance_tier=ModelPerformanceTier.CHALLENGER_FAILING,
                status=EvaluationStatus.FAIL,
                issues=[f"Evaluation system error: {str(e)}"],
                recommendations=["Contact MLOps team to investigate evaluation system issue"],
                vs_champion=None
            )
    
    async def _get_evaluation_data(self, model_version: str, hours: int) -> Optional[pd.DataFrame]:
        """Get prediction and actual data for evaluation"""
        if not self.postgres:
            return None
            
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            with self.postgres.cursor(cursor_factory=RealDictCursor) as cur:
                # Get predictions with actual values (for accuracy calculation)
                cur.execute("""
                    SELECT 
                        p.symbol,
                        p.current_price,
                        p.predicted_price,
                        p.sentiment_score,
                        p.confidence,
                        p.created_at as prediction_time,
                        -- Get actual price from next day's data
                        sp.close as actual_next_price,
                        sp.date as actual_date,
                        -- Calculate latency (if available from ab_results)
                        ar.latency_ms
                    FROM predictions p
                    LEFT JOIN stock_prices sp ON p.symbol = sp.symbol 
                        AND sp.date::date = (p.created_at::date + INTERVAL '1 day')
                    LEFT JOIN ab_results ar ON p.symbol = ar.user_id 
                        AND ar.variant = %s
                        AND ar.recorded_at BETWEEN p.created_at - INTERVAL '1 minute' 
                        AND p.created_at + INTERVAL '1 minute'
                    WHERE p.model_version = %s 
                    AND p.created_at >= %s
                    ORDER BY p.created_at DESC
                """, (model_version, model_version, cutoff_time))
                
                rows = cur.fetchall()
                
                if not rows:
                    logger.warning(f"No evaluation data found for {model_version}")
                    return None
                
                df = pd.DataFrame(rows)
                
                # Calculate actual returns and prediction errors
                df['actual_return'] = (df['actual_next_price'] - df['current_price']) / df['current_price']
                df['predicted_return'] = (df['predicted_price'] - df['current_price']) / df['current_price']
                df['prediction_error'] = abs(df['predicted_price'] - df['actual_next_price'])
                df['relative_error'] = df['prediction_error'] / df['actual_next_price']
                
                # Filter out rows without actual values for core metrics
                core_df = df.dropna(subset=['actual_next_price'])
                
                logger.info(f"Loaded {len(df)} predictions, {len(core_df)} with actual values for evaluation")
                
                return df
                
        except Exception as e:
            logger.error(f"Failed to get evaluation data: {e}")
            return None
    
    async def _calculate_statistical_metrics(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate comprehensive statistical performance metrics"""
        metrics = {}
        
        # Filter data with actual values
        actual_data = data.dropna(subset=['actual_next_price'])
        
        if len(actual_data) == 0:
            return {'error': 'No actual values available for statistical metrics'}
        
        predictions = actual_data['predicted_price'].values
        actuals = actual_data['actual_next_price'].values
        
        # Core regression metrics
        metrics['mae'] = float(mean_absolute_error(actuals, predictions))
        metrics['rmse'] = float(np.sqrt(mean_squared_error(actuals, predictions)))
        metrics['r2_score'] = float(r2_score(actuals, predictions))
        
        # Percentage-based metrics
        relative_errors = np.abs(predictions - actuals) / np.abs(actuals)
        metrics['mape'] = float(np.mean(relative_errors))  # Mean Absolute Percentage Error
        metrics['median_ape'] = float(np.median(relative_errors))
        
        # Accuracy at different thresholds
        metrics['accuracy_1pct'] = float(np.mean(relative_errors <= 0.01))  # Within 1%
        metrics['accuracy_5pct'] = float(np.mean(relative_errors <= 0.05))  # Within 5%
        metrics['accuracy_10pct'] = float(np.mean(relative_errors <= 0.10)) # Within 10%
        
        # Direction accuracy (up/down prediction)
        actual_directions = np.sign(actual_data['actual_return'])
        predicted_directions = np.sign(actual_data['predicted_return'])
        metrics['direction_accuracy'] = float(np.mean(actual_directions == predicted_directions))
        
        # Error distribution statistics
        errors = predictions - actuals
        metrics['mean_bias'] = float(np.mean(errors))  # Systematic bias
        metrics['error_std'] = float(np.std(errors))
        metrics['error_skewness'] = float(stats.skew(errors))
        metrics['error_kurtosis'] = float(stats.kurtosis(errors))
        
        # Confidence calibration (if available)
        if 'confidence' in actual_data.columns:
            confidence = actual_data['confidence'].values
            # Correlation between confidence and accuracy
            accuracy_indicators = (relative_errors <= 0.05).astype(float)
            if np.std(confidence) > 0:
                metrics['confidence_correlation'] = float(np.corrcoef(confidence, accuracy_indicators)[0, 1])
            else:
                metrics['confidence_correlation'] = 0.0
        
        # Sample size
        metrics['sample_size'] = len(actual_data)
        metrics['evaluation_coverage'] = len(actual_data) / len(data)  # Fraction with actual values
        
        return metrics
    
    async def _perform_statistical_tests(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Perform statistical tests on model predictions"""
        tests = {}
        
        actual_data = data.dropna(subset=['actual_next_price'])
        if len(actual_data) < 30:
            return {'error': 'Insufficient sample size for statistical tests'}
        
        predictions = actual_data['predicted_price'].values
        actuals = actual_data['actual_next_price'].values
        errors = predictions - actuals
        
        # Test 1: Normality of errors (Jarque-Bera test)
        try:
            jb_stat, jb_pvalue = jarque_bera(errors)
            tests['error_normality'] = {
                'test': 'jarque_bera',
                'statistic': float(jb_stat),
                'p_value': float(jb_pvalue),
                'is_normal': jb_pvalue > 0.05,
                'interpretation': 'Errors are normally distributed' if jb_pvalue > 0.05 else 'Errors are not normally distributed'
            }
        except:
            tests['error_normality'] = {'error': 'Failed to compute normality test'}
        
        # Test 2: Zero bias test (t-test)
        try:
            t_stat, t_pvalue = stats.ttest_1samp(errors, 0)
            tests['zero_bias'] = {
                'test': 'one_sample_t_test',
                'statistic': float(t_stat),
                'p_value': float(t_pvalue),
                'is_unbiased': t_pvalue > 0.05,
                'mean_bias': float(np.mean(errors)),
                'interpretation': 'Model is unbiased' if t_pvalue > 0.05 else 'Model has systematic bias'
            }
        except:
            tests['zero_bias'] = {'error': 'Failed to compute bias test'}
        
        # Test 3: Heteroscedasticity test (Breusch-Pagan proxy)
        try:
            # Simple proxy: correlation between absolute errors and predicted values
            abs_errors = np.abs(errors)
            corr_coef, corr_pvalue = stats.pearsonr(abs_errors, predictions)
            tests['heteroscedasticity'] = {
                'test': 'error_prediction_correlation',
                'correlation': float(corr_coef),
                'p_value': float(corr_pvalue),
                'is_homoscedastic': abs(corr_coef) < 0.1 or corr_pvalue > 0.05,
                'interpretation': 'Constant error variance' if abs(corr_coef) < 0.1 else 'Error variance changes with predictions'
            }
        except:
            tests['heteroscedasticity'] = {'error': 'Failed to compute heteroscedasticity test'}
        
        return tests
    
    async def _calculate_business_metrics(self, data: pd.DataFrame, model_version: str) -> Dict[str, float]:
        """Calculate business impact and trading performance metrics"""
        metrics = {}
        
        # Filter data with actual values
        trading_data = data.dropna(subset=['actual_next_price', 'predicted_return', 'actual_return'])
        
        if len(trading_data) == 0:
            return {'error': 'No trading data available for business metrics'}
        
        predicted_returns = trading_data['predicted_return'].values
        actual_returns = trading_data['actual_return'].values
        
        # Trading simulation (simple strategy: buy if predicted return > threshold)
        buy_threshold = 0.01  # 1% minimum predicted return
        sell_threshold = -0.01  # -1% maximum predicted loss
        
        # Generate trading signals
        buy_signals = predicted_returns > buy_threshold
        sell_signals = predicted_returns < sell_threshold
        hold_signals = ~(buy_signals | sell_signals)
        
        # Calculate returns for each strategy
        buy_returns = actual_returns[buy_signals] if np.any(buy_signals) else np.array([])
        sell_returns = -actual_returns[sell_signals] if np.any(sell_signals) else np.array([])  # Short selling
        all_trading_returns = np.concatenate([buy_returns, sell_returns]) if len(buy_returns) > 0 or len(sell_returns) > 0 else np.array([])
        
        # Performance metrics
        if len(all_trading_returns) > 0:
            metrics['total_trades'] = len(all_trading_returns)
            metrics['avg_return_per_trade'] = float(np.mean(all_trading_returns))
            metrics['win_rate'] = float(np.mean(all_trading_returns > 0))
            metrics['total_return'] = float(np.sum(all_trading_returns))
            
            # Risk metrics
            if len(all_trading_returns) > 1:
                metrics['volatility'] = float(np.std(all_trading_returns))
                metrics['sharpe_ratio'] = float(metrics['avg_return_per_trade'] / metrics['volatility']) if metrics['volatility'] > 0 else 0.0
                
                # Maximum drawdown calculation
                cumulative_returns = np.cumsum(all_trading_returns)
                running_max = np.maximum.accumulate(cumulative_returns)
                drawdowns = running_max - cumulative_returns
                metrics['max_drawdown'] = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0
            else:
                metrics['volatility'] = 0.0
                metrics['sharpe_ratio'] = 0.0
                metrics['max_drawdown'] = 0.0
        else:
            # No trades generated
            metrics.update({
                'total_trades': 0,
                'avg_return_per_trade': 0.0,
                'win_rate': 0.0,
                'total_return': 0.0,
                'volatility': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0
            })
        
        # Buy and hold benchmark
        benchmark_return = float(np.mean(actual_returns))
        metrics['benchmark_return'] = benchmark_return
        metrics['excess_return'] = metrics['avg_return_per_trade'] - benchmark_return
        
        # Trade distribution
        metrics['buy_signals_pct'] = float(np.mean(buy_signals))
        metrics['sell_signals_pct'] = float(np.mean(sell_signals))
        metrics['hold_signals_pct'] = float(np.mean(hold_signals))
        
        # Signal quality
        if np.any(buy_signals):
            buy_accuracy = np.mean(actual_returns[buy_signals] > 0)
            metrics['buy_signal_accuracy'] = float(buy_accuracy)
        
        if np.any(sell_signals):
            sell_accuracy = np.mean(actual_returns[sell_signals] < 0)  # Correct when actual is negative
            metrics['sell_signal_accuracy'] = float(sell_accuracy)
        
        return metrics
    
    async def _calculate_roi_analysis(self, data: pd.DataFrame, model_version: str) -> Dict[str, float]:
        """Calculate detailed ROI analysis with transaction costs"""
        roi_metrics = {}
        
        # Trading simulation with realistic assumptions
        trading_data = data.dropna(subset=['actual_next_price', 'predicted_return', 'actual_return'])
        
        if len(trading_data) == 0:
            return {'error': 'No data for ROI analysis'}
        
        # Trading assumptions
        initial_capital = 100000  # $100k starting capital
        position_size_pct = 0.1   # 10% position size per trade
        transaction_cost_pct = 0.001  # 0.1% transaction cost (bid-ask spread + commission)
        
        predicted_returns = trading_data['predicted_return'].values
        actual_returns = trading_data['actual_return'].values
        
        # Conservative trading thresholds
        buy_threshold = 0.015   # 1.5% minimum predicted return
        confidence_threshold = 0.6  # Minimum confidence for trading
        
        # Apply confidence filter if available
        if 'confidence' in trading_data.columns:
            confidence_mask = trading_data['confidence'] >= confidence_threshold
        else:
            confidence_mask = np.ones(len(trading_data), dtype=bool)
        
        # Trading signals
        trading_signals = (predicted_returns > buy_threshold) & confidence_mask
        
        # Calculate portfolio performance
        portfolio_value = initial_capital
        trade_history = []
        
        for i, (should_trade, actual_return) in enumerate(zip(trading_signals, actual_returns)):
            if should_trade:
                position_size = portfolio_value * position_size_pct
                gross_return = actual_return
                transaction_costs = position_size * transaction_cost_pct * 2  # Buy and sell
                net_return = gross_return - (transaction_costs / position_size)
                
                trade_pnl = position_size * net_return
                portfolio_value += trade_pnl
                
                trade_history.append({
                    'gross_return': gross_return,
                    'net_return': net_return,
                    'trade_pnl': trade_pnl,
                    'portfolio_value': portfolio_value
                })
        
        # ROI Analysis
        if trade_history:
            total_return = (portfolio_value - initial_capital) / initial_capital
            roi_metrics['total_roi'] = float(total_return)
            roi_metrics['annualized_roi'] = float(total_return * (365 / len(trading_data)))  # Rough annualization
            roi_metrics['total_trades'] = len(trade_history)
            roi_metrics['avg_trade_return'] = float(np.mean([t['net_return'] for t in trade_history]))
            roi_metrics['trade_win_rate'] = float(np.mean([t['net_return'] > 0 for t in trade_history]))
            
            # Risk-adjusted metrics
            trade_returns = np.array([t['net_return'] for t in trade_history])
            roi_metrics['volatility'] = float(np.std(trade_returns))
            roi_metrics['sharpe_ratio'] = float(roi_metrics['avg_trade_return'] / roi_metrics['volatility']) if roi_metrics['volatility'] > 0 else 0.0
            
            # Drawdown analysis
            portfolio_values = np.array([t['portfolio_value'] for t in trade_history])
            running_max = np.maximum.accumulate(portfolio_values)
            drawdowns = (running_max - portfolio_values) / running_max
            roi_metrics['max_drawdown'] = float(np.max(drawdowns))
            
            # Transaction cost impact
            gross_pnl = sum([t['gross_return'] * initial_capital * position_size_pct for t in trade_history])
            net_pnl = portfolio_value - initial_capital
            roi_metrics['transaction_cost_impact'] = float((gross_pnl - net_pnl) / initial_capital)
        else:
            roi_metrics.update({
                'total_roi': 0.0,
                'annualized_roi': 0.0,
                'total_trades': 0,
                'avg_trade_return': 0.0,
                'trade_win_rate': 0.0,
                'volatility': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'transaction_cost_impact': 0.0
            })
        
        # Benchmark comparison (buy and hold)
        benchmark_return = float(np.mean(actual_returns) * len(actual_returns))
        roi_metrics['benchmark_roi'] = benchmark_return
        roi_metrics['excess_roi'] = roi_metrics['total_roi'] - benchmark_return
        
        return roi_metrics
    
    async def _calculate_operational_metrics(self, data: pd.DataFrame, evaluation_period_hours: int) -> Dict[str, float]:
        """Calculate operational performance metrics"""
        metrics = {}
        
        # Latency metrics (if available)
        latency_data = data.dropna(subset=['latency_ms'])
        if len(latency_data) > 0:
            latencies = latency_data['latency_ms'].values
            metrics['avg_latency_ms'] = float(np.mean(latencies))
            metrics['median_latency_ms'] = float(np.median(latencies))
            metrics['p95_latency_ms'] = float(np.percentile(latencies, 95))
            metrics['p99_latency_ms'] = float(np.percentile(latencies, 99))
            metrics['max_latency_ms'] = float(np.max(latencies))
        else:
            metrics.update({
                'avg_latency_ms': 0.0,
                'median_latency_ms': 0.0,
                'p95_latency_ms': 0.0,
                'p99_latency_ms': 0.0,
                'max_latency_ms': 0.0
            })
        
        # Throughput metrics
        total_predictions = len(data)
        metrics['total_predictions'] = total_predictions
        metrics['predictions_per_hour'] = total_predictions / evaluation_period_hours
        
        # Availability metrics (based on prediction frequency)
        if total_predictions > 0:
            # Estimate expected predictions (assuming one prediction per symbol per hour)
            unique_symbols = data['symbol'].nunique() if 'symbol' in data.columns else 1
            expected_predictions = unique_symbols * evaluation_period_hours
            metrics['prediction_availability'] = min(1.0, total_predictions / expected_predictions)
        else:
            metrics['prediction_availability'] = 0.0
        
        # Error rate
        predictions_with_errors = data[data['prediction_error'].isna() | (data['prediction_error'] < 0)]
        metrics['error_rate'] = len(predictions_with_errors) / len(data) if len(data) > 0 else 1.0
        metrics['success_rate'] = 1.0 - metrics['error_rate']
        
        # Data completeness
        required_fields = ['predicted_price', 'current_price', 'sentiment_score']
        completeness_scores = []
        for field in required_fields:
            if field in data.columns:
                completeness = (1 - data[field].isna().mean())
                completeness_scores.append(completeness)
                metrics[f'{field}_completeness'] = float(completeness)
        
        metrics['overall_completeness'] = float(np.mean(completeness_scores)) if completeness_scores else 0.0
        
        return metrics
    
    async def _assess_data_quality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Comprehensive data quality assessment"""
        quality = {}
        
        # Basic statistics
        quality['total_records'] = len(data)
        quality['records_with_actuals'] = len(data.dropna(subset=['actual_next_price']))
        quality['data_completeness_pct'] = (quality['records_with_actuals'] / quality['total_records']) * 100 if quality['total_records'] > 0 else 0
        
        # Data freshness
        if 'prediction_time' in data.columns:
            latest_prediction = pd.to_datetime(data['prediction_time']).max()
            hours_since_latest = (datetime.now() - latest_prediction).total_seconds() / 3600
            quality['hours_since_latest_prediction'] = float(hours_since_latest)
            quality['data_is_fresh'] = hours_since_latest <= self.thresholds.min_data_freshness_hours
        else:
            quality['hours_since_latest_prediction'] = float('inf')
            quality['data_is_fresh'] = False
        
        # Data distribution checks
        if 'predicted_price' in data.columns:
            predictions = data['predicted_price'].dropna()
            if len(predictions) > 0:
                quality['prediction_stats'] = {
                    'mean': float(predictions.mean()),
                    'std': float(predictions.std()),
                    'min': float(predictions.min()),
                    'max': float(predictions.max()),
                    'zeros': int((predictions == 0).sum()),
                    'negative': int((predictions < 0).sum())
                }
                
                # Check for anomalies
                z_scores = np.abs(stats.zscore(predictions))
                quality['prediction_outliers'] = int((z_scores > 3).sum())
                quality['prediction_outlier_rate'] = quality['prediction_outliers'] / len(predictions)
        
        # Actual values quality
        if 'actual_next_price' in data.columns:
            actuals = data['actual_next_price'].dropna()
            if len(actuals) > 0:
                quality['actual_stats'] = {
                    'mean': float(actuals.mean()),
                    'std': float(actuals.std()),
                    'min': float(actuals.min()),
                    'max': float(actuals.max())
                }
        
        # Missing data analysis
        missing_data = {}
        for column in data.columns:
            missing_count = data[column].isna().sum()
            missing_pct = (missing_count / len(data)) * 100
            missing_data[column] = {
                'count': int(missing_count),
                'percentage': float(missing_pct)
            }
        
        quality['missing_data'] = missing_data
        
        # Overall quality score (0-1)
        quality_factors = [
            min(1.0, quality['data_completeness_pct'] / 100),  # Completeness
            1.0 if quality['data_is_fresh'] else 0.5,  # Freshness
            max(0.0, 1.0 - quality.get('prediction_outlier_rate', 1.0))  # Outlier rate
        ]
        
        quality['overall_quality_score'] = float(np.mean(quality_factors))
        quality['quality_issues'] = []
        
        if quality['data_completeness_pct'] < 50:
            quality['quality_issues'].append('Low data completeness (<50%)')
        
        if not quality['data_is_fresh']:
            quality['quality_issues'].append(f'Stale data (>{self.thresholds.min_data_freshness_hours}h old)')
        
        if quality.get('prediction_outlier_rate', 0) > 0.1:
            quality['quality_issues'].append('High outlier rate in predictions (>10%)')
        
        return quality