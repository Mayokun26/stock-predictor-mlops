#!/usr/bin/env python3
"""
Enhanced Model Comparison Framework
Integrates backtesting, statistical evaluation, and A/B testing for comprehensive model assessment
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass, asdict
import json
from scipy import stats
import mlflow
import mlflow.sklearn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .backtesting_engine import BacktestingEngine, BacktestResult
from .model_evaluator import ModelEvaluator, EvaluationResult, EvaluationStatus

logger = logging.getLogger(__name__)

@dataclass
class ModelComparisonResult:
    """Comprehensive model comparison results"""
    
    # Model identification
    champion_model: str
    challenger_model: str
    comparison_date: str
    
    # Statistical comparison
    statistical_significance: bool
    p_value: float
    confidence_interval: Tuple[float, float]
    effect_size: float
    
    # Backtesting comparison
    champion_backtest: BacktestResult
    challenger_backtest: BacktestResult
    
    # Performance metrics
    performance_improvement: Dict[str, float]
    risk_improvement: Dict[str, float]
    
    # Business impact
    projected_roi_improvement: float
    projected_sharpe_improvement: float
    risk_adjusted_improvement: float
    
    # Decision recommendation
    recommendation: str
    confidence_score: float
    reasoning: List[str]
    
    # Deployment readiness
    deployment_ready: bool
    deployment_conditions: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        # Convert BacktestResult objects to dicts
        result['champion_backtest'] = self.champion_backtest.to_dict()
        result['challenger_backtest'] = self.challenger_backtest.to_dict()
        return result
    
    def summary_report(self) -> str:
        """Generate executive summary report"""
        
        status = "✅ RECOMMENDED" if self.deployment_ready else "⚠️ NOT READY"
        
        return f"""
Model Comparison Report - {self.comparison_date}
===============================================

{status} for Deployment

Champion: {self.champion_model}
Challenger: {self.challenger_model}

KEY FINDINGS:
📊 Statistical Significance: {'Yes' if self.statistical_significance else 'No'} (p-value: {self.p_value:.4f})
💰 ROI Improvement: {self.projected_roi_improvement:.2%}
📈 Sharpe Improvement: {self.projected_sharpe_improvement:.2f}
⚡ Risk Adjustment: {self.risk_adjusted_improvement:.2%}

PERFORMANCE COMPARISON:
  Annualized Return: {self.performance_improvement.get('return', 0):.2%} improvement
  Max Drawdown: {self.risk_improvement.get('drawdown', 0):.2%} improvement
  Directional Accuracy: {self.performance_improvement.get('accuracy', 0):.2%} improvement

RECOMMENDATION: {self.recommendation}
Confidence: {self.confidence_score:.1%}

REASONING:
{chr(10).join(f"  • {reason}" for reason in self.reasoning)}

DEPLOYMENT CONDITIONS:
{chr(10).join(f"  ✓ {condition}" for condition in self.deployment_conditions)}
        """


class EnhancedModelComparison:
    """
    Enhanced model comparison framework with backtesting and statistical rigor
    """
    
    def __init__(self, 
                 min_significance_level: float = 0.05,
                 min_effect_size: float = 0.1,
                 min_sharpe_improvement: float = 0.2,
                 min_sample_size: int = 100):
        """
        Initialize model comparison framework
        
        Args:
            min_significance_level: Minimum p-value for statistical significance
            min_effect_size: Minimum effect size for practical significance
            min_sharpe_improvement: Minimum Sharpe ratio improvement for deployment
            min_sample_size: Minimum sample size for valid comparison
        """
        self.min_significance_level = min_significance_level
        self.min_effect_size = min_effect_size
        self.min_sharpe_improvement = min_sharpe_improvement
        self.min_sample_size = min_sample_size
        
        # Initialize sub-components
        self.backtesting_engine = BacktestingEngine()
        self.evaluator = ModelEvaluator()
        
        # Track comparison history
        self.comparison_history: List[ModelComparisonResult] = []
        
    def comprehensive_model_comparison(self,
                                     champion_model,
                                     challenger_model,
                                     symbol: str,
                                     start_date: str,
                                     end_date: str,
                                     validation_data: Optional[pd.DataFrame] = None) -> ModelComparisonResult:
        """
        Perform comprehensive model comparison using multiple evaluation methods
        
        Args:
            champion_model: Current production model
            challenger_model: New model to evaluate
            symbol: Stock symbol for backtesting
            start_date: Start date for evaluation period
            end_date: End date for evaluation period
            validation_data: Optional validation dataset
            
        Returns:
            ModelComparisonResult with comprehensive analysis
        """
        logger.info(f"Starting comprehensive model comparison for {symbol}")
        
        # 1. Statistical Evaluation
        statistical_results = self._statistical_comparison(
            champion_model, challenger_model, validation_data
        )
        
        # 2. Backtesting Evaluation
        champion_backtest = self.backtesting_engine.walk_forward_backtest(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            model_class=lambda: champion_model
        )
        
        challenger_backtest = self.backtesting_engine.walk_forward_backtest(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            model_class=lambda: challenger_model
        )
        
        # 3. Performance Comparison
        performance_improvement = self._calculate_performance_improvement(
            champion_backtest, challenger_backtest
        )
        
        risk_improvement = self._calculate_risk_improvement(
            champion_backtest, challenger_backtest
        )
        
        # 4. Business Impact Analysis
        business_impact = self._analyze_business_impact(
            champion_backtest, challenger_backtest
        )
        
        # 5. Deployment Decision
        decision = self._make_deployment_decision(
            statistical_results=statistical_results,
            performance_improvement=performance_improvement,
            risk_improvement=risk_improvement,
            business_impact=business_impact,
            champion_backtest=champion_backtest,
            challenger_backtest=challenger_backtest
        )
        
        # Create comprehensive result
        result = ModelComparisonResult(
            champion_model=str(champion_model),
            challenger_model=str(challenger_model),
            comparison_date=datetime.now().isoformat(),
            statistical_significance=statistical_results['significant'],
            p_value=statistical_results['p_value'],
            confidence_interval=statistical_results['confidence_interval'],
            effect_size=statistical_results['effect_size'],
            champion_backtest=champion_backtest,
            challenger_backtest=challenger_backtest,
            performance_improvement=performance_improvement,
            risk_improvement=risk_improvement,
            projected_roi_improvement=business_impact['roi_improvement'],
            projected_sharpe_improvement=business_impact['sharpe_improvement'],
            risk_adjusted_improvement=business_impact['risk_adjusted_improvement'],
            recommendation=decision['recommendation'],
            confidence_score=decision['confidence'],
            reasoning=decision['reasoning'],
            deployment_ready=decision['deployment_ready'],
            deployment_conditions=decision['conditions']
        )
        
        self.comparison_history.append(result)
        
        # Log results to MLflow
        self._log_comparison_to_mlflow(result)
        
        logger.info("Model comparison completed")
        return result
    
    def _statistical_comparison(self, 
                              champion_model, 
                              challenger_model, 
                              validation_data: Optional[pd.DataFrame]) -> Dict[str, Any]:
        """Perform statistical comparison of model predictions"""
        
        if validation_data is None:
            logger.warning("No validation data provided for statistical comparison")
            return {
                'significant': False,
                'p_value': 1.0,
                'confidence_interval': (0, 0),
                'effect_size': 0
            }
        
        # Generate predictions from both models
        try:
            champion_predictions = champion_model.predict(validation_data.drop('target', axis=1))
            challenger_predictions = challenger_model.predict(validation_data.drop('target', axis=1))
            actual_values = validation_data['target'].values
            
            # Calculate prediction errors
            champion_errors = np.abs(champion_predictions - actual_values)
            challenger_errors = np.abs(challenger_predictions - actual_values)
            
            # Paired t-test for statistical significance
            t_stat, p_value = stats.ttest_rel(champion_errors, challenger_errors)
            
            # Effect size (Cohen's d)
            pooled_std = np.sqrt((np.var(champion_errors) + np.var(challenger_errors)) / 2)
            effect_size = (np.mean(challenger_errors) - np.mean(champion_errors)) / pooled_std
            
            # Confidence interval for effect size
            se_effect_size = np.sqrt((len(champion_errors) + len(challenger_errors)) / 
                                   (len(champion_errors) * len(challenger_errors)) + 
                                   (effect_size ** 2) / (2 * (len(champion_errors) + len(challenger_errors))))
            
            ci_lower = effect_size - 1.96 * se_effect_size
            ci_upper = effect_size + 1.96 * se_effect_size
            
            return {
                'significant': p_value < self.min_significance_level and abs(effect_size) > self.min_effect_size,
                'p_value': p_value,
                'confidence_interval': (ci_lower, ci_upper),
                'effect_size': effect_size
            }
            
        except Exception as e:
            logger.error(f"Statistical comparison failed: {e}")
            return {
                'significant': False,
                'p_value': 1.0,
                'confidence_interval': (0, 0),
                'effect_size': 0
            }
    
    def _calculate_performance_improvement(self, 
                                         champion: BacktestResult, 
                                         challenger: BacktestResult) -> Dict[str, float]:
        """Calculate performance improvements"""
        
        return {
            'return': challenger.annualized_return - champion.annualized_return,
            'sharpe': challenger.sharpe_ratio - champion.sharpe_ratio,
            'accuracy': challenger.directional_accuracy - champion.directional_accuracy,
            'win_rate': challenger.win_rate - champion.win_rate,
            'profit_factor': challenger.profit_factor - champion.profit_factor
        }
    
    def _calculate_risk_improvement(self, 
                                  champion: BacktestResult, 
                                  challenger: BacktestResult) -> Dict[str, float]:
        """Calculate risk metric improvements"""
        
        return {
            'drawdown': champion.max_drawdown - challenger.max_drawdown,  # Positive = improvement
            'volatility': champion.volatility - challenger.volatility,
            'var_95': champion.var_95 - challenger.var_95,
            'sortino': challenger.sortino_ratio - champion.sortino_ratio
        }
    
    def _analyze_business_impact(self, 
                               champion: BacktestResult, 
                               challenger: BacktestResult) -> Dict[str, float]:
        """Analyze business impact of model switch"""
        
        # Calculate improvements
        roi_improvement = challenger.total_return - champion.total_return
        sharpe_improvement = challenger.sharpe_ratio - champion.sharpe_ratio
        
        # Risk-adjusted improvement (combines return and risk)
        champion_risk_adj = champion.annualized_return / max(abs(champion.max_drawdown), 0.01)
        challenger_risk_adj = challenger.annualized_return / max(abs(challenger.max_drawdown), 0.01)
        risk_adjusted_improvement = challenger_risk_adj - champion_risk_adj
        
        return {
            'roi_improvement': roi_improvement,
            'sharpe_improvement': sharpe_improvement,
            'risk_adjusted_improvement': risk_adjusted_improvement
        }
    
    def _make_deployment_decision(self, 
                                statistical_results: Dict,
                                performance_improvement: Dict,
                                risk_improvement: Dict,
                                business_impact: Dict,
                                champion_backtest: BacktestResult,
                                challenger_backtest: BacktestResult) -> Dict[str, Any]:
        """Make data-driven deployment decision"""
        
        reasoning = []
        confidence_factors = []
        deployment_conditions = []
        
        # Check statistical significance
        if statistical_results['significant']:
            reasoning.append("Statistical significance achieved in prediction accuracy")
            confidence_factors.append(0.3)
        else:
            reasoning.append("No statistical significance in prediction accuracy")
            confidence_factors.append(-0.2)
        
        # Check Sharpe ratio improvement
        if business_impact['sharpe_improvement'] >= self.min_sharpe_improvement:
            reasoning.append(f"Sharpe ratio improved by {business_impact['sharpe_improvement']:.2f}")
            confidence_factors.append(0.4)
            deployment_conditions.append("Sharpe ratio improvement criterion met")
        else:
            reasoning.append("Insufficient Sharpe ratio improvement")
            confidence_factors.append(-0.3)
        
        # Check drawdown improvement
        if risk_improvement['drawdown'] > 0.01:  # 1% improvement in drawdown
            reasoning.append("Maximum drawdown reduced")
            confidence_factors.append(0.2)
            deployment_conditions.append("Risk reduction achieved")
        elif risk_improvement['drawdown'] < -0.05:  # 5% worse drawdown
            reasoning.append("Maximum drawdown increased significantly")
            confidence_factors.append(-0.4)
        
        # Check return improvement
        if performance_improvement['return'] > 0.02:  # 2% annual return improvement
            reasoning.append("Significant return improvement")
            confidence_factors.append(0.3)
            deployment_conditions.append("Return improvement criterion met")
        elif performance_improvement['return'] < -0.02:
            reasoning.append("Return performance degraded")
            confidence_factors.append(-0.3)
        
        # Check directional accuracy
        if performance_improvement['accuracy'] > 0.05:  # 5% improvement
            reasoning.append("Directional accuracy improved")
            confidence_factors.append(0.2)
        
        # Calculate overall confidence
        base_confidence = 0.5
        confidence_adjustment = sum(confidence_factors) / len(confidence_factors) if confidence_factors else 0
        overall_confidence = max(0.1, min(0.95, base_confidence + confidence_adjustment))
        
        # Make recommendation
        if (business_impact['sharpe_improvement'] >= self.min_sharpe_improvement and
            risk_improvement['drawdown'] >= 0 and
            statistical_results['significant']):
            recommendation = "DEPLOY - Strong evidence for improvement"
            deployment_ready = True
            deployment_conditions.extend([
                "A/B test recommended for production validation",
                "Monitor for 30 days with gradual traffic increase",
                "Set up automated rollback triggers"
            ])
        elif (business_impact['sharpe_improvement'] >= self.min_sharpe_improvement * 0.5 and
              risk_improvement['drawdown'] >= -0.02):
            recommendation = "CAUTIOUS DEPLOY - Moderate improvement with monitoring"
            deployment_ready = True
            deployment_conditions.extend([
                "Deploy with 10% traffic split initially",
                "Extended monitoring period required",
                "Manual review after 7 days"
            ])
        else:
            recommendation = "DO NOT DEPLOY - Insufficient improvement or increased risk"
            deployment_ready = False
            deployment_conditions = [
                "Model needs further optimization",
                "Consider additional features or training data",
                "Re-evaluate after model improvements"
            ]
        
        return {
            'recommendation': recommendation,
            'confidence': overall_confidence,
            'reasoning': reasoning,
            'deployment_ready': deployment_ready,
            'conditions': deployment_conditions
        }
    
    def _log_comparison_to_mlflow(self, result: ModelComparisonResult):
        """Log comparison results to MLflow"""
        
        try:
            with mlflow.start_run(run_name=f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                # Log parameters
                mlflow.log_param("champion_model", result.champion_model)
                mlflow.log_param("challenger_model", result.challenger_model)
                mlflow.log_param("comparison_date", result.comparison_date)
                
                # Log metrics
                mlflow.log_metric("statistical_significance", 1 if result.statistical_significance else 0)
                mlflow.log_metric("p_value", result.p_value)
                mlflow.log_metric("effect_size", result.effect_size)
                mlflow.log_metric("roi_improvement", result.projected_roi_improvement)
                mlflow.log_metric("sharpe_improvement", result.projected_sharpe_improvement)
                mlflow.log_metric("confidence_score", result.confidence_score)
                
                # Log performance improvements
                for metric, value in result.performance_improvement.items():
                    mlflow.log_metric(f"perf_improvement_{metric}", value)
                
                # Log risk improvements
                for metric, value in result.risk_improvement.items():
                    mlflow.log_metric(f"risk_improvement_{metric}", value)
                
                # Log deployment decision
                mlflow.log_param("deployment_ready", result.deployment_ready)
                mlflow.log_param("recommendation", result.recommendation)
                
                # Save detailed results as artifact
                result_file = "comparison_result.json"
                with open(result_file, 'w') as f:
                    json.dump(result.to_dict(), f, indent=2, default=str)
                mlflow.log_artifact(result_file)
                
        except Exception as e:
            logger.error(f"Failed to log comparison to MLflow: {e}")
    
    def generate_comparison_report(self, result: ModelComparisonResult) -> str:
        """Generate detailed comparison report"""
        
        report = result.summary_report()
        
        # Add detailed metrics comparison
        report += "\n\nDETAILED METRICS COMPARISON:\n"
        report += "=" * 50 + "\n"
        
        # Champion metrics
        report += f"\nChampion Model ({result.champion_model}):\n"
        report += f"  Annualized Return: {result.champion_backtest.annualized_return:.2%}\n"
        report += f"  Sharpe Ratio: {result.champion_backtest.sharpe_ratio:.2f}\n"
        report += f"  Max Drawdown: {result.champion_backtest.max_drawdown:.2%}\n"
        report += f"  Win Rate: {result.champion_backtest.win_rate:.2%}\n"
        
        # Challenger metrics  
        report += f"\nChallenger Model ({result.challenger_model}):\n"
        report += f"  Annualized Return: {result.challenger_backtest.annualized_return:.2%}\n"
        report += f"  Sharpe Ratio: {result.challenger_backtest.sharpe_ratio:.2f}\n"
        report += f"  Max Drawdown: {result.challenger_backtest.max_drawdown:.2%}\n"
        report += f"  Win Rate: {result.challenger_backtest.win_rate:.2%}\n"
        
        return report

    def batch_model_comparison(self, 
                             models: List[Tuple[str, Any]], 
                             symbols: List[str],
                             start_date: str,
                             end_date: str) -> Dict[str, List[ModelComparisonResult]]:
        """Run batch comparison across multiple models and symbols"""
        
        results = {}
        
        for i in range(len(models) - 1):
            champion_name, champion_model = models[i]
            challenger_name, challenger_model = models[i + 1]
            
            symbol_results = []
            
            for symbol in symbols:
                try:
                    result = self.comprehensive_model_comparison(
                        champion_model=champion_model,
                        challenger_model=challenger_model,
                        symbol=symbol,
                        start_date=start_date,
                        end_date=end_date
                    )
                    symbol_results.append(result)
                    
                except Exception as e:
                    logger.error(f"Comparison failed for {champion_name} vs {challenger_name} on {symbol}: {e}")
                    continue
            
            results[f"{champion_name}_vs_{challenger_name}"] = symbol_results
        
        return results


if __name__ == "__main__":
    # Example usage
    from backtesting_engine import SimpleLinearModel, SimpleMovingAverageModel
    
    comparator = EnhancedModelComparison()
    
    # Compare two simple models
    champion = SimpleMovingAverageModel()
    challenger = SimpleLinearModel()
    
    result = comparator.comprehensive_model_comparison(
        champion_model=champion,
        challenger_model=challenger,
        symbol="AAPL",
        start_date="2023-01-01",
        end_date="2023-12-31"
    )
    
    print(result.summary_report())