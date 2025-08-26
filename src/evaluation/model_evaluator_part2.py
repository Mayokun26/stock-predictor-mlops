#!/usr/bin/env python3
"""
Model Evaluator Part 2 - Scoring, Classification and Decision Logic
Continuation of the ModelEvaluator class implementation
"""

def _calculate_overall_score(self, 
                           statistical_metrics: Dict[str, float],
                           business_metrics: Dict[str, float], 
                           operational_metrics: Dict[str, float],
                           data_quality: Dict[str, Any]) -> float:
    """Calculate weighted overall performance score (0-1)"""
    
    # Statistical performance score (40% weight)
    statistical_score = 0.0
    if 'error' not in statistical_metrics:
        # Accuracy component (higher is better)
        accuracy_5pct = statistical_metrics.get('accuracy_5pct', 0)
        direction_accuracy = statistical_metrics.get('direction_accuracy', 0)
        r2_score = max(0, statistical_metrics.get('r2_score', 0))  # Can be negative
        
        # Error component (lower is better, so invert)
        mape = statistical_metrics.get('mape', 1.0)
        mape_score = max(0, 1 - (mape / 0.2))  # 20% MAPE = 0 score
        
        statistical_score = (accuracy_5pct * 0.3 + direction_accuracy * 0.3 + r2_score * 0.2 + mape_score * 0.2)
    
    # Business performance score (35% weight)  
    business_score = 0.0
    if 'error' not in business_metrics:
        sharpe_ratio = business_metrics.get('sharpe_ratio', 0)
        sharpe_score = min(1.0, max(0, sharpe_ratio / 2.0))  # Sharpe of 2.0 = max score
        
        win_rate = business_metrics.get('win_rate', 0)
        
        max_drawdown = business_metrics.get('max_drawdown', 1.0)
        drawdown_score = max(0, 1 - (max_drawdown / 0.2))  # 20% drawdown = 0 score
        
        business_score = (sharpe_score * 0.4 + win_rate * 0.3 + drawdown_score * 0.3)
    
    # Operational performance score (15% weight)
    operational_score = 0.0
    if operational_metrics:
        # Latency score (lower is better)
        p95_latency = operational_metrics.get('p95_latency_ms', 5000)  # Default high latency
        latency_score = max(0, 1 - (p95_latency / 2000))  # 2000ms = 0 score
        
        # Availability score
        availability = operational_metrics.get('prediction_availability', 0)
        
        # Success rate
        success_rate = operational_metrics.get('success_rate', 0)
        
        operational_score = (latency_score * 0.4 + availability * 0.3 + success_rate * 0.3)
    
    # Data quality score (10% weight)
    quality_score = data_quality.get('overall_quality_score', 0)
    
    # Weighted overall score
    overall_score = (
        statistical_score * 0.40 +
        business_score * 0.35 + 
        operational_score * 0.15 +
        quality_score * 0.10
    )
    
    return float(min(1.0, max(0.0, overall_score)))

def _classify_performance_tier(self,
                             overall_score: float,
                             statistical_metrics: Dict[str, float],
                             business_metrics: Dict[str, float]) -> ModelPerformanceTier:
    """Classify model performance tier based on comprehensive metrics"""
    
    # Champion criteria (must meet all)
    champion_criteria = [
        overall_score >= 0.8,
        statistical_metrics.get('accuracy_5pct', 0) >= 0.7,
        business_metrics.get('sharpe_ratio', 0) >= 1.0,
        business_metrics.get('max_drawdown', 1.0) <= 0.1
    ]
    
    if all(champion_criteria):
        return ModelPerformanceTier.CHAMPION
    
    # Challenger winning (strong performance, potential champion)
    winning_criteria = [
        overall_score >= 0.7,
        statistical_metrics.get('accuracy_5pct', 0) >= 0.6,
        business_metrics.get('sharpe_ratio', 0) >= 0.5
    ]
    
    if all(winning_criteria):
        return ModelPerformanceTier.CHALLENGER_WINNING
    
    # Challenger acceptable (meets minimum requirements)
    acceptable_criteria = [
        overall_score >= 0.5,
        statistical_metrics.get('accuracy_5pct', 0) >= 0.4,
        business_metrics.get('sharpe_ratio', 0) >= 0.0  # At least break-even
    ]
    
    if all(acceptable_criteria):
        return ModelPerformanceTier.CHALLENGER_ACCEPTABLE
    
    # Below acceptable performance
    return ModelPerformanceTier.CHALLENGER_FAILING

def _identify_issues(self,
                   statistical_metrics: Dict[str, float],
                   business_metrics: Dict[str, float],
                   operational_metrics: Dict[str, float],
                   data_quality: Dict[str, Any]) -> List[str]:
    """Identify specific performance issues"""
    issues = []
    
    # Statistical issues
    if statistical_metrics.get('accuracy_5pct', 0) < self.thresholds.min_accuracy_5pct:
        issues.append(f"Low prediction accuracy: {statistical_metrics.get('accuracy_5pct', 0):.1%} < {self.thresholds.min_accuracy_5pct:.1%}")
    
    if statistical_metrics.get('mape', 1.0) > self.thresholds.min_mape_threshold:
        issues.append(f"High MAPE: {statistical_metrics.get('mape', 0):.1%} > {self.thresholds.min_mape_threshold:.1%}")
    
    if statistical_metrics.get('r2_score', -1) < self.thresholds.min_r2_score:
        issues.append(f"Low R² score: {statistical_metrics.get('r2_score', 0):.3f} < {self.thresholds.min_r2_score:.3f}")
    
    # Business issues
    if business_metrics.get('sharpe_ratio', 0) < self.thresholds.min_sharpe_ratio:
        issues.append(f"Low Sharpe ratio: {business_metrics.get('sharpe_ratio', 0):.2f} < {self.thresholds.min_sharpe_ratio:.2f}")
    
    if business_metrics.get('max_drawdown', 0) > self.thresholds.max_drawdown_threshold:
        issues.append(f"High maximum drawdown: {business_metrics.get('max_drawdown', 0):.1%} > {self.thresholds.max_drawdown_threshold:.1%}")
    
    if business_metrics.get('total_roi', 0) < self.thresholds.min_roi_threshold:
        issues.append(f"Low ROI: {business_metrics.get('total_roi', 0):.1%} < {self.thresholds.min_roi_threshold:.1%}")
    
    # Operational issues
    if operational_metrics.get('p95_latency_ms', 0) > self.thresholds.max_latency_p95_ms:
        issues.append(f"High P95 latency: {operational_metrics.get('p95_latency_ms', 0):.0f}ms > {self.thresholds.max_latency_p95_ms}ms")
    
    if operational_metrics.get('prediction_availability', 0) < self.thresholds.min_availability:
        issues.append(f"Low availability: {operational_metrics.get('prediction_availability', 0):.1%} < {self.thresholds.min_availability:.1%}")
    
    # Data quality issues  
    quality_issues = data_quality.get('quality_issues', [])
    issues.extend([f"Data quality: {issue}" for issue in quality_issues])
    
    if data_quality.get('total_records', 0) < self.thresholds.min_sample_size:
        issues.append(f"Insufficient sample size: {data_quality.get('total_records', 0)} < {self.thresholds.min_sample_size}")
    
    return issues

def _generate_recommendations(self,
                            issues: List[str],
                            statistical_metrics: Dict[str, float],
                            business_metrics: Dict[str, float]) -> List[str]:
    """Generate actionable recommendations based on identified issues"""
    recommendations = []
    
    # Statistical recommendations
    if statistical_metrics.get('accuracy_5pct', 0) < 0.5:
        recommendations.append("Improve feature engineering - consider additional technical indicators or sentiment features")
    
    if statistical_metrics.get('mean_bias', 0) > 0.02:  # More than 2% systematic bias
        recommendations.append("Address systematic bias - model consistently over/under predicts")
    
    if statistical_metrics.get('direction_accuracy', 0) < 0.6:
        recommendations.append("Focus on directional accuracy - consider classification approach for up/down prediction")
    
    # Business recommendations  
    if business_metrics.get('win_rate', 0) < 0.5:
        recommendations.append("Improve trade selection - increase prediction confidence thresholds")
    
    if business_metrics.get('sharpe_ratio', 0) < 0.5:
        recommendations.append("Enhance risk-adjusted returns - consider position sizing optimization")
    
    if business_metrics.get('max_drawdown', 0) > 0.15:
        recommendations.append("Implement better risk management - add stop-loss mechanisms or position limits")
    
    # Model architecture recommendations
    if len([i for i in issues if 'accuracy' in i.lower()]) > 2:
        recommendations.append("Consider advanced model architectures - ensemble methods or deep learning approaches")
    
    # Data recommendations
    if any('data quality' in issue.lower() for issue in issues):
        recommendations.append("Improve data pipeline - ensure timely and complete feature updates")
    
    # Operational recommendations
    if any('latency' in issue.lower() for issue in issues):
        recommendations.append("Optimize model serving - consider model compression or caching strategies")
    
    # General recommendations
    if len(issues) > 5:
        recommendations.append("Comprehensive model redesign recommended - multiple critical issues identified")
    elif len(issues) == 0:
        recommendations.append("Excellent performance - consider promoting to champion status")
    
    return recommendations

async def _compare_with_champion(self,
                               challenger_version: str,
                               challenger_statistical: Dict[str, float],
                               challenger_business: Dict[str, float]) -> Optional[Dict[str, Any]]:
    """Compare challenger model with current champion"""
    try:
        # Get current champion model performance
        champion_data = await self._get_champion_baseline()
        
        if not champion_data:
            return {'note': 'No champion baseline available for comparison'}
        
        comparison = {
            'champion_version': champion_data['version'],
            'challenger_version': challenger_version,
            'comparison_timestamp': datetime.now().isoformat(),
            'statistical_comparison': {},
            'business_comparison': {},
            'recommendation': 'insufficient_data'
        }
        
        # Statistical comparison
        for metric in ['accuracy_5pct', 'mape', 'r2_score', 'direction_accuracy']:
            champion_value = champion_data['statistical_metrics'].get(metric, 0)
            challenger_value = challenger_statistical.get(metric, 0)
            
            if metric == 'mape':  # Lower is better for MAPE
                improvement = (champion_value - challenger_value) / champion_value if champion_value > 0 else 0
            else:  # Higher is better for others
                improvement = (challenger_value - champion_value) / champion_value if champion_value > 0 else 0
            
            comparison['statistical_comparison'][metric] = {
                'champion_value': champion_value,
                'challenger_value': challenger_value,
                'improvement_pct': improvement * 100,
                'challenger_better': improvement > 0
            }
        
        # Business comparison
        for metric in ['sharpe_ratio', 'total_roi', 'win_rate', 'max_drawdown']:
            champion_value = champion_data['business_metrics'].get(metric, 0)
            challenger_value = challenger_business.get(metric, 0)
            
            if metric == 'max_drawdown':  # Lower is better for drawdown
                improvement = (champion_value - challenger_value) / champion_value if champion_value > 0 else 0
            else:  # Higher is better for others
                improvement = (challenger_value - champion_value) / champion_value if champion_value > 0 else 0
            
            comparison['business_comparison'][metric] = {
                'champion_value': champion_value,
                'challenger_value': challenger_value,
                'improvement_pct': improvement * 100,
                'challenger_better': improvement > 0
            }
        
        # Overall recommendation
        statistical_improvements = sum(1 for m in comparison['statistical_comparison'].values() if m['challenger_better'])
        business_improvements = sum(1 for m in comparison['business_comparison'].values() if m['challenger_better'])
        
        total_metrics = len(comparison['statistical_comparison']) + len(comparison['business_comparison'])
        total_improvements = statistical_improvements + business_improvements
        
        improvement_ratio = total_improvements / total_metrics
        
        # Decision logic
        if improvement_ratio >= 0.75:  # 75% of metrics improved
            comparison['recommendation'] = 'promote_to_champion'
        elif improvement_ratio >= 0.5:  # 50% of metrics improved
            comparison['recommendation'] = 'continue_testing'
        else:
            comparison['recommendation'] = 'reject_challenger'
        
        # Add confidence based on sample sizes and significance
        comparison['confidence'] = 'medium'  # Simplified for now
        
        return comparison
        
    except Exception as e:
        logger.error(f"Champion comparison failed: {e}")
        return {'error': f'Champion comparison failed: {str(e)}'}

async def _get_champion_baseline(self) -> Optional[Dict[str, Any]]:
    """Get current champion model baseline performance"""
    if not self.postgres:
        return None
        
    try:
        # Get most recent champion evaluation
        with self.postgres.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT 
                    model_version,
                    statistical_metrics,
                    business_metrics,
                    overall_score,
                    evaluation_timestamp
                FROM model_evaluations 
                WHERE performance_tier = 'champion' 
                OR model_version = 'champion'
                ORDER BY evaluation_timestamp DESC 
                LIMIT 1
            """)
            
            row = cur.fetchone()
            
            if row:
                return {
                    'version': row['model_version'],
                    'statistical_metrics': row['statistical_metrics'],
                    'business_metrics': row['business_metrics'],
                    'overall_score': row['overall_score'],
                    'evaluation_timestamp': row['evaluation_timestamp']
                }
                
    except Exception as e:
        logger.error(f"Failed to get champion baseline: {e}")
    
    return None

def _determine_evaluation_status(self,
                               overall_score: float,
                               issues: List[str],
                               performance_tier: ModelPerformanceTier) -> EvaluationStatus:
    """Determine overall evaluation status"""
    
    # Critical issues that cause failure
    critical_keywords = ['insufficient sample', 'system error', 'data quality']
    has_critical_issues = any(any(keyword in issue.lower() for keyword in critical_keywords) for issue in issues)
    
    if has_critical_issues:
        return EvaluationStatus.FAIL
    
    # Performance-based status
    if performance_tier in [ModelPerformanceTier.CHAMPION, ModelPerformanceTier.CHALLENGER_WINNING]:
        return EvaluationStatus.PASS
    elif performance_tier == ModelPerformanceTier.CHALLENGER_ACCEPTABLE:
        return EvaluationStatus.WARNING
    else:
        return EvaluationStatus.FAIL

async def _store_evaluation_result(self, result: EvaluationResult) -> bool:
    """Store evaluation result in database"""
    if not self.postgres:
        logger.warning("Cannot store evaluation result - no database connection")
        return False
        
    try:
        with self.postgres.cursor() as cur:
            cur.execute("""
                INSERT INTO model_evaluations 
                (id, model_version, evaluation_timestamp, statistical_metrics, 
                 business_metrics, operational_metrics, data_quality, overall_score,
                 performance_tier, status, issues, recommendations, vs_champion)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    overall_score = EXCLUDED.overall_score,
                    performance_tier = EXCLUDED.performance_tier,
                    status = EXCLUDED.status,
                    updated_at = CURRENT_TIMESTAMP
            """, (
                result.evaluation_id, result.model_version, result.timestamp,
                json.dumps(result.statistical_metrics), json.dumps(result.business_metrics),
                json.dumps(result.operational_metrics), json.dumps(result.data_quality),
                result.overall_score, result.performance_tier.value, result.status.value,
                result.issues, result.recommendations, 
                json.dumps(result.vs_champion) if result.vs_champion else None
            ))
        
        self.postgres.commit()
        logger.info(f"✅ Evaluation result stored: {result.evaluation_id}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to store evaluation result: {e}")
        return False

async def get_evaluation_history(self,
                               model_version: str = None,
                               days: int = 30) -> List[Dict[str, Any]]:
    """Get evaluation history for analysis and trending"""
    if not self.postgres:
        return []
        
    try:
        cutoff_date = datetime.now() - timedelta(days=days)
        
        with self.postgres.cursor(cursor_factory=RealDictCursor) as cur:
            if model_version:
                cur.execute("""
                    SELECT * FROM model_evaluations 
                    WHERE model_version = %s 
                    AND evaluation_timestamp >= %s
                    ORDER BY evaluation_timestamp DESC
                """, (model_version, cutoff_date))
            else:
                cur.execute("""
                    SELECT * FROM model_evaluations 
                    WHERE evaluation_timestamp >= %s
                    ORDER BY evaluation_timestamp DESC
                """, (cutoff_date,))
            
            rows = cur.fetchall()
            return [dict(row) for row in rows]
            
    except Exception as e:
        logger.error(f"Failed to get evaluation history: {e}")
        return []

async def automated_model_decision(self, evaluation_result: EvaluationResult) -> Dict[str, Any]:
    """Make automated decisions based on evaluation results"""
    decision = {
        'model_version': evaluation_result.model_version,
        'decision_timestamp': datetime.now().isoformat(),
        'recommended_action': 'no_action',
        'confidence': 'low',
        'reasoning': []
    }
    
    # Champion promotion logic
    if (evaluation_result.performance_tier == ModelPerformanceTier.CHALLENGER_WINNING and
        evaluation_result.status == EvaluationStatus.PASS and
        evaluation_result.vs_champion and
        evaluation_result.vs_champion.get('recommendation') == 'promote_to_champion'):
        
        decision['recommended_action'] = 'promote_to_champion'
        decision['confidence'] = 'high'
        decision['reasoning'].append('Challenger significantly outperforms champion across multiple metrics')
    
    # Model retirement logic
    elif (evaluation_result.performance_tier == ModelPerformanceTier.CHALLENGER_FAILING and
          evaluation_result.status == EvaluationStatus.FAIL and
          len(evaluation_result.issues) > 3):
        
        decision['recommended_action'] = 'retire_model'
        decision['confidence'] = 'medium'
        decision['reasoning'].append('Model failing on multiple critical metrics')
    
    # Continue testing logic
    elif (evaluation_result.performance_tier in [ModelPerformanceTier.CHALLENGER_WINNING, 
                                                ModelPerformanceTier.CHALLENGER_ACCEPTABLE] and
          evaluation_result.status in [EvaluationStatus.PASS, EvaluationStatus.WARNING]):
        
        decision['recommended_action'] = 'continue_testing'
        decision['confidence'] = 'medium'
        decision['reasoning'].append('Model shows promise but needs more data for conclusive decision')
    
    # Add specific reasoning based on metrics
    if evaluation_result.statistical_metrics.get('accuracy_5pct', 0) > 0.7:
        decision['reasoning'].append('High prediction accuracy detected')
    
    if evaluation_result.business_metrics.get('sharpe_ratio', 0) > 1.5:
        decision['reasoning'].append('Excellent risk-adjusted returns')
    
    if len(evaluation_result.issues) == 0:
        decision['reasoning'].append('No significant performance issues identified')
    
    return decision

# Add these methods to the ModelEvaluator class
ModelEvaluator._calculate_overall_score = _calculate_overall_score
ModelEvaluator._classify_performance_tier = _classify_performance_tier
ModelEvaluator._identify_issues = _identify_issues
ModelEvaluator._generate_recommendations = _generate_recommendations
ModelEvaluator._compare_with_champion = _compare_with_champion
ModelEvaluator._get_champion_baseline = _get_champion_baseline
ModelEvaluator._determine_evaluation_status = _determine_evaluation_status
ModelEvaluator._store_evaluation_result = _store_evaluation_result
ModelEvaluator.get_evaluation_history = get_evaluation_history
ModelEvaluator.automated_model_decision = automated_model_decision