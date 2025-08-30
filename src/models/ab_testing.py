"""
Enterprise A/B Testing Framework with Statistical Significance Testing
Production-grade A/B testing for ML model comparison and optimization
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass, asdict
from enum import Enum
from scipy import stats
from scipy.stats import ttest_ind
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestStatus(Enum):
    """A/B test status"""
    DRAFT = "draft"
    RUNNING = "running"
    COMPLETED = "completed"
    PAUSED = "paused"
    STOPPED = "stopped"

class TestResult(Enum):
    """A/B test result"""
    CHAMPION_WINS = "champion_wins"
    CHALLENGER_WINS = "challenger_wins"
    INCONCLUSIVE = "inconclusive"
    INSUFFICIENT_DATA = "insufficient_data"

@dataclass
class ABTestConfig:
    """A/B test configuration"""
    name: str
    description: str
    champion_model_id: str
    challenger_model_id: str
    traffic_split: float = 0.5
    success_metric: str = "accuracy"
    minimum_sample_size: int = 1000
    significance_level: float = 0.05
    power: float = 0.8
    minimum_effect_size: float = 0.02
    max_duration_days: int = 30
    early_stopping: bool = True
    metadata: Dict[str, Any] = None

@dataclass
class TestMetrics:
    """Metrics for A/B test analysis"""
    metric_name: str
    champion_value: float
    challenger_value: float
    champion_samples: int
    challenger_samples: int
    difference: float
    relative_difference: float
    confidence_interval: Tuple[float, float]
    p_value: float
    significance_level: float
    is_significant: bool
    effect_size: float
    power: float

@dataclass
class ABTestResult:
    """Complete A/B test result"""
    test_name: str
    status: TestStatus
    result: TestResult
    start_date: datetime
    end_date: Optional[datetime]
    duration_days: int
    champion_model_id: str
    challenger_model_id: str
    primary_metric: TestMetrics
    secondary_metrics: List[TestMetrics]
    total_samples: int
    traffic_split: float
    statistical_power: float
    recommendation: str
    confidence_score: float
    metadata: Dict[str, Any]

class StatisticalTest:
    """Statistical testing utilities"""
    
    @staticmethod
    def welch_t_test(group_a: np.ndarray, group_b: np.ndarray, alpha: float = 0.05) -> Dict[str, Any]:
        """Welch's t-test for unequal variances"""
        try:
            statistic, p_value = ttest_ind(group_a, group_b, equal_var=False)
            
            # Effect size (Cohen's d)
            pooled_std = np.sqrt((np.var(group_a, ddof=1) + np.var(group_b, ddof=1)) / 2)
            effect_size = (np.mean(group_a) - np.mean(group_b)) / pooled_std if pooled_std != 0 else 0
            
            # Confidence interval for difference in means
            diff_mean = np.mean(group_a) - np.mean(group_b)
            se_diff = np.sqrt(np.var(group_a, ddof=1) / len(group_a) + np.var(group_b, ddof=1) / len(group_b))
            
            t_critical = stats.t.ppf(1 - alpha/2, max(len(group_a) + len(group_b) - 2, 1))
            ci_lower = diff_mean - t_critical * se_diff
            ci_upper = diff_mean + t_critical * se_diff
            
            return {
                'statistic': statistic,
                'p_value': p_value,
                'effect_size': effect_size,
                'confidence_interval': (ci_lower, ci_upper),
                'is_significant': p_value < alpha
            }
        except Exception as e:
            logger.error(f"Error in Welch's t-test: {e}")
            return {
                'statistic': 0, 
                'p_value': 1.0, 
                'effect_size': 0, 
                'confidence_interval': (0, 0), 
                'is_significant': False
            }

class ABTestManager:
    """A/B test manager for ML model comparison"""
    
    def __init__(self, database_manager=None):
        self.database_manager = database_manager
        self.active_tests = {}
        self.test_results = {}
        self.statistical_test = StatisticalTest()
    
    def create_test(self, config: ABTestConfig) -> str:
        """Create a new A/B test"""
        try:
            test_id = f"test_{config.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            if not self._validate_config(config):
                raise ValueError("Invalid A/B test configuration")
            
            test_data = {
                'test_id': test_id,
                'config': config,
                'status': TestStatus.DRAFT,
                'created_at': datetime.now(),
                'champion_data': [],
                'challenger_data': [],
                'metrics_history': []
            }
            
            self.active_tests[test_id] = test_data
            logger.info(f"Created A/B test: {test_id}")
            
            return test_id
            
        except Exception as e:
            logger.error(f"Error creating A/B test: {e}")
            return ""
    
    def start_test(self, test_id: str) -> bool:
        """Start an A/B test"""
        try:
            if test_id not in self.active_tests:
                return False
            
            test_data = self.active_tests[test_id]
            test_data['status'] = TestStatus.RUNNING
            test_data['start_time'] = datetime.now()
            
            logger.info(f"Started A/B test: {test_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error starting test: {e}")
            return False
    
    def record_observation(self, test_id: str, model_variant: str, metric_name: str,
                          metric_value: float, timestamp: datetime = None) -> bool:
        """Record a single observation for A/B test"""
        try:
            if test_id not in self.active_tests:
                return False
            
            test_data = self.active_tests[test_id]
            
            if test_data['status'] != TestStatus.RUNNING:
                return False
            
            observation = {
                'metric_name': metric_name,
                'metric_value': metric_value,
                'timestamp': timestamp or datetime.now(),
                'metadata': {}
            }
            
            if model_variant == 'champion':
                test_data['champion_data'].append(observation)
            elif model_variant == 'challenger':
                test_data['challenger_data'].append(observation)
            else:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error recording observation: {e}")
            return False
    
    def analyze_test(self, test_id: str) -> Optional[ABTestResult]:
        """Analyze A/B test results"""
        try:
            if test_id not in self.active_tests:
                return None
            
            test_data = self.active_tests[test_id]
            config = test_data['config']
            
            # Extract primary metric data
            champion_values = [obs['metric_value'] for obs in test_data['champion_data'] 
                             if obs['metric_name'] == config.success_metric]
            challenger_values = [obs['metric_value'] for obs in test_data['challenger_data'] 
                               if obs['metric_name'] == config.success_metric]
            
            if len(champion_values) == 0 or len(challenger_values) == 0:
                return self._create_insufficient_data_result(test_data)
            
            # Primary metric analysis
            primary_metric = self._analyze_metric(
                config.success_metric,
                np.array(champion_values),
                np.array(challenger_values),
                config.significance_level
            )
            
            # Determine result
            result = self._determine_test_result(primary_metric, config)
            
            # Generate recommendation
            recommendation = self._generate_recommendation(primary_metric, config)
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(primary_metric)
            
            # Create result object
            ab_result = ABTestResult(
                test_name=config.name,
                status=test_data['status'],
                result=result,
                start_date=test_data.get('start_time', test_data['created_at']),
                end_date=datetime.now() if test_data['status'] == TestStatus.COMPLETED else None,
                duration_days=(datetime.now() - test_data.get('start_time', test_data['created_at'])).days,
                champion_model_id=config.champion_model_id,
                challenger_model_id=config.challenger_model_id,
                primary_metric=primary_metric,
                secondary_metrics=[],
                total_samples=len(champion_values) + len(challenger_values),
                traffic_split=config.traffic_split,
                statistical_power=primary_metric.power,
                recommendation=recommendation,
                confidence_score=confidence_score,
                metadata=config.metadata or {}
            )
            
            return ab_result
            
        except Exception as e:
            logger.error(f"Error analyzing test: {e}")
            return None
    
    def _validate_config(self, config: ABTestConfig) -> bool:
        """Validate A/B test configuration"""
        try:
            if not config.name or not config.champion_model_id or not config.challenger_model_id:
                return False
            
            if config.traffic_split <= 0 or config.traffic_split >= 1:
                return False
            
            if config.significance_level <= 0 or config.significance_level >= 1:
                return False
            
            return True
        except:
            return False
    
    def _analyze_metric(self, metric_name: str, champion_data: np.ndarray, 
                       challenger_data: np.ndarray, alpha: float) -> TestMetrics:
        """Analyze a single metric"""
        try:
            # Basic statistics
            champion_mean = np.mean(champion_data)
            challenger_mean = np.mean(challenger_data)
            difference = challenger_mean - champion_mean
            relative_difference = difference / champion_mean if champion_mean != 0 else 0
            
            # Statistical test
            test_result = self.statistical_test.welch_t_test(challenger_data, champion_data, alpha)
            
            # Effect size
            effect_size = test_result.get('effect_size', 0)
            
            # Simple power estimation
            power = min(0.99, max(0.05, 1 - test_result.get('p_value', 1.0)))
            
            return TestMetrics(
                metric_name=metric_name,
                champion_value=champion_mean,
                challenger_value=challenger_mean,
                champion_samples=len(champion_data),
                challenger_samples=len(challenger_data),
                difference=difference,
                relative_difference=relative_difference,
                confidence_interval=test_result.get('confidence_interval', (0, 0)),
                p_value=test_result.get('p_value', 1.0),
                significance_level=alpha,
                is_significant=test_result.get('is_significant', False),
                effect_size=effect_size,
                power=power
            )
            
        except Exception as e:
            logger.error(f"Error analyzing metric {metric_name}: {e}")
            return TestMetrics(
                metric_name=metric_name,
                champion_value=0.0,
                challenger_value=0.0,
                champion_samples=0,
                challenger_samples=0,
                difference=0.0,
                relative_difference=0.0,
                confidence_interval=(0.0, 0.0),
                p_value=1.0,
                significance_level=alpha,
                is_significant=False,
                effect_size=0.0,
                power=0.0
            )
    
    def _determine_test_result(self, primary_metric: TestMetrics, config: ABTestConfig) -> TestResult:
        """Determine A/B test result"""
        if primary_metric.challenger_samples + primary_metric.champion_samples < config.minimum_sample_size:
            return TestResult.INSUFFICIENT_DATA
        
        if not primary_metric.is_significant:
            return TestResult.INCONCLUSIVE
        
        if primary_metric.challenger_value > primary_metric.champion_value:
            return TestResult.CHALLENGER_WINS
        else:
            return TestResult.CHAMPION_WINS
    
    def _generate_recommendation(self, primary_metric: TestMetrics, config: ABTestConfig) -> str:
        """Generate recommendation based on test results"""
        if primary_metric.challenger_samples + primary_metric.champion_samples < config.minimum_sample_size:
            remaining = config.minimum_sample_size - (primary_metric.challenger_samples + primary_metric.champion_samples)
            return f"Continue test - need {remaining} more samples"
        
        if primary_metric.is_significant:
            winner = "challenger" if primary_metric.challenger_value > primary_metric.champion_value else "champion"
            improvement = abs(primary_metric.relative_difference) * 100
            confidence = (1-primary_metric.p_value)*100
            return f"Deploy {winner} model - {improvement:.1f}% improvement with {confidence:.1f}% confidence"
        else:
            return f"No significant difference detected (p={primary_metric.p_value:.3f}). Consider running longer."
    
    def _calculate_confidence_score(self, primary_metric: TestMetrics) -> float:
        """Calculate overall confidence score for the test"""
        try:
            p_score = 1 - primary_metric.p_value
            effect_score = min(abs(primary_metric.effect_size), 1.0)
            sample_score = min((primary_metric.challenger_samples + primary_metric.champion_samples) / 1000, 1.0)
            
            confidence = 0.5 * p_score + 0.3 * effect_score + 0.2 * sample_score
            return max(0.0, min(1.0, confidence))
        except:
            return 0.5
    
    def _create_insufficient_data_result(self, test_data: Dict) -> ABTestResult:
        """Create result for insufficient data case"""
        config = test_data['config']
        
        return ABTestResult(
            test_name=config.name,
            status=test_data['status'],
            result=TestResult.INSUFFICIENT_DATA,
            start_date=test_data.get('start_time', test_data['created_at']),
            end_date=None,
            duration_days=(datetime.now() - test_data.get('start_time', test_data['created_at'])).days,
            champion_model_id=config.champion_model_id,
            challenger_model_id=config.challenger_model_id,
            primary_metric=TestMetrics(
                metric_name=config.success_metric,
                champion_value=0.0,
                challenger_value=0.0,
                champion_samples=len(test_data.get('champion_data', [])),
                challenger_samples=len(test_data.get('challenger_data', [])),
                difference=0.0,
                relative_difference=0.0,
                confidence_interval=(0.0, 0.0),
                p_value=1.0,
                significance_level=config.significance_level,
                is_significant=False,
                effect_size=0.0,
                power=0.0
            ),
            secondary_metrics=[],
            total_samples=len(test_data.get('champion_data', [])) + len(test_data.get('challenger_data', [])),
            traffic_split=config.traffic_split,
            statistical_power=0.0,
            recommendation="Insufficient data - continue collecting samples",
            confidence_score=0.0,
            metadata=config.metadata or {}
        )
    
    def stop_test(self, test_id: str) -> bool:
        """Stop an A/B test"""
        try:
            if test_id in self.active_tests:
                self.active_tests[test_id]['status'] = TestStatus.COMPLETED
                self.active_tests[test_id]['end_time'] = datetime.now()
                
                result = self.analyze_test(test_id)
                if result:
                    self.test_results[test_id] = result
                
                logger.info(f"Stopped A/B test: {test_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error stopping test: {e}")
            return False
    
    def get_test_summary(self, test_id: str) -> Dict[str, Any]:
        """Get test summary"""
        if test_id not in self.active_tests:
            return {}
        
        test_data = self.active_tests[test_id]
        config = test_data['config']
        
        return {
            'test_id': test_id,
            'name': config.name,
            'status': test_data['status'].value,
            'champion_samples': len(test_data.get('champion_data', [])),
            'challenger_samples': len(test_data.get('challenger_data', [])),
            'total_samples': len(test_data.get('champion_data', [])) + len(test_data.get('challenger_data', [])),
            'required_samples': config.minimum_sample_size,
            'duration_days': (datetime.now() - test_data.get('start_time', test_data['created_at'])).days,
            'traffic_split': config.traffic_split
        }
    
    def assign_user_to_test(self, user_id: str, test_id: str = None) -> str:
        """Assign user to A/B test variant"""
        try:
            # Simple hash-based assignment for consistent user experience
            import hashlib
            
            # If no specific test, use default behavior (champion)
            if test_id and test_id in self.active_tests:
                config = self.active_tests[test_id]['config']
                
                # Hash user ID to get consistent assignment
                user_hash = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
                assignment_value = (user_hash % 100) / 100.0
                
                if assignment_value < config.traffic_split:
                    return 'challenger'
                else:
                    return 'champion'
            else:
                # Default to champion if no active test
                return 'champion'
                
        except Exception as e:
            logger.error(f"Error assigning user to test: {e}")
            return 'champion'  # Default to champion on error

def test_ab_testing():
    """Test A/B testing framework"""
    ab_manager = ABTestManager()
    
    config = ABTestConfig(
        name="xgboost_vs_ensemble",
        description="Compare XGBoost model against ensemble",
        champion_model_id="xgboost_v1",
        challenger_model_id="ensemble_v1",
        traffic_split=0.5,
        success_metric="accuracy",
        minimum_sample_size=100,
        significance_level=0.05
    )
    
    test_id = ab_manager.create_test(config)
    print(f"Created test: {test_id}")
    
    ab_manager.start_test(test_id)
    
    # Simulate observations
    np.random.seed(42)
    
    for _ in range(500):
        accuracy = np.random.normal(0.75, 0.05)
        ab_manager.record_observation(test_id, 'champion', 'accuracy', accuracy)
    
    for _ in range(500):
        accuracy = np.random.normal(0.77, 0.05)
        ab_manager.record_observation(test_id, 'challenger', 'accuracy', accuracy)
    
    result = ab_manager.analyze_test(test_id)
    if result:
        print(f"Test Result: {result.result.value}")
        print(f"Champion: {result.primary_metric.champion_value:.4f}")
        print(f"Challenger: {result.primary_metric.challenger_value:.4f}")
        print(f"P-value: {result.primary_metric.p_value:.4f}")
        print(f"Recommendation: {result.recommendation}")

if __name__ == "__main__":
    test_ab_testing()