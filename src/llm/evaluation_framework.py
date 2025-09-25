#!/usr/bin/env python3
"""
LLM Evaluation Framework for Financial Applications
Advanced evaluation metrics and A/B testing for model comparison
"""
import os
import json
import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import hashlib
import statistics

import numpy as np
from scipy import stats
import redis.asyncio as aioredis

logger = logging.getLogger(__name__)

@dataclass
class ABTestConfig:
    """A/B test configuration"""
    test_name: str
    model_a: str
    model_b: str
    traffic_split: float = 0.5  # 50/50 split
    min_samples: int = 100
    confidence_level: float = 0.95
    max_duration_days: int = 14
    
@dataclass
class ABTestResult:
    """A/B test results"""
    test_name: str
    model_a_performance: Dict[str, float]
    model_b_performance: Dict[str, float]
    statistical_significance: bool
    p_value: float
    confidence_interval: Tuple[float, float]
    winner: Optional[str]
    recommendation: str

class FinancialEvaluationMetrics:
    """Domain-specific evaluation metrics for financial LLM responses"""
    
    @staticmethod
    def calculate_financial_accuracy(response: str, expected_metrics: Dict[str, float]) -> float:
        """Evaluate financial accuracy by checking numerical predictions"""
        
        # Extract numbers from response
        import re
        numbers = re.findall(r'-?\d+\.?\d*', response)
        
        if not numbers or not expected_metrics:
            return 0.0
        
        accuracy_scores = []
        
        # Compare extracted numbers with expected values
        for key, expected_value in expected_metrics.items():
            if key.lower() in response.lower():
                # Find closest number in response
                try:
                    response_numbers = [float(n) for n in numbers]
                    closest_number = min(response_numbers, key=lambda x: abs(x - expected_value))
                    
                    # Calculate relative accuracy
                    if expected_value != 0:
                        relative_error = abs((closest_number - expected_value) / expected_value)
                        accuracy = max(0, 1 - relative_error)
                        accuracy_scores.append(accuracy)
                except ValueError:
                    continue
        
        return statistics.mean(accuracy_scores) if accuracy_scores else 0.0
    
    @staticmethod
    def evaluate_risk_awareness(response: str) -> float:
        """Evaluate risk management content quality"""
        
        risk_concepts = {
            "volatility": 0.2,
            "risk management": 0.25,
            "diversification": 0.2,
            "downside risk": 0.25,
            "position sizing": 0.2,
            "stop loss": 0.15,
            "correlation": 0.15,
            "liquidity risk": 0.2,
            "market risk": 0.2,
            "credit risk": 0.15
        }
        
        score = 0.0
        response_lower = response.lower()
        
        for concept, weight in risk_concepts.items():
            if concept in response_lower:
                score += weight
        
        # Bonus for mentioning specific risk percentages
        import re
        if re.search(r'\d+%.*risk', response_lower):
            score += 0.1
        
        return min(score, 1.0)
    
    @staticmethod
    def evaluate_actionability(response: str) -> float:
        """Evaluate if response provides actionable investment insights"""
        
        actionable_terms = {
            "buy": 0.3,
            "sell": 0.3,
            "hold": 0.2,
            "recommend": 0.25,
            "target price": 0.3,
            "stop loss": 0.25,
            "entry point": 0.25,
            "exit strategy": 0.25,
            "position size": 0.2,
            "allocation": 0.2,
            "rebalance": 0.15
        }
        
        score = 0.0
        response_lower = response.lower()
        
        for term, weight in actionable_terms.items():
            if term in response_lower:
                score += weight
        
        return min(score, 1.0)
    
    @staticmethod
    def evaluate_market_context(response: str) -> float:
        """Evaluate incorporation of broader market context"""
        
        context_indicators = {
            "market conditions": 0.25,
            "economic indicators": 0.25,
            "sector performance": 0.2,
            "market sentiment": 0.2,
            "macro environment": 0.2,
            "fed policy": 0.15,
            "interest rates": 0.15,
            "inflation": 0.15,
            "gdp": 0.1,
            "employment": 0.1
        }
        
        score = 0.0
        response_lower = response.lower()
        
        for indicator, weight in context_indicators.items():
            if indicator in response_lower:
                score += weight
        
        return min(score, 1.0)

class ModelComparisonFramework:
    """Framework for comparing different LLM models"""
    
    def __init__(self, redis_client=None):
        self.redis_client = redis_client
        self.evaluation_metrics = FinancialEvaluationMetrics()
        
    async def compare_model_responses(self, 
                                    prompt: str,
                                    responses: Dict[str, str],
                                    ground_truth: Dict[str, Any] = None) -> Dict[str, Dict[str, float]]:
        """Compare multiple model responses on same prompt"""
        
        comparison_results = {}
        
        for model_name, response in responses.items():
            scores = {
                "financial_accuracy": 0.0,
                "risk_awareness": self.evaluation_metrics.evaluate_risk_awareness(response),
                "actionability": self.evaluation_metrics.evaluate_actionability(response),
                "market_context": self.evaluation_metrics.evaluate_market_context(response),
                "length_score": self._evaluate_response_length(response),
                "clarity_score": self._evaluate_clarity(response)
            }
            
            if ground_truth:
                scores["financial_accuracy"] = self.evaluation_metrics.calculate_financial_accuracy(
                    response, ground_truth.get("expected_metrics", {})
                )
            
            # Composite score
            scores["composite_score"] = statistics.mean([
                scores["risk_awareness"],
                scores["actionability"], 
                scores["market_context"],
                scores["clarity_score"]
            ])
            
            comparison_results[model_name] = scores
        
        return comparison_results
    
    def _evaluate_response_length(self, response: str) -> float:
        """Evaluate if response length is appropriate"""
        word_count = len(response.split())
        
        if 50 <= word_count <= 200:
            return 1.0
        elif 200 < word_count <= 300:
            return 0.8
        elif 30 <= word_count < 50:
            return 0.7
        elif word_count > 300:
            return 0.6
        else:
            return 0.3
    
    def _evaluate_clarity(self, response: str) -> float:
        """Evaluate response clarity and structure"""
        score = 0.5
        
        # Structure indicators
        if any(marker in response for marker in ["1.", "2.", "3.", "•", "-"]):
            score += 0.2
        
        # Clear sections
        if any(header in response.lower() for header in ["analysis", "recommendation", "conclusion"]):
            score += 0.2
        
        # Logical flow indicators
        if any(connector in response.lower() for connector in ["therefore", "however", "moreover", "consequently"]):
            score += 0.1
        
        return min(score, 1.0)

class ABTestingFramework:
    """A/B testing framework for model comparison"""
    
    def __init__(self, redis_client=None):
        self.redis_client = redis_client
        self.active_tests = {}
        
    async def create_ab_test(self, config: ABTestConfig) -> str:
        """Create new A/B test"""
        test_id = hashlib.md5(f"{config.test_name}_{datetime.utcnow()}".encode()).hexdigest()[:10]
        
        test_data = {
            "config": asdict(config),
            "start_time": datetime.utcnow().isoformat(),
            "status": "active",
            "results_a": [],
            "results_b": [],
            "total_requests": 0
        }
        
        if self.redis_client:
            await self.redis_client.setex(
                f"ab_test:{test_id}",
                86400 * config.max_duration_days,
                json.dumps(test_data)
            )
        
        self.active_tests[test_id] = test_data
        logger.info(f"🧪 Created A/B test: {config.test_name} ({test_id})")
        
        return test_id
    
    async def record_test_result(self, 
                               test_id: str,
                               model_used: str,
                               performance_metrics: Dict[str, float]) -> None:
        """Record result for A/B test"""
        
        test_data = await self._get_test_data(test_id)
        if not test_data:
            return
        
        config = ABTestConfig(**test_data["config"])
        test_data["total_requests"] += 1
        
        # Determine which bucket this result belongs to
        if model_used == config.model_a:
            test_data["results_a"].append(performance_metrics)
        elif model_used == config.model_b:
            test_data["results_b"].append(performance_metrics)
        
        # Store updated data
        if self.redis_client:
            await self.redis_client.setex(
                f"ab_test:{test_id}",
                86400 * config.max_duration_days,
                json.dumps(test_data)
            )
    
    async def analyze_ab_test(self, test_id: str) -> ABTestResult:
        """Analyze A/B test results"""
        
        test_data = await self._get_test_data(test_id)
        if not test_data:
            raise ValueError(f"Test {test_id} not found")
        
        config = ABTestConfig(**test_data["config"])
        results_a = test_data["results_a"]
        results_b = test_data["results_b"]
        
        if len(results_a) < config.min_samples or len(results_b) < config.min_samples:
            return ABTestResult(
                test_name=config.test_name,
                model_a_performance={},
                model_b_performance={},
                statistical_significance=False,
                p_value=1.0,
                confidence_interval=(0.0, 0.0),
                winner=None,
                recommendation="Insufficient data - continue test"
            )
        
        # Calculate performance metrics for each model
        model_a_perf = self._aggregate_performance(results_a)
        model_b_perf = self._aggregate_performance(results_b)
        
        # Statistical significance test
        a_scores = [r.get("composite_score", 0) for r in results_a]
        b_scores = [r.get("composite_score", 0) for r in results_b]
        
        statistic, p_value = stats.ttest_ind(a_scores, b_scores)
        
        is_significant = p_value < (1 - config.confidence_level)
        
        # Determine winner
        winner = None
        if is_significant:
            if model_a_perf["composite_score"] > model_b_perf["composite_score"]:
                winner = config.model_a
            else:
                winner = config.model_b
        
        # Confidence interval for difference in means
        diff_mean = model_b_perf["composite_score"] - model_a_perf["composite_score"]
        pooled_std = np.sqrt((np.var(a_scores) + np.var(b_scores)) / 2)
        margin_error = stats.t.ppf(config.confidence_level, len(a_scores) + len(b_scores) - 2) * pooled_std
        conf_interval = (diff_mean - margin_error, diff_mean + margin_error)
        
        # Generate recommendation
        recommendation = self._generate_recommendation(
            is_significant, winner, model_a_perf, model_b_perf, p_value
        )
        
        return ABTestResult(
            test_name=config.test_name,
            model_a_performance=model_a_perf,
            model_b_performance=model_b_perf,
            statistical_significance=is_significant,
            p_value=p_value,
            confidence_interval=conf_interval,
            winner=winner,
            recommendation=recommendation
        )
    
    async def _get_test_data(self, test_id: str) -> Dict[str, Any]:
        """Get test data from Redis or memory"""
        if self.redis_client:
            data = await self.redis_client.get(f"ab_test:{test_id}")
            if data:
                return json.loads(data)
        
        return self.active_tests.get(test_id)
    
    def _aggregate_performance(self, results: List[Dict[str, float]]) -> Dict[str, float]:
        """Aggregate performance metrics"""
        if not results:
            return {}
        
        metrics = {}
        for key in results[0].keys():
            values = [r.get(key, 0) for r in results]
            metrics[key] = statistics.mean(values)
        
        return metrics
    
    def _generate_recommendation(self, 
                               is_significant: bool,
                               winner: Optional[str],
                               perf_a: Dict[str, float],
                               perf_b: Dict[str, float],
                               p_value: float) -> str:
        """Generate actionable recommendation"""
        
        if not is_significant:
            return f"No statistically significant difference (p={p_value:.3f}). Continue testing or investigate other factors."
        
        if winner:
            a_score = perf_a.get("composite_score", 0)
            b_score = perf_b.get("composite_score", 0)
            improvement = abs((b_score - a_score) / a_score) * 100 if a_score > 0 else 0
            
            return f"Winner: {winner} with {improvement:.1f}% better performance (p={p_value:.3f}). Recommend full deployment."
        
        return "Inconclusive results despite statistical significance. Review test design."

class PromptOptimizationFramework:
    """Framework for optimizing prompts through systematic testing"""
    
    def __init__(self, redis_client=None):
        self.redis_client = redis_client
        
    async def test_prompt_variants(self, 
                                 base_prompt: str,
                                 variants: Dict[str, str],
                                 test_cases: List[Dict[str, Any]],
                                 model_generator) -> Dict[str, float]:
        """Test multiple prompt variants"""
        
        results = {}
        
        for variant_name, prompt_template in variants.items():
            variant_scores = []
            
            for test_case in test_cases:
                # Format prompt with test case
                formatted_prompt = prompt_template.format(**test_case.get("inputs", {}))
                
                try:
                    # Generate response
                    response = await model_generator(formatted_prompt)
                    
                    # Evaluate response
                    score = await self._evaluate_prompt_response(
                        response, test_case.get("expected_output")
                    )
                    variant_scores.append(score)
                    
                except Exception as e:
                    logger.error(f"Error testing variant {variant_name}: {e}")
                    variant_scores.append(0.0)
            
            results[variant_name] = statistics.mean(variant_scores) if variant_scores else 0.0
        
        return results
    
    async def _evaluate_prompt_response(self, 
                                      response: str,
                                      expected: Dict[str, Any] = None) -> float:
        """Evaluate prompt response quality"""
        
        # Basic quality metrics
        score = 0.0
        
        # Length appropriateness
        word_count = len(response.split())
        if 30 <= word_count <= 300:
            score += 0.3
        
        # Structure quality
        if any(marker in response for marker in ["1.", "2.", "•", "-"]):
            score += 0.2
        
        # Financial content
        financial_terms = ["analysis", "risk", "market", "investment", "price", "performance"]
        financial_score = sum(1 for term in financial_terms if term in response.lower()) / len(financial_terms)
        score += financial_score * 0.3
        
        # Clarity indicators
        if any(word in response.lower() for word in ["therefore", "however", "conclusion", "recommendation"]):
            score += 0.2
        
        return min(score, 1.0)

# Global instances
evaluation_framework = None
comparison_framework = None
ab_testing_framework = None
prompt_optimization = None

def get_evaluation_frameworks():
    """Get global evaluation framework instances"""
    global evaluation_framework, comparison_framework, ab_testing_framework, prompt_optimization
    
    if evaluation_framework is None:
        evaluation_framework = FinancialEvaluationMetrics()
        comparison_framework = ModelComparisonFramework()
        ab_testing_framework = ABTestingFramework()
        prompt_optimization = PromptOptimizationFramework()
    
    return evaluation_framework, comparison_framework, ab_testing_framework, prompt_optimization