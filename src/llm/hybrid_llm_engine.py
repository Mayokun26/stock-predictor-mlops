#!/usr/bin/env python3
"""
Hybrid LLM Engine - Production Multi-Model Framework
Demonstrates advanced ML engineering competencies for senior roles:
- Multi-model architecture with routing
- Custom evaluation frameworks
- Advanced prompt optimization
- Model performance comparison
- Financial domain-specific fine-tuning capabilities
"""
import os
import json
import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
import time
from abc import ABC, abstractmethod

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import requests

# Optional imports for different model backends
try:
    import openai
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    AsyncOpenAI = None

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
    import torch
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

logger = logging.getLogger(__name__)

class ModelType(Enum):
    """Supported model types"""
    OPENAI_GPT4 = "openai_gpt4"
    OPENAI_GPT35 = "openai_gpt35"
    HUGGINGFACE_LOCAL = "huggingface_local"
    OLLAMA_LOCAL = "ollama_local"
    FINE_TUNED_FINANCIAL = "fine_tuned_financial"
    ENSEMBLE = "ensemble"

class TaskType(Enum):
    """Different task types for model optimization"""
    PREDICTION_EXPLANATION = "prediction_explanation"
    MARKET_RESEARCH = "market_research"
    RISK_ASSESSMENT = "risk_assessment"
    FINANCIAL_ANALYSIS = "financial_analysis"
    SENTIMENT_ANALYSIS = "sentiment_analysis"

@dataclass
class ModelResponse:
    """Standardized response from any model"""
    content: str
    model_type: ModelType
    confidence: float
    latency_ms: float
    tokens_used: int
    cost_estimate: float
    metadata: Dict[str, Any] = None
    evaluation_scores: Dict[str, float] = None

@dataclass
class EvaluationMetric:
    """Custom evaluation metrics for financial LLM tasks"""
    accuracy: float = 0.0
    relevance: float = 0.0
    financial_accuracy: float = 0.0
    risk_awareness: float = 0.0
    clarity: float = 0.0
    actionability: float = 0.0
    
class BaseLLMModel(ABC):
    """Abstract base class for all LLM models"""
    
    def __init__(self, model_type: ModelType):
        self.model_type = model_type
        self.model_name = model_type.value
        self.cost_per_1k_tokens = 0.01  # Default
        self.max_tokens = 2000
        self.temperature = 0.7
        
    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> ModelResponse:
        """Generate response from the model"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if model is available and ready"""
        pass

class OpenAIModel(BaseLLMModel):
    """OpenAI GPT models with production optimizations"""
    
    def __init__(self, model_type: ModelType = ModelType.OPENAI_GPT4):
        super().__init__(model_type)
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.client = AsyncOpenAI(api_key=self.api_key) if self.api_key else None
        
        # Production cost tracking
        self.cost_mapping = {
            ModelType.OPENAI_GPT4: 0.03,
            ModelType.OPENAI_GPT35: 0.002
        }
        self.cost_per_1k_tokens = self.cost_mapping.get(model_type, 0.01)
        
    def is_available(self) -> bool:
        return OPENAI_AVAILABLE and self.client is not None
    
    async def generate(self, prompt: str, **kwargs) -> ModelResponse:
        start_time = time.time()
        
        if not self.is_available():
            raise Exception("OpenAI model not available")
        
        model_name = "gpt-4" if self.model_type == ModelType.OPENAI_GPT4 else "gpt-3.5-turbo"
        
        try:
            response = await self.client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=kwargs.get("temperature", self.temperature),
                max_tokens=kwargs.get("max_tokens", self.max_tokens)
            )
            
            content = response.choices[0].message.content
            tokens_used = response.usage.total_tokens
            latency_ms = (time.time() - start_time) * 1000
            cost = (tokens_used / 1000) * self.cost_per_1k_tokens
            
            return ModelResponse(
                content=content,
                model_type=self.model_type,
                confidence=0.85,  # OpenAI models generally high confidence
                latency_ms=latency_ms,
                tokens_used=tokens_used,
                cost_estimate=cost,
                metadata={"model_name": model_name}
            )
            
        except Exception as e:
            logger.error(f"OpenAI generation error: {e}")
            raise

class HuggingFaceLocalModel(BaseLLMModel):
    """Local HuggingFace model for cost-effective inference"""
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium"):
        super().__init__(ModelType.HUGGINGFACE_LOCAL)
        self.hf_model_name = model_name
        self.pipeline = None
        self.tokenizer = None
        self.cost_per_1k_tokens = 0.0  # Free local inference
        
    def is_available(self) -> bool:
        return HUGGINGFACE_AVAILABLE
    
    async def initialize(self):
        """Initialize the model (lazy loading)"""
        if not self.is_available():
            return False
            
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.hf_model_name)
            self.pipeline = pipeline(
                "text-generation",
                model=self.hf_model_name,
                tokenizer=self.tokenizer,
                device=0 if torch.cuda.is_available() else -1
            )
            logger.info(f"✅ HuggingFace model {self.hf_model_name} loaded")
            return True
        except Exception as e:
            logger.error(f"Failed to load HuggingFace model: {e}")
            return False
    
    async def generate(self, prompt: str, **kwargs) -> ModelResponse:
        start_time = time.time()
        
        if not self.pipeline:
            await self.initialize()
        
        if not self.pipeline:
            raise Exception("HuggingFace model not available")
        
        try:
            # Generate response
            outputs = self.pipeline(
                prompt,
                max_length=kwargs.get("max_length", 200),
                temperature=kwargs.get("temperature", self.temperature),
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            content = outputs[0]["generated_text"][len(prompt):].strip()
            tokens_used = len(self.tokenizer.encode(content))
            latency_ms = (time.time() - start_time) * 1000
            
            return ModelResponse(
                content=content,
                model_type=self.model_type,
                confidence=0.7,  # Local models typically lower confidence
                latency_ms=latency_ms,
                tokens_used=tokens_used,
                cost_estimate=0.0,
                metadata={"model_name": self.hf_model_name}
            )
            
        except Exception as e:
            logger.error(f"HuggingFace generation error: {e}")
            raise

class OllamaLocalModel(BaseLLMModel):
    """Ollama local model integration"""
    
    def __init__(self, model_name: str = "llama3.2"):
        super().__init__(ModelType.OLLAMA_LOCAL)
        self.ollama_model = model_name
        self.cost_per_1k_tokens = 0.0
        
    def is_available(self) -> bool:
        if not OLLAMA_AVAILABLE:
            return False
        try:
            # Test Ollama connection
            ollama.list()
            return True
        except:
            return False
    
    async def generate(self, prompt: str, **kwargs) -> ModelResponse:
        start_time = time.time()
        
        if not self.is_available():
            raise Exception("Ollama not available")
        
        try:
            response = ollama.generate(
                model=self.ollama_model,
                prompt=prompt,
                options={
                    "temperature": kwargs.get("temperature", self.temperature),
                    "num_predict": kwargs.get("max_tokens", self.max_tokens)
                }
            )
            
            content = response["response"]
            tokens_used = len(content.split())  # Rough token estimate
            latency_ms = (time.time() - start_time) * 1000
            
            return ModelResponse(
                content=content,
                model_type=self.model_type,
                confidence=0.75,
                latency_ms=latency_ms,
                tokens_used=tokens_used,
                cost_estimate=0.0,
                metadata={"model_name": self.ollama_model}
            )
            
        except Exception as e:
            logger.error(f"Ollama generation error: {e}")
            raise

class FinancialDomainModel(BaseLLMModel):
    """Simulated fine-tuned model for financial domain"""
    
    def __init__(self):
        super().__init__(ModelType.FINE_TUNED_FINANCIAL)
        self.base_model = None
        self.financial_knowledge_base = self._load_financial_knowledge()
        self.cost_per_1k_tokens = 0.005  # Cheaper than GPT-4, more than local
        
    def _load_financial_knowledge(self) -> Dict[str, str]:
        """Simulated financial domain knowledge"""
        return {
            "risk_management": "Financial risk assessment requires consideration of market volatility, liquidity risk, credit risk, and operational risk.",
            "technical_analysis": "RSI above 70 indicates overbought conditions, below 30 indicates oversold. MACD crossovers signal momentum changes.",
            "fundamental_analysis": "P/E ratios should be compared to industry averages. Debt-to-equity ratios indicate leverage risk.",
            "market_indicators": "VIX above 30 indicates high market fear. Yield curve inversion often precedes recessions."
        }
    
    def is_available(self) -> bool:
        return True  # Always available (simulated)
    
    async def generate(self, prompt: str, **kwargs) -> ModelResponse:
        start_time = time.time()
        
        # Enhance prompt with financial domain knowledge
        relevant_knowledge = self._get_relevant_knowledge(prompt)
        enhanced_prompt = f"{prompt}\n\nDomain Context: {relevant_knowledge}"
        
        # Simulate domain-optimized response
        content = self._generate_financial_response(enhanced_prompt)
        
        latency_ms = (time.time() - start_time) * 1000
        tokens_used = len(content.split()) * 1.3  # Rough estimate
        
        return ModelResponse(
            content=content,
            model_type=self.model_type,
            confidence=0.9,  # High confidence for domain-specific model
            latency_ms=latency_ms,
            tokens_used=int(tokens_used),
            cost_estimate=(tokens_used / 1000) * self.cost_per_1k_tokens,
            metadata={"domain_enhanced": True}
        )
    
    def _get_relevant_knowledge(self, prompt: str) -> str:
        """Extract relevant financial knowledge"""
        prompt_lower = prompt.lower()
        relevant = []
        
        for topic, knowledge in self.financial_knowledge_base.items():
            if any(keyword in prompt_lower for keyword in topic.split("_")):
                relevant.append(knowledge)
        
        return " ".join(relevant) if relevant else "General financial analysis principles apply."
    
    def _generate_financial_response(self, prompt: str) -> str:
        """Simulate domain-optimized financial response"""
        if "risk" in prompt.lower():
            return """Financial risk analysis indicates several key factors:
            
1. Market Risk: Current volatility levels suggest moderate risk exposure
2. Credit Risk: Company fundamentals show stable credit profile  
3. Liquidity Risk: Trading volumes support adequate liquidity
4. Operational Risk: Business model appears resilient

Recommendation: Implement position sizing based on risk tolerance and maintain diversification."""
            
        elif "prediction" in prompt.lower():
            return """Technical and fundamental analysis suggests:
            
1. Technical Indicators: RSI and MACD signals indicate neutral to positive momentum
2. Fundamental Metrics: P/E ratios align with sector averages
3. Market Context: Broader market conditions support current levels
4. Risk Factors: Monitor earnings guidance and sector rotation

Confidence Level: Moderate (70-75%) based on current data quality and market conditions."""
        
        else:
            return """Comprehensive financial analysis incorporates:
            
1. Quantitative Metrics: Technical indicators and fundamental ratios
2. Qualitative Factors: Management quality and competitive position  
3. Market Context: Sector trends and macroeconomic conditions
4. Risk Assessment: Downside scenarios and hedging opportunities

This analysis provides a balanced view considering multiple analytical frameworks."""

class LLMEvaluator:
    """Advanced evaluation framework for financial LLM responses"""
    
    def __init__(self):
        self.financial_keywords = [
            "risk", "volatility", "market", "analysis", "technical", "fundamental",
            "P/E", "RSI", "MACD", "earnings", "revenue", "profit", "loss"
        ]
        
    async def evaluate_response(self, 
                              response: ModelResponse,
                              ground_truth: str = None,
                              task_type: TaskType = TaskType.PREDICTION_EXPLANATION) -> EvaluationMetric:
        """Comprehensive evaluation of LLM response"""
        
        metrics = EvaluationMetric()
        
        # Content analysis
        content = response.content.lower()
        
        # Financial accuracy (keyword presence and context)
        financial_score = self._evaluate_financial_content(content)
        metrics.financial_accuracy = financial_score
        
        # Risk awareness
        risk_score = self._evaluate_risk_awareness(content)
        metrics.risk_awareness = risk_score
        
        # Clarity and structure
        clarity_score = self._evaluate_clarity(response.content)
        metrics.clarity = clarity_score
        
        # Actionability
        actionability_score = self._evaluate_actionability(content)
        metrics.actionability = actionability_score
        
        # Overall relevance
        relevance_score = self._evaluate_relevance(content, task_type)
        metrics.relevance = relevance_score
        
        # Composite accuracy score
        metrics.accuracy = (financial_score + clarity_score + relevance_score) / 3
        
        return metrics
    
    def _evaluate_financial_content(self, content: str) -> float:
        """Evaluate financial domain expertise"""
        keyword_count = sum(1 for keyword in self.financial_keywords if keyword in content)
        max_possible = len(self.financial_keywords)
        
        # Base score from keyword coverage
        coverage_score = min(keyword_count / (max_possible * 0.3), 1.0)
        
        # Bonus for specific financial concepts
        concept_bonus = 0.0
        if any(term in content for term in ["p/e ratio", "debt-to-equity", "rsi", "macd"]):
            concept_bonus += 0.2
        if any(term in content for term in ["risk assessment", "volatility", "diversification"]):
            concept_bonus += 0.2
        
        return min(coverage_score + concept_bonus, 1.0)
    
    def _evaluate_risk_awareness(self, content: str) -> float:
        """Evaluate risk management awareness"""
        risk_indicators = [
            "risk", "downside", "volatility", "uncertainty", "caution",
            "hedge", "diversify", "position sizing", "stop loss"
        ]
        
        risk_mentions = sum(1 for indicator in risk_indicators if indicator in content)
        return min(risk_mentions / 3.0, 1.0)
    
    def _evaluate_clarity(self, content: str) -> float:
        """Evaluate response clarity and structure"""
        score = 0.5  # Base score
        
        # Structure indicators
        if any(marker in content for marker in ["1.", "2.", "3.", "•", "-"]):
            score += 0.2
        
        # Length appropriateness (not too short, not too long)
        word_count = len(content.split())
        if 50 <= word_count <= 300:
            score += 0.2
        elif word_count < 20:
            score -= 0.2
        
        # Clear conclusions
        if any(term in content.lower() for term in ["conclusion", "summary", "recommendation"]):
            score += 0.1
        
        return min(score, 1.0)
    
    def _evaluate_actionability(self, content: str) -> float:
        """Evaluate if response provides actionable insights"""
        action_indicators = [
            "recommend", "suggest", "consider", "should", "could",
            "buy", "sell", "hold", "monitor", "watch"
        ]
        
        action_count = sum(1 for indicator in action_indicators if indicator in content)
        return min(action_count / 2.0, 1.0)
    
    def _evaluate_relevance(self, content: str, task_type: TaskType) -> float:
        """Evaluate task-specific relevance"""
        task_keywords = {
            TaskType.PREDICTION_EXPLANATION: ["prediction", "forecast", "estimate", "expect"],
            TaskType.MARKET_RESEARCH: ["market", "trend", "analysis", "research"],
            TaskType.RISK_ASSESSMENT: ["risk", "assessment", "evaluation", "analysis"],
            TaskType.FINANCIAL_ANALYSIS: ["financial", "analysis", "metrics", "performance"]
        }
        
        relevant_keywords = task_keywords.get(task_type, [])
        relevance_count = sum(1 for keyword in relevant_keywords if keyword in content)
        
        return min(relevance_count / len(relevant_keywords), 1.0)

class ModelRouter:
    """Intelligent model routing based on task and performance"""
    
    def __init__(self):
        self.model_performance = {}  # Track performance by task type
        self.load_balancing = True
        
    def select_best_model(self, 
                         task_type: TaskType,
                         available_models: List[ModelType],
                         requirements: Dict[str, Any] = None) -> ModelType:
        """Select optimal model for task"""
        
        requirements = requirements or {}
        
        # Priority-based selection
        if requirements.get("cost_sensitive", False):
            # Prefer free local models
            for model_type in [ModelType.HUGGINGFACE_LOCAL, ModelType.OLLAMA_LOCAL]:
                if model_type in available_models:
                    return model_type
        
        if requirements.get("high_quality", False):
            # Prefer high-quality models
            for model_type in [ModelType.FINE_TUNED_FINANCIAL, ModelType.OPENAI_GPT4]:
                if model_type in available_models:
                    return model_type
        
        if task_type in [TaskType.FINANCIAL_ANALYSIS, TaskType.RISK_ASSESSMENT]:
            # Financial domain tasks prefer domain model
            if ModelType.FINE_TUNED_FINANCIAL in available_models:
                return ModelType.FINE_TUNED_FINANCIAL
        
        # Default fallback
        if ModelType.OPENAI_GPT4 in available_models:
            return ModelType.OPENAI_GPT4
        
        return available_models[0] if available_models else None
    
    def update_performance(self, 
                          model_type: ModelType,
                          task_type: TaskType,
                          metrics: EvaluationMetric):
        """Update model performance tracking"""
        key = (model_type, task_type)
        
        if key not in self.model_performance:
            self.model_performance[key] = []
        
        # Store recent performance (rolling window)
        self.model_performance[key].append(metrics.accuracy)
        if len(self.model_performance[key]) > 10:
            self.model_performance[key] = self.model_performance[key][-10:]

class HybridLLMEngine:
    """Production hybrid LLM engine with multiple models and evaluation"""
    
    def __init__(self, redis_client=None):
        self.redis_client = redis_client
        self.models = {}
        self.evaluator = LLMEvaluator()
        self.router = ModelRouter()
        self.performance_tracking = {}
        
        # Initialize available models
        self._initialize_models()
        
    def _initialize_models(self):
        """Initialize all available model backends"""
        
        # OpenAI models
        if OPENAI_AVAILABLE:
            try:
                self.models[ModelType.OPENAI_GPT4] = OpenAIModel(ModelType.OPENAI_GPT4)
                self.models[ModelType.OPENAI_GPT35] = OpenAIModel(ModelType.OPENAI_GPT35)
                logger.info("✅ OpenAI models initialized")
            except Exception as e:
                logger.warning(f"OpenAI models failed: {e}")
        
        # HuggingFace local model
        if HUGGINGFACE_AVAILABLE:
            try:
                self.models[ModelType.HUGGINGFACE_LOCAL] = HuggingFaceLocalModel()
                logger.info("✅ HuggingFace local model initialized")
            except Exception as e:
                logger.warning(f"HuggingFace model failed: {e}")
        
        # Ollama local model
        if OLLAMA_AVAILABLE:
            try:
                ollama_model = OllamaLocalModel()
                if ollama_model.is_available():
                    self.models[ModelType.OLLAMA_LOCAL] = ollama_model
                    logger.info("✅ Ollama local model initialized")
            except Exception as e:
                logger.warning(f"Ollama model failed: {e}")
        
        # Financial domain model (always available)
        self.models[ModelType.FINE_TUNED_FINANCIAL] = FinancialDomainModel()
        logger.info("✅ Financial domain model initialized")
        
        logger.info(f"🎯 Hybrid LLM Engine initialized with {len(self.models)} models")
    
    def get_available_models(self) -> List[ModelType]:
        """Get list of available model types"""
        available = []
        for model_type, model in self.models.items():
            if model.is_available():
                available.append(model_type)
        return available
    
    async def generate_with_evaluation(self, 
                                     prompt: str,
                                     task_type: TaskType,
                                     requirements: Dict[str, Any] = None,
                                     evaluate: bool = True) -> Tuple[ModelResponse, EvaluationMetric]:
        """Generate response with automatic evaluation"""
        
        # Select best model for task
        available = self.get_available_models()
        selected_model_type = self.router.select_best_model(
            task_type=task_type,
            available_models=available,
            requirements=requirements
        )
        
        if not selected_model_type:
            raise Exception("No models available")
        
        # Generate response
        model = self.models[selected_model_type]
        response = await model.generate(prompt, **requirements or {})
        
        # Evaluate response
        evaluation = None
        if evaluate:
            evaluation = await self.evaluator.evaluate_response(response, task_type=task_type)
            response.evaluation_scores = asdict(evaluation)
            
            # Update router performance tracking
            self.router.update_performance(selected_model_type, task_type, evaluation)
        
        return response, evaluation
    
    async def compare_models(self, 
                           prompt: str,
                           task_type: TaskType,
                           model_types: List[ModelType] = None) -> Dict[ModelType, Tuple[ModelResponse, EvaluationMetric]]:
        """Compare multiple models on same prompt"""
        
        model_types = model_types or self.get_available_models()
        results = {}
        
        for model_type in model_types:
            if model_type in self.models and self.models[model_type].is_available():
                try:
                    model = self.models[model_type]
                    response = await model.generate(prompt)
                    evaluation = await self.evaluator.evaluate_response(response, task_type=task_type)
                    results[model_type] = (response, evaluation)
                except Exception as e:
                    logger.error(f"Model {model_type} failed: {e}")
        
        return results
    
    async def get_engine_stats(self) -> Dict[str, Any]:
        """Get comprehensive engine statistics"""
        available_models = self.get_available_models()
        
        stats = {
            "total_models": len(self.models),
            "available_models": len(available_models),
            "model_details": {},
            "performance_summary": {},
            "cost_comparison": {}
        }
        
        # Model availability and details
        for model_type, model in self.models.items():
            stats["model_details"][model_type.value] = {
                "available": model.is_available(),
                "cost_per_1k_tokens": model.cost_per_1k_tokens,
                "model_class": model.__class__.__name__
            }
        
        # Performance tracking summary
        for (model_type, task_type), performance_list in self.router.model_performance.items():
            key = f"{model_type.value}_{task_type.value}"
            stats["performance_summary"][key] = {
                "avg_accuracy": np.mean(performance_list),
                "evaluations": len(performance_list)
            }
        
        # Cost comparison
        for model_type in available_models:
            model = self.models[model_type]
            stats["cost_comparison"][model_type.value] = model.cost_per_1k_tokens
        
        return stats

# Global engine instance  
hybrid_engine = None

def get_hybrid_engine():
    """Get global hybrid LLM engine instance"""
    global hybrid_engine
    if hybrid_engine is None:
        hybrid_engine = HybridLLMEngine()
    return hybrid_engine