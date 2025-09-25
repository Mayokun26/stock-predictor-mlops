#!/usr/bin/env python3
"""
LLM Service for Prediction Explanations and Market Analysis
Supports OpenAI, Anthropic, and local models with fallback strategies
"""
import os
import json
import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from dataclasses import dataclass, asdict
import hashlib

import openai
from openai import AsyncOpenAI
import tiktoken

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEFAULT_MODEL = os.getenv("LLM_DEFAULT_MODEL", "gpt-4")
MAX_RETRIES = 3
REQUEST_TIMEOUT = 30

logger = logging.getLogger(__name__)

@dataclass
class LLMRequest:
    """Structured LLM request with caching support"""
    prompt: str
    model: str = DEFAULT_MODEL
    temperature: float = 0.7
    max_tokens: int = 500
    system_prompt: Optional[str] = None
    context: Optional[Dict] = None
    
    def cache_key(self) -> str:
        """Generate cache key for this request"""
        content = f"{self.prompt}_{self.model}_{self.temperature}_{self.max_tokens}_{self.system_prompt}"
        return hashlib.md5(content.encode()).hexdigest()

@dataclass
class LLMResponse:
    """Structured LLM response with metadata"""
    content: str
    model: str
    tokens_used: int
    cost_estimate: float
    cached: bool = False
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()

class PromptTemplate:
    """Template management for consistent prompting"""
    
    PREDICTION_EXPLANATION = """
    You are a senior quantitative analyst explaining stock price predictions.
    
    PREDICTION DATA:
    - Symbol: {symbol}
    - Current Price: ${current_price:.2f}
    - Predicted Price: ${predicted_price:.2f}
    - Confidence: {confidence:.1%}
    - Price Change: {price_change:+.2f} ({price_change_pct:+.1%})
    
    TECHNICAL INDICATORS:
    {technical_indicators}
    
    MARKET SENTIMENT:
    - News Headlines: {news_headlines}
    - Sentiment Score: {sentiment_score:.2f}
    
    MODEL DETAILS:
    - Model Type: {model_type}
    - Feature Importance: {feature_importance}
    - Prediction Timestamp: {timestamp}
    
    Provide a clear, professional explanation of this prediction in 2-3 paragraphs:
    1. Key factors driving this prediction
    2. Technical analysis reasoning
    3. Risk factors and confidence assessment
    
    Be specific about numbers and maintain professional financial analysis tone.
    """
    
    MARKET_RESEARCH = """
    You are a research analyst providing market context for trading decisions.
    
    QUERY: {query}
    
    RELEVANT CONTEXT:
    {context}
    
    MARKET DATA:
    {market_data}
    
    Provide comprehensive market analysis including:
    1. Current market conditions
    2. Key trends and patterns
    3. Risk factors and opportunities
    4. Actionable insights
    
    Focus on factual analysis with specific data points and clear reasoning.
    """
    
    RISK_ASSESSMENT = """
    Analyze the risk profile of this prediction:
    
    PREDICTION: {symbol} from ${current_price:.2f} to ${predicted_price:.2f}
    VOLATILITY: {volatility:.2f}
    MARKET CAP: ${market_cap:,.0f}
    SECTOR: {sector}
    
    RECENT PERFORMANCE:
    {performance_data}
    
    Provide risk assessment covering:
    1. Probability of target achievement
    2. Downside risk scenarios
    3. Key risk factors to monitor
    4. Recommended position sizing
    """

class LLMService:
    """Production-grade LLM service with hybrid multi-model support"""
    
    def __init__(self, redis_client=None, rag_service=None):
        self.redis_client = redis_client
        self.rag_service = rag_service
        self.openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
        self.tokenizer = tiktoken.encoding_for_model("gpt-4") if tiktoken else None
        
        # Initialize hybrid engine
        try:
            from hybrid_llm_engine import HybridLLMEngine, TaskType
            self.hybrid_engine = HybridLLMEngine(redis_client)
            self.TaskType = TaskType
            self.hybrid_available = True
            logger.info("✅ Hybrid LLM engine initialized")
        except ImportError as e:
            self.hybrid_engine = None
            self.TaskType = None
            self.hybrid_available = False
            logger.warning(f"Hybrid engine not available: {e}")
        
        # Cost tracking (estimated prices per 1K tokens)
        self.cost_per_1k_tokens = {
            "gpt-4": 0.03,
            "gpt-4-turbo": 0.01,
            "gpt-3.5-turbo": 0.002,
        }
        
        if not self.openai_client and not self.hybrid_available:
            logger.warning("No LLM backends available - check configuration")
    
    async def generate_explanation(self, 
                                 prediction_data: Dict[str, Any],
                                 use_cache: bool = True,
                                 use_hybrid: bool = True) -> LLMResponse:
        """Generate explanation for stock prediction"""
        
        # Format technical indicators for display
        tech_indicators = self._format_technical_indicators(
            prediction_data.get('technical_indicators', {})
        )
        
        # Create prompt from template
        prompt = PromptTemplate.PREDICTION_EXPLANATION.format(
            symbol=prediction_data.get('symbol', 'UNKNOWN'),
            current_price=prediction_data.get('current_price', 0),
            predicted_price=prediction_data.get('predicted_price', 0),
            confidence=prediction_data.get('confidence', 0),
            price_change=prediction_data.get('predicted_price', 0) - prediction_data.get('current_price', 0),
            price_change_pct=(prediction_data.get('predicted_price', 0) - prediction_data.get('current_price', 0)) / prediction_data.get('current_price', 1),
            technical_indicators=tech_indicators,
            news_headlines=prediction_data.get('news_headlines', []),
            sentiment_score=prediction_data.get('sentiment_score', 0),
            model_type=prediction_data.get('model_type', 'Unknown'),
            feature_importance=prediction_data.get('feature_importance', {}),
            timestamp=datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
        )
        
        request = LLMRequest(
            prompt=prompt,
            system_prompt="You are a professional quantitative analyst with expertise in stock market prediction and technical analysis.",
            temperature=0.3,  # Lower temperature for more consistent explanations
            max_tokens=600
        )
        
        # Use hybrid engine if available and requested
        if use_hybrid and self.hybrid_available:
            try:
                response, evaluation = await self.hybrid_engine.generate_with_evaluation(
                    prompt=request.prompt,
                    task_type=self.TaskType.PREDICTION_EXPLANATION,
                    requirements={"temperature": request.temperature},
                    evaluate=True
                )
                
                return LLMResponse(
                    content=response.content,
                    model=response.model_type.value,
                    tokens_used=response.tokens_used,
                    cost_estimate=response.cost_estimate,
                    cached=False,
                    timestamp=datetime.utcnow()
                )
                
            except Exception as e:
                logger.error(f"Hybrid engine failed, falling back: {e}")
        
        return await self._execute_request(request, use_cache=use_cache)
    
    async def analyze_market_research(self, 
                                    query: str,
                                    context_data: Dict[str, Any],
                                    use_cache: bool = True,
                                    use_rag: bool = True) -> LLMResponse:
        """Analyze market research query with context and RAG enhancement"""
        
        # Enhance query with RAG if available
        enhanced_context = context_data
        if use_rag and self.rag_service:
            try:
                symbols = context_data.get('symbols', [])
                rag_enhancement = await self.rag_service.enhance_market_query(
                    query=query,
                    symbols=symbols,
                    max_contexts=3
                )
                
                # Merge RAG context with existing context
                enhanced_context = {
                    **context_data,
                    'rag_contexts': rag_enhancement['enhanced_contexts'],
                    'context_sources': rag_enhancement['context_sources'],
                    'search_performed': rag_enhancement['search_performed']
                }
                
            except Exception as e:
                logger.error(f"RAG enhancement failed: {e}")
                # Continue without RAG enhancement
        
        # Format enhanced context for prompt
        rag_context = ""
        if enhanced_context.get('rag_contexts'):
            rag_context = "\n\nRELEVANT MARKET CONTEXT:\n"
            for i, ctx in enumerate(enhanced_context['rag_contexts'], 1):
                rag_context += f"{i}. {ctx['text'][:300]}...\n"
                rag_context += f"   Source: {ctx['source']} (Relevance: {ctx['relevance']:.2f})\n\n"
        
        prompt = PromptTemplate.MARKET_RESEARCH.format(
            query=query,
            context=json.dumps(enhanced_context.get('context', {}), indent=2) + rag_context,
            market_data=json.dumps(enhanced_context.get('market_data', {}), indent=2)
        )
        
        request = LLMRequest(
            prompt=prompt,
            system_prompt="You are a senior market research analyst providing comprehensive market analysis.",
            temperature=0.5,
            max_tokens=800
        )
        
        return await self._execute_request(request, use_cache=use_cache)
    
    async def assess_risk(self,
                         prediction_data: Dict[str, Any],
                         use_cache: bool = True) -> LLMResponse:
        """Generate risk assessment for prediction"""
        
        prompt = PromptTemplate.RISK_ASSESSMENT.format(
            symbol=prediction_data.get('symbol', 'UNKNOWN'),
            current_price=prediction_data.get('current_price', 0),
            predicted_price=prediction_data.get('predicted_price', 0),
            volatility=prediction_data.get('volatility', 0),
            market_cap=prediction_data.get('market_cap', 0),
            sector=prediction_data.get('sector', 'Unknown'),
            performance_data=json.dumps(prediction_data.get('performance_data', {}), indent=2)
        )
        
        request = LLMRequest(
            prompt=prompt,
            system_prompt="You are a risk management specialist analyzing investment risks.",
            temperature=0.2,  # Very low temperature for consistent risk analysis
            max_tokens=500
        )
        
        return await self._execute_request(request, use_cache=use_cache)
    
    async def _execute_request(self, request: LLMRequest, use_cache: bool = True) -> LLMResponse:
        """Execute LLM request with caching and fallbacks"""
        
        # Check cache first
        if use_cache and self.redis_client:
            cached_response = await self._get_cached_response(request)
            if cached_response:
                return cached_response
        
        # Execute request with retries
        for attempt in range(MAX_RETRIES):
            try:
                response = await self._call_openai(request)
                
                # Cache successful response
                if use_cache and self.redis_client:
                    await self._cache_response(request, response)
                
                return response
                
            except Exception as e:
                logger.error(f"LLM request attempt {attempt + 1} failed: {e}")
                if attempt == MAX_RETRIES - 1:
                    # Return fallback response on final failure
                    return self._generate_fallback_response(request)
                
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
        
        return self._generate_fallback_response(request)
    
    async def _call_openai(self, request: LLMRequest) -> LLMResponse:
        """Call OpenAI API"""
        if not self.openai_client:
            raise Exception("OpenAI client not available")
        
        messages = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": request.prompt})
        
        response = await self.openai_client.chat.completions.create(
            model=request.model,
            messages=messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            timeout=REQUEST_TIMEOUT
        )
        
        content = response.choices[0].message.content
        tokens_used = response.usage.total_tokens
        cost_estimate = (tokens_used / 1000) * self.cost_per_1k_tokens.get(request.model, 0.01)
        
        return LLMResponse(
            content=content,
            model=request.model,
            tokens_used=tokens_used,
            cost_estimate=cost_estimate
        )
    
    async def _get_cached_response(self, request: LLMRequest) -> Optional[LLMResponse]:
        """Get cached response if available"""
        try:
            cache_key = f"llm_cache:{request.cache_key()}"
            cached_data = await self.redis_client.get(cache_key)
            
            if cached_data:
                data = json.loads(cached_data)
                response = LLMResponse(**data)
                response.cached = True
                return response
                
        except Exception as e:
            logger.error(f"Cache retrieval failed: {e}")
        
        return None
    
    async def _cache_response(self, request: LLMRequest, response: LLMResponse, ttl: int = 3600):
        """Cache response for future use"""
        try:
            cache_key = f"llm_cache:{request.cache_key()}"
            cache_data = asdict(response)
            cache_data['timestamp'] = cache_data['timestamp'].isoformat()
            
            await self.redis_client.setex(
                cache_key, 
                ttl,  # 1 hour default TTL
                json.dumps(cache_data)
            )
            
        except Exception as e:
            logger.error(f"Cache storage failed: {e}")
    
    def _generate_fallback_response(self, request: LLMRequest) -> LLMResponse:
        """Generate fallback response when LLM fails"""
        fallback_content = """
        Analysis temporarily unavailable due to service limitations.
        
        This prediction is based on quantitative models analyzing:
        • Technical indicators and price patterns
        • Market sentiment from news and social media
        • Historical performance and volatility metrics
        
        Please refer to the technical metrics and indicators for detailed analysis.
        Consider consulting additional sources for comprehensive market analysis.
        """
        
        return LLMResponse(
            content=fallback_content.strip(),
            model="fallback",
            tokens_used=0,
            cost_estimate=0.0
        )
    
    def _format_technical_indicators(self, indicators: Dict[str, float]) -> str:
        """Format technical indicators for display"""
        if not indicators:
            return "No technical indicators available"
        
        formatted = []
        for name, value in indicators.items():
            if isinstance(value, (int, float)):
                formatted.append(f"• {name}: {value:.2f}")
            else:
                formatted.append(f"• {name}: {value}")
        
        return "\n".join(formatted)
    
    async def compare_models(self, 
                           prompt: str,
                           task_type: str = "prediction_explanation") -> Dict[str, Any]:
        """Compare multiple models on same prompt"""
        if not self.hybrid_available:
            return {"error": "Hybrid engine not available"}
        
        try:
            # Map string task type to enum
            task_enum = getattr(self.TaskType, task_type.upper(), self.TaskType.PREDICTION_EXPLANATION)
            
            results = await self.hybrid_engine.compare_models(
                prompt=prompt,
                task_type=task_enum
            )
            
            comparison = {}
            for model_type, (response, evaluation) in results.items():
                comparison[model_type.value] = {
                    "response": response.content[:200] + "..." if len(response.content) > 200 else response.content,
                    "latency_ms": response.latency_ms,
                    "tokens_used": response.tokens_used,
                    "cost_estimate": response.cost_estimate,
                    "evaluation_scores": {
                        "accuracy": evaluation.accuracy,
                        "financial_accuracy": evaluation.financial_accuracy,
                        "risk_awareness": evaluation.risk_awareness,
                        "clarity": evaluation.clarity
                    }
                }
            
            return {
                "prompt": prompt[:100] + "..." if len(prompt) > 100 else prompt,
                "task_type": task_type,
                "model_comparisons": comparison,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Model comparison failed: {e}")
            return {"error": str(e)}
    
    async def get_hybrid_engine_stats(self) -> Dict[str, Any]:
        """Get comprehensive hybrid engine statistics"""
        if not self.hybrid_available:
            return {"status": "unavailable", "message": "Hybrid engine not initialized"}
        
        try:
            stats = await self.hybrid_engine.get_engine_stats()
            return {
                "hybrid_engine_status": "operational",
                "engine_stats": stats,
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            return {
                "hybrid_engine_status": "error",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def evaluate_response_quality(self, 
                                      response_text: str,
                                      task_type: str = "prediction_explanation") -> Dict[str, Any]:
        """Evaluate quality of a given response"""
        if not self.hybrid_available:
            return {"error": "Hybrid engine not available"}
        
        try:
            # Map string to enum
            task_enum = getattr(self.TaskType, task_type.upper(), self.TaskType.PREDICTION_EXPLANATION)
            
            # Create mock response for evaluation
            from hybrid_llm_engine import ModelResponse, ModelType
            mock_response = ModelResponse(
                content=response_text,
                model_type=ModelType.OPENAI_GPT4,  # Default for evaluation
                confidence=0.8,
                latency_ms=0,
                tokens_used=len(response_text.split()),
                cost_estimate=0.0
            )
            
            evaluation = await self.hybrid_engine.evaluator.evaluate_response(
                response=mock_response,
                task_type=task_enum
            )
            
            return {
                "response_length": len(response_text),
                "word_count": len(response_text.split()),
                "evaluation_scores": {
                    "overall_accuracy": evaluation.accuracy,
                    "financial_accuracy": evaluation.financial_accuracy,
                    "risk_awareness": evaluation.risk_awareness,
                    "clarity": evaluation.clarity,
                    "actionability": evaluation.actionability,
                    "relevance": evaluation.relevance
                },
                "task_type": task_type,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Response evaluation failed: {e}")
            return {"error": str(e)}
    
    async def get_usage_stats(self) -> Dict[str, Any]:
        """Get LLM usage statistics"""
        try:
            if not self.redis_client:
                return {"error": "Redis not available"}
            
            # Get basic usage metrics from cache keys
            cache_keys = []
            async for key in self.redis_client.scan_iter(match="llm_cache:*"):
                cache_keys.append(key)
            
            return {
                "cached_responses": len(cache_keys),
                "cache_hit_rate": "Available in production metrics",
                "estimated_cost_savings": len(cache_keys) * 0.01,  # Rough estimate
                "status": "operational" if self.openai_client else "degraded"
            }
            
        except Exception as e:
            return {"error": str(e)}

# Global LLM service instance (initialized with Redis in main app)
llm_service = None

def get_llm_service():
    """Get global LLM service instance"""
    global llm_service
    if llm_service is None:
        llm_service = LLMService()
    return llm_service