#!/usr/bin/env python3
"""
Test script for LLM integration functionality
Tests both the LLM service and API endpoints
"""
import asyncio
import json
import os
import sys
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import aiohttp
import requests

async def test_llm_service_directly():
    """Test LLM service directly"""
    print("🧪 Testing LLM Service Directly...")
    
    try:
        from llm.llm_service import LLMService
        
        # Initialize service (without Redis for testing)
        llm_service = LLMService()
        
        # Test data
        prediction_data = {
            "symbol": "AAPL",
            "current_price": 150.0,
            "predicted_price": 155.0,
            "confidence": 0.75,
            "technical_indicators": {
                "RSI": 45.2,
                "SMA_20": 149.5,
                "Volume": 1000000,
                "MACD": 0.5
            },
            "news_headlines": ["Strong earnings reported", "New product launch announced"],
            "sentiment_score": 0.65,
            "model_type": "RandomForest",
            "feature_importance": {"RSI": 0.3, "Volume": 0.2, "SMA_20": 0.15}
        }
        
        print("📊 Testing explanation generation...")
        explanation = await llm_service.generate_explanation(
            prediction_data=prediction_data,
            use_cache=False  # Don't use cache for testing
        )
        
        print(f"✅ Generated explanation ({explanation.tokens_used} tokens, ${explanation.cost_estimate:.4f}):")
        print(f"   {explanation.content[:200]}...")
        
        print("\n🔍 Testing risk assessment...")
        risk_assessment = await llm_service.assess_risk(
            prediction_data=prediction_data,
            use_cache=False
        )
        
        print(f"✅ Generated risk assessment ({risk_assessment.tokens_used} tokens, ${risk_assessment.cost_estimate:.4f}):")
        print(f"   {risk_assessment.content[:200]}...")
        
        print("\n📈 Testing market research...")
        context_data = {
            "context": {"query": "AAPL stock outlook", "analysis_type": "technical"},
            "market_data": {"AAPL": {"current_price": 150.0, "volume": 1000000}}
        }
        
        research = await llm_service.analyze_market_research(
            query="What are the key factors affecting AAPL stock price today?",
            context_data=context_data,
            use_cache=False
        )
        
        print(f"✅ Generated market research ({research.tokens_used} tokens, ${research.cost_estimate:.4f}):")
        print(f"   {research.content[:200]}...")
        
        total_cost = explanation.cost_estimate + risk_assessment.cost_estimate + research.cost_estimate
        total_tokens = explanation.tokens_used + risk_assessment.tokens_used + research.tokens_used
        
        print(f"\n📊 Total usage: {total_tokens} tokens, ${total_cost:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ LLM service test failed: {e}")
        return False

def test_api_endpoints():
    """Test API endpoints"""
    print("\n🌐 Testing API Endpoints...")
    
    base_url = "http://localhost:8000"
    
    # Test basic health
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            print("✅ Basic API health check passed")
        else:
            print("❌ API not responding correctly")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ API server not running. Start with: python src/api/production_api.py")
        return False
    
    # Test explanation endpoint
    print("\n📝 Testing prediction explanation endpoint...")
    explanation_data = {
        "symbol": "AAPL",
        "include_risk_assessment": True,
        "user_id": "test_user"
    }
    
    try:
        response = requests.post(
            f"{base_url}/predict/explain",
            json=explanation_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Explanation generated: {len(result.get('explanation', ''))} chars")
            print(f"   Model: {result.get('model_used', 'unknown')}")
            print(f"   Tokens: {result.get('tokens_used', 0)}")
            print(f"   Cost: ${result.get('cost_estimate', 0):.4f}")
            print(f"   Cached: {result.get('cached', False)}")
        else:
            print(f"❌ Explanation endpoint failed: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Explanation endpoint error: {e}")
        return False
    
    # Test market research endpoint
    print("\n📊 Testing market research endpoint...")
    research_data = {
        "query": "What are the market trends for tech stocks today?",
        "symbols": ["AAPL", "MSFT"],
        "include_sentiment": True,
        "user_id": "test_user"
    }
    
    try:
        response = requests.post(
            f"{base_url}/market/research",
            json=research_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Market research generated: {len(result.get('analysis', ''))} chars")
            print(f"   Model: {result.get('model_used', 'unknown')}")
            print(f"   Symbols analyzed: {result.get('symbols_analyzed', [])}")
        else:
            print(f"❌ Market research endpoint failed: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Market research endpoint error: {e}")
        return False
    
    # Test LLM stats endpoint
    print("\n📈 Testing LLM stats endpoint...")
    try:
        response = requests.get(f"{base_url}/llm/stats")
        if response.status_code == 200:
            result = response.json()
            print(f"✅ LLM stats: {result.get('service_status', 'unknown')}")
        else:
            print(f"❌ LLM stats endpoint failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ LLM stats endpoint error: {e}")
    
    return True

def test_environment_setup():
    """Test environment and dependencies"""
    print("🔧 Testing Environment Setup...")
    
    # Check OpenAI API key
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        print(f"✅ OpenAI API key found: {openai_key[:8]}...")
    else:
        print("⚠️  OpenAI API key not found. Set OPENAI_API_KEY environment variable")
        print("   LLM features will use fallback responses")
    
    # Check required imports
    try:
        import openai
        print("✅ OpenAI library available")
    except ImportError:
        print("❌ OpenAI library not installed. Run: pip install openai tiktoken")
        return False
    
    try:
        import tiktoken
        print("✅ Tiktoken library available")
    except ImportError:
        print("⚠️  Tiktoken library not found. Install for token counting: pip install tiktoken")
    
    return True

def main():
    """Main test execution"""
    print("🚀 LLM Integration Test Suite")
    print("=" * 50)
    
    # Test environment
    env_ok = test_environment_setup()
    if not env_ok:
        print("\n❌ Environment setup failed. Please fix dependencies first.")
        return
    
    # Test service directly
    print("\n" + "=" * 50)
    asyncio.run(test_llm_service_directly())
    
    # Test API endpoints
    print("\n" + "=" * 50)
    test_api_endpoints()
    
    print("\n🎉 LLM Integration Testing Complete!")
    print("\nNext steps:")
    print("1. Start the API server: python src/api/production_api.py")
    print("2. Test explanations: curl -X POST http://localhost:8000/predict/explain -H 'Content-Type: application/json' -d '{\"symbol\":\"AAPL\"}'")
    print("3. Check LLM metrics at: http://localhost:8000/metrics")

if __name__ == "__main__":
    main()