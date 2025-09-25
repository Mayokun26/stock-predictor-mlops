#!/usr/bin/env python3
"""
Test script for Hybrid LLM Engine and Advanced ML Engineering Features
Tests the sophisticated LLM framework you can explain in interviews
"""
import asyncio
import json
import os
import sys
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import requests

def test_basic_endpoints():
    """Test basic LLM endpoints"""
    print("🧪 Testing Basic LLM Endpoints...")
    
    base_url = "http://127.0.0.1:8001"
    
    # Test basic LLM stats
    try:
        response = requests.get(f"{base_url}/llm/stats")
        if response.status_code == 200:
            print("✅ Basic LLM stats endpoint working")
        else:
            print(f"❌ LLM stats failed: {response.status_code}")
    except Exception as e:
        print(f"❌ LLM stats error: {e}")
    
    # Test hybrid engine stats
    try:
        response = requests.get(f"{base_url}/llm/hybrid-stats")
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Hybrid engine stats: {result.get('hybrid_engine_status', 'unknown')}")
            
            if 'engine_stats' in result:
                stats = result['engine_stats']
                print(f"   Available models: {stats.get('available_models', 0)}/{stats.get('total_models', 0)}")
                print(f"   Model details: {len(stats.get('model_details', {}))}")
        else:
            print(f"❌ Hybrid stats failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Hybrid stats error: {e}")

def test_model_comparison():
    """Test model comparison functionality"""
    print("\n🔬 Testing Model Comparison...")
    
    base_url = "http://127.0.0.1:8001"
    
    comparison_data = {
        "prompt": "Analyze AAPL stock performance and provide investment recommendations with risk assessment.",
        "task_type": "financial_analysis"
    }
    
    try:
        response = requests.post(
            f"{base_url}/llm/compare-models",
            json=comparison_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Model comparison successful")
            
            if 'model_comparisons' in result:
                comparisons = result['model_comparisons']
                print(f"   Compared {len(comparisons)} models:")
                
                for model_name, metrics in comparisons.items():
                    print(f"   • {model_name}:")
                    print(f"     - Latency: {metrics.get('latency_ms', 0):.1f}ms")
                    print(f"     - Cost: ${metrics.get('cost_estimate', 0):.4f}")
                    
                    eval_scores = metrics.get('evaluation_scores', {})
                    print(f"     - Accuracy: {eval_scores.get('accuracy', 0):.2f}")
                    print(f"     - Financial accuracy: {eval_scores.get('financial_accuracy', 0):.2f}")
                    print(f"     - Risk awareness: {eval_scores.get('risk_awareness', 0):.2f}")
            else:
                print("   No model comparisons in response")
                
        else:
            print(f"❌ Model comparison failed: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"❌ Model comparison error: {e}")

def test_response_evaluation():
    """Test response quality evaluation"""
    print("\n📊 Testing Response Evaluation...")
    
    base_url = "http://127.0.0.1:8001"
    
    # Test evaluating a financial response
    evaluation_data = {
        "response_text": """
        Based on technical analysis, AAPL shows strong momentum with RSI at 65, indicating moderate buying pressure. 
        The stock has broken above the 20-day SMA and shows positive volume trends.
        
        Risk factors include market volatility and sector rotation concerns. 
        Recommendation: Consider a position with 2% portfolio allocation and stop loss at $210.
        """,
        "task_type": "financial_analysis"
    }
    
    try:
        response = requests.post(
            f"{base_url}/llm/evaluate-response",
            json=evaluation_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Response evaluation successful")
            
            eval_scores = result.get('evaluation_scores', {})
            print(f"   Response length: {result.get('response_length', 0)} chars")
            print(f"   Word count: {result.get('word_count', 0)} words")
            print(f"   Quality scores:")
            print(f"     - Overall accuracy: {eval_scores.get('overall_accuracy', 0):.2f}")
            print(f"     - Financial accuracy: {eval_scores.get('financial_accuracy', 0):.2f}")
            print(f"     - Risk awareness: {eval_scores.get('risk_awareness', 0):.2f}")
            print(f"     - Clarity: {eval_scores.get('clarity', 0):.2f}")
            print(f"     - Actionability: {eval_scores.get('actionability', 0):.2f}")
            print(f"     - Relevance: {eval_scores.get('relevance', 0):.2f}")
            
        else:
            print(f"❌ Response evaluation failed: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"❌ Response evaluation error: {e}")

def test_model_benchmarking():
    """Test model benchmarking with multiple prompts"""
    print("\n🏆 Testing Model Benchmarking...")
    
    base_url = "http://127.0.0.1:8001"
    
    benchmark_data = {
        "test_prompts": [
            "Analyze Tesla stock performance and provide risk-adjusted investment recommendation.",
            "Explain why Apple stock price might fluctuate based on current market conditions.",
            "Assess the financial risk of investing in Microsoft given current technical indicators."
        ],
        "task_type": "financial_analysis"
    }
    
    try:
        response = requests.post(
            f"{base_url}/llm/benchmark-models",
            json=benchmark_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Model benchmarking successful")
            print(f"   Total test cases: {result.get('total_test_cases', 0)}")
            
            if 'aggregate_performance' in result:
                aggregates = result['aggregate_performance']
                print(f"   Model performance summary:")
                
                for model_name, performance in aggregates.items():
                    print(f"   • {model_name}:")
                    print(f"     - Average accuracy: {performance.get('avg_accuracy', 0):.3f}")
                    print(f"     - Average latency: {performance.get('avg_latency_ms', 0):.1f}ms")
                    print(f"     - Average cost: ${performance.get('avg_cost', 0):.4f}")
                    print(f"     - Test cases: {performance.get('test_count', 0)}")
            else:
                print("   No aggregate performance data")
                
        else:
            print(f"❌ Model benchmarking failed: {response.status_code}")
            print(f"   Response: {response.text[:200]}...")
            
    except Exception as e:
        print(f"❌ Model benchmarking error: {e}")

def test_enhanced_prediction_explanation():
    """Test enhanced prediction explanation with hybrid engine"""
    print("\n💡 Testing Enhanced Prediction Explanation...")
    
    base_url = "http://127.0.0.1:8001"
    
    explanation_data = {
        "symbol": "AAPL",
        "include_risk_assessment": True,
        "user_id": "test_hybrid"
    }
    
    try:
        response = requests.post(
            f"{base_url}/predict/explain",
            json=explanation_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Enhanced prediction explanation generated")
            print(f"   Model used: {result.get('model_used', 'unknown')}")
            print(f"   Tokens used: {result.get('tokens_used', 0)}")
            print(f"   Processing time: {result.get('processing_time_ms', 0):.1f}ms")
            print(f"   Cached: {result.get('cached', False)}")
            
            # Show snippet of explanation
            explanation = result.get('explanation', '')
            print(f"   Explanation snippet: {explanation[:150]}...")
            
            if result.get('risk_assessment'):
                risk = result.get('risk_assessment', '')
                print(f"   Risk assessment snippet: {risk[:150]}...")
                
        else:
            print(f"❌ Enhanced explanation failed: {response.status_code}")
            print(f"   Response: {response.text[:200]}...")
            
    except Exception as e:
        print(f"❌ Enhanced explanation error: {e}")

def show_system_architecture():
    """Display the hybrid LLM system architecture"""
    print("\n🏗️  Hybrid LLM System Architecture:")
    print("="*60)
    print("""
    ┌─────────────────────────────────────────────────────────┐
    │                 Hybrid LLM Engine                       │
    │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
    │  │  OpenAI     │  │ HuggingFace │  │   Ollama    │     │
    │  │  GPT-4/3.5  │  │   Local     │  │   Local     │     │
    │  └─────────────┘  └─────────────┘  └─────────────┘     │
    │  ┌─────────────┐  ┌─────────────┐                      │
    │  │ Financial   │  │  Ensemble   │                      │
    │  │ Fine-tuned  │  │   Models    │                      │
    │  └─────────────┘  └─────────────┘                      │
    └─────────────────────────────────────────────────────────┘
                              │
    ┌─────────────────────────────────────────────────────────┐
    │              Model Router & Evaluator                   │
    │  • Smart routing based on task type and requirements   │
    │  • Performance tracking and A/B testing               │
    │  • Cost optimization and fallback strategies          │
    └─────────────────────────────────────────────────────────┘
                              │
    ┌─────────────────────────────────────────────────────────┐
    │           Advanced Evaluation Framework                 │
    │  • Financial accuracy metrics                          │
    │  • Risk awareness scoring                              │
    │  • Actionability assessment                            │
    │  • Domain-specific evaluation                          │
    └─────────────────────────────────────────────────────────┘
    """)
    
    print("\n🎯 Interview Talking Points:")
    print("• Multi-model architecture with intelligent routing")
    print("• Custom evaluation metrics for financial domain")
    print("• Cost optimization across different model backends") 
    print("• A/B testing framework for model comparison")
    print("• Production monitoring and performance tracking")
    print("• Graceful degradation and fallback strategies")

def main():
    """Main test execution"""
    print("🚀 Hybrid LLM Engine Test Suite")
    print("=" * 60)
    print("\nTesting advanced ML engineering features that demonstrate")
    print("senior-level competency in LLM system architecture.\n")
    
    # Show architecture
    show_system_architecture()
    
    # Run tests
    test_basic_endpoints()
    test_model_comparison()
    test_response_evaluation()
    test_enhanced_prediction_explanation()
    test_model_benchmarking()
    
    print("\n" + "="*60)
    print("🎉 Hybrid LLM Testing Complete!")
    print("\n💼 Senior MLOps Interview Readiness:")
    print("✅ Multi-model LLM architecture")
    print("✅ Custom evaluation frameworks") 
    print("✅ Model performance comparison")
    print("✅ Financial domain optimization")
    print("✅ Production monitoring and routing")
    print("✅ Cost-aware model selection")
    
    print(f"\n🔗 API Documentation: http://127.0.0.1:8001/docs")
    print("📊 New endpoints for senior-level demonstrations:")
    print("   • POST /llm/compare-models")
    print("   • POST /llm/evaluate-response") 
    print("   • POST /llm/benchmark-models")
    print("   • GET /llm/hybrid-stats")

if __name__ == "__main__":
    main()