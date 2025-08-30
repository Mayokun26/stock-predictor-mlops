#!/usr/bin/env python3
"""
Test script for backtesting engine
"""

import sys
import os
sys.path.append('src')

from evaluation.backtesting_engine import BacktestingEngine, SimpleLinearModel, SimpleMovingAverageModel
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

def test_backtesting():
    """Test the backtesting engine with a simple example"""
    
    print("🚀 Testing MLOps Backtesting Engine")
    print("=" * 50)
    
    # Initialize backtesting engine
    engine = BacktestingEngine(
        initial_capital=10000,  # Smaller capital for testing
        transaction_cost=0.001
    )
    
    # Test with Apple stock - shorter time period for quick testing
    print("\n📊 Running backtest for AAPL (2023-2024)...")
    
    try:
        result = engine.walk_forward_backtest(
            symbol="AAPL",
            start_date="2023-01-01", 
            end_date="2024-01-01",
            model_class=SimpleLinearModel
        )
        
        print("\n✅ Backtest completed successfully!")
        print(result.summary())
        
        # Test multiple symbols
        print("\n📈 Testing multiple symbols...")
        symbols = ["AAPL", "MSFT"]
        
        results = engine.run_multi_symbol_backtest(
            symbols=symbols,
            start_date="2023-06-01",
            end_date="2023-12-01", 
            model_class=SimpleMovingAverageModel
        )
        
        print(f"\n✅ Multi-symbol backtest completed for {len(results)} symbols")
        
        # Generate report
        report = engine.generate_performance_report(results)
        print(report)
        
        return True
        
    except Exception as e:
        print(f"❌ Backtest failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_backtesting()
    if success:
        print("\n🎉 All backtesting tests passed!")
    else:
        print("\n💥 Backtesting tests failed!")
        sys.exit(1)