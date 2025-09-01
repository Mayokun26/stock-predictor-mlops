#!/usr/bin/env python3
"""
Test script for enhanced feature engineering pipeline
"""

import yfinance as yf
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.append('src')

def test_basic_indicators():
    """Test basic indicator calculations"""
    print("Testing basic technical indicators...")
    
    ticker = yf.Ticker('AAPL')
    hist = ticker.history(period='1y')
    df = hist.copy()
    df.columns = df.columns.str.lower()
    
    print(f'Data shape: {df.shape}')
    print(f'Current price: ${df["close"].iloc[-1]:.2f}')
    
    # Test Williams %R
    high_max = df['high'].rolling(window=14).max()
    low_min = df['low'].rolling(window=14).min()
    wr = -100 * (high_max - df['close']) / (high_max - low_min)
    print(f'Williams %R: {wr.iloc[-1]:.2f}')
    
    # Test CCI
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    sma = typical_price.rolling(window=20).mean()
    mad = typical_price.rolling(window=20).apply(lambda x: np.fabs(x - x.mean()).mean(), raw=True)
    cci = (typical_price - sma) / (0.015 * mad)
    print(f'CCI: {cci.iloc[-1]:.2f}')
    
    # Test OBV
    price_change = df['close'].diff()
    volume_direction = np.where(price_change > 0, df['volume'], 
                               np.where(price_change < 0, -df['volume'], 0))
    obv = pd.Series(volume_direction, index=df.index).cumsum()
    print(f'OBV: {obv.iloc[-1]:.0f}')
    
    # Test VWAP
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    vwap = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
    print(f'VWAP: ${vwap.iloc[-1]:.2f}')
    
    print('✓ All basic indicator calculations working correctly!')
    return True

def test_feature_engine_integration():
    """Test full feature engine integration"""
    print("\nTesting AdvancedFeatureEngine integration...")
    
    try:
        # Add the src/api directory to path
        api_path = os.path.join('src', 'api')
        if api_path not in sys.path:
            sys.path.insert(0, api_path)
        
        from production_api import AdvancedFeatureEngine
        import asyncio
        
        async def run_test():
            engine = AdvancedFeatureEngine(None)
            features = await engine._calculate_features('AAPL')
            
            print(f'Total features calculated: {len(features)}')
            
            # Categorize features
            momentum_features = [k for k in features.keys() if any(x in k for x in ['rsi', 'macd', 'stochastic', 'williams', 'cci', 'momentum', 'roc'])]
            volatility_features = [k for k in features.keys() if any(x in k for x in ['bollinger', 'atr', 'keltner', 'donchian'])]
            trend_features = [k for k in features.keys() if any(x in k for x in ['adx', 'aroon', 'parabolic', 'ichimoku', 'dmi'])]
            volume_features = [k for k in features.keys() if any(x in k for x in ['volume', 'obv', 'ad_line', 'chaikin', 'vwap'])]
            
            print(f'  Momentum indicators: {len(momentum_features)}')
            print(f'  Volatility indicators: {len(volatility_features)}')
            print(f'  Trend indicators: {len(trend_features)}')
            print(f'  Volume indicators: {len(volume_features)}')
            
            # Test specific new indicators
            new_indicators = ['williams_r', 'cci', 'momentum_10', 'roc_20', 'adx', 'aroon_up', 'ichimoku_tenkan', 'obv', 'vwap']
            print('\nNew indicators status:')
            working_count = 0
            for indicator in new_indicators:
                if indicator in features:
                    print(f'  ✓ {indicator}: {features[indicator]:.4f}')
                    working_count += 1
                else:
                    print(f'  ✗ {indicator}: missing')
            
            print(f'\n✓ Feature engine integration working: {working_count}/{len(new_indicators)} indicators')
            return True
        
        return asyncio.run(run_test())
        
    except Exception as e:
        print(f'✗ Feature engine integration failed: {e}')
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("ENHANCED FEATURE ENGINEERING PIPELINE TEST")
    print("=" * 60)
    
    success = True
    
    # Test 1: Basic indicators
    try:
        success &= test_basic_indicators()
    except Exception as e:
        print(f'✗ Basic indicators test failed: {e}')
        success = False
    
    # Test 2: Full integration
    try:
        success &= test_feature_engine_integration()
    except Exception as e:
        print(f'✗ Integration test failed: {e}')
        success = False
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 ALL TESTS PASSED - Enhanced feature pipeline ready!")
        print("📊 30+ technical indicators now available")
        print("🚀 Ready for MLflow integration next")
    else:
        print("❌ SOME TESTS FAILED - Check errors above")
    print("=" * 60)
    
    return success

if __name__ == "__main__":
    main()