#!/usr/bin/env python3
"""
Technical Indicators Calculator
Production-grade implementation with comprehensive indicator suite
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

class TechnicalIndicators:
    """
    Comprehensive technical analysis indicators calculator
    Optimized for real-time feature engineering in production systems
    """
    
    @staticmethod
    def calculate_all(df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Calculate all technical indicators for a price DataFrame
        
        Args:
            df: DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
            
        Returns:
            Dict of indicator name -> pandas Series
        """
        if df.empty or len(df) < 20:
            logger.warning("Insufficient data for technical indicators")
            return {}
        
        indicators = {}
        
        # Price-based indicators
        indicators['sma_5'] = TechnicalIndicators.sma(df['close'], 5)
        indicators['sma_10'] = TechnicalIndicators.sma(df['close'], 10)
        indicators['sma_20'] = TechnicalIndicators.sma(df['close'], 20)
        indicators['sma_50'] = TechnicalIndicators.sma(df['close'], 50)
        
        indicators['ema_5'] = TechnicalIndicators.ema(df['close'], 5)
        indicators['ema_12'] = TechnicalIndicators.ema(df['close'], 12)
        indicators['ema_26'] = TechnicalIndicators.ema(df['close'], 26)
        
        # Momentum indicators
        indicators['rsi'] = TechnicalIndicators.rsi(df['close'], 14)
        indicators['rsi_6'] = TechnicalIndicators.rsi(df['close'], 6)
        indicators['rsi_21'] = TechnicalIndicators.rsi(df['close'], 21)
        
        # MACD
        macd_line, macd_signal, macd_hist = TechnicalIndicators.macd(df['close'])
        indicators['macd_line'] = macd_line
        indicators['macd_signal'] = macd_signal
        indicators['macd_histogram'] = macd_hist
        
        # Volatility indicators
        indicators['volatility'] = TechnicalIndicators.volatility(df['close'], 20)
        indicators['atr'] = TechnicalIndicators.atr(df, 14)
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = TechnicalIndicators.bollinger_bands(df['close'])
        indicators['bb_upper'] = bb_upper
        indicators['bb_middle'] = bb_middle
        indicators['bb_lower'] = bb_lower
        indicators['bb_width'] = (bb_upper - bb_lower) / bb_middle
        indicators['bb_position'] = (df['close'] - bb_lower) / (bb_upper - bb_lower)
        
        # Volume indicators
        indicators['volume_sma'] = TechnicalIndicators.sma(df['volume'], 20)
        indicators['volume_ratio'] = df['volume'] / indicators['volume_sma']
        indicators['obv'] = TechnicalIndicators.obv(df['close'], df['volume'])
        
        # Price action indicators
        indicators['price_change_1d'] = df['close'].pct_change(1)
        indicators['price_change_5d'] = df['close'].pct_change(5)
        indicators['price_change_20d'] = df['close'].pct_change(20)
        
        # Support/Resistance levels
        indicators['support_20'] = TechnicalIndicators.support_resistance(df['low'], 20, 'support')
        indicators['resistance_20'] = TechnicalIndicators.support_resistance(df['high'], 20, 'resistance')
        
        # Trend indicators
        indicators['adx'] = TechnicalIndicators.adx(df, 14)
        
        # Custom composite indicators
        indicators['momentum_composite'] = TechnicalIndicators.momentum_composite(indicators)
        indicators['trend_strength'] = TechnicalIndicators.trend_strength(indicators)
        indicators['volatility_regime'] = TechnicalIndicators.volatility_regime(indicators)
        
        return indicators
    
    @staticmethod
    def sma(series: pd.Series, window: int) -> pd.Series:
        """Simple Moving Average"""
        return series.rolling(window=window, min_periods=1).mean()
    
    @staticmethod
    def ema(series: pd.Series, window: int) -> pd.Series:
        """Exponential Moving Average"""
        return series.ewm(span=window, adjust=False).mean()
    
    @staticmethod
    def rsi(series: pd.Series, window: int = 14) -> pd.Series:
        """Relative Strength Index"""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
        """MACD (Moving Average Convergence Divergence)"""
        ema_fast = TechnicalIndicators.ema(series, fast)
        ema_slow = TechnicalIndicators.ema(series, slow)
        
        macd_line = ema_fast - ema_slow
        macd_signal = TechnicalIndicators.ema(macd_line, signal)
        macd_histogram = macd_line - macd_signal
        
        return macd_line, macd_signal, macd_histogram
    
    @staticmethod
    def volatility(series: pd.Series, window: int = 20) -> pd.Series:
        """Historical Volatility (rolling standard deviation)"""
        returns = series.pct_change()
        return returns.rolling(window=window).std() * np.sqrt(252)  # Annualized
    
    @staticmethod
    def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
        """Average True Range"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return true_range.rolling(window=window).mean()
    
    @staticmethod
    def bollinger_bands(series: pd.Series, window: int = 20, std: float = 2) -> tuple:
        """Bollinger Bands"""
        middle = TechnicalIndicators.sma(series, window)
        std_dev = series.rolling(window=window).std()
        
        upper = middle + (std_dev * std)
        lower = middle - (std_dev * std)
        
        return upper, middle, lower
    
    @staticmethod
    def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """On-Balance Volume"""
        obv = pd.Series(index=close.index, dtype=float)
        obv.iloc[0] = volume.iloc[0]
        
        for i in range(1, len(close)):
            if close.iloc[i] > close.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
            elif close.iloc[i] < close.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        
        return obv
    
    @staticmethod
    def support_resistance(series: pd.Series, window: int, level_type: str) -> pd.Series:
        """Calculate dynamic support/resistance levels"""
        if level_type == 'support':
            return series.rolling(window=window).min()
        else:  # resistance
            return series.rolling(window=window).max()
    
    @staticmethod
    def adx(df: pd.DataFrame, window: int = 14) -> pd.Series:
        """Average Directional Index (trend strength)"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        # Calculate directional movement
        dm_pos = pd.Series(np.where((high.diff() > low.diff().abs()) & (high.diff() > 0), 
                                   high.diff(), 0), index=df.index)
        dm_neg = pd.Series(np.where((low.diff().abs() > high.diff()) & (low.diff() < 0), 
                                   low.diff().abs(), 0), index=df.index)
        
        # Calculate true range
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Smooth the values
        atr = true_range.rolling(window=window).mean()
        di_pos = 100 * (dm_pos.rolling(window=window).mean() / atr)
        di_neg = 100 * (dm_neg.rolling(window=window).mean() / atr)
        
        # Calculate ADX
        dx = 100 * np.abs(di_pos - di_neg) / (di_pos + di_neg)
        adx = dx.rolling(window=window).mean()
        
        return adx
    
    @staticmethod
    def momentum_composite(indicators: Dict[str, pd.Series]) -> pd.Series:
        """Composite momentum score combining multiple indicators"""
        try:
            # Normalize RSI (0-100 to -1 to 1)
            rsi_norm = (indicators['rsi'] - 50) / 50
            
            # Normalize MACD histogram
            macd_norm = indicators['macd_histogram'] / indicators['macd_histogram'].abs().rolling(20).max()
            
            # Price vs moving average
            price_ma_momentum = (indicators['sma_5'] - indicators['sma_20']) / indicators['sma_20']
            
            # Combine with weights
            momentum = (0.4 * rsi_norm + 0.3 * macd_norm + 0.3 * price_ma_momentum)
            
            return momentum.fillna(0)
        
        except Exception as e:
            logger.warning(f"Failed to calculate momentum composite: {e}")
            return pd.Series(0, index=indicators['rsi'].index)
    
    @staticmethod
    def trend_strength(indicators: Dict[str, pd.Series]) -> pd.Series:
        """Measure overall trend strength"""
        try:
            # ADX component (trend strength)
            adx_norm = indicators['adx'] / 100
            
            # Moving average alignment
            ma_alignment = np.where(
                (indicators['sma_5'] > indicators['sma_20']) & 
                (indicators['sma_20'] > indicators['sma_50']), 1,
                np.where(
                    (indicators['sma_5'] < indicators['sma_20']) & 
                    (indicators['sma_20'] < indicators['sma_50']), -1, 0
                )
            )
            ma_alignment = pd.Series(ma_alignment, index=indicators['adx'].index)
            
            # Bollinger Band position
            bb_trend = (indicators['bb_position'] - 0.5) * 2  # -1 to 1
            
            # Combine
            trend = (0.5 * adx_norm + 0.3 * ma_alignment + 0.2 * bb_trend)
            
            return trend.fillna(0)
        
        except Exception as e:
            logger.warning(f"Failed to calculate trend strength: {e}")
            return pd.Series(0, index=indicators['adx'].index)
    
    @staticmethod
    def volatility_regime(indicators: Dict[str, pd.Series]) -> pd.Series:
        """Classify volatility regime (low/normal/high)"""
        try:
            vol = indicators['volatility']
            vol_ma = vol.rolling(252).mean()  # 1-year average
            vol_std = vol.rolling(252).std()
            
            # Standardize current volatility
            vol_z_score = (vol - vol_ma) / vol_std
            
            # Classify regime
            regime = pd.Series(index=vol.index, dtype=float)
            regime = np.where(vol_z_score < -0.5, -1,  # Low vol
                             np.where(vol_z_score > 0.5, 1,  # High vol
                                     0))  # Normal vol
            
            return pd.Series(regime, index=vol.index)
        
        except Exception as e:
            logger.warning(f"Failed to calculate volatility regime: {e}")
            return pd.Series(0, index=indicators['volatility'].index)
    
    @staticmethod
    def get_latest_values(indicators: Dict[str, pd.Series]) -> Dict[str, float]:
        """Extract the latest value from each indicator for feature serving"""
        latest = {}
        
        for name, series in indicators.items():
            if not series.empty and not pd.isna(series.iloc[-1]):
                latest[name] = float(series.iloc[-1])
            else:
                latest[name] = 0.0
        
        return latest
    
    @staticmethod
    def validate_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
        """Validate input data quality for indicator calculation"""
        quality = {
            'is_valid': True,
            'issues': [],
            'completeness': 1.0,
            'row_count': len(df)
        }
        
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            quality['is_valid'] = False
            quality['issues'].append(f"Missing columns: {missing_cols}")
        
        if len(df) < 20:
            quality['is_valid'] = False
            quality['issues'].append(f"Insufficient data: {len(df)} rows (need at least 20)")
        
        # Check for missing values
        missing_pct = df[required_cols].isnull().sum().sum() / (len(df) * len(required_cols))
        quality['completeness'] = 1.0 - missing_pct
        
        if missing_pct > 0.1:  # More than 10% missing
            quality['issues'].append(f"High missing data: {missing_pct:.1%}")
        
        # Check for data consistency
        if 'high' in df.columns and 'low' in df.columns:
            invalid_ranges = (df['high'] < df['low']).sum()
            if invalid_ranges > 0:
                quality['issues'].append(f"Invalid high/low ranges: {invalid_ranges} rows")
        
        return quality