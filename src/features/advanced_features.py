"""
Advanced Feature Engineering with FinBERT Sentiment Analysis
Enterprise-grade feature engineering for financial ML models
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import logging
import asyncio
from dataclasses import dataclass
import yfinance as yf
from textblob import TextBlob
import re
import warnings
warnings.filterwarnings('ignore')

# Advanced NLP imports
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("Transformers not available. Using TextBlob for sentiment analysis.")

# Technical analysis imports
try:
    import ta
    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False
    logging.warning("TA-Lib not available. Using manual calculations.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SentimentResult:
    """Sentiment analysis result"""
    score: float  # -1 to 1
    confidence: float  # 0 to 1
    label: str  # positive, negative, neutral
    model_used: str

class FinBERTSentimentAnalyzer:
    """Financial sentiment analysis using FinBERT"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize FinBERT model"""
        try:
            if TRANSFORMERS_AVAILABLE:
                # Try to load FinBERT for financial sentiment
                model_name = "ProsusAI/finbert"
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
                self.pipeline = pipeline(
                    "sentiment-analysis",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    framework="pt"
                )
                logger.info("FinBERT model loaded successfully")
            else:
                logger.warning("Using TextBlob fallback for sentiment analysis")
        except Exception as e:
            logger.error(f"Error loading FinBERT: {e}. Using TextBlob fallback.")
            self.pipeline = None
    
    def analyze_sentiment(self, text: str) -> SentimentResult:
        """Analyze sentiment of financial text"""
        try:
            # Clean text
            cleaned_text = self._clean_financial_text(text)
            
            if self.pipeline:
                # Use FinBERT
                result = self.pipeline(cleaned_text)[0]
                
                # Convert FinBERT labels to standardized format
                score = self._convert_finbert_score(result['label'], result['score'])
                
                return SentimentResult(
                    score=score,
                    confidence=result['score'],
                    label=result['label'].lower(),
                    model_used='finbert'
                )
            else:
                # Use TextBlob fallback
                blob = TextBlob(cleaned_text)
                sentiment = blob.sentiment
                
                # Convert TextBlob polarity to label
                if sentiment.polarity > 0.1:
                    label = 'positive'
                elif sentiment.polarity < -0.1:
                    label = 'negative'
                else:
                    label = 'neutral'
                
                return SentimentResult(
                    score=sentiment.polarity,
                    confidence=sentiment.subjectivity,
                    label=label,
                    model_used='textblob'
                )
                
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            return SentimentResult(
                score=0.0,
                confidence=0.0,
                label='neutral',
                model_used='error'
            )
    
    def _clean_financial_text(self, text: str) -> str:
        """Clean financial text for better sentiment analysis"""
        if not text:
            return ""
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove stock tickers in brackets
        text = re.sub(r'\$[A-Z]{1,5}', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Truncate to max length for BERT
        if len(text) > 512:
            text = text[:512]
        
        return text.strip()
    
    def _convert_finbert_score(self, label: str, confidence: float) -> float:
        """Convert FinBERT output to standardized sentiment score"""
        label = label.lower()
        
        if label == 'positive':
            return confidence
        elif label == 'negative':
            return -confidence
        else:  # neutral
            return 0.0
    
    def batch_analyze(self, texts: List[str]) -> List[SentimentResult]:
        """Analyze sentiment for multiple texts"""
        results = []
        
        for text in texts:
            result = self.analyze_sentiment(text)
            results.append(result)
        
        return results

class AdvancedFeatureEngineer:
    """Advanced feature engineering for financial ML"""
    
    def __init__(self):
        self.sentiment_analyzer = FinBERTSentimentAnalyzer()
        
    def create_technical_features(self, price_data: pd.DataFrame) -> Dict[str, float]:
        """Create comprehensive technical indicators"""
        try:
            if len(price_data) < 50:
                return {}
            
            features = {}
            
            # Ensure we have the required columns
            if 'Close' not in price_data.columns:
                if 'price' in price_data.columns:
                    price_data['Close'] = price_data['price']
                else:
                    return {}
            
            if 'Volume' not in price_data.columns:
                price_data['Volume'] = 1000000  # Default volume
            
            prices = price_data['Close']
            volumes = price_data['Volume']
            
            # Moving Averages
            features.update(self._calculate_moving_averages(prices))
            
            # Momentum Indicators
            features.update(self._calculate_momentum_indicators(prices))
            
            # Volatility Indicators
            features.update(self._calculate_volatility_indicators(prices))
            
            # Volume Indicators
            features.update(self._calculate_volume_indicators(prices, volumes))
            
            # Pattern Recognition
            features.update(self._calculate_pattern_features(prices))
            
            # Market Structure
            features.update(self._calculate_market_structure(price_data))
            
            return features
            
        except Exception as e:
            logger.error(f"Error creating technical features: {e}")
            return {}
    
    def _calculate_moving_averages(self, prices: pd.Series) -> Dict[str, float]:
        """Calculate moving average features"""
        features = {}
        
        try:
            # Simple Moving Averages
            sma_periods = [5, 10, 20, 50, 100, 200]
            for period in sma_periods:
                if len(prices) >= period:
                    sma = prices.rolling(window=period).mean().iloc[-1]
                    features[f'sma_{period}'] = float(sma) if pd.notna(sma) else 0.0
                    
                    # Price relative to SMA
                    features[f'price_vs_sma_{period}'] = (prices.iloc[-1] / sma - 1) if sma != 0 else 0.0
            
            # Exponential Moving Averages
            ema_periods = [12, 26, 50]
            for period in ema_periods:
                if len(prices) >= period:
                    ema = prices.ewm(span=period).mean().iloc[-1]
                    features[f'ema_{period}'] = float(ema) if pd.notna(ema) else 0.0
            
            # Moving Average Crossovers
            if 'sma_20' in features and 'sma_50' in features:
                features['ma_crossover_20_50'] = features['sma_20'] - features['sma_50']
            
        except Exception as e:
            logger.error(f"Error calculating moving averages: {e}")
        
        return features
    
    def _calculate_momentum_indicators(self, prices: pd.Series) -> Dict[str, float]:
        """Calculate momentum indicators"""
        features = {}
        
        try:
            # RSI
            if len(prices) >= 15:
                delta = prices.diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                features['rsi'] = float(rsi.iloc[-1]) if pd.notna(rsi.iloc[-1]) else 50.0
            
            # MACD
            if len(prices) >= 26:
                exp1 = prices.ewm(span=12).mean()
                exp2 = prices.ewm(span=26).mean()
                macd = exp1 - exp2
                signal = macd.ewm(span=9).mean()
                histogram = macd - signal
                
                features['macd'] = float(macd.iloc[-1]) if pd.notna(macd.iloc[-1]) else 0.0
                features['macd_signal'] = float(signal.iloc[-1]) if pd.notna(signal.iloc[-1]) else 0.0
                features['macd_histogram'] = float(histogram.iloc[-1]) if pd.notna(histogram.iloc[-1]) else 0.0
            
            # Stochastic Oscillator
            if len(prices) >= 14:
                low_14 = prices.rolling(window=14).min()
                high_14 = prices.rolling(window=14).max()
                k_percent = 100 * ((prices - low_14) / (high_14 - low_14))
                features['stoch_k'] = float(k_percent.iloc[-1]) if pd.notna(k_percent.iloc[-1]) else 50.0
            
            # Rate of Change
            roc_periods = [1, 5, 10, 20]
            for period in roc_periods:
                if len(prices) > period:
                    roc = (prices.iloc[-1] - prices.iloc[-period-1]) / prices.iloc[-period-1] * 100
                    features[f'roc_{period}'] = float(roc) if pd.notna(roc) else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating momentum indicators: {e}")
        
        return features
    
    def _calculate_volatility_indicators(self, prices: pd.Series) -> Dict[str, float]:
        """Calculate volatility indicators"""
        features = {}
        
        try:
            # Bollinger Bands
            if len(prices) >= 20:
                sma = prices.rolling(window=20).mean()
                std = prices.rolling(window=20).std()
                bb_upper = sma + (std * 2)
                bb_lower = sma - (std * 2)
                bb_middle = sma
                
                features['bb_upper'] = float(bb_upper.iloc[-1]) if pd.notna(bb_upper.iloc[-1]) else 0.0
                features['bb_lower'] = float(bb_lower.iloc[-1]) if pd.notna(bb_lower.iloc[-1]) else 0.0
                features['bb_width'] = float((bb_upper - bb_lower).iloc[-1]) if pd.notna((bb_upper - bb_lower).iloc[-1]) else 0.0
                
                # Bollinger Band position
                bb_position = (prices.iloc[-1] - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1])
                features['bb_position'] = float(bb_position) if pd.notna(bb_position) else 0.5
            
            # Average True Range (ATR)
            if len(prices) >= 14:
                high = prices  # Using close as proxy for high
                low = prices   # Using close as proxy for low
                close = prices
                
                tr1 = high - low
                tr2 = abs(high - close.shift(1))
                tr3 = abs(low - close.shift(1))
                true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                atr = true_range.rolling(window=14).mean()
                features['atr'] = float(atr.iloc[-1]) if pd.notna(atr.iloc[-1]) else 0.0
            
            # Volatility measures
            volatility_periods = [10, 20, 30]
            for period in volatility_periods:
                if len(prices) >= period:
                    returns = prices.pct_change()
                    vol = returns.rolling(window=period).std() * np.sqrt(252)  # Annualized
                    features[f'volatility_{period}d'] = float(vol.iloc[-1]) if pd.notna(vol.iloc[-1]) else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating volatility indicators: {e}")
        
        return features
    
    def _calculate_volume_indicators(self, prices: pd.Series, volumes: pd.Series) -> Dict[str, float]:
        """Calculate volume-based indicators"""
        features = {}
        
        try:
            # Volume moving averages
            if len(volumes) >= 20:
                volume_sma = volumes.rolling(window=20).mean()
                features['volume_sma_20'] = float(volume_sma.iloc[-1]) if pd.notna(volume_sma.iloc[-1]) else 0.0
                features['volume_ratio'] = volumes.iloc[-1] / volume_sma.iloc[-1] if volume_sma.iloc[-1] != 0 else 1.0
            
            # On Balance Volume (OBV)
            if len(prices) == len(volumes) and len(prices) > 1:
                price_changes = prices.diff()
                obv = pd.Series(index=prices.index, dtype=float)
                obv.iloc[0] = volumes.iloc[0]
                
                for i in range(1, len(prices)):
                    if price_changes.iloc[i] > 0:
                        obv.iloc[i] = obv.iloc[i-1] + volumes.iloc[i]
                    elif price_changes.iloc[i] < 0:
                        obv.iloc[i] = obv.iloc[i-1] - volumes.iloc[i]
                    else:
                        obv.iloc[i] = obv.iloc[i-1]
                
                features['obv'] = float(obv.iloc[-1])
                
                # OBV trend
                if len(obv) >= 10:
                    obv_sma = obv.rolling(window=10).mean()
                    features['obv_trend'] = (obv.iloc[-1] - obv_sma.iloc[-1]) / obv_sma.iloc[-1] if obv_sma.iloc[-1] != 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating volume indicators: {e}")
        
        return features
    
    def _calculate_pattern_features(self, prices: pd.Series) -> Dict[str, float]:
        """Calculate pattern recognition features"""
        features = {}
        
        try:
            if len(prices) >= 50:
                # Support and Resistance levels
                recent_prices = prices.tail(20)
                features['support_level'] = float(recent_prices.min())
                features['resistance_level'] = float(recent_prices.max())
                features['price_vs_support'] = (prices.iloc[-1] - features['support_level']) / features['support_level'] if features['support_level'] != 0 else 0.0
                features['price_vs_resistance'] = (prices.iloc[-1] - features['resistance_level']) / features['resistance_level'] if features['resistance_level'] != 0 else 0.0
                
                # Trend strength
                returns = prices.pct_change().dropna()
                if len(returns) >= 20:
                    positive_returns = len(returns[returns > 0])
                    features['trend_strength'] = (positive_returns / len(returns)) * 2 - 1  # Convert to -1 to 1 scale
                
                # Price momentum patterns
                if len(prices) >= 5:
                    recent_trend = np.polyfit(range(5), prices.tail(5), 1)[0]
                    features['short_term_trend'] = float(recent_trend)
                
        except Exception as e:
            logger.error(f"Error calculating pattern features: {e}")
        
        return features
    
    def _calculate_market_structure(self, price_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate market microstructure features"""
        features = {}
        
        try:
            if 'Open' in price_data.columns and 'Close' in price_data.columns:
                # Gap analysis
                gaps = price_data['Open'] - price_data['Close'].shift(1)
                if len(gaps) > 1:
                    features['avg_gap'] = float(gaps.mean()) if pd.notna(gaps.mean()) else 0.0
                    features['gap_volatility'] = float(gaps.std()) if pd.notna(gaps.std()) else 0.0
            
            # Intraday returns
            if 'High' in price_data.columns and 'Low' in price_data.columns and 'Close' in price_data.columns:
                intraday_returns = (price_data['High'] - price_data['Low']) / price_data['Close']
                if len(intraday_returns) > 0:
                    features['avg_intraday_return'] = float(intraday_returns.mean()) if pd.notna(intraday_returns.mean()) else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating market structure: {e}")
        
        return features
    
    def create_sentiment_features(self, news_data: List[Dict], social_data: List[Dict]) -> Dict[str, float]:
        """Create sentiment-based features from news and social data"""
        try:
            all_sentiments = []
            news_sentiments = []
            social_sentiments = []
            
            # Process news data
            for item in news_data:
                text = f"{item.get('title', '')} {item.get('content', '')}"
                sentiment = self.sentiment_analyzer.analyze_sentiment(text)
                all_sentiments.append(sentiment.score)
                news_sentiments.append(sentiment.score)
            
            # Process social data
            for item in social_data:
                text = item.get('content', '')
                sentiment = self.sentiment_analyzer.analyze_sentiment(text)
                all_sentiments.append(sentiment.score)
                social_sentiments.append(sentiment.score)
            
            features = {}
            
            # Overall sentiment features
            if all_sentiments:
                features.update(self._calculate_sentiment_stats(all_sentiments, 'overall'))
            
            # News-specific features
            if news_sentiments:
                features.update(self._calculate_sentiment_stats(news_sentiments, 'news'))
            
            # Social-specific features
            if social_sentiments:
                features.update(self._calculate_sentiment_stats(social_sentiments, 'social'))
            
            # Cross-source sentiment analysis
            if news_sentiments and social_sentiments:
                features['sentiment_divergence'] = abs(np.mean(news_sentiments) - np.mean(social_sentiments))
            
            return features
            
        except Exception as e:
            logger.error(f"Error creating sentiment features: {e}")
            return {}
    
    def _calculate_sentiment_stats(self, sentiments: List[float], prefix: str) -> Dict[str, float]:
        """Calculate statistical features from sentiment scores"""
        if not sentiments:
            return {}
        
        scores = np.array(sentiments)
        
        return {
            f'{prefix}_sentiment_mean': float(scores.mean()),
            f'{prefix}_sentiment_std': float(scores.std()),
            f'{prefix}_sentiment_min': float(scores.min()),
            f'{prefix}_sentiment_max': float(scores.max()),
            f'{prefix}_sentiment_median': float(np.median(scores)),
            f'{prefix}_sentiment_skew': float(self._calculate_skew(scores)),
            f'{prefix}_sentiment_positive_ratio': float(len(scores[scores > 0.1]) / len(scores)),
            f'{prefix}_sentiment_negative_ratio': float(len(scores[scores < -0.1]) / len(scores)),
            f'{prefix}_sentiment_neutral_ratio': float(len(scores[abs(scores) <= 0.1]) / len(scores)),
            f'{prefix}_sentiment_extreme_ratio': float(len(scores[abs(scores) > 0.7]) / len(scores)),
            f'{prefix}_sentiment_momentum': float(self._calculate_sentiment_momentum(scores))
        }
    
    def _calculate_skew(self, data: np.ndarray) -> float:
        """Calculate skewness"""
        try:
            if len(data) < 3:
                return 0.0
            
            mean = data.mean()
            std = data.std()
            
            if std == 0:
                return 0.0
            
            skew = ((data - mean) ** 3).mean() / (std ** 3)
            return skew
        except:
            return 0.0
    
    def _calculate_sentiment_momentum(self, scores: np.ndarray) -> float:
        """Calculate sentiment momentum (trend over time)"""
        try:
            if len(scores) < 2:
                return 0.0
            
            # Simple linear regression slope
            x = np.arange(len(scores))
            slope = np.polyfit(x, scores, 1)[0]
            return slope
        except:
            return 0.0
    
    def create_macro_features(self, macro_data: Dict[str, Any]) -> Dict[str, float]:
        """Create macroeconomic features"""
        features = {}
        
        try:
            # Interest rates
            if 'interest_rate' in macro_data:
                features['interest_rate'] = float(macro_data['interest_rate'])
            
            # VIX (volatility index)
            if 'vix' in macro_data:
                features['vix'] = float(macro_data['vix'])
                features['vix_normalized'] = (macro_data['vix'] - 15) / 25  # Normalize around typical range
            
            # Dollar index
            if 'dxy' in macro_data:
                features['dxy'] = float(macro_data['dxy'])
            
            # Sector performance
            if 'sector_performance' in macro_data:
                sector_data = macro_data['sector_performance']
                if isinstance(sector_data, dict):
                    for sector, performance in sector_data.items():
                        features[f'sector_{sector}_performance'] = float(performance)
            
        except Exception as e:
            logger.error(f"Error creating macro features: {e}")
        
        return features

# Test function
def test_advanced_features():
    """Test advanced feature engineering"""
    engineer = AdvancedFeatureEngineer()
    
    # Create sample price data
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    prices = 100 + np.cumsum(np.random.randn(100) * 0.02)
    volumes = np.random.randint(1000000, 5000000, 100)
    
    price_data = pd.DataFrame({
        'Date': dates,
        'Close': prices,
        'Volume': volumes,
        'Open': prices + np.random.randn(100) * 0.01,
        'High': prices + abs(np.random.randn(100) * 0.02),
        'Low': prices - abs(np.random.randn(100) * 0.02)
    })
    
    # Test technical features
    tech_features = engineer.create_technical_features(price_data)
    print(f"Technical features: {len(tech_features)} created")
    print(f"Sample: {list(tech_features.items())[:5]}")
    
    # Test sentiment features
    news_data = [
        {'title': 'Stock market rallies on positive economic news', 'content': 'Markets are optimistic'},
        {'title': 'Concerns over inflation impact', 'content': 'Investors worried about rising prices'}
    ]
    
    social_data = [
        {'content': 'Great earnings report! Very bullish on this stock'},
        {'content': 'Market looks scary, might be time to sell'}
    ]
    
    sentiment_features = engineer.create_sentiment_features(news_data, social_data)
    print(f"Sentiment features: {len(sentiment_features)} created")
    print(f"Sample: {list(sentiment_features.items())[:5]}")

if __name__ == "__main__":
    test_advanced_features()