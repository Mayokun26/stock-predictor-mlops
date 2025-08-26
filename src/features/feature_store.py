"""
Redis-Based Feature Store for Real-Time ML Feature Serving
Enterprise-grade feature store with versioning, TTL, and real-time serving capabilities
"""
import redis
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
import logging
from dataclasses import dataclass, asdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class FeatureMetadata:
    """Metadata for features in the store"""
    name: str
    version: str
    data_type: str
    description: str
    source: str
    timestamp: datetime
    ttl_seconds: int
    tags: List[str]
    validation_rules: Dict[str, Any]

@dataclass
class FeatureValue:
    """Feature value with metadata"""
    entity_id: str
    feature_name: str
    value: Any
    timestamp: datetime
    version: str
    metadata: Optional[Dict[str, Any]] = None

class RedisFeatureStore:
    """Production-grade Redis-based feature store"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url
        self.feature_metadata: Dict[str, FeatureMetadata] = {}
        try:
            self.redis_client = redis.Redis.from_url(redis_url, decode_responses=True)
            # Test connection
            self.redis_client.ping()
            logger.info(f"Connected to Redis at {redis_url}")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}. Running in mock mode.")
            self.redis_client = None
    
    def _generate_feature_key(self, entity_id: str, feature_name: str, version: str = "latest") -> str:
        """Generate Redis key for feature"""
        return f"feature:{entity_id}:{feature_name}:{version}"
    
    def _generate_metadata_key(self, feature_name: str, version: str = "latest") -> str:
        """Generate Redis key for feature metadata"""
        return f"metadata:{feature_name}:{version}"
    
    def register_feature(self, metadata: FeatureMetadata) -> bool:
        """Register a new feature with metadata"""
        try:
            if self.redis_client:
                metadata_key = self._generate_metadata_key(metadata.name, metadata.version)
                metadata_json = json.dumps(asdict(metadata), default=str)
                
                if metadata.ttl_seconds > 0:
                    self.redis_client.setex(metadata_key, metadata.ttl_seconds, metadata_json)
                else:
                    self.redis_client.set(metadata_key, metadata_json)
            
            # Store in local cache
            self.feature_metadata[f"{metadata.name}:{metadata.version}"] = metadata
            logger.info(f"Registered feature: {metadata.name} v{metadata.version}")
            return True
                
        except Exception as e:
            logger.error(f"Error registering feature {metadata.name}: {e}")
            return False
    
    def write_feature(self, feature: FeatureValue) -> bool:
        """Write a single feature value"""
        try:
            if not self.redis_client:
                logger.info(f"Mock write: {feature.entity_id}:{feature.feature_name} = {feature.value}")
                return True
                
            feature_key = self._generate_feature_key(
                feature.entity_id, 
                feature.feature_name, 
                feature.version
            )
            
            feature_data = {
                'value': feature.value,
                'timestamp': feature.timestamp.isoformat(),
                'version': feature.version,
                'metadata': feature.metadata or {}
            }
            
            feature_json = json.dumps(feature_data, default=str)
            
            # Default TTL
            ttl = 3600
            metadata_key = f"{feature.feature_name}:{feature.version}"
            if metadata_key in self.feature_metadata:
                ttl = self.feature_metadata[metadata_key].ttl_seconds
            
            self.redis_client.setex(feature_key, ttl, feature_json)
            
            # Also store as 'latest' version
            latest_key = self._generate_feature_key(
                feature.entity_id, 
                feature.feature_name, 
                "latest"
            )
            self.redis_client.setex(latest_key, ttl, feature_json)
            
            return True
                
        except Exception as e:
            logger.error(f"Error writing feature {feature.feature_name}: {e}")
            return False
    
    def read_feature(self, entity_id: str, feature_name: str, version: str = "latest") -> Optional[FeatureValue]:
        """Read a single feature value"""
        try:
            if not self.redis_client:
                return None
                
            feature_key = self._generate_feature_key(entity_id, feature_name, version)
            feature_data = self.redis_client.get(feature_key)
            
            if not feature_data:
                return None
            
            data = json.loads(feature_data)
            
            return FeatureValue(
                entity_id=entity_id,
                feature_name=feature_name,
                value=data['value'],
                timestamp=datetime.fromisoformat(data['timestamp']),
                version=data['version'],
                metadata=data.get('metadata')
            )
                
        except Exception as e:
            logger.error(f"Error reading feature {feature_name}: {e}")
            return None
    
    def health_check(self) -> Dict[str, Any]:
        """Check feature store health"""
        try:
            if self.redis_client:
                self.redis_client.ping()
                return {
                    'status': 'healthy',
                    'feature_count': len(self.feature_metadata),
                    'redis_url': self.redis_url
                }
            else:
                return {
                    'status': 'mock_mode',
                    'feature_count': len(self.feature_metadata)
                }
                
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e)
            }

class FeatureEngineering:
    """Feature engineering utilities"""
    
    @staticmethod
    def calculate_technical_indicators(prices: pd.Series) -> Dict[str, float]:
        """Calculate technical indicators from price series"""
        try:
            if len(prices) < 20:
                return {}
            
            # Simple Moving Averages
            sma_5 = prices.rolling(window=5).mean().iloc[-1]
            sma_20 = prices.rolling(window=20).mean().iloc[-1]
            sma_50 = prices.rolling(window=50).mean().iloc[-1] if len(prices) >= 50 else None
            
            # RSI
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            sma = prices.rolling(window=20).mean()
            std = prices.rolling(window=20).std()
            bb_upper = sma + (std * 2)
            bb_lower = sma - (std * 2)
            
            return {
                'sma_5': float(sma_5) if pd.notna(sma_5) else 0.0,
                'sma_20': float(sma_20) if pd.notna(sma_20) else 0.0,
                'sma_50': float(sma_50) if pd.notna(sma_50) and sma_50 is not None else 0.0,
                'rsi': float(rsi.iloc[-1]) if pd.notna(rsi.iloc[-1]) else 50.0,
                'bb_upper': float(bb_upper.iloc[-1]) if pd.notna(bb_upper.iloc[-1]) else 0.0,
                'bb_lower': float(bb_lower.iloc[-1]) if pd.notna(bb_lower.iloc[-1]) else 0.0,
                'price_change_1d': float((prices.iloc[-1] - prices.iloc[-2]) / prices.iloc[-2]) if len(prices) > 1 and prices.iloc[-2] != 0 else 0.0,
                'volatility_20d': float(prices.rolling(window=20).std().iloc[-1]) if pd.notna(prices.rolling(window=20).std().iloc[-1]) else 0.0
            }
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
            return {}
    
    @staticmethod
    def calculate_sentiment_features(sentiment_scores: List[float]) -> Dict[str, float]:
        """Calculate sentiment-based features"""
        try:
            if not sentiment_scores:
                return {}
            
            scores = np.array(sentiment_scores)
            
            return {
                'sentiment_mean': float(scores.mean()),
                'sentiment_std': float(scores.std()),
                'sentiment_positive_ratio': float(len(scores[scores > 0]) / len(scores)),
                'sentiment_negative_ratio': float(len(scores[scores < 0]) / len(scores)),
                'sentiment_trend': float(np.polyfit(range(len(scores)), scores, 1)[0]) if len(scores) > 1 else 0.0
            }
            
        except Exception as e:
            logger.error(f"Error calculating sentiment features: {e}")
            return {}

# Test function
def test_feature_store():
    """Test the feature store"""
    store = RedisFeatureStore()
    
    # Register a feature
    metadata = FeatureMetadata(
        name="stock_price",
        version="v1.0",
        data_type="float",
        description="Current stock price",
        source="yahoo_finance",
        timestamp=datetime.now(),
        ttl_seconds=1800,
        tags=["finance", "price"],
        validation_rules={"min_value": 0.0}
    )
    
    store.register_feature(metadata)
    
    # Write a feature
    feature = FeatureValue(
        entity_id="AAPL",
        feature_name="stock_price",
        value=150.25,
        timestamp=datetime.now(),
        version="v1.0"
    )
    
    success = store.write_feature(feature)
    print(f"Feature write success: {success}")
    
    # Read the feature
    read_feature = store.read_feature("AAPL", "stock_price")
    print(f"Read feature: {read_feature}")
    
    # Health check
    health = store.health_check()
    print(f"Health check: {health}")

if __name__ == "__main__":
    test_feature_store()