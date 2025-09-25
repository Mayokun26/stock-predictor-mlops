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
import hashlib
import uuid
from dataclasses import dataclass, asdict
from collections import defaultdict

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
    # Enhanced lineage tracking
    parent_features: List[str]  # Features this depends on
    transformation_code: str     # Code used to generate feature
    feature_group: str           # Logical grouping
    owner: str                   # Feature owner
    approval_status: str         # approved, pending, deprecated
    quality_score: Optional[float] = None  # Data quality score
    schema_version: str = "1.0"   # Schema evolution tracking

@dataclass
class FeatureValue:
    """Feature value with metadata"""
    entity_id: str
    feature_name: str
    value: Any
    timestamp: datetime
    version: str
    metadata: Optional[Dict[str, Any]] = None
    # Enhanced tracking
    compute_id: str = None       # Unique computation ID
    data_quality_checks: Dict[str, bool] = None  # Quality validation results
    lineage_hash: str = None     # Hash of input features

class RedisFeatureStore:
    """Production-grade Redis-based feature store with advanced lineage tracking"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url
        self.feature_metadata: Dict[str, FeatureMetadata] = {}
        self.feature_lineage: Dict[str, List[str]] = defaultdict(list)  # Feature dependency graph
        self.feature_versions: Dict[str, List[str]] = defaultdict(list)  # Version history
        
        try:
            self.redis_client = redis.Redis.from_url(redis_url, decode_responses=True)
            # Test connection
            self.redis_client.ping()
            logger.info(f"Connected to Redis at {redis_url}")
            
            # Initialize lineage tracking structures
            self._initialize_lineage_tracking()
            
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}. Running in mock mode.")
            self.redis_client = None
    
    def _generate_feature_key(self, entity_id: str, feature_name: str, version: str = "latest") -> str:
        """Generate Redis key for feature"""
        return f"feature:{entity_id}:{feature_name}:{version}"
    
    def _generate_metadata_key(self, feature_name: str, version: str = "latest") -> str:
        """Generate Redis key for feature metadata"""
        return f"metadata:{feature_name}:{version}"
    
    def _initialize_lineage_tracking(self):
        """Initialize lineage tracking structures in Redis"""
        try:
            if self.redis_client:
                # Check if lineage structures exist, create if not
                if not self.redis_client.exists("feature_lineage"):
                    self.redis_client.hset("feature_lineage", mapping={})
                    
                if not self.redis_client.exists("feature_versions"):
                    self.redis_client.hset("feature_versions", mapping={})
                    
                logger.info("Lineage tracking initialized")
        except Exception as e:
            logger.error(f"Error initializing lineage tracking: {e}")
    
    def _generate_lineage_hash(self, input_features: List[str], transformation_code: str) -> str:
        """Generate hash for feature lineage tracking"""
        lineage_input = f"{sorted(input_features)}:{transformation_code}"
        return hashlib.sha256(lineage_input.encode()).hexdigest()[:16]
    
    def _update_feature_lineage(self, feature_name: str, parent_features: List[str], version: str):
        """Update feature lineage graph"""
        try:
            if self.redis_client:
                # Store lineage information
                lineage_key = f"lineage:{feature_name}:{version}"
                lineage_data = {
                    'parents': json.dumps(parent_features),
                    'timestamp': datetime.now().isoformat(),
                    'downstream_features': json.dumps([])  # Will be updated when features depend on this one
                }
                self.redis_client.hset(lineage_key, mapping=lineage_data)
                
                # Update parent features' downstream dependencies
                for parent in parent_features:
                    parent_lineage_key = f"lineage:{parent}:latest"
                    if self.redis_client.exists(parent_lineage_key):
                        downstream = self.redis_client.hget(parent_lineage_key, 'downstream_features')
                        if downstream:
                            downstream_list = json.loads(downstream)
                            if feature_name not in downstream_list:
                                downstream_list.append(feature_name)
                                self.redis_client.hset(parent_lineage_key, 'downstream_features', json.dumps(downstream_list))
                
                logger.info(f"Updated lineage for {feature_name}")
                
        except Exception as e:
            logger.error(f"Error updating lineage for {feature_name}: {e}")
    
    def _update_version_history(self, feature_name: str, version: str):
        """Update version history for a feature"""
        try:
            if self.redis_client:
                version_key = f"versions:{feature_name}"
                versions = self.redis_client.lrange(version_key, 0, -1)
                if version not in versions:
                    self.redis_client.lpush(version_key, version)
                    # Keep only last 50 versions
                    self.redis_client.ltrim(version_key, 0, 49)
                    
        except Exception as e:
            logger.error(f"Error updating version history for {feature_name}: {e}")
    
    def register_feature(self, metadata: FeatureMetadata) -> bool:
        """Register a new feature with enhanced metadata and lineage tracking"""
        try:
            if self.redis_client:
                metadata_key = self._generate_metadata_key(metadata.name, metadata.version)
                metadata_json = json.dumps(asdict(metadata), default=str)
                
                if metadata.ttl_seconds > 0:
                    self.redis_client.setex(metadata_key, metadata.ttl_seconds, metadata_json)
                else:
                    self.redis_client.set(metadata_key, metadata_json)
                
                # Update lineage tracking
                self._update_feature_lineage(metadata.name, metadata.parent_features, metadata.version)
                
                # Update version history
                self._update_version_history(metadata.name, metadata.version)
            
            # Store in local cache
            self.feature_metadata[f"{metadata.name}:{metadata.version}"] = metadata
            logger.info(f"Registered feature: {metadata.name} v{metadata.version} with {len(metadata.parent_features)} dependencies")
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
    
    # Advanced Feature Store Methods
    
    def get_feature_lineage(self, feature_name: str, version: str = "latest") -> Dict[str, Any]:
        """Get complete lineage information for a feature"""
        try:
            if not self.redis_client:
                return {}
                
            lineage_key = f"lineage:{feature_name}:{version}"
            lineage_data = self.redis_client.hgetall(lineage_key)
            
            if not lineage_data:
                return {}
            
            return {
                'feature_name': feature_name,
                'version': version,
                'parent_features': json.loads(lineage_data.get('parents', '[]')),
                'downstream_features': json.loads(lineage_data.get('downstream_features', '[]')),
                'last_updated': lineage_data.get('timestamp', ''),
                'lineage_depth': self._calculate_lineage_depth(feature_name, version)
            }
            
        except Exception as e:
            logger.error(f"Error getting lineage for {feature_name}: {e}")
            return {}
    
    def _calculate_lineage_depth(self, feature_name: str, version: str, visited: set = None) -> int:
        """Calculate the depth of feature dependencies (recursive)"""
        if visited is None:
            visited = set()
        
        if feature_name in visited:
            return 0  # Prevent infinite recursion
        
        visited.add(feature_name)
        
        try:
            if not self.redis_client:
                return 0
                
            lineage_key = f"lineage:{feature_name}:{version}"
            lineage_data = self.redis_client.hget(lineage_key, 'parents')
            
            if not lineage_data:
                return 0
            
            parents = json.loads(lineage_data)
            if not parents:
                return 0
            
            max_depth = 0
            for parent in parents:
                parent_depth = self._calculate_lineage_depth(parent, "latest", visited.copy())
                max_depth = max(max_depth, parent_depth)
            
            return max_depth + 1
            
        except Exception as e:
            logger.error(f"Error calculating lineage depth for {feature_name}: {e}")
            return 0
    
    def get_feature_versions(self, feature_name: str) -> List[str]:
        """Get all versions of a feature"""
        try:
            if not self.redis_client:
                return []
                
            version_key = f"versions:{feature_name}"
            versions = self.redis_client.lrange(version_key, 0, -1)
            return versions
            
        except Exception as e:
            logger.error(f"Error getting versions for {feature_name}: {e}")
            return []
    
    def get_feature_impact_analysis(self, feature_name: str) -> Dict[str, Any]:
        """Analyze the impact of a feature across the system"""
        try:
            lineage = self.get_feature_lineage(feature_name)
            
            impact_analysis = {
                'feature_name': feature_name,
                'direct_dependencies': len(lineage.get('downstream_features', [])),
                'dependency_tree': self._build_dependency_tree(feature_name),
                'total_impacted_features': 0,
                'critical_paths': [],
                'recommendation': ""
            }
            
            # Calculate total impact
            def count_downstream(fname: str, visited: set = None) -> int:
                if visited is None:
                    visited = set()
                if fname in visited:
                    return 0
                visited.add(fname)
                
                downstream = self.get_feature_lineage(fname).get('downstream_features', [])
                count = len(downstream)
                for ds_feature in downstream:
                    count += count_downstream(ds_feature, visited.copy())
                return count
            
            impact_analysis['total_impacted_features'] = count_downstream(feature_name)
            
            # Generate recommendation
            if impact_analysis['total_impacted_features'] > 10:
                impact_analysis['recommendation'] = "HIGH IMPACT: Changes require careful coordination and testing"
            elif impact_analysis['total_impacted_features'] > 3:
                impact_analysis['recommendation'] = "MEDIUM IMPACT: Review downstream features before changes"
            else:
                impact_analysis['recommendation'] = "LOW IMPACT: Changes are relatively safe"
            
            return impact_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing impact for {feature_name}: {e}")
            return {}
    
    def _build_dependency_tree(self, feature_name: str, visited: set = None) -> Dict[str, Any]:
        """Build a recursive dependency tree"""
        if visited is None:
            visited = set()
        
        if feature_name in visited:
            return {'name': feature_name, 'circular': True}
        
        visited.add(feature_name)
        
        lineage = self.get_feature_lineage(feature_name)
        downstream = lineage.get('downstream_features', [])
        
        tree = {
            'name': feature_name,
            'children': []
        }
        
        for child in downstream:
            tree['children'].append(self._build_dependency_tree(child, visited.copy()))
        
        return tree

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

# Enhanced Test function
def test_advanced_feature_store():
    """Test the advanced feature store with lineage tracking"""
    print("🚀 Testing Advanced Feature Store with Lineage Tracking")
    store = RedisFeatureStore()
    
    # Register base features (no dependencies)
    base_metadata = FeatureMetadata(
        name="raw_stock_price",
        version="v1.0",
        data_type="float",
        description="Raw stock price from market data",
        source="yahoo_finance",
        timestamp=datetime.now(),
        ttl_seconds=1800,
        tags=["finance", "price", "raw"],
        validation_rules={"min_value": 0.0},
        parent_features=[],  # No dependencies
        transformation_code="direct_api_fetch()",
        feature_group="market_data",
        owner="data_team",
        approval_status="approved"
    )
    
    # Register derived feature (depends on base)
    derived_metadata = FeatureMetadata(
        name="price_moving_average",
        version="v1.0", 
        data_type="float",
        description="20-day moving average of stock price",
        source="feature_engineering",
        timestamp=datetime.now(),
        ttl_seconds=3600,
        tags=["finance", "price", "derived", "technical_indicator"],
        validation_rules={"min_value": 0.0},
        parent_features=["raw_stock_price"],  # Depends on raw price
        transformation_code="calculate_moving_average(raw_stock_price, window=20)",
        feature_group="technical_indicators",
        owner="ml_team",
        approval_status="approved"
    )
    
    # Register advanced derived feature (depends on derived)
    advanced_metadata = FeatureMetadata(
        name="price_momentum_signal",
        version="v1.0",
        data_type="float", 
        description="Momentum signal based on price vs moving average",
        source="feature_engineering",
        timestamp=datetime.now(),
        ttl_seconds=1800,
        tags=["finance", "signal", "momentum"],
        validation_rules={"min_value": -1.0, "max_value": 1.0},
        parent_features=["raw_stock_price", "price_moving_average"],  # Depends on both
        transformation_code="calculate_momentum_signal(raw_stock_price, price_moving_average)",
        feature_group="trading_signals",
        owner="quant_team", 
        approval_status="approved"
    )
    
    # Register all features
    store.register_feature(base_metadata)
    store.register_feature(derived_metadata)
    store.register_feature(advanced_metadata)
    
    # Write sample feature values
    features = [
        FeatureValue("AAPL", "raw_stock_price", 150.25, datetime.now(), "v1.0"),
        FeatureValue("AAPL", "price_moving_average", 148.75, datetime.now(), "v1.0"),
        FeatureValue("AAPL", "price_momentum_signal", 0.35, datetime.now(), "v1.0")
    ]
    
    for feature in features:
        success = store.write_feature(feature)
        print(f"✅ {feature.feature_name}: {success}")
    
    print("\n📊 Advanced Feature Store Capabilities:")
    
    # Test lineage tracking
    print("\n🔗 Feature Lineage:")
    for feature_name in ["raw_stock_price", "price_moving_average", "price_momentum_signal"]:
        lineage = store.get_feature_lineage(feature_name)
        print(f"  {feature_name}:")
        print(f"    - Parents: {lineage.get('parent_features', [])}")
        print(f"    - Downstream: {lineage.get('downstream_features', [])}")
        print(f"    - Depth: {lineage.get('lineage_depth', 0)}")
    
    # Test impact analysis
    print("\n📈 Impact Analysis:")
    impact = store.get_feature_impact_analysis("raw_stock_price")
    print(f"  raw_stock_price impact:")
    print(f"    - Direct dependencies: {impact.get('direct_dependencies', 0)}")
    print(f"    - Total impacted: {impact.get('total_impacted_features', 0)}")
    print(f"    - Recommendation: {impact.get('recommendation', 'N/A')}")
    
    # Test version tracking
    print("\n🔄 Version History:")
    for feature_name in ["raw_stock_price", "price_moving_average"]:
        versions = store.get_feature_versions(feature_name)
        print(f"  {feature_name}: {versions}")
    
    # Health check
    health = store.health_check()
    print(f"\n💊 Health Check: {health}")
    
    print("\n🎯 Advanced Feature Store Test Complete!")

def test_feature_store():
    """Original basic test - kept for compatibility"""
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
        validation_rules={"min_value": 0.0},
        parent_features=[],
        transformation_code="api_fetch()",
        feature_group="market",
        owner="data_team",
        approval_status="approved"
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
    test_advanced_feature_store()