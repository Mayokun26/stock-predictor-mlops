"""
PostgreSQL Database Schema for MLOps Stock Prediction System
Enterprise-grade schema with connection pooling, migrations, and monitoring
"""
from sqlalchemy import (
    create_engine, Column, String, Float, Integer, DateTime, Boolean, 
    Text, JSON, Index, ForeignKey, UniqueConstraint, CheckConstraint
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.pool import QueuePool
from sqlalchemy.dialects.postgresql import UUID, JSONB
import uuid
from datetime import datetime
from typing import Optional, Dict, Any, List
import logging
import os
from contextlib import contextmanager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

Base = declarative_base()

# Core Tables
class MarketData(Base):
    __tablename__ = 'market_data'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    symbol = Column(String(10), nullable=False, index=True)
    source = Column(String(50), nullable=False)  # yahoo, kalshi, etc.
    price = Column(Float, nullable=False)
    volume = Column(Integer, default=0)
    bid = Column(Float)
    ask = Column(Float)
    spread = Column(Float)
    timestamp = Column(DateTime, nullable=False, index=True)
    metadata = Column(JSONB, default={})
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_market_data_symbol_timestamp', 'symbol', 'timestamp'),
        Index('idx_market_data_source_timestamp', 'source', 'timestamp'),
    )

class NewsData(Base):
    __tablename__ = 'news_data'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(Text, nullable=False)
    content = Column(Text)
    source = Column(String(100), nullable=False)
    url = Column(Text, unique=True)
    author = Column(String(200))
    published_at = Column(DateTime, nullable=False, index=True)
    sentiment_score = Column(Float)
    relevance_score = Column(Float)
    symbols_mentioned = Column(JSONB, default=[])  # Array of stock symbols
    entities = Column(JSONB, default={})  # Named entities extracted
    metadata = Column(JSONB, default={})
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_news_published_at', 'published_at'),
        Index('idx_news_sentiment', 'sentiment_score'),
        Index('gin_symbols_mentioned', 'symbols_mentioned', postgresql_using='gin'),
    )

class SocialData(Base):
    __tablename__ = 'social_data'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    platform = Column(String(50), nullable=False)  # reddit, twitter, etc.
    post_id = Column(String(200), unique=True)
    content = Column(Text, nullable=False)
    author = Column(String(200))
    published_at = Column(DateTime, nullable=False, index=True)
    sentiment_score = Column(Float)
    engagement_score = Column(Integer, default=0)  # upvotes, likes, etc.
    symbols_mentioned = Column(JSONB, default=[])
    subreddit = Column(String(100))  # For Reddit
    hashtags = Column(JSONB, default=[])  # For Twitter
    metadata = Column(JSONB, default={})
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_social_platform_published', 'platform', 'published_at'),
        Index('idx_social_sentiment', 'sentiment_score'),
        Index('gin_social_symbols', 'symbols_mentioned', postgresql_using='gin'),
    )

# Feature Tables
class Features(Base):
    __tablename__ = 'features'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    entity_id = Column(String(50), nullable=False, index=True)  # Symbol or entity
    feature_name = Column(String(100), nullable=False)
    feature_value = Column(Float, nullable=False)
    feature_type = Column(String(50), nullable=False)  # technical, sentiment, macro, etc.
    version = Column(String(20), default='1.0')
    timestamp = Column(DateTime, nullable=False, index=True)
    metadata = Column(JSONB, default={})
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_features_entity_name_timestamp', 'entity_id', 'feature_name', 'timestamp'),
        Index('idx_features_type', 'feature_type'),
        UniqueConstraint('entity_id', 'feature_name', 'timestamp', 'version'),
    )

# Model Tables
class Models(Base):
    __tablename__ = 'models'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(200), nullable=False)
    version = Column(String(50), nullable=False)
    model_type = Column(String(100), nullable=False)  # xgboost, lstm, etc.
    status = Column(String(50), default='training')  # training, deployed, archived
    hyperparameters = Column(JSONB, default={})
    metrics = Column(JSONB, default={})  # accuracy, precision, etc.
    features_used = Column(JSONB, default=[])
    training_data_from = Column(DateTime)
    training_data_to = Column(DateTime)
    trained_at = Column(DateTime, default=datetime.utcnow)
    deployed_at = Column(DateTime)
    artifact_path = Column(Text)  # MLflow artifact path
    mlflow_run_id = Column(String(100), unique=True)
    created_by = Column(String(100), default='system')
    metadata = Column(JSONB, default={})
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_models_name_version', 'name', 'version'),
        Index('idx_models_status', 'status'),
        UniqueConstraint('name', 'version'),
    )

class Predictions(Base):
    __tablename__ = 'predictions'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_id = Column(UUID(as_uuid=True), ForeignKey('models.id'), nullable=False)
    entity_id = Column(String(50), nullable=False, index=True)  # Symbol
    prediction_type = Column(String(50), nullable=False)  # price, direction, volatility
    predicted_value = Column(Float, nullable=False)
    confidence_score = Column(Float)
    prediction_horizon = Column(String(20))  # 1h, 1d, 1w, etc.
    features_used = Column(JSONB, default={})
    timestamp = Column(DateTime, nullable=False, index=True)
    actual_value = Column(Float)  # Filled later for evaluation
    actual_timestamp = Column(DateTime)
    error = Column(Float)  # prediction - actual
    metadata = Column(JSONB, default={})
    created_at = Column(DateTime, default=datetime.utcnow)
    
    model = relationship("Models", back_populates="predictions")
    
    __table_args__ = (
        Index('idx_predictions_entity_timestamp', 'entity_id', 'timestamp'),
        Index('idx_predictions_model_timestamp', 'model_id', 'timestamp'),
    )

Models.predictions = relationship("Predictions", back_populates="model")

# A/B Testing Tables
class ABTests(Base):
    __tablename__ = 'ab_tests'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(200), nullable=False, unique=True)
    description = Column(Text)
    champion_model_id = Column(UUID(as_uuid=True), ForeignKey('models.id'), nullable=False)
    challenger_model_id = Column(UUID(as_uuid=True), ForeignKey('models.id'), nullable=False)
    traffic_split = Column(Float, default=0.5)  # 0.5 = 50/50 split
    status = Column(String(50), default='running')  # running, completed, paused
    start_date = Column(DateTime, nullable=False)
    end_date = Column(DateTime)
    success_metric = Column(String(100), nullable=False)  # accuracy, mse, roi, etc.
    statistical_significance = Column(Float)
    winner = Column(String(20))  # champion, challenger, inconclusive
    metadata = Column(JSONB, default={})
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    champion_model = relationship("Models", foreign_keys=[champion_model_id])
    challenger_model = relationship("Models", foreign_keys=[challenger_model_id])
    
    __table_args__ = (
        Index('idx_ab_tests_status', 'status'),
        CheckConstraint('traffic_split >= 0 AND traffic_split <= 1', name='traffic_split_range'),
    )

class ABTestResults(Base):
    __tablename__ = 'ab_test_results'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    test_id = Column(UUID(as_uuid=True), ForeignKey('ab_tests.id'), nullable=False)
    model_variant = Column(String(20), nullable=False)  # champion or challenger
    metric_name = Column(String(100), nullable=False)
    metric_value = Column(Float, nullable=False)
    sample_size = Column(Integer, nullable=False)
    timestamp = Column(DateTime, nullable=False, index=True)
    metadata = Column(JSONB, default={})
    created_at = Column(DateTime, default=datetime.utcnow)
    
    test = relationship("ABTests", back_populates="results")
    
    __table_args__ = (
        Index('idx_ab_results_test_timestamp', 'test_id', 'timestamp'),
        Index('idx_ab_results_variant', 'model_variant'),
    )

ABTests.results = relationship("ABTestResults", back_populates="test")

# Monitoring Tables
class ModelMetrics(Base):
    __tablename__ = 'model_metrics'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_id = Column(UUID(as_uuid=True), ForeignKey('models.id'), nullable=False)
    metric_name = Column(String(100), nullable=False)
    metric_value = Column(Float, nullable=False)
    metric_type = Column(String(50), nullable=False)  # accuracy, latency, throughput, etc.
    window_start = Column(DateTime, nullable=False)
    window_end = Column(DateTime, nullable=False)
    sample_size = Column(Integer)
    metadata = Column(JSONB, default={})
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    
    model = relationship("Models")
    
    __table_args__ = (
        Index('idx_model_metrics_model_name', 'model_id', 'metric_name'),
        Index('idx_model_metrics_created', 'created_at'),
    )

class DataQualityChecks(Base):
    __tablename__ = 'data_quality_checks'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    table_name = Column(String(100), nullable=False)
    check_name = Column(String(100), nullable=False)
    check_type = Column(String(50), nullable=False)  # completeness, uniqueness, range, etc.
    status = Column(String(20), nullable=False)  # passed, failed, warning
    expected_value = Column(Float)
    actual_value = Column(Float)
    threshold = Column(Float)
    details = Column(JSONB, default={})
    checked_at = Column(DateTime, nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_dq_checks_table_status', 'table_name', 'status'),
        Index('idx_dq_checks_checked_at', 'checked_at'),
    )

# System Tables
class SystemLogs(Base):
    __tablename__ = 'system_logs'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    service = Column(String(100), nullable=False)
    level = Column(String(20), nullable=False, index=True)  # INFO, WARNING, ERROR
    message = Column(Text, nullable=False)
    details = Column(JSONB, default={})
    timestamp = Column(DateTime, nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_system_logs_service_level', 'service', 'level'),
    )

# Database Manager Class
class DatabaseManager:
    """Manages database connections with pooling and health checks"""
    
    def __init__(self, database_url: str = None):
        self.database_url = database_url or os.getenv(
            'DATABASE_URL', 
            'postgresql://postgres:password@localhost:5432/betting_mlops'
        )
        
        # Create engine with connection pooling
        self.engine = create_engine(
            self.database_url,
            poolclass=QueuePool,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,  # Verify connections before use
            pool_recycle=3600,   # Recycle connections every hour
            echo=False  # Set to True for SQL debugging
        )
        
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine
        )
        
        logger.info(f"Database manager initialized with URL: {self.database_url}")
    
    def create_tables(self):
        """Create all tables"""
        try:
            Base.metadata.create_all(bind=self.engine)
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(f"Error creating tables: {e}")
            raise
    
    def drop_tables(self):
        """Drop all tables (use with caution!)"""
        try:
            Base.metadata.drop_all(bind=self.engine)
            logger.info("Database tables dropped")
        except Exception as e:
            logger.error(f"Error dropping tables: {e}")
            raise
    
    @contextmanager
    def get_session(self):
        """Get database session with automatic cleanup"""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
    
    def health_check(self) -> Dict[str, Any]:
        """Check database health"""
        try:
            with self.get_session() as session:
                # Simple query to test connection
                result = session.execute("SELECT 1 as health_check")
                health_value = result.scalar()
                
                # Get connection pool stats
                pool = self.engine.pool
                
                return {
                    'status': 'healthy' if health_value == 1 else 'unhealthy',
                    'pool_size': pool.size(),
                    'checked_in_connections': pool.checkedin(),
                    'checked_out_connections': pool.checkedout(),
                    'overflow_connections': pool.overflow(),
                    'invalid_connections': pool.invalid(),
                }
                
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e)
            }
    
    def get_table_stats(self) -> Dict[str, int]:
        """Get row counts for all tables"""
        stats = {}
        try:
            with self.get_session() as session:
                for table_name in Base.metadata.tables.keys():
                    result = session.execute(f"SELECT COUNT(*) FROM {table_name}")
                    stats[table_name] = result.scalar()
        except Exception as e:
            logger.error(f"Error getting table stats: {e}")
            stats['error'] = str(e)
        
        return stats

# Helper functions for common operations
class DatabaseOperations:
    """Common database operations"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
    
    def insert_market_data(self, symbol: str, source: str, price: float, 
                          volume: int = 0, timestamp: datetime = None, **kwargs) -> str:
        """Insert market data record"""
        try:
            with self.db.get_session() as session:
                record = MarketData(
                    symbol=symbol,
                    source=source,
                    price=price,
                    volume=volume,
                    timestamp=timestamp or datetime.utcnow(),
                    bid=kwargs.get('bid'),
                    ask=kwargs.get('ask'),
                    spread=kwargs.get('spread'),
                    metadata=kwargs.get('metadata', {})
                )
                session.add(record)
                session.flush()  # Get ID without committing
                return str(record.id)
        except Exception as e:
            logger.error(f"Error inserting market data: {e}")
            raise
    
    def get_latest_price(self, symbol: str, source: str = None) -> Optional[float]:
        """Get latest price for symbol"""
        try:
            with self.db.get_session() as session:
                query = session.query(MarketData).filter(MarketData.symbol == symbol)
                if source:
                    query = query.filter(MarketData.source == source)
                
                latest = query.order_by(MarketData.timestamp.desc()).first()
                return latest.price if latest else None
                
        except Exception as e:
            logger.error(f"Error getting latest price: {e}")
            return None
    
    def insert_prediction(self, model_id: str, entity_id: str, prediction_type: str,
                         predicted_value: float, confidence_score: float = None,
                         features_used: Dict = None) -> str:
        """Insert prediction record"""
        try:
            with self.db.get_session() as session:
                record = Predictions(
                    model_id=model_id,
                    entity_id=entity_id,
                    prediction_type=prediction_type,
                    predicted_value=predicted_value,
                    confidence_score=confidence_score,
                    features_used=features_used or {},
                    timestamp=datetime.utcnow()
                )
                session.add(record)
                session.flush()
                return str(record.id)
        except Exception as e:
            logger.error(f"Error inserting prediction: {e}")
            raise

# Test function
def test_database():
    """Test database operations"""
    # Initialize database
    db_manager = DatabaseManager()
    db_ops = DatabaseOperations(db_manager)
    
    try:
        # Create tables
        db_manager.create_tables()
        
        # Health check
        health = db_manager.health_check()
        print(f"Database health: {health}")
        
        # Insert test data
        market_id = db_ops.insert_market_data(
            symbol="AAPL",
            source="test",
            price=150.25,
            volume=1000000
        )
        print(f"Inserted market data with ID: {market_id}")
        
        # Get latest price
        price = db_ops.get_latest_price("AAPL")
        print(f"Latest AAPL price: {price}")
        
        # Table stats
        stats = db_manager.get_table_stats()
        print(f"Table stats: {stats}")
        
    except Exception as e:
        logger.error(f"Database test failed: {e}")

if __name__ == "__main__":
    test_database()