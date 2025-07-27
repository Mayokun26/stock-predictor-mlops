"""
SQLAlchemy models for stock market prediction system
"""

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime

Base = declarative_base()


class Stock(Base):
    """Stock master data"""
    __tablename__ = 'stocks'
    
    symbol = Column(String(10), primary_key=True)
    name = Column(String(255), nullable=False)
    sector = Column(String(100))
    industry = Column(String(100))
    market_cap = Column(Float)
    exchange = Column(String(10))
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    prices = relationship("StockPrice", back_populates="stock")
    fundamentals = relationship("StockFundamentals", back_populates="stock")


class StockPrice(Base):
    """Daily stock price and volume data"""
    __tablename__ = 'stock_prices'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(10), ForeignKey('stocks.symbol'), nullable=False)
    date = Column(DateTime, nullable=False)
    open_price = Column(Float)
    high_price = Column(Float)
    low_price = Column(Float)
    close_price = Column(Float)
    adj_close_price = Column(Float)
    volume = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship
    stock = relationship("Stock", back_populates="prices")


class StockFundamentals(Base):
    """Fundamental analysis data"""
    __tablename__ = 'stock_fundamentals'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(10), ForeignKey('stocks.symbol'), nullable=False)
    date = Column(DateTime, nullable=False)
    
    # Valuation metrics
    pe_ratio = Column(Float)
    pb_ratio = Column(Float)
    ps_ratio = Column(Float)
    peg_ratio = Column(Float)
    ev_ebitda = Column(Float)
    
    # Profitability metrics
    roe = Column(Float)  # Return on Equity
    roa = Column(Float)  # Return on Assets
    roic = Column(Float)  # Return on Invested Capital
    gross_margin = Column(Float)
    operating_margin = Column(Float)
    net_margin = Column(Float)
    
    # Financial strength
    debt_to_equity = Column(Float)
    current_ratio = Column(Float)
    quick_ratio = Column(Float)
    debt_to_assets = Column(Float)
    
    # Growth metrics
    revenue_growth = Column(Float)
    earnings_growth = Column(Float)
    book_value_growth = Column(Float)
    
    # Size metrics
    market_cap = Column(Float)
    enterprise_value = Column(Float)
    shares_outstanding = Column(Float)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship
    stock = relationship("Stock", back_populates="fundamentals")


class NewsArticle(Base):
    """News articles for sentiment analysis"""
    __tablename__ = 'news_articles'
    
    id = Column(Integer, primary_key=True)
    title = Column(Text, nullable=False)
    content = Column(Text)
    url = Column(Text)
    source = Column(String(100))
    published_at = Column(DateTime, nullable=False)
    
    # Sentiment analysis results
    sentiment_score = Column(Float)  # -1 to 1 (negative to positive)
    sentiment_magnitude = Column(Float)  # 0 to 1 (confidence)
    sentiment_label = Column(String(20))  # positive, negative, neutral
    
    # Stock relevance
    symbols_mentioned = Column(Text)  # JSON array of stock symbols
    
    created_at = Column(DateTime, default=datetime.utcnow)


class TechnicalIndicator(Base):
    """Technical analysis indicators"""
    __tablename__ = 'technical_indicators'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(10), ForeignKey('stocks.symbol'), nullable=False)
    date = Column(DateTime, nullable=False)
    
    # Moving averages
    sma_20 = Column(Float)  # 20-day simple moving average
    sma_50 = Column(Float)  # 50-day simple moving average
    sma_200 = Column(Float)  # 200-day simple moving average
    ema_12 = Column(Float)  # 12-day exponential moving average
    ema_26 = Column(Float)  # 26-day exponential moving average
    
    # Momentum indicators
    rsi = Column(Float)  # Relative Strength Index
    macd = Column(Float)  # MACD line
    macd_signal = Column(Float)  # MACD signal line
    macd_histogram = Column(Float)  # MACD histogram
    
    # Volatility indicators
    bollinger_upper = Column(Float)
    bollinger_middle = Column(Float)
    bollinger_lower = Column(Float)
    
    # Volume indicators
    volume_sma_20 = Column(Float)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship
    stock = relationship("Stock")


class Prediction(Base):
    """ML model predictions"""
    __tablename__ = 'predictions'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(10), ForeignKey('stocks.symbol'), nullable=False)
    prediction_date = Column(DateTime, nullable=False)
    target_date = Column(DateTime, nullable=False)  # Date being predicted
    
    # Prediction details
    model_name = Column(String(100), nullable=False)
    model_version = Column(String(50))
    prediction_type = Column(String(50))  # price_direction, price_target, volatility
    
    # Prediction values
    predicted_value = Column(Float)
    confidence_score = Column(Float)
    prediction_label = Column(String(20))  # up, down, hold
    
    # Actual outcome (filled in later)
    actual_value = Column(Float)
    was_correct = Column(Boolean)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship
    stock = relationship("Stock")