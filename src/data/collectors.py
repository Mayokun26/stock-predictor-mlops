"""
Enterprise Data Collection Pipeline
Collects data from multiple sources: Kalshi API, News APIs, Market Data
"""
import asyncio
import aiohttp
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging
from dataclasses import dataclass
import os
from dotenv import load_dotenv
import yfinance as yf
import requests
from textblob import TextBlob
import redis
import json
import time

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MarketData:
    symbol: str
    price: float
    volume: int
    timestamp: datetime
    bid: Optional[float] = None
    ask: Optional[float] = None
    spread: Optional[float] = None

@dataclass 
class NewsData:
    title: str
    content: str
    source: str
    timestamp: datetime
    sentiment_score: float
    url: str
    relevance_score: Optional[float] = None

@dataclass
class SocialData:
    platform: str
    content: str
    author: str
    timestamp: datetime
    sentiment_score: float
    engagement_score: int
    symbol_mentions: List[str]

class KalshiCollector:
    """Collect market data from Kalshi prediction markets"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv('KALSHI_API_KEY')
        self.base_url = "https://trading-api.kalshi.com/trade-api/v2"
        self.session = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def get_markets(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Fetch active prediction markets"""
        try:
            url = f"{self.base_url}/markets"
            params = {"limit": limit, "status": "open"}
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('markets', [])
                else:
                    logger.error(f"Kalshi API error: {response.status}")
                    return []
        except Exception as e:
            logger.error(f"Error fetching Kalshi markets: {e}")
            return []
    
    async def get_market_data(self, ticker: str) -> Optional[MarketData]:
        """Get specific market data"""
        try:
            url = f"{self.base_url}/markets/{ticker}"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    market = data.get('market', {})
                    
                    return MarketData(
                        symbol=ticker,
                        price=float(market.get('last_price', 0)),
                        volume=int(market.get('volume', 0)),
                        timestamp=datetime.now(),
                        bid=market.get('yes_bid'),
                        ask=market.get('yes_ask')
                    )
        except Exception as e:
            logger.error(f"Error fetching market data for {ticker}: {e}")
            return None

class NewsAPICollector:
    """Collect news data from multiple news APIs"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv('NEWS_API_KEY')
        self.base_url = "https://newsapi.org/v2"
        
    async def get_stock_news(self, symbols: List[str], hours_back: int = 24) -> List[NewsData]:
        """Fetch news for given stock symbols"""
        news_items = []
        
        for symbol in symbols:
            try:
                url = f"{self.base_url}/everything"
                from_date = (datetime.now() - timedelta(hours=hours_back)).strftime('%Y-%m-%d')
                
                params = {
                    'q': f'{symbol} OR stock OR market',
                    'from': from_date,
                    'sortBy': 'publishedAt',
                    'apiKey': self.api_key,
                    'language': 'en',
                    'pageSize': 50
                }
                
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, params=params) as response:
                        if response.status == 200:
                            data = await response.json()
                            articles = data.get('articles', [])
                            
                            for article in articles:
                                # Calculate sentiment using TextBlob
                                text = f"{article.get('title', '')} {article.get('description', '')}"
                                sentiment = TextBlob(text).sentiment.polarity
                                
                                news_item = NewsData(
                                    title=article.get('title', ''),
                                    content=article.get('description', ''),
                                    source=article.get('source', {}).get('name', 'Unknown'),
                                    timestamp=datetime.fromisoformat(
                                        article.get('publishedAt', '').replace('Z', '+00:00')
                                    ),
                                    sentiment_score=sentiment,
                                    url=article.get('url', '')
                                )
                                news_items.append(news_item)
                                
            except Exception as e:
                logger.error(f"Error fetching news for {symbol}: {e}")
                
        return news_items

class YahooFinanceCollector:
    """Collect financial data from Yahoo Finance"""
    
    async def get_stock_data(self, symbols: List[str], period: str = "1d") -> List[MarketData]:
        """Fetch stock price data"""
        market_data = []
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period=period)
                
                if not hist.empty:
                    latest = hist.iloc[-1]
                    data = MarketData(
                        symbol=symbol,
                        price=float(latest['Close']),
                        volume=int(latest['Volume']),
                        timestamp=datetime.now()
                    )
                    market_data.append(data)
                    
            except Exception as e:
                logger.error(f"Error fetching Yahoo Finance data for {symbol}: {e}")
                
        return market_data
    
    async def get_technical_indicators(self, symbol: str, period: str = "3mo") -> Dict[str, float]:
        """Calculate technical indicators"""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period)
            
            if len(hist) < 50:
                return {}
                
            # Simple moving averages
            sma_20 = hist['Close'].rolling(window=20).mean().iloc[-1]
            sma_50 = hist['Close'].rolling(window=50).mean().iloc[-1]
            
            # RSI calculation
            delta = hist['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            sma = hist['Close'].rolling(window=20).mean()
            std = hist['Close'].rolling(window=20).std()
            upper_band = sma + (std * 2)
            lower_band = sma - (std * 2)
            
            return {
                'sma_20': sma_20,
                'sma_50': sma_50,
                'rsi': rsi.iloc[-1],
                'bb_upper': upper_band.iloc[-1],
                'bb_lower': lower_band.iloc[-1],
                'price': hist['Close'].iloc[-1],
                'volume': hist['Volume'].iloc[-1]
            }
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators for {symbol}: {e}")
            return {}

class RedditCollector:
    """Collect sentiment data from Reddit (using pushshift API)"""
    
    def __init__(self):
        self.base_url = "https://api.pushshift.io/reddit/search/submission"
        
    async def get_stock_sentiment(self, symbols: List[str], hours_back: int = 24) -> List[SocialData]:
        """Get Reddit sentiment for stock symbols"""
        social_data = []
        
        # Popular investing subreddits
        subreddits = ["investing", "stocks", "SecurityAnalysis", "ValueInvesting"]
        
        for symbol in symbols:
            for subreddit in subreddits:
                try:
                    after = int((datetime.now() - timedelta(hours=hours_back)).timestamp())
                    
                    params = {
                        'subreddit': subreddit,
                        'q': symbol,
                        'after': after,
                        'size': 25,
                        'sort_type': 'score'
                    }
                    
                    async with aiohttp.ClientSession() as session:
                        async with session.get(self.base_url, params=params) as response:
                            if response.status == 200:
                                data = await response.json()
                                posts = data.get('data', [])
                                
                                for post in posts:
                                    text = f"{post.get('title', '')} {post.get('selftext', '')}"
                                    sentiment = TextBlob(text).sentiment.polarity
                                    
                                    social_item = SocialData(
                                        platform='reddit',
                                        content=text[:500],  # Truncate for storage
                                        author=post.get('author', 'unknown'),
                                        timestamp=datetime.fromtimestamp(post.get('created_utc', 0)),
                                        sentiment_score=sentiment,
                                        engagement_score=post.get('score', 0),
                                        symbol_mentions=[symbol]
                                    )
                                    social_data.append(social_item)
                                    
                except Exception as e:
                    logger.error(f"Error fetching Reddit data for {symbol}: {e}")
                    
        return social_data

class DataPipeline:
    """Main data pipeline orchestrator"""
    
    def __init__(self, redis_host: str = "localhost", redis_port: int = 6379):
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        self.collectors = {
            'kalshi': KalshiCollector(),
            'news': NewsAPICollector(),
            'yahoo': YahooFinanceCollector(),
            'reddit': RedditCollector()
        }
        
    async def collect_all_data(self, symbols: List[str]) -> Dict[str, Any]:
        """Orchestrate data collection from all sources"""
        logger.info(f"Starting data collection for symbols: {symbols}")
        
        # Collect data concurrently
        tasks = []
        
        # Market data
        tasks.append(self.collectors['yahoo'].get_stock_data(symbols))
        
        # News data
        tasks.append(self.collectors['news'].get_stock_news(symbols))
        
        # Social data
        tasks.append(self.collectors['reddit'].get_stock_sentiment(symbols))
        
        # Kalshi data (if configured)
        if os.getenv('KALSHI_API_KEY'):
            async with self.collectors['kalshi'] as kalshi:
                markets = await kalshi.get_markets()
                relevant_markets = [m for m in markets if any(s.lower() in m.get('ticker', '').lower() for s in symbols)]
                tasks.append(asyncio.create_task(self._get_kalshi_data(relevant_markets)))
        
        # Wait for all data collection to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Organize results
        collected_data = {
            'market_data': results[0] if not isinstance(results[0], Exception) else [],
            'news_data': results[1] if not isinstance(results[1], Exception) else [],
            'social_data': results[2] if not isinstance(results[2], Exception) else [],
            'kalshi_data': results[3] if len(results) > 3 and not isinstance(results[3], Exception) else [],
            'timestamp': datetime.now(),
            'symbols': symbols
        }
        
        # Cache in Redis
        await self._cache_data(collected_data)
        
        logger.info(f"Data collection completed: {len(collected_data['market_data'])} market, {len(collected_data['news_data'])} news, {len(collected_data['social_data'])} social items")
        
        return collected_data
    
    async def _get_kalshi_data(self, markets: List[Dict]) -> List[MarketData]:
        """Helper to collect Kalshi data"""
        kalshi_data = []
        async with self.collectors['kalshi'] as kalshi:
            for market in markets[:10]:  # Limit to avoid rate limits
                data = await kalshi.get_market_data(market.get('ticker'))
                if data:
                    kalshi_data.append(data)
        return kalshi_data
    
    async def _cache_data(self, data: Dict[str, Any]):
        """Cache collected data in Redis"""
        try:
            # Store raw data
            cache_key = f"raw_data:{datetime.now().strftime('%Y-%m-%d-%H')}"
            self.redis_client.setex(cache_key, 3600, json.dumps(data, default=str))
            
            # Store latest data for each symbol
            for symbol in data['symbols']:
                symbol_key = f"latest:{symbol}"
                symbol_data = {
                    'market': [d for d in data['market_data'] if d.symbol == symbol],
                    'news': [d for d in data['news_data'] if symbol.lower() in d.title.lower() or symbol.lower() in d.content.lower()],
                    'social': [d for d in data['social_data'] if symbol in d.symbol_mentions],
                    'timestamp': str(datetime.now())
                }
                self.redis_client.setex(symbol_key, 1800, json.dumps(symbol_data, default=str))
                
        except Exception as e:
            logger.error(f"Error caching data: {e}")

# Example usage and testing
async def main():
    """Test the data pipeline"""
    pipeline = DataPipeline()
    
    # Test symbols
    test_symbols = ['AAPL', 'TSLA', 'MSFT', 'GOOGL']
    
    # Collect data
    data = await pipeline.collect_all_data(test_symbols)
    
    print(f"\n=== DATA COLLECTION RESULTS ===")
    print(f"Market data points: {len(data['market_data'])}")
    print(f"News articles: {len(data['news_data'])}")
    print(f"Social media posts: {len(data['social_data'])}")
    print(f"Kalshi markets: {len(data['kalshi_data'])}")
    
    # Sample data
    if data['market_data']:
        print(f"\nSample market data: {data['market_data'][0]}")
    if data['news_data']:
        print(f"\nSample news: {data['news_data'][0].title}")
    if data['social_data']:
        print(f"\nSample social: {data['social_data'][0].content[:100]}...")

if __name__ == "__main__":
    asyncio.run(main())