#!/usr/bin/env python3
"""
Test script to verify API connections are working
"""

from src.data.kalshi import KalshiClient
from src.data.news import NewsClient


def test_kalshi():
    """Test Kalshi API connection"""
    print("Testing Kalshi API...")
    client = KalshiClient()
    
    if client.test_connection():
        print("✅ Kalshi API connection successful")
        
        # Get a few markets to verify
        markets = client.get_markets(limit=3)
        print(f"Found {len(markets)} markets")
        for market in markets[:2]:
            print(f"  - {market.get('title', 'Unknown')}")
    else:
        print("❌ Kalshi API connection failed")


def test_news():
    """Test News API connection"""
    print("\nTesting News API...")
    client = NewsClient()
    
    if client.test_connection():
        print("✅ News API connection successful")
        
        # Get some headlines to verify
        headlines = client.get_headlines(query="politics", days_back=1)
        print(f"Found {len(headlines)} recent headlines")
        for article in headlines[:2]:
            print(f"  - {article.get('title', 'Unknown')}")
    else:
        print("❌ News API connection failed")


if __name__ == "__main__":
    test_kalshi()
    test_news()
    print("\nAPI testing complete!")