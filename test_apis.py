#!/usr/bin/env python3
"""
Test script to verify API connections are working
"""

from src.data.alpaca import AlpacaClient
from src.data.news import NewsClient


def test_alpaca():
    """Test Alpaca API connection"""
    print("Testing Alpaca API...")
    client = AlpacaClient()
    
    if client.test_connection():
        # Get some positions and popular stocks
        positions = client.get_positions()
        stocks = client.get_top_stocks(5)
        
        print(f"Current positions: {len(positions)}")
        for pos in positions:
            print(f"  - {pos['symbol']}: {pos['qty']} shares, ${pos['market_value']:.2f}")
            
        print(f"Popular stocks for testing: {', '.join(stocks)}")
    else:
        print("❌ Alpaca API connection failed")


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
    test_alpaca()
    test_news()
    print("\nAPI testing complete!")