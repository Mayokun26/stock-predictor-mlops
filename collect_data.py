#!/usr/bin/env python3
"""
Data collection script for stock market data
"""

from src.data.yahoo_finance import YahooFinanceClient
from src.database.models import Stock
from src.database.connection import get_db_session


def main():
    """Run data collection for all stocks"""
    print("Starting data collection...")
    
    # Get all stocks from database
    session = get_db_session()
    stocks = session.query(Stock).filter(Stock.is_active == True).all()
    symbols = [stock.symbol for stock in stocks]
    session.close()
    
    print(f"Collecting data for {len(symbols)} stocks: {', '.join(symbols)}")
    
    # Initialize Yahoo Finance client
    yf_client = YahooFinanceClient()
    
    # Collect all data
    results = yf_client.collect_all_data(symbols)
    
    # Summary
    print("\n=== Collection Summary ===")
    total_prices = sum(r['prices_added'] for r in results.values())
    info_updated = sum(1 for r in results.values() if r['info_updated'])
    fundamentals_stored = sum(1 for r in results.values() if r['fundamentals_stored'])
    
    print(f"Stocks processed: {len(results)}")
    print(f"Info updated: {info_updated}/{len(results)}")
    print(f"Price records added: {total_prices}")
    print(f"Fundamentals stored: {fundamentals_stored}/{len(results)}")
    
    # Show any failures
    failures = [symbol for symbol, result in results.items() 
                if not result['info_updated'] or result['prices_added'] == 0]
    
    if failures:
        print(f"\nFailed to collect data for: {', '.join(failures)}")
    else:
        print("\n✅ All data collection successful!")


if __name__ == "__main__":
    main()