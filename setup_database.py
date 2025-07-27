#!/usr/bin/env python3
"""
Set up the database schema and initial data
"""

from src.database.connection import create_tables
from src.database.models import Stock, StockPrice, StockFundamentals
from src.database.connection import get_db_session


def setup_initial_stocks():
    """Add initial stock symbols to track"""
    
    # Top stocks by market cap (popular for demo)
    initial_stocks = [
        ('AAPL', 'Apple Inc.', 'Technology'),
        ('MSFT', 'Microsoft Corporation', 'Technology'),
        ('GOOGL', 'Alphabet Inc.', 'Technology'),
        ('AMZN', 'Amazon.com Inc.', 'Consumer Discretionary'),
        ('TSLA', 'Tesla Inc.', 'Consumer Discretionary'),
        ('META', 'Meta Platforms Inc.', 'Technology'),
        ('NVDA', 'NVIDIA Corporation', 'Technology'),
        ('NFLX', 'Netflix Inc.', 'Communication Services'),
        ('AMD', 'Advanced Micro Devices', 'Technology'),
        ('INTC', 'Intel Corporation', 'Technology'),
        
        # Add some value stocks for diversity
        ('BRK.B', 'Berkshire Hathaway Inc.', 'Financial Services'),
        ('JNJ', 'Johnson & Johnson', 'Healthcare'),
        ('PG', 'Procter & Gamble Co.', 'Consumer Staples'),
        ('KO', 'The Coca-Cola Company', 'Consumer Staples'),
        ('WMT', 'Walmart Inc.', 'Consumer Staples'),
        ('JPM', 'JPMorgan Chase & Co.', 'Financial Services'),
        ('BAC', 'Bank of America Corp.', 'Financial Services'),
        ('XOM', 'Exxon Mobil Corporation', 'Energy'),
        ('CVX', 'Chevron Corporation', 'Energy'),
        ('PFE', 'Pfizer Inc.', 'Healthcare'),
    ]
    
    session = get_db_session()
    
    try:
        for symbol, name, sector in initial_stocks:
            # Check if stock already exists
            existing = session.query(Stock).filter(Stock.symbol == symbol).first()
            if not existing:
                stock = Stock(
                    symbol=symbol,
                    name=name,
                    sector=sector,
                    exchange='NASDAQ' if symbol in ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX', 'AMD', 'INTC'] else 'NYSE'
                )
                session.add(stock)
        
        session.commit()
        print(f"Added {len(initial_stocks)} stocks to database")
        
    except Exception as e:
        session.rollback()
        print(f"Error adding stocks: {e}")
        
    finally:
        session.close()


def show_database_info():
    """Display information about the created database"""
    session = get_db_session()
    
    try:
        stock_count = session.query(Stock).count()
        print(f"\nDatabase setup complete!")
        print(f"Stocks in database: {stock_count}")
        
        if stock_count > 0:
            print("\nSample stocks:")
            stocks = session.query(Stock).limit(5).all()
            for stock in stocks:
                print(f"  {stock.symbol}: {stock.name} ({stock.sector})")
                
    finally:
        session.close()


if __name__ == "__main__":
    print("Setting up database...")
    
    # Create tables
    create_tables()
    
    # Add initial stock data
    setup_initial_stocks()
    
    # Show summary
    show_database_info()
    
    print("\nNext steps:")
    print("1. Run data collection to populate price and fundamental data")
    print("2. Set up news collection pipeline")
    print("3. Calculate technical indicators")