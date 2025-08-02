#!/usr/bin/env python3
"""
Quick data collection script for immediate ML model development
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import sqlite3
import os

def create_simple_db():
    """Create a simple SQLite database for quick development"""
    conn = sqlite3.connect('stocks.db')
    
    # Create tables
    conn.execute('''
        CREATE TABLE IF NOT EXISTS stock_prices (
            symbol TEXT,
            date TEXT,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INTEGER,
            PRIMARY KEY (symbol, date)
        )
    ''')
    
    conn.execute('''
        CREATE TABLE IF NOT EXISTS stock_info (
            symbol TEXT PRIMARY KEY,
            name TEXT,
            sector TEXT,
            market_cap REAL,
            pe_ratio REAL,
            dividend_yield REAL
        )
    ''')
    
    conn.commit()
    return conn

def collect_stock_data(symbols, days=90):
    """Quickly collect essential stock data"""
    conn = create_simple_db()
    
    print(f"Collecting {days} days of data for {len(symbols)} stocks...")
    
    for i, symbol in enumerate(symbols, 1):
        print(f"[{i}/{len(symbols)}] {symbol}...", end="")
        
        try:
            ticker = yf.Ticker(symbol)
            
            # Get price data
            hist = ticker.history(period=f"{days}d")
            if len(hist) > 0:
                # Insert price data
                for date, row in hist.iterrows():
                    conn.execute('''
                        INSERT OR REPLACE INTO stock_prices 
                        (symbol, date, open, high, low, close, volume)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (symbol, date.strftime('%Y-%m-%d'), 
                          row['Open'], row['High'], row['Low'], 
                          row['Close'], row['Volume']))
                
                # Get basic info
                info = ticker.info
                conn.execute('''
                    INSERT OR REPLACE INTO stock_info 
                    (symbol, name, sector, market_cap, pe_ratio, dividend_yield)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (symbol, 
                      info.get('longName', symbol),
                      info.get('sector', 'Unknown'),
                      info.get('marketCap', 0),
                      info.get('trailingPE', 0),
                      info.get('dividendYield', 0)))
                
                print(f" OK ({len(hist)} days)")
            else:
                print(" No data")
                
        except Exception as e:
            print(f" Error: {str(e)[:30]}")
            continue
    
    conn.commit()
    conn.close()
    print(f"\n✅ Data saved to stocks.db")

def main():
    """Main execution"""
    # Top 10 liquid stocks for quick testing
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 
               'META', 'NVDA', 'JPM', 'V', 'JNJ']
    
    collect_stock_data(symbols, days=90)
    
    # Quick verification
    conn = sqlite3.connect('stocks.db')
    price_count = conn.execute('SELECT COUNT(*) FROM stock_prices').fetchone()[0]
    stock_count = conn.execute('SELECT COUNT(*) FROM stock_info').fetchone()[0]
    conn.close()
    
    print(f"\nDatabase created:")
    print(f"  {stock_count} stocks")
    print(f"  {price_count} price records")
    print(f"  Ready for ML model development!")

if __name__ == "__main__":
    main()