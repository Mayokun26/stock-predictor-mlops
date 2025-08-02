#!/usr/bin/env python3
import sqlite3

conn = sqlite3.connect('stocks.db')
stock_count = conn.execute("SELECT COUNT(*) FROM stock_info").fetchone()[0]
price_count = conn.execute("SELECT COUNT(*) FROM stock_prices").fetchone()[0]
conn.close()

print(f"Stocks: {stock_count}")
print(f"Prices: {price_count}")