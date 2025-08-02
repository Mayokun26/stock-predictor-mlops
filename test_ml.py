#!/usr/bin/env python3
import sqlite3
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

print("Loading data from SQLite...")
conn = sqlite3.connect('stocks.db')
df = pd.read_sql('SELECT * FROM stock_prices WHERE symbol="AAPL" ORDER BY date', conn)
conn.close()

if len(df) == 0:
    print("❌ No AAPL data found. Run quick_collect.py first")
    exit(1)

print(f"Loaded {len(df)} AAPL price records")

# Create features
df['target'] = df['close'].shift(-1)  # Next day's close price
df['returns'] = df['close'].pct_change()
df['sma_5'] = df['close'].rolling(5).mean()
df['volatility'] = df['close'].rolling(10).std()
df = df.dropna()

print(f"Created features for {len(df)} trading days")

# Prepare data
X = df[['open', 'high', 'low', 'volume', 'returns', 'sma_5', 'volatility']]
y = df['target']

split = int(len(df) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples")

# Train model
model = RandomForestRegressor(n_estimators=50, random_state=42)
model.fit(X_train, y_train)

# Evaluate
pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, pred))
mae = np.mean(np.abs(y_test - pred))

print(f"AAPL Price Prediction Results:")
print(f"  RMSE: ${rmse:.2f}")
print(f"  MAE: ${mae:.2f}")
print(f"  Mean price: ${y_test.mean():.2f}")
print("✅ Basic ML model working!")