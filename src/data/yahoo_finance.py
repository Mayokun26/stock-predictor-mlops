"""
Yahoo Finance data collection client
"""

import yfinance as yf
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from ..database.models import Stock, StockPrice, StockFundamentals
from ..database.connection import get_db_session


class YahooFinanceClient:
    """Client for collecting stock data from Yahoo Finance"""
    
    def __init__(self):
        self.session = get_db_session()
    
    def get_stock_info(self, symbol: str) -> Dict:
        """Get basic stock information"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            return info
        except Exception as e:
            print(f"Error getting info for {symbol}: {e}")
            return {}
    
    def get_price_history(self, symbol: str, period: str = "1y") -> pd.DataFrame:
        """Get historical price data"""
        try:
            ticker = yf.Ticker(symbol)
            history = ticker.history(period=period)
            return history
        except Exception as e:
            print(f"Error getting price history for {symbol}: {e}")
            return pd.DataFrame()
    
    def update_stock_info(self, symbol: str) -> bool:
        """Update stock master information"""
        try:
            info = self.get_stock_info(symbol)
            if not info:
                return False
            
            # Update stock record
            stock = self.session.query(Stock).filter(Stock.symbol == symbol).first()
            if stock:
                stock.name = info.get('longName', stock.name)
                stock.sector = info.get('sector', stock.sector)
                stock.industry = info.get('industry', stock.industry)
                stock.market_cap = info.get('marketCap')
                stock.exchange = info.get('exchange', stock.exchange)
                
                self.session.commit()
                return True
            
        except Exception as e:
            print(f"Error updating stock info for {symbol}: {e}")
            self.session.rollback()
            
        return False
    
    def store_price_data(self, symbol: str, period: str = "1y") -> int:
        """Store historical price data in database"""
        try:
            history = self.get_price_history(symbol, period)
            if history.empty:
                return 0
            
            count = 0
            for date, row in history.iterrows():
                # Check if record already exists
                existing = self.session.query(StockPrice).filter(
                    StockPrice.symbol == symbol,
                    StockPrice.date == date.date()
                ).first()
                
                if not existing:
                    price_record = StockPrice(
                        symbol=symbol,
                        date=date.date(),
                        open_price=row['Open'],
                        high_price=row['High'],
                        low_price=row['Low'],
                        close_price=row['Close'],
                        adj_close_price=row['Close'],  # YFinance already adjusts
                        volume=int(row['Volume']) if pd.notna(row['Volume']) else None
                    )
                    self.session.add(price_record)
                    count += 1
            
            self.session.commit()
            return count
            
        except Exception as e:
            print(f"Error storing price data for {symbol}: {e}")
            self.session.rollback()
            return 0
    
    def store_fundamentals(self, symbol: str) -> bool:
        """Store fundamental analysis data"""
        try:
            info = self.get_stock_info(symbol)
            if not info:
                return False
            
            # Check if we already have recent fundamentals
            existing = self.session.query(StockFundamentals).filter(
                StockFundamentals.symbol == symbol,
                StockFundamentals.date >= datetime.now().date() - timedelta(days=7)
            ).first()
            
            if existing:
                return True  # Already have recent data
            
            # Extract fundamental metrics
            fundamentals = StockFundamentals(
                symbol=symbol,
                date=datetime.now().date(),
                
                # Valuation metrics
                pe_ratio=self._safe_float(info.get('trailingPE')),
                pb_ratio=self._safe_float(info.get('priceToBook')),
                ps_ratio=self._safe_float(info.get('priceToSalesTrailing12Months')),
                peg_ratio=self._safe_float(info.get('pegRatio')),
                ev_ebitda=self._safe_float(info.get('enterpriseToEbitda')),
                
                # Profitability metrics
                roe=self._safe_float(info.get('returnOnEquity')),
                roa=self._safe_float(info.get('returnOnAssets')),
                gross_margin=self._safe_float(info.get('grossMargins')),
                operating_margin=self._safe_float(info.get('operatingMargins')),
                net_margin=self._safe_float(info.get('profitMargins')),
                
                # Financial strength
                debt_to_equity=self._safe_float(info.get('debtToEquity')),
                current_ratio=self._safe_float(info.get('currentRatio')),
                quick_ratio=self._safe_float(info.get('quickRatio')),
                
                # Growth metrics
                revenue_growth=self._safe_float(info.get('revenueGrowth')),
                earnings_growth=self._safe_float(info.get('earningsGrowth')),
                
                # Size metrics
                market_cap=self._safe_float(info.get('marketCap')),
                enterprise_value=self._safe_float(info.get('enterpriseValue')),
                shares_outstanding=self._safe_float(info.get('sharesOutstanding'))
            )
            
            self.session.add(fundamentals)
            self.session.commit()
            return True
            
        except Exception as e:
            print(f"Error storing fundamentals for {symbol}: {e}")
            self.session.rollback()
            return False
    
    def _safe_float(self, value) -> Optional[float]:
        """Safely convert value to float"""
        if value is None or pd.isna(value):
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None
    
    def collect_all_data(self, symbols: List[str] = None) -> Dict[str, Dict]:
        """Collect all data for specified symbols"""
        if symbols is None:
            # Get all active stocks from database
            stocks = self.session.query(Stock).filter(Stock.is_active == True).all()
            symbols = [stock.symbol for stock in stocks]
        
        results = {}
        
        for symbol in symbols:
            print(f"Collecting data for {symbol}...")
            
            result = {
                'info_updated': False,
                'prices_added': 0,
                'fundamentals_stored': False
            }
            
            # Update stock info
            result['info_updated'] = self.update_stock_info(symbol)
            
            # Store price data
            result['prices_added'] = self.store_price_data(symbol)
            
            # Store fundamentals
            result['fundamentals_stored'] = self.store_fundamentals(symbol)
            
            results[symbol] = result
            print(f"  Info: {result['info_updated']}, Prices: {result['prices_added']}, Fundamentals: {result['fundamentals_stored']}")
        
        return results
    
    def __del__(self):
        """Clean up database session"""
        if hasattr(self, 'session'):
            self.session.close()