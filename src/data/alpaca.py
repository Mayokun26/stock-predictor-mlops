import alpaca_trade_api as tradeapi
from typing import Dict, List, Optional
from ..config import ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL


class AlpacaClient:
    """Client for Alpaca paper trading API"""
    
    def __init__(self):
        self.api = tradeapi.REST(
            ALPACA_API_KEY,
            ALPACA_SECRET_KEY,
            ALPACA_BASE_URL,
            api_version='v2'
        )
        
    def test_connection(self) -> bool:
        """Test if Alpaca API connection works"""
        try:
            account = self.api.get_account()
            print(f"✅ Alpaca API connected - Account: {account.status}")
            print(f"  Portfolio value: ${float(account.portfolio_value):,.2f}")
            print(f"  Buying power: ${float(account.buying_power):,.2f}")
            return True
        except Exception as e:
            print(f"❌ Alpaca API connection failed: {e}")
            return False
    
    def get_positions(self) -> List[Dict]:
        """Get current portfolio positions"""
        try:
            positions = self.api.list_positions()
            return [
                {
                    'symbol': pos.symbol,
                    'qty': float(pos.qty),
                    'market_value': float(pos.market_value),
                    'unrealized_pl': float(pos.unrealized_pl)
                }
                for pos in positions
            ]
        except Exception as e:
            print(f"Error fetching positions: {e}")
            return []
    
    def get_top_stocks(self, limit: int = 10) -> List[str]:
        """Get list of popular stocks for testing"""
        # Popular stocks for demo purposes
        return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX', 'AMD', 'INTC'][:limit]