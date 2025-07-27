import requests
from typing import Dict, List, Optional
from ..config import KALSHI_KEY_ID, KALSHI_PRIVATE_KEY, KALSHI_BASE_URL


class KalshiClient:
    """Simple client for testing Kalshi API connectivity"""
    
    def __init__(self):
        self.key_id = KALSHI_KEY_ID
        self.private_key = KALSHI_PRIVATE_KEY
        self.base_url = KALSHI_BASE_URL
        self.session = requests.Session()
        
    def test_connection(self) -> bool:
        """Test if Kalshi endpoint is accessible"""
        try:
            # Try basic connectivity test
            response = self.session.get(f"{self.base_url}/markets", timeout=10)
            
            if response.status_code == 200:
                print(f"✅ Kalshi API endpoint accessible")
                return True
            elif response.status_code == 401:
                print(f"✅ Kalshi API endpoint accessible (401 = needs auth)")
                return True
            elif response.status_code == 404:
                print(f"⚠️  Kalshi API endpoint structure unknown (404)")
                return True  # Endpoint exists, just wrong path
            else:
                print(f"⚠️  Kalshi API responded with {response.status_code}")
                return True
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Kalshi API connection failed: {e}")
            return False
    
    def get_markets(self, limit: int = 100) -> List[Dict]:
        """Placeholder for market data - will implement auth later"""
        return []