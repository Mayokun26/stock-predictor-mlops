import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from ..config import NEWS_API_KEY, NEWS_API_BASE_URL


class NewsClient:
    """Client for fetching news data"""
    
    def __init__(self):
        self.api_key = NEWS_API_KEY
        self.base_url = NEWS_API_BASE_URL
        self.session = requests.Session()
        
    def get_headlines(self, 
                     query: str = "election", 
                     days_back: int = 1,
                     page_size: int = 100) -> List[Dict]:
        """Fetch recent headlines related to betting markets"""
        
        from_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
        
        params = {
            'q': query,
            'from': from_date,
            'sortBy': 'publishedAt',
            'apiKey': self.api_key,
            'pageSize': page_size,
            'language': 'en'
        }
        
        endpoint = f"{self.base_url}/everything"
        response = self.session.get(endpoint, params=params)
        response.raise_for_status()
        
        return response.json().get("articles", [])
    
    def get_top_headlines(self, category: str = "general") -> List[Dict]:
        """Get top headlines for sentiment analysis"""
        params = {
            'category': category,
            'country': 'us',
            'apiKey': self.api_key,
            'pageSize': 50
        }
        
        endpoint = f"{self.base_url}/top-headlines"
        response = self.session.get(endpoint, params=params)
        response.raise_for_status()
        
        return response.json().get("articles", [])
    
    def test_connection(self) -> bool:
        """Test if News API connection is working"""
        try:
            headlines = self.get_top_headlines()
            return len(headlines) > 0
        except Exception as e:
            print(f"News API connection failed: {e}")
            return False