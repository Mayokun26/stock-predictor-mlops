import os
from dotenv import load_dotenv

load_dotenv()

# API Configuration
KALSHI_KEY_ID = os.getenv('KALSHI_KEY_ID')
KALSHI_PRIVATE_KEY = os.getenv('KALSHI_PRIVATE_KEY')
NEWS_API_KEY = os.getenv('NEWS_API_KEY')

# API URLs
KALSHI_BASE_URL = "https://api.elections.kalshi.com"
NEWS_API_BASE_URL = "https://newsapi.org/v2"

# Validate required environment variables
required_vars = ['KALSHI_KEY_ID', 'KALSHI_PRIVATE_KEY', 'NEWS_API_KEY']
missing_vars = [var for var in required_vars if not os.getenv(var)]

if missing_vars:
    raise ValueError(f"Missing required environment variables: {missing_vars}")