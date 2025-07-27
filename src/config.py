import os
from dotenv import load_dotenv

load_dotenv()

# API Configuration
ALPACA_API_KEY = os.getenv('ALPACA_API_KEY')
ALPACA_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY')
ALPACA_BASE_URL = os.getenv('ALPACA_BASE_URL')
NEWS_API_KEY = os.getenv('NEWS_API_KEY')

# API URLs
NEWS_API_BASE_URL = "https://newsapi.org/v2"

# Validate required environment variables
required_vars = ['ALPACA_API_KEY', 'ALPACA_SECRET_KEY', 'ALPACA_BASE_URL', 'NEWS_API_KEY']
missing_vars = [var for var in required_vars if not os.getenv(var)]

if missing_vars:
    raise ValueError(f"Missing required environment variables: {missing_vars}")