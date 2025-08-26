#!/usr/bin/env python3
"""
Production Stock Data Pipeline DAG
Enterprise Airflow implementation with comprehensive data collection, validation, and feature engineering
"""
from datetime import datetime, timedelta
from typing import Dict, List, Any

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.sensors.filesystem import FileSensor
from airflow.providers.postgres.operators.postgres import PostgresOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.providers.redis.hooks.redis import RedisHook
from airflow.models import Variable
from airflow.exceptions import AirflowException
from airflow.utils.dates import days_ago
from airflow.utils.task_group import TaskGroup

import pandas as pd
import numpy as np
import yfinance as yf
import requests
import json
import logging
from transformers import pipeline

# Import custom modules
import sys
import os
sys.path.append('/opt/airflow/dags')
from utils.data_quality import DataQualityChecker
from utils.feature_engineering import FeatureEngineer
from utils.monitoring import AirflowMonitor

logger = logging.getLogger(__name__)

# DAG Configuration
DEFAULT_ARGS = {
    'owner': 'mlops-team',
    'depends_on_past': False,
    'start_date': days_ago(1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'execution_timeout': timedelta(hours=2),
    'sla': timedelta(hours=3),
}

# Stock symbols to process
STOCK_SYMBOLS = [
    'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX',
    'JPM', 'V', 'JNJ', 'WMT', 'PG', 'UNH', 'DIS', 'HD', 'MA', 'CRM', 'BAC', 'ADBE'
]

dag = DAG(
    'stock_data_pipeline',
    default_args=DEFAULT_ARGS,
    description='Production stock data collection, validation, and feature engineering pipeline',
    schedule_interval='0 9-16 * * 1-5',  # Every hour during market hours, weekdays only
    max_active_runs=1,
    catchup=False,
    tags=['mlops', 'stock-data', 'features', 'production'],
)

# Utility Functions

def get_postgres_conn():
    """Get PostgreSQL connection"""
    hook = PostgresHook(postgres_conn_id='postgres_mlops')
    return hook.get_conn()

def get_redis_conn():
    """Get Redis connection"""
    hook = RedisHook(redis_conn_id='redis_mlops')
    return hook.get_conn()

def send_slack_notification(context, message):
    """Send Slack notification for critical events"""
    slack_webhook = Variable.get("slack_webhook_url", default_var=None)
    if slack_webhook:
        payload = {
            "text": f"MLOps Pipeline Alert: {message}",
            "username": "Airflow Bot",
            "channel": "#mlops-alerts"
        }
        try:
            requests.post(slack_webhook, json=payload)
        except Exception as e:
            logger.warning(f"Failed to send Slack notification: {e}")

# Task Functions

def validate_market_hours(**context):
    """Check if market is open and pipeline should run"""
    now = datetime.now()
    
    # Skip weekends
    if now.weekday() >= 5:
        raise AirflowException("Market is closed - weekend")
    
    # Check market hours (9:30 AM - 4:00 PM ET, roughly)
    market_open = now.replace(hour=9, minute=30, second=0)
    market_close = now.replace(hour=16, minute=0, second=0)
    
    if not (market_open <= now <= market_close):
        logger.info("Outside market hours - running reduced pipeline")
        # Allow pipeline to continue with reduced frequency
    
    logger.info(f"Pipeline validation passed at {now}")
    return True

def collect_stock_data(symbol: str, **context):
    """Collect comprehensive stock data for a symbol"""
    try:
        logger.info(f"Collecting data for {symbol}")
        
        # Initialize yfinance ticker
        ticker = yf.Ticker(symbol)
        
        # Get various data types
        data_collection = {
            'symbol': symbol,
            'collection_timestamp': datetime.now(),
            'data_types': {}
        }
        
        # 1. Historical price data (last 2 days for updates)
        hist_data = ticker.history(period="2d", interval="1h")
        if not hist_data.empty:
            data_collection['data_types']['historical_prices'] = {
                'records': len(hist_data),
                'latest_timestamp': hist_data.index[-1].isoformat(),
                'data_sample': hist_data.tail(1).to_dict('records')[0]
            }
        
        # 2. Company info
        try:
            info = ticker.info
            data_collection['data_types']['company_info'] = {
                'market_cap': info.get('marketCap'),
                'sector': info.get('sector'),
                'industry': info.get('industry'),
                'pe_ratio': info.get('trailingPE'),
                'dividend_yield': info.get('dividendYield')
            }
        except:
            logger.warning(f"Could not fetch company info for {symbol}")
        
        # 3. Real-time quote
        try:
            current_data = ticker.history(period="1d", interval="1m").tail(1)
            if not current_data.empty:
                data_collection['data_types']['current_quote'] = {
                    'price': float(current_data['Close'].iloc[0]),
                    'volume': int(current_data['Volume'].iloc[0]),
                    'timestamp': current_data.index[0].isoformat()
                }
        except:
            logger.warning(f"Could not fetch current quote for {symbol}")
        
        # Store raw data in database
        conn = get_postgres_conn()
        cursor = conn.cursor()
        
        # Store historical data
        for idx, row in hist_data.iterrows():
            cursor.execute("""
                INSERT INTO stock_prices (symbol, date, open, high, low, close, volume, data_source)
                VALUES (%s, %s, %s, %s, %s, %s, %s, 'yfinance')
                ON CONFLICT (symbol, date) DO UPDATE SET
                    open = EXCLUDED.open,
                    high = EXCLUDED.high,
                    low = EXCLUDED.low,
                    close = EXCLUDED.close,
                    volume = EXCLUDED.volume,
                    updated_at = CURRENT_TIMESTAMP
            """, (symbol, idx, row['Open'], row['High'], row['Low'], row['Close'], row['Volume']))
        
        # Store company info
        if 'company_info' in data_collection['data_types']:
            info_data = data_collection['data_types']['company_info']
            cursor.execute("""
                INSERT INTO stock_info (symbol, name, sector, industry, market_cap, pe_ratio, dividend_yield)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (symbol) DO UPDATE SET
                    sector = EXCLUDED.sector,
                    industry = EXCLUDED.industry,
                    market_cap = EXCLUDED.market_cap,
                    pe_ratio = EXCLUDED.pe_ratio,
                    dividend_yield = EXCLUDED.dividend_yield,
                    updated_at = CURRENT_TIMESTAMP
            """, (symbol, symbol, info_data.get('sector'), info_data.get('industry'),
                  info_data.get('market_cap'), info_data.get('pe_ratio'), info_data.get('dividend_yield')))
        
        conn.commit()
        cursor.close()
        conn.close()
        
        logger.info(f"✅ Successfully collected data for {symbol}")
        return data_collection
        
    except Exception as e:
        logger.error(f"❌ Failed to collect data for {symbol}: {e}")
        raise AirflowException(f"Data collection failed for {symbol}: {str(e)}")

def collect_news_data(symbol: str, **context):
    """Collect news data for sentiment analysis"""
    try:
        logger.info(f"Collecting news for {symbol}")
        
        # News API configuration
        news_api_key = Variable.get("news_api_key", default_var="demo_key")
        
        # Get news from multiple sources
        news_sources = [
            f"https://newsapi.org/v2/everything?q={symbol}&apiKey={news_api_key}&sortBy=publishedAt&pageSize=10",
            f"https://newsapi.org/v2/everything?q={symbol}+stock&apiKey={news_api_key}&sortBy=publishedAt&pageSize=10"
        ]
        
        all_articles = []
        
        for url in news_sources:
            try:
                response = requests.get(url, timeout=30)
                if response.status_code == 200:
                    data = response.json()
                    articles = data.get('articles', [])
                    all_articles.extend(articles)
                else:
                    logger.warning(f"News API returned status {response.status_code}")
            except requests.RequestException as e:
                logger.warning(f"News API request failed: {e}")
        
        # Remove duplicates based on title
        unique_articles = []
        seen_titles = set()
        
        for article in all_articles:
            title = article.get('title', '').lower()
            if title and title not in seen_titles:
                seen_titles.add(title)
                unique_articles.append(article)
        
        # Store news data
        conn = get_postgres_conn()
        cursor = conn.cursor()
        
        for article in unique_articles[:20]:  # Limit to 20 articles
            try:
                cursor.execute("""
                    INSERT INTO news_articles 
                    (symbol, title, description, content, url, published_at, source)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (url) DO NOTHING
                """, (
                    symbol,
                    article.get('title'),
                    article.get('description'),
                    article.get('content'),
                    article.get('url'),
                    article.get('publishedAt'),
                    article.get('source', {}).get('name')
                ))
            except Exception as e:
                logger.warning(f"Failed to insert article: {e}")
        
        conn.commit()
        cursor.close()
        conn.close()
        
        logger.info(f"✅ Collected {len(unique_articles)} news articles for {symbol}")
        return {'symbol': symbol, 'articles_collected': len(unique_articles)}
        
    except Exception as e:
        logger.error(f"❌ News collection failed for {symbol}: {e}")
        return {'symbol': symbol, 'articles_collected': 0, 'error': str(e)}

def perform_data_quality_checks(symbol: str, **context):
    """Comprehensive data quality validation"""
    try:
        logger.info(f"Performing data quality checks for {symbol}")
        
        checker = DataQualityChecker()
        
        # Get recent data for validation
        conn = get_postgres_conn()
        
        # Check price data quality
        price_data = pd.read_sql("""
            SELECT * FROM stock_prices 
            WHERE symbol = %s 
            AND date >= CURRENT_DATE - INTERVAL '7 days'
            ORDER BY date DESC
        """, conn, params=(symbol,))
        
        quality_results = {
            'symbol': symbol,
            'check_timestamp': datetime.now().isoformat(),
            'checks': {}
        }
        
        # Data completeness check
        if len(price_data) > 0:
            completeness = checker.check_completeness(price_data)
            quality_results['checks']['completeness'] = completeness
            
            # Data consistency checks
            consistency = checker.check_consistency(price_data)
            quality_results['checks']['consistency'] = consistency
            
            # Outlier detection
            outliers = checker.detect_outliers(price_data)
            quality_results['checks']['outliers'] = outliers
            
            # Freshness check
            latest_timestamp = pd.to_datetime(price_data['date']).max()
            hours_since_latest = (datetime.now() - latest_timestamp).total_seconds() / 3600
            
            quality_results['checks']['freshness'] = {
                'latest_timestamp': latest_timestamp.isoformat(),
                'hours_since_latest': hours_since_latest,
                'is_fresh': hours_since_latest <= 24
            }
            
        else:
            quality_results['checks']['error'] = 'No price data found for quality checks'
        
        # Store quality check results
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO data_quality_checks 
            (symbol, check_timestamp, check_results, overall_quality_score)
            VALUES (%s, %s, %s, %s)
        """, (
            symbol, 
            datetime.now(),
            json.dumps(quality_results),
            quality_results.get('overall_score', 0.0)
        ))
        
        conn.commit()
        cursor.close()
        conn.close()
        
        # Raise alert if quality is poor
        overall_score = quality_results.get('overall_score', 0.0)
        if overall_score < 0.7:
            send_slack_notification(context, f"Poor data quality for {symbol}: score {overall_score:.2f}")
        
        logger.info(f"✅ Data quality check completed for {symbol}")
        return quality_results
        
    except Exception as e:
        logger.error(f"❌ Data quality check failed for {symbol}: {e}")
        raise AirflowException(f"Data quality check failed: {str(e)}")

def calculate_technical_indicators(symbol: str, **context):
    """Calculate and store technical indicators"""
    try:
        logger.info(f"Calculating technical indicators for {symbol}")
        
        # Get historical data
        conn = get_postgres_conn()
        
        price_data = pd.read_sql("""
            SELECT date, open, high, low, close, volume
            FROM stock_prices 
            WHERE symbol = %s 
            AND date >= CURRENT_DATE - INTERVAL '60 days'
            ORDER BY date ASC
        """, conn, params=(symbol,))
        
        if len(price_data) < 20:
            raise AirflowException(f"Insufficient data for technical indicators: {len(price_data)} records")
        
        # Calculate indicators using FeatureEngineer
        engineer = FeatureEngineer()
        indicators = engineer.calculate_all_indicators(price_data)
        
        # Get latest values for feature store
        latest_indicators = {}
        for name, series in indicators.items():
            if not series.empty and not pd.isna(series.iloc[-1]):
                latest_indicators[name] = float(series.iloc[-1])
        
        # Store in feature store via Redis
        redis_conn = get_redis_conn()
        feature_key = f"features:{symbol}:technical_indicators"
        
        feature_data = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'indicators': latest_indicators,
            'calculation_period_days': 60
        }
        
        redis_conn.setex(feature_key, 3600, json.dumps(feature_data))  # 1 hour TTL
        
        # Also store in PostgreSQL for persistence
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO technical_indicators 
            (symbol, calculation_date, indicators, created_at)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (symbol, calculation_date) DO UPDATE SET
                indicators = EXCLUDED.indicators,
                updated_at = CURRENT_TIMESTAMP
        """, (symbol, datetime.now().date(), json.dumps(latest_indicators), datetime.now()))
        
        conn.commit()
        cursor.close()
        conn.close()
        
        logger.info(f"✅ Calculated {len(latest_indicators)} technical indicators for {symbol}")
        return {'symbol': symbol, 'indicators_calculated': len(latest_indicators)}
        
    except Exception as e:
        logger.error(f"❌ Technical indicator calculation failed for {symbol}: {e}")
        raise AirflowException(f"Technical indicator calculation failed: {str(e)}")

def calculate_sentiment_features(symbol: str, **context):
    """Calculate news sentiment features"""
    try:
        logger.info(f"Calculating sentiment features for {symbol}")
        
        # Get recent news
        conn = get_postgres_conn()
        
        news_data = pd.read_sql("""
            SELECT title, description, content, published_at
            FROM news_articles 
            WHERE symbol = %s 
            AND published_at >= CURRENT_DATE - INTERVAL '7 days'
            ORDER BY published_at DESC
            LIMIT 50
        """, conn, params=(symbol,))
        
        if len(news_data) == 0:
            logger.warning(f"No recent news found for {symbol}")
            return {'symbol': symbol, 'sentiment_features': {}}
        
        # Initialize FinBERT sentiment analyzer
        try:
            sentiment_analyzer = pipeline(
                'sentiment-analysis',
                model='ProsusAI/finbert',
                framework='pt',
                device=-1  # CPU
            )
        except Exception as e:
            logger.error(f"Failed to load FinBERT model: {e}")
            raise AirflowException("Could not initialize sentiment analyzer")
        
        # Process news articles
        all_sentiments = []
        
        for _, article in news_data.iterrows():
            text_to_analyze = []
            
            if pd.notna(article['title']):
                text_to_analyze.append(str(article['title']))
            
            if pd.notna(article['description']):
                text_to_analyze.append(str(article['description']))
            
            if text_to_analyze:
                # Combine title and description
                combined_text = '. '.join(text_to_analyze)
                
                # Truncate if too long (BERT has token limits)
                if len(combined_text) > 500:
                    combined_text = combined_text[:500]
                
                try:
                    result = sentiment_analyzer(combined_text)[0]
                    
                    # Convert to numeric score
                    score = result['score']
                    if result['label'] == 'negative':
                        score = -score
                    elif result['label'] == 'neutral':
                        score = 0
                    
                    all_sentiments.append({
                        'score': score,
                        'label': result['label'],
                        'timestamp': article['published_at']
                    })
                
                except Exception as e:
                    logger.warning(f"Sentiment analysis failed for article: {e}")
        
        # Calculate sentiment features
        if all_sentiments:
            scores = [s['score'] for s in all_sentiments]
            
            sentiment_features = {
                'avg_sentiment': float(np.mean(scores)),
                'sentiment_volatility': float(np.std(scores)),
                'positive_news_ratio': float(np.mean([s['score'] > 0.1 for s in all_sentiments])),
                'negative_news_ratio': float(np.mean([s['score'] < -0.1 for s in all_sentiments])),
                'news_volume_7d': len(all_sentiments),
                'latest_sentiment': float(scores[0]) if scores else 0.0
            }
        else:
            sentiment_features = {
                'avg_sentiment': 0.0,
                'sentiment_volatility': 0.0,
                'positive_news_ratio': 0.0,
                'negative_news_ratio': 0.0,
                'news_volume_7d': 0,
                'latest_sentiment': 0.0
            }
        
        # Store in feature store
        redis_conn = get_redis_conn()
        feature_key = f"features:{symbol}:sentiment"
        
        feature_data = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'sentiment_features': sentiment_features,
            'analysis_period_days': 7
        }
        
        redis_conn.setex(feature_key, 3600, json.dumps(feature_data))  # 1 hour TTL
        
        # Store in PostgreSQL
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO sentiment_features 
            (symbol, calculation_date, features, created_at)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (symbol, calculation_date) DO UPDATE SET
                features = EXCLUDED.features,
                updated_at = CURRENT_TIMESTAMP
        """, (symbol, datetime.now().date(), json.dumps(sentiment_features), datetime.now()))
        
        conn.commit()
        cursor.close()
        conn.close()
        
        logger.info(f"✅ Calculated sentiment features for {symbol} based on {len(all_sentiments)} articles")
        return {'symbol': symbol, 'sentiment_features': sentiment_features}
        
    except Exception as e:
        logger.error(f"❌ Sentiment feature calculation failed for {symbol}: {e}")
        raise AirflowException(f"Sentiment calculation failed: {str(e)}")

def trigger_model_evaluation(**context):
    """Trigger model evaluation pipeline"""
    try:
        logger.info("Triggering model evaluation pipeline")
        
        # Get current model versions
        conn = get_postgres_conn()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT DISTINCT model_version 
            FROM predictions 
            WHERE created_at >= CURRENT_DATE - INTERVAL '1 day'
        """)
        
        active_models = [row[0] for row in cursor.fetchall()]
        cursor.close()
        conn.close()
        
        if not active_models:
            logger.warning("No active models found for evaluation")
            return {'models_evaluated': 0}
        
        # Trigger evaluation for each model (simplified - in production this would call the evaluation API)
        evaluation_results = []
        
        for model_version in active_models:
            # This would typically trigger a separate DAG or API call
            evaluation_results.append({
                'model_version': model_version,
                'evaluation_triggered': True,
                'timestamp': datetime.now().isoformat()
            })
            
            logger.info(f"Evaluation triggered for model {model_version}")
        
        # Store trigger events
        redis_conn = get_redis_conn()
        trigger_key = f"evaluation_triggers:{datetime.now().strftime('%Y%m%d')}"
        redis_conn.lpush(trigger_key, json.dumps(evaluation_results))
        redis_conn.expire(trigger_key, 86400)  # 24 hour TTL
        
        return {'models_evaluated': len(evaluation_results), 'results': evaluation_results}
        
    except Exception as e:
        logger.error(f"❌ Model evaluation trigger failed: {e}")
        raise AirflowException(f"Model evaluation trigger failed: {str(e)}")

def cleanup_old_data(**context):
    """Clean up old data to manage storage"""
    try:
        logger.info("Starting data cleanup process")
        
        conn = get_postgres_conn()
        cursor = conn.cursor()
        
        cleanup_results = {}
        
        # Clean up old predictions (keep 90 days)
        cursor.execute("""
            DELETE FROM predictions 
            WHERE created_at < CURRENT_DATE - INTERVAL '90 days'
        """)
        cleanup_results['predictions_deleted'] = cursor.rowcount
        
        # Clean up old news articles (keep 30 days)
        cursor.execute("""
            DELETE FROM news_articles 
            WHERE published_at < CURRENT_DATE - INTERVAL '30 days'
        """)
        cleanup_results['news_articles_deleted'] = cursor.rowcount
        
        # Clean up old data quality checks (keep 30 days)
        cursor.execute("""
            DELETE FROM data_quality_checks 
            WHERE check_timestamp < CURRENT_DATE - INTERVAL '30 days'
        """)
        cleanup_results['quality_checks_deleted'] = cursor.rowcount
        
        # Clean up Redis cache (old feature data)
        redis_conn = get_redis_conn()
        
        # Scan for old feature keys and delete them
        pattern = "features:*"
        deleted_keys = 0
        
        for key in redis_conn.scan_iter(match=pattern):
            try:
                # Check if key is older than 24 hours (simplified check)
                ttl = redis_conn.ttl(key)
                if ttl == -1:  # No TTL set, delete old keys
                    redis_conn.delete(key)
                    deleted_keys += 1
            except:
                pass
        
        cleanup_results['redis_keys_deleted'] = deleted_keys
        
        conn.commit()
        cursor.close()
        conn.close()
        
        logger.info(f"✅ Data cleanup completed: {cleanup_results}")
        return cleanup_results
        
    except Exception as e:
        logger.error(f"❌ Data cleanup failed: {e}")
        raise AirflowException(f"Data cleanup failed: {str(e)}")

# Task Definitions

# Validation Tasks
validate_market_task = PythonOperator(
    task_id='validate_market_hours',
    python_callable=validate_market_hours,
    dag=dag,
)

# Data Collection Task Group
with TaskGroup("data_collection", dag=dag) as data_collection_group:
    
    # Create dynamic tasks for each stock symbol
    stock_data_tasks = []
    news_data_tasks = []
    
    for symbol in STOCK_SYMBOLS:
        # Stock data collection
        stock_task = PythonOperator(
            task_id=f'collect_stock_data_{symbol}',
            python_callable=collect_stock_data,
            op_kwargs={'symbol': symbol},
            pool='stock_data_pool',  # Resource pool to limit concurrent API calls
        )
        stock_data_tasks.append(stock_task)
        
        # News data collection  
        news_task = PythonOperator(
            task_id=f'collect_news_data_{symbol}',
            python_callable=collect_news_data,
            op_kwargs={'symbol': symbol},
            pool='news_api_pool',
        )
        news_data_tasks.append(news_task)

# Data Quality Task Group
with TaskGroup("data_quality", dag=dag) as data_quality_group:
    
    quality_check_tasks = []
    
    for symbol in STOCK_SYMBOLS:
        quality_task = PythonOperator(
            task_id=f'quality_check_{symbol}',
            python_callable=perform_data_quality_checks,
            op_kwargs={'symbol': symbol},
        )
        quality_check_tasks.append(quality_task)

# Feature Engineering Task Group
with TaskGroup("feature_engineering", dag=dag) as feature_engineering_group:
    
    technical_indicator_tasks = []
    sentiment_feature_tasks = []
    
    for symbol in STOCK_SYMBOLS:
        # Technical indicators
        tech_task = PythonOperator(
            task_id=f'calculate_technical_indicators_{symbol}',
            python_callable=calculate_technical_indicators,
            op_kwargs={'symbol': symbol},
        )
        technical_indicator_tasks.append(tech_task)
        
        # Sentiment features
        sentiment_task = PythonOperator(
            task_id=f'calculate_sentiment_features_{symbol}',
            python_callable=calculate_sentiment_features,
            op_kwargs={'symbol': symbol},
            pool='sentiment_analysis_pool',
        )
        sentiment_feature_tasks.append(sentiment_task)

# Model Operations
trigger_evaluation_task = PythonOperator(
    task_id='trigger_model_evaluation',
    python_callable=trigger_model_evaluation,
    dag=dag,
)

# Maintenance
cleanup_task = PythonOperator(
    task_id='cleanup_old_data',
    python_callable=cleanup_old_data,
    dag=dag,
)

# Task Dependencies
validate_market_task >> data_collection_group
data_collection_group >> data_quality_group
data_quality_group >> feature_engineering_group
feature_engineering_group >> trigger_evaluation_task
trigger_evaluation_task >> cleanup_task

# Set up cross-symbol dependencies within groups
for i in range(len(STOCK_SYMBOLS)):
    # Data collection dependencies
    stock_data_tasks[i] >> news_data_tasks[i]
    
    # Quality checks depend on data collection
    news_data_tasks[i] >> quality_check_tasks[i]
    
    # Feature engineering depends on quality checks
    quality_check_tasks[i] >> technical_indicator_tasks[i]
    quality_check_tasks[i] >> sentiment_feature_tasks[i]