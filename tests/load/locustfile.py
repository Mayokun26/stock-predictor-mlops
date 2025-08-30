#!/usr/bin/env python3
"""
Load Testing Configuration for MLOps Stock Prediction API
Comprehensive load testing scenarios with realistic usage patterns
"""

from locust import HttpUser, task, between, events
import json
import random
import time
from datetime import datetime, timedelta
import uuid
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MLOpsAPIUser(HttpUser):
    """
    Simulates realistic user behavior for the MLOps API
    """
    
    wait_time = between(1, 5)  # Wait 1-5 seconds between requests
    
    def on_start(self):
        """Initialize user session"""
        self.user_id = str(uuid.uuid4())
        self.symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "NVDA", "META", "NFLX"]
        self.news_headlines = [
            "Tech stocks surge on positive earnings",
            "Federal Reserve signals rate stability",
            "AI companies show strong growth momentum",
            "Market volatility expected amid economic uncertainty",
            "Quarterly earnings exceed analyst expectations",
            "Technology sector leads market rally",
            "Consumer spending remains robust despite inflation",
            "Global supply chain improvements boost manufacturing"
        ]
        
        logger.info(f"User {self.user_id} started load test session")
    
    @task(1)
    def health_check(self):
        """Basic health check - lightweight operation"""
        with self.client.get("/", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Health check failed: {response.status_code}")
    
    @task(2)
    def detailed_health_check(self):
        """Detailed health check - more comprehensive"""
        with self.client.get("/health", catch_response=True) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    if data.get("status") in ["healthy", "degraded"]:
                        response.success()
                    else:
                        response.failure(f"Unhealthy status: {data.get('status')}")
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
            else:
                response.failure(f"Health check failed: {response.status_code}")
    
    @task(10)
    def predict_stock_price(self):
        """Main prediction endpoint - high frequency operation"""
        symbol = random.choice(self.symbols)
        
        # Create realistic request payload
        payload = {
            "symbol": symbol,
            "news_headlines": random.sample(self.news_headlines, random.randint(1, 3)),
            "user_id": self.user_id
        }
        
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "LoadTest/1.0",
            "X-Request-ID": str(uuid.uuid4())
        }
        
        start_time = time.time()
        
        with self.client.post("/predict", 
                             json=payload, 
                             headers=headers,
                             catch_response=True) as response:
            
            response_time = round((time.time() - start_time) * 1000, 2)
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    
                    # Validate response structure
                    required_fields = [
                        "symbol", "current_price", "predicted_price", 
                        "confidence", "model_version"
                    ]
                    
                    missing_fields = [field for field in required_fields if field not in data]
                    if missing_fields:
                        response.failure(f"Missing fields: {missing_fields}")
                        return
                    
                    # Validate data quality
                    if data["confidence"] < 0 or data["confidence"] > 1:
                        response.failure(f"Invalid confidence score: {data['confidence']}")
                        return
                    
                    if data["current_price"] <= 0 or data["predicted_price"] <= 0:
                        response.failure(f"Invalid price values")
                        return
                    
                    # Log slow responses
                    if response_time > 2000:  # 2 seconds
                        logger.warning(f"Slow prediction response: {response_time}ms for {symbol}")
                    
                    response.success()
                    
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
                except KeyError as e:
                    response.failure(f"Missing required field: {e}")
                    
            elif response.status_code == 429:
                # Rate limiting - expected behavior
                logger.info("Rate limit hit - expected behavior")
                response.success()
                
            elif response.status_code == 404:
                response.failure(f"Symbol not found: {symbol}")
                
            else:
                response.failure(f"Prediction failed: {response.status_code} - {response.text}")
    
    @task(1)
    def get_metrics(self):
        """Access metrics endpoint"""
        with self.client.get("/metrics", catch_response=True) as response:
            if response.status_code == 200:
                # Verify prometheus metrics format
                content = response.text
                if "predictions_total" in content or "prediction_request_duration" in content:
                    response.success()
                else:
                    response.failure("Metrics format invalid")
            else:
                response.failure(f"Metrics failed: {response.status_code}")

class HighVolumeUser(HttpUser):
    """
    Simulates high-volume trading scenarios
    """
    
    wait_time = between(0.1, 1.0)  # Very fast requests
    
    def on_start(self):
        self.user_id = f"hv_user_{uuid.uuid4()}"
        self.symbols = ["AAPL", "GOOGL", "MSFT"]  # Focus on top symbols
    
    @task
    def rapid_predictions(self):
        """Rapid-fire prediction requests"""
        symbol = random.choice(self.symbols)
        
        payload = {
            "symbol": symbol,
            "user_id": self.user_id,
            "news_headlines": []  # Minimal payload for speed
        }
        
        with self.client.post("/predict", json=payload, catch_response=True) as response:
            if response.status_code in [200, 429]:  # Accept rate limits
                response.success()
            else:
                response.failure(f"Rapid prediction failed: {response.status_code}")

class StressTestUser(HttpUser):
    """
    Stress testing with invalid requests and edge cases
    """
    
    wait_time = between(0.5, 2.0)
    
    @task(5)
    def valid_request(self):
        """Normal valid request for baseline"""
        payload = {
            "symbol": "AAPL",
            "user_id": "stress_test_user"
        }
        
        self.client.post("/predict", json=payload)
    
    @task(1)
    def invalid_symbol(self):
        """Test with invalid symbol"""
        payload = {
            "symbol": "INVALID_SYMBOL_12345",
            "user_id": "stress_test_user"
        }
        
        with self.client.post("/predict", json=payload, catch_response=True) as response:
            # Expect this to fail gracefully
            if response.status_code in [400, 404, 422]:
                response.success()  # Proper error handling
            elif response.status_code == 500:
                response.failure("Server error on invalid symbol")
            else:
                response.success()
    
    @task(1)
    def malformed_request(self):
        """Test with malformed JSON"""
        malformed_json = '{"symbol": "AAPL", "user_id": "test"'  # Missing closing brace
        
        with self.client.post("/predict", 
                             data=malformed_json,
                             headers={"Content-Type": "application/json"},
                             catch_response=True) as response:
            if response.status_code in [400, 422]:
                response.success()  # Proper error handling
            else:
                response.failure(f"Unexpected response to malformed JSON: {response.status_code}")
    
    @task(1)
    def large_payload(self):
        """Test with unusually large payload"""
        large_headlines = ["Very long headline " * 100] * 50  # Large payload
        
        payload = {
            "symbol": "AAPL",
            "user_id": "stress_test_user",
            "news_headlines": large_headlines
        }
        
        with self.client.post("/predict", json=payload, catch_response=True) as response:
            # Should handle large payloads gracefully
            if response.status_code in [200, 413, 422]:  # OK, too large, or validation error
                response.success()
            else:
                response.failure(f"Poor handling of large payload: {response.status_code}")

# Event handlers for detailed reporting
@events.request.add_listener
def on_request(request_type, name, response_time, response_length, exception, context, **kwargs):
    """Log request details"""
    if exception:
        logger.error(f"Request failed: {request_type} {name} - {exception}")
    elif response_time > 5000:  # Log slow requests > 5s
        logger.warning(f"Slow request: {request_type} {name} - {response_time}ms")

@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Log test start"""
    logger.info(f"Load test starting with {environment.runner.user_count} users")

@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Log test completion and results"""
    stats = environment.runner.stats
    
    logger.info(f"""
Load Test Completed:
- Total requests: {stats.total.num_requests}
- Total failures: {stats.total.num_failures}
- Average response time: {stats.total.avg_response_time:.2f}ms
- Max response time: {stats.total.max_response_time:.2f}ms
- Requests per second: {stats.total.total_rps:.2f}
- Failure rate: {(stats.total.num_failures / max(stats.total.num_requests, 1) * 100):.2f}%
    """)

# Custom load shapes for different testing scenarios
from locust.env import Environment
from locust import LoadTestShape

class StepLoadShape(LoadTestShape):
    """
    Step load pattern: gradually increase load
    """
    
    step_time = 60  # Duration of each step in seconds
    step_load = 10   # Users to add per step
    spawn_rate = 2   # Users spawned per second
    time_limit = 600 # Total test duration
    
    def tick(self):
        run_time = self.get_run_time()
        
        if run_time > self.time_limit:
            return None
        
        current_step = run_time // self.step_time
        user_count = (current_step + 1) * self.step_load
        
        return (user_count, self.spawn_rate)

class SpikeLoadShape(LoadTestShape):
    """
    Spike load pattern: sudden increases in load
    """
    
    def tick(self):
        run_time = self.get_run_time()
        
        if run_time > 300:  # 5 minutes total
            return None
        
        # Spike pattern: low, high, low, high
        if run_time < 60:
            return (10, 2)
        elif run_time < 120:
            return (100, 10)  # Sudden spike
        elif run_time < 180:
            return (20, 5)    # Back to low
        elif run_time < 240:
            return (150, 15)  # Bigger spike
        else:
            return (30, 5)    # Cool down

# Performance test configuration
class PerformanceTestConfig:
    """Configuration for different test scenarios"""
    
    BASELINE = {
        "users": 50,
        "spawn_rate": 2,
        "duration": "10m",
        "description": "Baseline performance test"
    }
    
    LOAD_TEST = {
        "users": 200,
        "spawn_rate": 5,
        "duration": "15m",
        "description": "Standard load test"
    }
    
    STRESS_TEST = {
        "users": 500,
        "spawn_rate": 10,
        "duration": "20m",
        "description": "Stress test with high concurrency"
    }
    
    SPIKE_TEST = {
        "users": 1000,
        "spawn_rate": 50,
        "duration": "5m",
        "description": "Spike test with sudden load increases"
    }
    
    ENDURANCE_TEST = {
        "users": 100,
        "spawn_rate": 2,
        "duration": "60m",
        "description": "Endurance test for stability"
    }

if __name__ == "__main__":
    print("Load testing configuration loaded")
    print("Available test scenarios:")
    for name, config in PerformanceTestConfig.__dict__.items():
        if isinstance(config, dict) and 'description' in config:
            print(f"  - {name}: {config['description']}")
    
    print("\nExample usage:")
    print("locust -f locustfile.py --host=http://localhost:8000 --users 100 --spawn-rate 5 --run-time 300s")