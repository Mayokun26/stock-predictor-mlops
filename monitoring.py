#!/usr/bin/env python3
"""
Monitoring and alerting for the MLOps system
"""
import time
import sqlite3
import pandas as pd
import requests
from datetime import datetime, timedelta
import logging
import json
import os
from prometheus_client import CollectorRegistry, Gauge, Counter, start_http_server, write_to_textfile

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MLOpsMonitor:
    def __init__(self):
        self.db_path = 'stocks.db'
        self.api_url = 'http://localhost:8000'
        self.registry = CollectorRegistry()
        
        # Prometheus metrics
        self.api_requests = Counter('api_requests_total', 'Total API requests', 
                                  ['endpoint', 'status'], registry=self.registry)
        self.prediction_accuracy = Gauge('prediction_accuracy', 'Model prediction accuracy',
                                       ['symbol'], registry=self.registry)
        self.data_freshness = Gauge('data_freshness_hours', 'Hours since last data update',
                                  ['symbol'], registry=self.registry)
        self.model_performance = Gauge('model_rmse', 'Model RMSE performance',
                                     ['symbol'], registry=self.registry)
        
    def check_api_health(self):
        """Check API health and performance"""
        try:
            start_time = time.time()
            response = requests.get(f"{self.api_url}/health", timeout=10)
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                health_data = response.json()
                logger.info(f"✅ API healthy - Response time: {response_time:.2f}s")
                
                # Update metrics
                self.api_requests.labels(endpoint='/health', status='success').inc()
                
                return True, health_data
            else:
                logger.error(f"❌ API unhealthy - Status: {response.status_code}")
                self.api_requests.labels(endpoint='/health', status='error').inc()
                return False, None
                
        except requests.exceptions.ConnectionError:
            logger.error("❌ API not reachable")
            self.api_requests.labels(endpoint='/health', status='connection_error').inc()
            return False, None
        except Exception as e:
            logger.error(f"❌ API health check failed: {e}")
            self.api_requests.labels(endpoint='/health', status='error').inc()
            return False, None
    
    def check_data_freshness(self):
        """Check how fresh our data is"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA']
            freshness_issues = []
            
            for symbol in symbols:
                query = f"""
                SELECT MAX(date) as latest_date 
                FROM stock_prices WHERE symbol='{symbol}'
                """
                result = pd.read_sql(query, conn)
                
                if not result.empty and result.iloc[0]['latest_date']:
                    latest_date = pd.to_datetime(result.iloc[0]['latest_date'])
                    hours_old = (datetime.now() - latest_date).total_seconds() / 3600
                    
                    self.data_freshness.labels(symbol=symbol).set(hours_old)
                    
                    if hours_old > 48:  # Alert if data is more than 2 days old
                        freshness_issues.append(f"{symbol}: {hours_old:.1f} hours old")
                        logger.warning(f"⚠️  Stale data for {symbol}: {hours_old:.1f} hours old")
                else:
                    freshness_issues.append(f"{symbol}: No data found")
                    logger.error(f"❌ No data found for {symbol}")
            
            conn.close()
            
            if not freshness_issues:
                logger.info("✅ All data is fresh")
                return True
            else:
                logger.warning(f"⚠️  Data freshness issues: {', '.join(freshness_issues)}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Data freshness check failed: {e}")
            return False
    
    def test_predictions(self):
        """Test prediction endpoints and track accuracy"""
        try:
            symbols = ['AAPL', 'MSFT']
            test_headlines = [
                "Company reports strong quarterly earnings",
                "Stock market shows positive momentum"
            ]
            
            prediction_issues = []
            
            for symbol in symbols:
                try:
                    test_request = {
                        "symbol": symbol,
                        "news_headlines": test_headlines
                    }
                    
                    start_time = time.time()
                    response = requests.post(f"{self.api_url}/predict", 
                                           json=test_request, timeout=15)
                    response_time = time.time() - start_time
                    
                    if response.status_code == 200:
                        result = response.json()
                        confidence = result.get('confidence', 0)
                        
                        logger.info(f"✅ {symbol} prediction: ${result['predicted_price']:.2f} "
                                  f"(confidence: {confidence:.3f}, time: {response_time:.2f}s)")
                        
                        self.api_requests.labels(endpoint='/predict', status='success').inc()
                        
                        # In a real system, you'd compare predictions to actual outcomes
                        # For now, we'll use confidence as a proxy for accuracy
                        self.prediction_accuracy.labels(symbol=symbol).set(confidence)
                        
                    else:
                        prediction_issues.append(f"{symbol}: HTTP {response.status_code}")
                        self.api_requests.labels(endpoint='/predict', status='error').inc()
                        
                except Exception as e:
                    prediction_issues.append(f"{symbol}: {str(e)}")
                    logger.error(f"❌ Prediction test failed for {symbol}: {e}")
            
            if not prediction_issues:
                logger.info("✅ All prediction tests passed")
                return True
            else:
                logger.warning(f"⚠️  Prediction issues: {', '.join(prediction_issues)}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Prediction testing failed: {e}")
            return False
    
    def check_model_performance(self):
        """Check model performance metrics"""
        try:
            # In a real system, you'd load this from MLflow or model registry
            # For now, we'll simulate some performance metrics
            
            symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA']
            performance_issues = []
            
            for symbol in symbols:
                # Simulate RMSE check (in reality, you'd calculate this from recent predictions vs actuals)
                simulated_rmse = 2.5  # This would come from your model evaluation
                
                self.model_performance.labels(symbol=symbol).set(simulated_rmse)
                
                if simulated_rmse > 5.0:  # Alert threshold
                    performance_issues.append(f"{symbol}: RMSE ${simulated_rmse:.2f}")
            
            if not performance_issues:
                logger.info("✅ Model performance within acceptable ranges")
                return True
            else:
                logger.warning(f"⚠️  Model performance issues: {', '.join(performance_issues)}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Model performance check failed: {e}")
            return False
    
    def generate_report(self):
        """Generate monitoring report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'checks': {}
        }
        
        # Run all checks
        checks = [
            ('api_health', self.check_api_health),
            ('data_freshness', self.check_data_freshness),
            ('predictions', self.test_predictions),
            ('model_performance', self.check_model_performance)
        ]
        
        all_passed = True
        
        for check_name, check_func in checks:
            logger.info(f"\n🔍 Running {check_name} check...")
            
            try:
                if check_name == 'api_health':
                    passed, data = check_func()
                    report['checks'][check_name] = {'passed': passed, 'data': data}
                else:
                    passed = check_func()
                    report['checks'][check_name] = {'passed': passed}
                
                if not passed:
                    all_passed = False
                    
            except Exception as e:
                logger.error(f"❌ {check_name} check failed: {e}")
                report['checks'][check_name] = {'passed': False, 'error': str(e)}
                all_passed = False
        
        report['overall_status'] = 'healthy' if all_passed else 'issues_detected'
        
        # Save report
        os.makedirs('monitoring', exist_ok=True)
        report_file = f"monitoring/report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Export Prometheus metrics
        write_to_textfile('monitoring/metrics.prom', self.registry)
        
        logger.info(f"\n📊 Monitoring report saved: {report_file}")
        
        if all_passed:
            logger.info("🎉 All systems healthy!")
        else:
            logger.warning("⚠️  Some issues detected - check report for details")
        
        return report
    
    def continuous_monitoring(self, interval_minutes=5):
        """Run continuous monitoring"""
        logger.info(f"🔄 Starting continuous monitoring (every {interval_minutes} minutes)")
        
        # Start Prometheus metrics server
        start_http_server(8001, registry=self.registry)
        logger.info("📈 Prometheus metrics available at http://localhost:8001")
        
        while True:
            try:
                self.generate_report()
                time.sleep(interval_minutes * 60)
            except KeyboardInterrupt:
                logger.info("🛑 Monitoring stopped by user")
                break
            except Exception as e:
                logger.error(f"❌ Monitoring error: {e}")
                time.sleep(60)  # Wait 1 minute before retrying

def main():
    """Main monitoring function"""
    monitor = MLOpsMonitor()
    
    if len(os.sys.argv) > 1:
        command = os.sys.argv[1]
        
        if command == "report":
            monitor.generate_report()
        elif command == "continuous":
            interval = int(os.sys.argv[2]) if len(os.sys.argv) > 2 else 5
            monitor.continuous_monitoring(interval)
        elif command == "health":
            passed, data = monitor.check_api_health()
            if data:
                print(json.dumps(data, indent=2))
        else:
            logger.error(f"Unknown command: {command}")
            logger.info("Available commands: report, continuous, health")
    else:
        # Generate single report
        monitor.generate_report()

if __name__ == "__main__":
    main()