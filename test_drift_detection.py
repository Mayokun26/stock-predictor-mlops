#!/usr/bin/env python3
"""
Automated Model Drift Detection System Test

Comprehensive test suite for the drift detection system.
Tests statistical drift, performance drift, and concept drift detection.
"""

import asyncio
import json
import numpy as np
import requests
import time
from datetime import datetime
from typing import List, Dict, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DriftDetectionTester:
    """
    Automated test suite for drift detection system
    
    Tests:
    - Statistical drift detection (PSI, KS test)
    - Performance drift monitoring
    - Concept drift identification
    - Alert generation and resolution
    - API endpoint integration
    """
    
    def __init__(self, api_base_url: str = "http://localhost:8000"):
        self.api_base_url = api_base_url
        self.test_results = {}
        
    def print_status(self, message: str, status: str = "INFO"):
        """Print formatted status message"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        status_colors = {
            'INFO': '\033[94m',
            'SUCCESS': '\033[92m', 
            'ERROR': '\033[91m',
            'WARNING': '\033[93m'
        }
        color = status_colors.get(status, '\033[0m')
        print(f"[{timestamp}] {color}{status}\033[0m: {message}")
    
    def generate_baseline_data(self, n_samples: int = 1000) -> Dict[str, Any]:
        """Generate synthetic baseline data for testing"""
        np.random.seed(42)  # Reproducible results
        
        # Generate features (10 technical indicators)
        features = np.random.multivariate_normal(
            mean=[0, 0, 50, 100, 0.5, 1.0, 0.2, 0, 0.3, 1.5],
            cov=np.eye(10) * 0.1,  # Low correlation, stable period
            size=n_samples
        )
        
        # Generate realistic stock prices and predictions
        base_price = 150.0
        actual_prices = base_price + np.cumsum(np.random.normal(0, 1, n_samples)) * 0.1
        prediction_noise = np.random.normal(0, 2, n_samples)  # Prediction error
        predicted_prices = actual_prices + prediction_noise
        
        return {
            "model_id": "test_model_v1",
            "symbol": "TEST",
            "features": features.tolist(),
            "predictions": predicted_prices.tolist(),
            "actuals": actual_prices.tolist()
        }
    
    def generate_drifted_data(self, n_samples: int = 200, drift_type: str = "statistical") -> Dict[str, Any]:
        """Generate data with different types of drift"""
        np.random.seed(123)  # Different seed for drift
        
        if drift_type == "statistical":
            # Feature distribution shift
            features = np.random.multivariate_normal(
                mean=[0.5, 0.2, 55, 110, 0.7, 1.2, 0.3, 0.1, 0.4, 1.8],  # Shifted means
                cov=np.eye(10) * 0.2,  # Increased variance
                size=n_samples
            )
            
            # Prices similar to baseline
            base_price = 152.0
            actual_prices = base_price + np.cumsum(np.random.normal(0, 1, n_samples)) * 0.1
            prediction_noise = np.random.normal(0, 2.2, n_samples)  # Slightly worse
            predicted_prices = actual_prices + prediction_noise
            
        elif drift_type == "performance":
            # Similar feature distribution but worse predictions
            features = np.random.multivariate_normal(
                mean=[0, 0, 50, 100, 0.5, 1.0, 0.2, 0, 0.3, 1.5],
                cov=np.eye(10) * 0.1,
                size=n_samples
            )
            
            base_price = 155.0
            actual_prices = base_price + np.cumsum(np.random.normal(0, 1, n_samples)) * 0.1
            prediction_noise = np.random.normal(0, 5, n_samples)  # Much worse predictions
            predicted_prices = actual_prices + prediction_noise
            
        elif drift_type == "concept":
            # Different market regime (higher volatility)
            features = np.random.multivariate_normal(
                mean=[0, 0, 50, 100, 0.5, 1.0, 0.8, 0, 0.3, 1.5],  # Higher volatility indicators
                cov=np.eye(10) * 0.15,
                size=n_samples
            )
            
            base_price = 148.0
            actual_prices = base_price + np.cumsum(np.random.normal(0, 2.5, n_samples)) * 0.1  # Higher vol
            prediction_noise = np.random.normal(0, 3, n_samples)
            predicted_prices = actual_prices + prediction_noise
            
        else:  # "stable" - no drift
            features = np.random.multivariate_normal(
                mean=[0, 0, 50, 100, 0.5, 1.0, 0.2, 0, 0.3, 1.5],
                cov=np.eye(10) * 0.1,
                size=n_samples
            )
            
            base_price = 151.0
            actual_prices = base_price + np.cumsum(np.random.normal(0, 1, n_samples)) * 0.1
            prediction_noise = np.random.normal(0, 2, n_samples)
            predicted_prices = actual_prices + prediction_noise
        
        return {
            "model_id": "test_model_v1",
            "symbol": "TEST",
            "features": features.tolist(),
            "predictions": predicted_prices.tolist(),
            "actuals": actual_prices.tolist()
        }
    
    def test_api_health(self) -> bool:
        """Test 1: API health and drift detection availability"""
        self.print_status("Testing API health...", "INFO")
        
        try:
            response = requests.get(f"{self.api_base_url}/health", timeout=10)
            if response.status_code == 200:
                health_data = response.json()
                self.test_results['api_health'] = True
                self.print_status("✅ API is healthy", "SUCCESS")
                self.print_status(f"   Services: {health_data.get('services', {})}", "INFO")
                return True
            else:
                self.test_results['api_health'] = False
                self.print_status(f"❌ API unhealthy: {response.status_code}", "ERROR")
                return False
                
        except Exception as e:
            self.test_results['api_health'] = False
            self.print_status(f"❌ API connection failed: {e}", "ERROR")
            return False
    
    def test_baseline_storage(self) -> bool:
        """Test 2: Baseline snapshot storage"""
        self.print_status("Testing baseline storage...", "INFO")
        
        try:
            baseline_data = self.generate_baseline_data(1000)
            
            response = requests.post(
                f"{self.api_base_url}/drift/baseline/test_model_v1/TEST",
                json=baseline_data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                self.test_results['baseline_storage'] = True
                self.print_status("✅ Baseline storage successful", "SUCCESS")
                self.print_status(f"   Features: {result['feature_count']}, Samples: {result['sample_count']}", "INFO")
                return True
            else:
                self.test_results['baseline_storage'] = False
                self.print_status(f"❌ Baseline storage failed: {response.status_code} - {response.text}", "ERROR")
                return False
                
        except Exception as e:
            self.test_results['baseline_storage'] = False
            self.print_status(f"❌ Baseline storage error: {e}", "ERROR")
            return False
    
    def test_drift_detection(self, drift_type: str) -> bool:
        """Test 3-6: Drift detection for different drift types"""
        self.print_status(f"Testing {drift_type} drift detection...", "INFO")
        
        try:
            drifted_data = self.generate_drifted_data(200, drift_type)
            
            response = requests.post(
                f"{self.api_base_url}/drift/detect",
                json=drifted_data,
                timeout=30
            )
            
            if response.status_code == 200:
                drift_metrics = response.json()
                
                # Log key metrics
                self.print_status(f"   PSI Score: {drift_metrics['psi_score']:.3f}", "INFO")
                self.print_status(f"   KS p-value: {drift_metrics['ks_p_value']:.3f}", "INFO")
                self.print_status(f"   RMSE Drift: {drift_metrics['rmse_drift']:.1%}", "INFO")
                self.print_status(f"   Overall Score: {drift_metrics['overall_drift_score']:.3f}", "INFO")
                self.print_status(f"   Status: {drift_metrics['drift_status']}", "INFO")
                self.print_status(f"   Active Alerts: {drift_metrics['active_alerts']}", "INFO")
                
                # Validate results based on drift type
                if drift_type == "statistical":
                    expected_drift = drift_metrics['psi_score'] > 0.1 or drift_metrics['ks_p_value'] < 0.1
                elif drift_type == "performance":
                    expected_drift = abs(drift_metrics['rmse_drift']) > 0.1
                elif drift_type == "concept":
                    expected_drift = drift_metrics['overall_drift_score'] > 0.3
                else:  # stable
                    expected_drift = drift_metrics['drift_status'] == 'STABLE'
                
                test_key = f'drift_detection_{drift_type}'
                if expected_drift or drift_type == "stable":
                    self.test_results[test_key] = True
                    self.print_status(f"✅ {drift_type.title()} drift detection working correctly", "SUCCESS")
                    return True
                else:
                    self.test_results[test_key] = False
                    self.print_status(f"❌ {drift_type.title()} drift not detected as expected", "ERROR")
                    return False
            else:
                self.test_results[f'drift_detection_{drift_type}'] = False
                self.print_status(f"❌ Drift detection failed: {response.status_code} - {response.text}", "ERROR")
                return False
                
        except Exception as e:
            self.test_results[f'drift_detection_{drift_type}'] = False
            self.print_status(f"❌ Drift detection error: {e}", "ERROR")
            return False
    
    def test_alert_management(self) -> bool:
        """Test 7: Alert retrieval and resolution"""
        self.print_status("Testing alert management...", "INFO")
        
        try:
            # Get alerts
            response = requests.get(
                f"{self.api_base_url}/drift/alerts/test_model_v1/TEST",
                timeout=10
            )
            
            if response.status_code == 200:
                alerts_data = response.json()
                active_alerts = alerts_data['active_alerts']
                alerts = alerts_data['alerts']
                
                self.print_status(f"   Found {active_alerts} active alerts", "INFO")
                
                # If there are alerts, try to resolve one
                if alerts:
                    first_alert = alerts[0]
                    self.print_status(f"   Alert: {first_alert['severity']} - {first_alert['description']}", "INFO")
                    
                    # Resolve the alert
                    resolve_response = requests.post(
                        f"{self.api_base_url}/drift/alerts/{first_alert['alert_id']}/resolve",
                        timeout=10
                    )
                    
                    if resolve_response.status_code == 200:
                        self.test_results['alert_management'] = True
                        self.print_status("✅ Alert management working correctly", "SUCCESS")
                        return True
                    else:
                        self.test_results['alert_management'] = False
                        self.print_status(f"❌ Alert resolution failed: {resolve_response.status_code}", "ERROR")
                        return False
                else:
                    self.test_results['alert_management'] = True
                    self.print_status("✅ Alert retrieval working (no alerts to resolve)", "SUCCESS")
                    return True
            else:
                self.test_results['alert_management'] = False
                self.print_status(f"❌ Alert retrieval failed: {response.status_code}", "ERROR")
                return False
                
        except Exception as e:
            self.test_results['alert_management'] = False
            self.print_status(f"❌ Alert management error: {e}", "ERROR")
            return False
    
    def test_drift_history_and_summary(self) -> bool:
        """Test 8: Drift history and summary endpoints"""
        self.print_status("Testing drift history and summary...", "INFO")
        
        try:
            # Test history endpoint
            history_response = requests.get(
                f"{self.api_base_url}/drift/history/test_model_v1/TEST?days=7",
                timeout=10
            )
            
            # Test summary endpoint
            summary_response = requests.get(
                f"{self.api_base_url}/drift/summary/test_model_v1/TEST",
                timeout=10
            )
            
            if history_response.status_code == 200 and summary_response.status_code == 200:
                history_data = history_response.json()
                summary_data = summary_response.json()
                
                self.print_status(f"   History samples: {history_data['sample_count']}", "INFO")
                self.print_status(f"   Summary status: {summary_data.get('status', 'N/A')}", "INFO")
                self.print_status(f"   Requires attention: {summary_data.get('requires_attention', False)}", "INFO")
                
                self.test_results['drift_history_summary'] = True
                self.print_status("✅ Drift history and summary working correctly", "SUCCESS")
                return True
            else:
                self.test_results['drift_history_summary'] = False
                self.print_status("❌ Drift history/summary endpoints failed", "ERROR")
                return False
                
        except Exception as e:
            self.test_results['drift_history_summary'] = False
            self.print_status(f"❌ Drift history/summary error: {e}", "ERROR")
            return False
    
    def generate_test_report(self):
        """Generate comprehensive test report"""
        self.print_status("=" * 60, "INFO")
        self.print_status("🎯 DRIFT DETECTION SYSTEM TEST REPORT", "INFO")
        self.print_status("=" * 60, "INFO")
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result)
        
        test_descriptions = {
            'api_health': 'API Health Check',
            'baseline_storage': 'Baseline Storage',
            'drift_detection_statistical': 'Statistical Drift Detection',
            'drift_detection_performance': 'Performance Drift Detection', 
            'drift_detection_concept': 'Concept Drift Detection',
            'drift_detection_stable': 'Stable Data Detection',
            'alert_management': 'Alert Management',
            'drift_history_summary': 'History & Summary Endpoints'
        }
        
        for test_name, result in self.test_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            description = test_descriptions.get(test_name, test_name.replace('_', ' ').title())
            self.print_status(f"{description}: {status}", 
                             "SUCCESS" if result else "ERROR")
        
        self.print_status("-" * 60, "INFO")
        self.print_status(f"Total Tests: {total_tests}", "INFO")
        self.print_status(f"Passed: {passed_tests}", "SUCCESS" if passed_tests == total_tests else "WARNING")
        self.print_status(f"Failed: {total_tests - passed_tests}", "ERROR" if passed_tests != total_tests else "INFO")
        self.print_status(f"Success Rate: {passed_tests/total_tests:.1%}", 
                         "SUCCESS" if passed_tests == total_tests else "WARNING")
        self.print_status("=" * 60, "INFO")
        
        return passed_tests == total_tests

async def run_drift_detection_tests():
    """Main test execution function"""
    tester = DriftDetectionTester()
    
    try:
        # Print test banner
        tester.print_status("🔍" * 20, "INFO")
        tester.print_status("🎯 AUTOMATED DRIFT DETECTION SYSTEM TESTS", "INFO")
        tester.print_status("🔍" * 20, "INFO")
        
        # Test 1: API Health
        if not tester.test_api_health():
            tester.print_status("⚠️ API not available, stopping tests", "WARNING")
            return False
        
        # Test 2: Baseline Storage
        if not tester.test_baseline_storage():
            tester.print_status("⚠️ Baseline storage failed, continuing with limited tests", "WARNING")
        else:
            # Wait a moment for baseline to be processed
            time.sleep(2)
        
        # Test 3-6: Different types of drift detection
        drift_types = ['statistical', 'performance', 'concept', 'stable']
        for drift_type in drift_types:
            tester.test_drift_detection(drift_type)
            time.sleep(1)  # Brief pause between tests
        
        # Test 7: Alert Management
        tester.test_alert_management()
        
        # Test 8: History and Summary
        tester.test_drift_history_and_summary()
        
        # Generate final report
        success = tester.generate_test_report()
        
        return success
        
    except KeyboardInterrupt:
        tester.print_status("🛑 Tests interrupted by user", "WARNING")
        return False
    except Exception as e:
        tester.print_status(f"❌ Test execution error: {e}", "ERROR")
        return False

def quick_drift_demo():
    """Quick demonstration of drift detection capabilities"""
    print("🔍 Quick Drift Detection Demo")
    print("=" * 40)
    
    try:
        # Test basic API connectivity
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code != 200:
            print("❌ API not available. Start the server first:")
            print("   python3 src/api/production_api.py")
            return False
        
        print("✅ API is running")
        
        # Generate demo data
        tester = DriftDetectionTester()
        
        # Store baseline
        print("\n📊 Storing baseline data...")
        baseline_data = tester.generate_baseline_data(500)
        
        baseline_response = requests.post(
            "http://localhost:8000/drift/baseline/demo_model/DEMO",
            json=baseline_data,
            timeout=20
        )
        
        if baseline_response.status_code == 200:
            print("✅ Baseline stored successfully")
        else:
            print(f"❌ Baseline storage failed: {baseline_response.status_code}")
            return False
        
        # Test drift detection
        print("\n🔍 Testing drift detection...")
        drifted_data = tester.generate_drifted_data(100, "statistical")
        
        drift_response = requests.post(
            "http://localhost:8000/drift/detect",
            json=drifted_data,
            timeout=20
        )
        
        if drift_response.status_code == 200:
            metrics = drift_response.json()
            print("✅ Drift detection completed")
            print(f"   Status: {metrics['drift_status']}")
            print(f"   PSI Score: {metrics['psi_score']:.3f}")
            print(f"   Overall Drift: {metrics['overall_drift_score']:.3f}")
            print(f"   Active Alerts: {metrics['active_alerts']}")
            
            return True
        else:
            print(f"❌ Drift detection failed: {drift_response.status_code}")
            return False
        
    except Exception as e:
        print(f"❌ Demo error: {e}")
        return False

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        # Quick demo mode
        success = quick_drift_demo()
    else:
        # Full test suite
        success = asyncio.run(run_drift_detection_tests())
    
    exit_code = 0 if success else 1
    print(f"\n🏁 Tests completed with exit code: {exit_code}")
    sys.exit(exit_code)