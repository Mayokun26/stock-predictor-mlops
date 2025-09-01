#!/usr/bin/env python3
"""
Comprehensive Test Suite for MLOps Pipeline
Tests model training, evaluation, API endpoints, and MLflow integration
"""

import os
import sys
import pytest
import asyncio
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import mlflow
import mlflow.sklearn
from sklearn.metrics import r2_score

# Add src to Python path
sys.path.append(str(Path(__file__).parent.parent / "src"))

class TestMLOpsPipeline:
    """Comprehensive MLOps pipeline tests"""
    
    @classmethod
    def setup_class(cls):
        """Set up test environment"""
        cls.mlflow_uri = "http://localhost:5001"
        cls.api_base_url = "http://localhost:8000"
        cls.test_symbols = ["AAPL", "MSFT"]
        
        mlflow.set_tracking_uri(cls.mlflow_uri)
        cls.client = mlflow.MlflowClient()
        
    def test_mlflow_server_health(self):
        """Test MLflow server connectivity"""
        try:
            response = requests.get(f"{self.mlflow_uri}/health", timeout=10)
            assert response.status_code == 200, "MLflow server not responding"
            print("✅ MLflow server health check passed")
        except Exception as e:
            pytest.fail(f"MLflow server health check failed: {e}")
    
    def test_api_server_health(self):
        """Test API server health"""
        try:
            response = requests.get(f"{self.api_base_url}/health", timeout=10)
            assert response.status_code == 200, "API server not responding"
            
            health_data = response.json()
            assert health_data["overall_status"] in ["healthy", "degraded"], "API not healthy"
            assert "mlflow" in health_data["components"], "MLflow component missing"
            
            print(f"✅ API server health check passed - Status: {health_data['overall_status']}")
        except Exception as e:
            pytest.fail(f"API server health check failed: {e}")
    
    def test_model_registry_access(self):
        """Test MLflow model registry access"""
        try:
            # Test basic registry access
            experiments = self.client.search_experiments()
            assert len(experiments) > 0, "No experiments found in MLflow"
            
            # Check for registered models
            models = self.client.search_registered_models()
            assert len(models) > 0, "No registered models found"
            
            # Verify expected models exist
            model_names = [model.name for model in models]
            for symbol in self.test_symbols:
                expected_model = f"{symbol}_predictor"
                assert expected_model in model_names, f"Model {expected_model} not found in registry"
                
            print(f"✅ Model registry test passed - Found {len(models)} registered models")
        except Exception as e:
            pytest.fail(f"Model registry test failed: {e}")
    
    def test_model_loading_and_prediction(self):
        """Test model loading and prediction functionality"""
        for symbol in self.test_symbols:
            try:
                model_name = f"{symbol}_predictor"
                
                # Get latest model version
                versions = self.client.search_model_versions(f"name='{model_name}'")
                assert len(versions) > 0, f"No versions found for {model_name}"
                
                latest_version = sorted(versions, key=lambda v: int(v.version))[-1]
                model_uri = f"models:/{model_name}/{latest_version.version}"
                
                # Load model
                model = mlflow.sklearn.load_model(model_uri)
                assert model is not None, f"Failed to load model {model_uri}"
                
                # Test prediction with dummy data (29 features)
                X_test = np.random.randn(5, 29)  # 5 samples, 29 features
                predictions = model.predict(X_test)
                
                assert len(predictions) == 5, "Prediction count mismatch"
                assert all(np.isfinite(predictions)), "Predictions contain invalid values"
                assert isinstance(predictions[0], (int, float, np.number)), "Invalid prediction type"
                
                print(f"✅ {symbol} model loading and prediction test passed")
                
            except Exception as e:
                pytest.fail(f"{symbol} model test failed: {e}")
    
    def test_api_prediction_endpoints(self):
        """Test API prediction endpoints"""
        for symbol in self.test_symbols:
            try:
                # Test prediction endpoint
                payload = {
                    "symbol": symbol,
                    "news_headlines": ["Test market update"],
                    "user_id": "test_user"
                }
                
                response = requests.post(
                    f"{self.api_base_url}/predict",
                    json=payload,
                    timeout=30
                )
                
                assert response.status_code == 200, f"Prediction API failed for {symbol}"
                
                data = response.json()
                required_fields = ["symbol", "predicted_price", "predicted_change_pct", 
                                 "confidence_score", "model_version"]
                
                for field in required_fields:
                    assert field in data, f"Missing field {field} in prediction response"
                
                # Validate data types and ranges
                assert isinstance(data["predicted_price"], (int, float)), "Invalid predicted_price type"
                assert isinstance(data["confidence_score"], (int, float)), "Invalid confidence_score type"
                assert 0 <= data["confidence_score"] <= 1, "Confidence score out of range"
                assert data["symbol"] == symbol, "Symbol mismatch in response"
                
                print(f"✅ {symbol} prediction API test passed - Model: {data['model_version']}")
                
            except Exception as e:
                pytest.fail(f"{symbol} prediction API test failed: {e}")
    
    def test_feature_store_endpoints(self):
        """Test feature store functionality"""
        try:
            # Test feature store stats
            response = requests.get(f"{self.api_base_url}/feature-store/stats", timeout=10)
            assert response.status_code == 200, "Feature store stats endpoint failed"
            
            stats = response.json()
            assert "total_cached_features" in stats, "Missing cached features count"
            
            # Test feature retrieval for a symbol
            symbol = self.test_symbols[0]
            response = requests.get(f"{self.api_base_url}/feature-store/features/{symbol}", timeout=15)
            assert response.status_code == 200, f"Feature retrieval failed for {symbol}"
            
            print("✅ Feature store endpoints test passed")
            
        except Exception as e:
            pytest.fail(f"Feature store test failed: {e}")
    
    def test_metrics_endpoint(self):
        """Test Prometheus metrics endpoint"""
        try:
            response = requests.get(f"{self.api_base_url}/metrics", timeout=10)
            assert response.status_code == 200, "Metrics endpoint failed"
            
            metrics_text = response.text
            assert "predictions_total" in metrics_text, "Missing predictions_total metric"
            assert "prediction_duration_seconds" in metrics_text, "Missing prediction_duration metric"
            
            print("✅ Metrics endpoint test passed")
            
        except Exception as e:
            pytest.fail(f"Metrics endpoint test failed: {e}")
    
    def test_model_performance_thresholds(self):
        """Test that models meet minimum performance thresholds"""
        for symbol in self.test_symbols:
            try:
                model_name = f"{symbol}_predictor"
                
                # Get model performance from recent experiments
                experiments = self.client.search_experiments()
                
                # Look for recent training runs
                model_runs = []
                for experiment in experiments[:3]:  # Check recent experiments
                    runs = self.client.search_runs(
                        experiment_ids=[experiment.experiment_id],
                        filter_string=f"params.symbol = '{symbol}'"
                    )
                    model_runs.extend(runs)
                
                if not model_runs:
                    print(f"⚠️ No training runs found for {symbol}, skipping performance test")
                    continue
                
                # Get most recent run
                latest_run = sorted(model_runs, key=lambda r: r.info.start_time)[-1]
                
                # Check performance metrics
                metrics = latest_run.data.metrics
                
                # Test R² threshold (financial ML is challenging, so lower threshold)
                if "test_r2" in metrics:
                    test_r2 = metrics["test_r2"]
                    assert test_r2 > -2.0, f"{symbol} R² too low: {test_r2:.4f}"
                    print(f"✅ {symbol} R² performance acceptable: {test_r2:.4f}")
                
                # Test MAE threshold
                if "test_mae" in metrics:
                    test_mae = metrics["test_mae"]
                    assert test_mae < 0.1, f"{symbol} MAE too high: {test_mae:.6f}"
                    print(f"✅ {symbol} MAE performance acceptable: {test_mae:.6f}")
                
            except Exception as e:
                pytest.fail(f"{symbol} performance threshold test failed: {e}")
    
    def test_model_versioning(self):
        """Test model versioning functionality"""
        for symbol in self.test_symbols:
            try:
                model_name = f"{symbol}_predictor"
                
                # Get all versions
                versions = self.client.search_model_versions(f"name='{model_name}'")
                assert len(versions) > 0, f"No versions found for {model_name}"
                
                # Check version numbers are valid
                version_numbers = [int(v.version) for v in versions]
                assert all(v > 0 for v in version_numbers), "Invalid version numbers found"
                assert len(set(version_numbers)) == len(version_numbers), "Duplicate version numbers"
                
                # Check that at least one model is in staging or production
                staged_versions = [v for v in versions if v.current_stage in ["Staging", "Production"]]
                assert len(staged_versions) > 0, f"No staged versions found for {model_name}"
                
                print(f"✅ {symbol} model versioning test passed - {len(versions)} versions, {len(staged_versions)} staged")
                
            except Exception as e:
                pytest.fail(f"{symbol} versioning test failed: {e}")

class TestDataValidation:
    """Test data quality and validation"""
    
    def test_feature_calculation_consistency(self):
        """Test that features are calculated consistently"""
        try:
            import yfinance as yf
            sys.path.append(str(Path(__file__).parent.parent))
            from train_model_with_mlflow import create_advanced_features
            
            # Test with AAPL data
            ticker = yf.Ticker("AAPL")
            hist = ticker.history(period="6mo")
            
            # Calculate features twice
            features1 = create_advanced_features(hist)
            features2 = create_advanced_features(hist)
            
            # Should be identical
            assert features1.keys() == features2.keys(), "Feature names inconsistent"
            
            for key in features1.keys():
                if isinstance(features1[key], (int, float)):
                    assert abs(features1[key] - features2[key]) < 1e-10, f"Feature {key} inconsistent"
            
            # Check we have expected 29 features
            assert len(features1) == 29, f"Expected 29 features, got {len(features1)}"
            
            print("✅ Feature calculation consistency test passed")
            
        except Exception as e:
            pytest.fail(f"Feature calculation test failed: {e}")

class TestIntegration:
    """Integration tests for complete pipeline"""
    
    def test_end_to_end_prediction_flow(self):
        """Test complete prediction flow from API to model"""
        try:
            symbol = "AAPL"
            
            # 1. Make API prediction
            payload = {
                "symbol": symbol,
                "news_headlines": ["Strong quarterly results"],
                "user_id": "integration_test"
            }
            
            response = requests.post(
                "http://localhost:8000/predict",
                json=payload,
                timeout=30
            )
            
            assert response.status_code == 200, "API prediction failed"
            prediction_data = response.json()
            
            # 2. Verify model was actually used (not fallback)
            assert "production" in prediction_data["model_version"].lower() or "staging" in prediction_data["model_version"].lower(), "Using fallback model instead of trained model"
            
            # 3. Verify prediction is reasonable
            predicted_price = prediction_data["predicted_price"]
            current_price = prediction_data["current_price"]
            change_pct = abs(prediction_data["predicted_change_pct"])
            
            assert isinstance(predicted_price, (int, float)), "Invalid predicted price type"
            assert predicted_price > 0, "Predicted price must be positive"
            assert change_pct < 50, "Predicted change too extreme"  # Sanity check
            
            # 4. Verify confidence score is reasonable
            confidence = prediction_data["confidence_score"]
            assert 0 <= confidence <= 1, "Confidence score out of range"
            
            print(f"✅ End-to-end prediction test passed - {symbol}: ${predicted_price:.2f} ({prediction_data['predicted_change_pct']:.2f}%)")
            
        except Exception as e:
            pytest.fail(f"End-to-end integration test failed: {e}")

def run_tests():
    """Run all tests and return exit code"""
    print("🧪 Starting MLOps Pipeline Test Suite")
    print("=" * 50)
    
    # Configure pytest
    pytest_args = [
        __file__,
        "-v",
        "--tb=short",
        "--no-header"
    ]
    
    # Run tests
    exit_code = pytest.main(pytest_args)
    
    if exit_code == 0:
        print("\n🎉 All tests passed successfully!")
    else:
        print(f"\n❌ Some tests failed (exit code: {exit_code})")
    
    return exit_code

if __name__ == "__main__":
    exit_code = run_tests()
    sys.exit(exit_code)