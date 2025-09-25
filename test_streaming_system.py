#!/usr/bin/env python3
"""
Real-Time Streaming ML System Test

Comprehensive test suite for the WebSocket-based streaming ML prediction system.
Tests server functionality, client connectivity, and real-time prediction flow.
"""

import asyncio
import json
import websockets
import subprocess
import time
import threading
from datetime import datetime
import sys
import os

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

class StreamingSystemTester:
    """
    Automated test suite for real-time streaming ML system
    
    Tests:
    - Server startup and health
    - WebSocket connectivity 
    - Real-time prediction streaming
    - Client message handling
    - Performance metrics
    """
    
    def __init__(self):
        self.server_url = "ws://localhost:8765"
        self.server_process = None
        self.test_results = {}
        self.received_predictions = []
        
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
    
    async def test_server_startup(self):
        """Test 1: Server startup and health check"""
        self.print_status("Testing server startup...", "INFO")
        
        try:
            # Start server in background
            cmd = [sys.executable, "src/streaming/real_time_predictor.py"]
            self.server_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=os.getcwd()
            )
            
            # Wait for server to initialize
            await asyncio.sleep(3)
            
            # Check if process is running
            if self.server_process.poll() is None:
                self.test_results['server_startup'] = True
                self.print_status("✅ Server started successfully", "SUCCESS")
            else:
                stderr = self.server_process.stderr.read().decode()
                self.test_results['server_startup'] = False
                self.print_status(f"❌ Server failed to start: {stderr}", "ERROR")
                return False
                
        except Exception as e:
            self.test_results['server_startup'] = False
            self.print_status(f"❌ Server startup error: {e}", "ERROR")
            return False
            
        return True
    
    async def test_websocket_connection(self):
        """Test 2: WebSocket connection establishment"""
        self.print_status("Testing WebSocket connection...", "INFO")
        
        try:
            websocket = await websockets.connect(
                self.server_url,
                ping_interval=20,
                ping_timeout=10
            )
            
            self.test_results['websocket_connection'] = True
            self.print_status("✅ WebSocket connection established", "SUCCESS")
            
            await websocket.close()
            return True
            
        except Exception as e:
            self.test_results['websocket_connection'] = False
            self.print_status(f"❌ WebSocket connection failed: {e}", "ERROR")
            return False
    
    async def test_status_command(self):
        """Test 3: Server status command"""
        self.print_status("Testing status command...", "INFO")
        
        try:
            websocket = await websockets.connect(self.server_url)
            
            # Send status command
            status_request = {"command": "get_status"}
            await websocket.send(json.dumps(status_request))
            
            # Wait for response
            response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
            data = json.loads(response)
            
            # Validate response
            required_fields = ['type', 'streaming', 'symbols', 'update_interval', 'client_count']
            if all(field in data for field in required_fields):
                self.test_results['status_command'] = True
                self.print_status("✅ Status command working correctly", "SUCCESS")
                self.print_status(f"   Streaming: {data['streaming']}", "INFO")
                self.print_status(f"   Symbols: {data['symbols']}", "INFO")
                self.print_status(f"   Update Interval: {data['update_interval']}s", "INFO")
            else:
                self.test_results['status_command'] = False
                self.print_status("❌ Status response missing required fields", "ERROR")
            
            await websocket.close()
            return self.test_results['status_command']
            
        except Exception as e:
            self.test_results['status_command'] = False
            self.print_status(f"❌ Status command test failed: {e}", "ERROR")
            return False
    
    async def test_real_time_predictions(self):
        """Test 4: Real-time prediction streaming"""
        self.print_status("Testing real-time predictions (60s)...", "INFO")
        
        try:
            websocket = await websockets.connect(self.server_url)
            
            # Listen for predictions for 60 seconds
            start_time = time.time()
            prediction_count = 0
            
            while time.time() - start_time < 60:  # 60 second test
                try:
                    # Wait for message with timeout
                    message = await asyncio.wait_for(websocket.recv(), timeout=35.0)
                    data = json.loads(message)
                    
                    if data.get('type') == 'live_predictions':
                        predictions = data.get('predictions', [])
                        prediction_count += len(predictions)
                        self.received_predictions.extend(predictions)
                        
                        # Log first prediction details
                        if predictions and prediction_count <= len(predictions):
                            pred = predictions[0]
                            self.print_status(
                                f"📊 Received prediction: {pred['symbol']} -> "
                                f"${pred['predicted_price']:.2f} "
                                f"({pred['predicted_change_pct']:+.2f}%)",
                                "SUCCESS"
                            )
                        
                        # Show running count
                        if prediction_count % 5 == 0:  # Every 5th prediction
                            self.print_status(f"   Total predictions received: {prediction_count}", "INFO")
                    
                    elif data.get('type') == 'initial_data':
                        init_predictions = data.get('predictions', [])
                        self.print_status(f"📡 Initial data received: {len(init_predictions)} predictions", "SUCCESS")
                        
                except asyncio.TimeoutError:
                    self.print_status("⚠️  Timeout waiting for predictions", "WARNING")
                    break
            
            await websocket.close()
            
            # Evaluate results
            if prediction_count >= 5:  # Expect at least 5 predictions in 60s
                self.test_results['real_time_predictions'] = True
                self.print_status(f"✅ Real-time predictions working ({prediction_count} received)", "SUCCESS")
            else:
                self.test_results['real_time_predictions'] = False
                self.print_status(f"❌ Insufficient predictions received ({prediction_count})", "ERROR")
            
            return self.test_results['real_time_predictions']
            
        except Exception as e:
            self.test_results['real_time_predictions'] = False
            self.print_status(f"❌ Real-time prediction test failed: {e}", "ERROR")
            return False
    
    async def test_multiple_clients(self):
        """Test 5: Multiple client connections"""
        self.print_status("Testing multiple client connections...", "INFO")
        
        clients = []
        try:
            # Connect 3 clients simultaneously
            for i in range(3):
                client = await websockets.connect(self.server_url)
                clients.append(client)
                await asyncio.sleep(0.5)  # Stagger connections
            
            self.print_status(f"✅ {len(clients)} clients connected", "SUCCESS")
            
            # Send status request from first client
            status_request = {"command": "get_status"}
            await clients[0].send(json.dumps(status_request))
            
            response = await asyncio.wait_for(clients[0].recv(), timeout=5.0)
            data = json.loads(response)
            
            client_count = data.get('client_count', 0)
            if client_count >= 3:
                self.test_results['multiple_clients'] = True
                self.print_status(f"✅ Multiple clients test passed (server reports {client_count} clients)", "SUCCESS")
            else:
                self.test_results['multiple_clients'] = False
                self.print_status(f"❌ Server reports {client_count} clients, expected 3+", "ERROR")
            
        except Exception as e:
            self.test_results['multiple_clients'] = False
            self.print_status(f"❌ Multiple clients test failed: {e}", "ERROR")
        finally:
            # Close all clients
            for client in clients:
                try:
                    await client.close()
                except:
                    pass
            
        return self.test_results.get('multiple_clients', False)
    
    def analyze_prediction_quality(self):
        """Test 6: Analyze prediction data quality"""
        self.print_status("Analyzing prediction quality...", "INFO")
        
        if not self.received_predictions:
            self.test_results['prediction_quality'] = False
            self.print_status("❌ No predictions to analyze", "ERROR")
            return False
        
        # Analyze prediction structure
        required_fields = [
            'symbol', 'timestamp', 'predicted_price', 'current_price',
            'predicted_change_pct', 'confidence_score', 'trend_direction',
            'volatility_score', 'features_used', 'model_version', 'processing_time_ms'
        ]
        
        valid_predictions = 0
        total_processing_time = 0
        confidence_scores = []
        
        for pred in self.received_predictions:
            # Check required fields
            if all(field in pred for field in required_fields):
                valid_predictions += 1
                
                # Collect metrics
                total_processing_time += pred.get('processing_time_ms', 0)
                confidence_scores.append(pred.get('confidence_score', 0))
        
        # Calculate quality metrics
        if valid_predictions > 0:
            avg_processing_time = total_processing_time / valid_predictions
            avg_confidence = sum(confidence_scores) / len(confidence_scores)
            
            self.print_status(f"📊 Prediction Quality Analysis:", "INFO")
            self.print_status(f"   Valid predictions: {valid_predictions}/{len(self.received_predictions)}", "INFO")
            self.print_status(f"   Average processing time: {avg_processing_time:.1f}ms", "INFO")
            self.print_status(f"   Average confidence: {avg_confidence:.1%}", "INFO")
            
            # Quality thresholds
            quality_passed = (
                valid_predictions == len(self.received_predictions) and  # All valid
                avg_processing_time < 2000 and  # Under 2 seconds
                0.3 <= avg_confidence <= 0.95  # Reasonable confidence range
            )
            
            if quality_passed:
                self.test_results['prediction_quality'] = True
                self.print_status("✅ Prediction quality analysis passed", "SUCCESS")
            else:
                self.test_results['prediction_quality'] = False
                self.print_status("❌ Prediction quality analysis failed", "ERROR")
        else:
            self.test_results['prediction_quality'] = False
            self.print_status("❌ No valid predictions found", "ERROR")
        
        return self.test_results.get('prediction_quality', False)
    
    def generate_test_report(self):
        """Generate comprehensive test report"""
        self.print_status("=" * 60, "INFO")
        self.print_status("🎯 STREAMING ML SYSTEM TEST REPORT", "INFO")
        self.print_status("=" * 60, "INFO")
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result)
        
        for test_name, result in self.test_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            self.print_status(f"{test_name.replace('_', ' ').title()}: {status}", 
                             "SUCCESS" if result else "ERROR")
        
        self.print_status("-" * 60, "INFO")
        self.print_status(f"Total Tests: {total_tests}", "INFO")
        self.print_status(f"Passed: {passed_tests}", "SUCCESS" if passed_tests == total_tests else "WARNING")
        self.print_status(f"Failed: {total_tests - passed_tests}", "ERROR" if passed_tests != total_tests else "INFO")
        self.print_status(f"Success Rate: {passed_tests/total_tests:.1%}", 
                         "SUCCESS" if passed_tests == total_tests else "WARNING")
        
        if self.received_predictions:
            self.print_status(f"Predictions Received: {len(self.received_predictions)}", "INFO")
        
        self.print_status("=" * 60, "INFO")
        
        return passed_tests == total_tests
    
    def cleanup(self):
        """Clean up test resources"""
        if self.server_process:
            self.print_status("🧹 Cleaning up server process...", "INFO")
            self.server_process.terminate()
            try:
                self.server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.server_process.kill()
        
        self.print_status("✅ Cleanup complete", "SUCCESS")

async def run_streaming_tests():
    """Main test execution function"""
    tester = StreamingSystemTester()
    
    try:
        # Print test banner
        tester.print_status("🚀" * 20, "INFO")
        tester.print_status("📊 REAL-TIME STREAMING ML SYSTEM TESTS", "INFO")
        tester.print_status("🚀" * 20, "INFO")
        
        # Run tests in sequence
        test_functions = [
            ("Server Startup", tester.test_server_startup),
            ("WebSocket Connection", tester.test_websocket_connection),
            ("Status Command", tester.test_status_command),
            ("Real-time Predictions", tester.test_real_time_predictions),
            ("Multiple Clients", tester.test_multiple_clients),
        ]
        
        for test_name, test_func in test_functions:
            tester.print_status(f"🎯 Running: {test_name}", "INFO")
            
            if not await test_func():
                tester.print_status(f"⚠️ Stopping tests due to {test_name} failure", "WARNING")
                break
            
            await asyncio.sleep(1)  # Brief pause between tests
        
        # Analyze prediction quality
        tester.analyze_prediction_quality()
        
        # Generate final report
        success = tester.generate_test_report()
        
        return success
        
    except KeyboardInterrupt:
        tester.print_status("🛑 Tests interrupted by user", "WARNING")
        return False
    except Exception as e:
        tester.print_status(f"❌ Test execution error: {e}", "ERROR")
        return False
    finally:
        tester.cleanup()

if __name__ == "__main__":
    print("🧪 Starting Streaming ML System Tests...")
    
    # Run async tests
    success = asyncio.run(run_streaming_tests())
    
    # Exit with appropriate code
    exit_code = 0 if success else 1
    print(f"🏁 Tests completed with exit code: {exit_code}")
    sys.exit(exit_code)