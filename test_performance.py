#!/usr/bin/env python3
"""
Comprehensive Performance Testing Suite
Automated performance testing, benchmarking, and optimization validation
"""

import asyncio
import time
import aiohttp
import json
import statistics
import sys
import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import logging
import psutil
import concurrent.futures
import matplotlib.pyplot as plt
import pandas as pd

# Add src to path
sys.path.append('src')

from utils.performance_optimization import (
    profiler, cache, initialize_performance_optimizations,
    resource_monitor, performance_monitor
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("performance_tests")

@dataclass
class PerformanceTestResult:
    """Results from performance testing"""
    
    test_name: str
    total_requests: int
    successful_requests: int
    failed_requests: int
    avg_response_time_ms: float
    min_response_time_ms: float
    max_response_time_ms: float
    p50_response_time_ms: float
    p95_response_time_ms: float
    p99_response_time_ms: float
    requests_per_second: float
    error_rate: float
    memory_usage_mb: float
    cpu_usage_percent: float
    duration_seconds: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "test_name": self.test_name,
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "avg_response_time_ms": round(self.avg_response_time_ms, 2),
            "min_response_time_ms": round(self.min_response_time_ms, 2),
            "max_response_time_ms": round(self.max_response_time_ms, 2),
            "p50_response_time_ms": round(self.p50_response_time_ms, 2),
            "p95_response_time_ms": round(self.p95_response_time_ms, 2),
            "p99_response_time_ms": round(self.p99_response_time_ms, 2),
            "requests_per_second": round(self.requests_per_second, 2),
            "error_rate": round(self.error_rate, 2),
            "memory_usage_mb": round(self.memory_usage_mb, 2),
            "cpu_usage_percent": round(self.cpu_usage_percent, 2),
            "duration_seconds": round(self.duration_seconds, 2)
        }
    
    def print_summary(self):
        """Print performance test summary"""
        print(f"\n{'='*60}")
        print(f"Performance Test Results: {self.test_name}")
        print(f"{'='*60}")
        print(f"Total Requests: {self.total_requests}")
        print(f"Successful: {self.successful_requests} ({((self.successful_requests/self.total_requests)*100):.1f}%)")
        print(f"Failed: {self.failed_requests} ({self.error_rate:.1f}%)")
        print(f"")
        print(f"Response Times (ms):")
        print(f"  Average: {self.avg_response_time_ms:.2f}")
        print(f"  Min: {self.min_response_time_ms:.2f}")
        print(f"  Max: {self.max_response_time_ms:.2f}")
        print(f"  50th percentile: {self.p50_response_time_ms:.2f}")
        print(f"  95th percentile: {self.p95_response_time_ms:.2f}")
        print(f"  99th percentile: {self.p99_response_time_ms:.2f}")
        print(f"")
        print(f"Throughput: {self.requests_per_second:.2f} requests/second")
        print(f"Test Duration: {self.duration_seconds:.2f} seconds")
        print(f"")
        print(f"Resource Usage:")
        print(f"  Memory: {self.memory_usage_mb:.2f} MB")
        print(f"  CPU: {self.cpu_usage_percent:.2f}%")

class PerformanceTester:
    """Comprehensive performance testing framework"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = None
        self.test_results: List[PerformanceTestResult] = []
        
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def single_request(self, endpoint: str, method: str = "GET", payload: Optional[Dict] = None) -> Dict[str, Any]:
        """Make a single request and measure performance"""
        start_time = time.time()
        
        try:
            if method.upper() == "GET":
                async with self.session.get(f"{self.base_url}{endpoint}") as response:
                    response_time = (time.time() - start_time) * 1000
                    content = await response.text()
                    
                    return {
                        "success": response.status < 400,
                        "status_code": response.status,
                        "response_time_ms": response_time,
                        "content_length": len(content),
                        "content": content[:500] if len(content) > 500 else content  # Truncate for memory
                    }
            
            elif method.upper() == "POST":
                headers = {"Content-Type": "application/json"}
                async with self.session.post(
                    f"{self.base_url}{endpoint}", 
                    json=payload,
                    headers=headers
                ) as response:
                    response_time = (time.time() - start_time) * 1000
                    content = await response.text()
                    
                    return {
                        "success": response.status < 400,
                        "status_code": response.status,
                        "response_time_ms": response_time,
                        "content_length": len(content),
                        "content": content[:500] if len(content) > 500 else content
                    }
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return {
                "success": False,
                "status_code": 0,
                "response_time_ms": response_time,
                "content_length": 0,
                "error": str(e)
            }
    
    async def concurrent_requests(
        self,
        endpoint: str,
        method: str = "GET",
        payload_generator: Optional[callable] = None,
        concurrent_users: int = 10,
        requests_per_user: int = 10,
        delay_between_requests: float = 0.1
    ) -> List[Dict[str, Any]]:
        """Generate concurrent requests to test performance under load"""
        
        async def user_session(user_id: int) -> List[Dict[str, Any]]:
            """Simulate a user making multiple requests"""
            results = []
            
            for i in range(requests_per_user):
                # Generate payload if generator provided
                payload = payload_generator(user_id, i) if payload_generator else None
                
                # Make request
                result = await self.single_request(endpoint, method, payload)
                result["user_id"] = user_id
                result["request_num"] = i
                results.append(result)
                
                # Delay between requests
                if delay_between_requests > 0:
                    await asyncio.sleep(delay_between_requests)
            
            return results
        
        # Run concurrent user sessions
        tasks = [user_session(user_id) for user_id in range(concurrent_users)]
        user_results = await asyncio.gather(*tasks)
        
        # Flatten results
        all_results = []
        for user_result in user_results:
            all_results.extend(user_result)
        
        return all_results
    
    def analyze_results(self, results: List[Dict[str, Any]], test_name: str, duration: float) -> PerformanceTestResult:
        """Analyze performance test results"""
        
        if not results:
            raise ValueError("No results to analyze")
        
        # Extract metrics
        response_times = [r["response_time_ms"] for r in results]
        successful_requests = [r for r in results if r["success"]]
        failed_requests = [r for r in results if not r["success"]]
        
        # Calculate percentiles
        response_times_sorted = sorted(response_times)
        p50 = statistics.median(response_times_sorted)
        p95 = response_times_sorted[int(len(response_times_sorted) * 0.95)] if response_times_sorted else 0
        p99 = response_times_sorted[int(len(response_times_sorted) * 0.99)] if response_times_sorted else 0
        
        # Get system resource usage
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        cpu_percent = process.cpu_percent()
        
        return PerformanceTestResult(
            test_name=test_name,
            total_requests=len(results),
            successful_requests=len(successful_requests),
            failed_requests=len(failed_requests),
            avg_response_time_ms=statistics.mean(response_times) if response_times else 0,
            min_response_time_ms=min(response_times) if response_times else 0,
            max_response_time_ms=max(response_times) if response_times else 0,
            p50_response_time_ms=p50,
            p95_response_time_ms=p95,
            p99_response_time_ms=p99,
            requests_per_second=len(results) / duration if duration > 0 else 0,
            error_rate=(len(failed_requests) / len(results)) * 100 if results else 0,
            memory_usage_mb=memory_mb,
            cpu_usage_percent=cpu_percent,
            duration_seconds=duration
        )
    
    async def test_health_endpoints(self) -> PerformanceTestResult:
        """Test health check endpoints performance"""
        logger.info("Testing health endpoints...")
        
        start_time = time.time()
        
        # Test both health endpoints
        results = []
        
        # Basic health check
        for _ in range(50):
            result = await self.single_request("/")
            results.append(result)
        
        # Detailed health check
        for _ in range(50):
            result = await self.single_request("/health")
            results.append(result)
        
        duration = time.time() - start_time
        return self.analyze_results(results, "Health Endpoints", duration)
    
    async def test_prediction_endpoint(self) -> PerformanceTestResult:
        """Test prediction endpoint performance"""
        logger.info("Testing prediction endpoint...")
        
        def payload_generator(user_id: int, request_num: int) -> Dict[str, Any]:
            symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]
            headlines = [
                "Tech stocks show strong performance",
                "Market volatility expected this week",
                "Earnings reports exceed expectations",
                "Federal Reserve maintains current rates"
            ]
            
            return {
                "symbol": symbols[request_num % len(symbols)],
                "news_headlines": headlines[:user_id % 3 + 1],
                "user_id": f"test_user_{user_id}"
            }
        
        start_time = time.time()
        
        results = await self.concurrent_requests(
            endpoint="/predict",
            method="POST",
            payload_generator=payload_generator,
            concurrent_users=20,
            requests_per_user=5,
            delay_between_requests=0.2
        )
        
        duration = time.time() - start_time
        return self.analyze_results(results, "Prediction Endpoint", duration)
    
    async def test_high_concurrency(self) -> PerformanceTestResult:
        """Test system under high concurrency"""
        logger.info("Testing high concurrency scenarios...")
        
        def payload_generator(user_id: int, request_num: int) -> Dict[str, Any]:
            return {
                "symbol": "AAPL",  # Use same symbol to test caching
                "user_id": f"concurrent_user_{user_id}",
                "news_headlines": []
            }
        
        start_time = time.time()
        
        results = await self.concurrent_requests(
            endpoint="/predict",
            method="POST",
            payload_generator=payload_generator,
            concurrent_users=50,
            requests_per_user=10,
            delay_between_requests=0.05  # Very fast requests
        )
        
        duration = time.time() - start_time
        return self.analyze_results(results, "High Concurrency", duration)
    
    async def test_burst_load(self) -> PerformanceTestResult:
        """Test system response to burst loads"""
        logger.info("Testing burst load scenarios...")
        
        def payload_generator(user_id: int, request_num: int) -> Dict[str, Any]:
            symbols = ["AAPL", "GOOGL", "MSFT"]
            return {
                "symbol": symbols[user_id % len(symbols)],
                "user_id": f"burst_user_{user_id}"
            }
        
        start_time = time.time()
        
        # Simulate burst: all requests at once
        results = await self.concurrent_requests(
            endpoint="/predict",
            method="POST",
            payload_generator=payload_generator,
            concurrent_users=100,
            requests_per_user=3,
            delay_between_requests=0.0  # No delay = burst
        )
        
        duration = time.time() - start_time
        return self.analyze_results(results, "Burst Load", duration)
    
    async def test_sustained_load(self) -> PerformanceTestResult:
        """Test system under sustained load"""
        logger.info("Testing sustained load...")
        
        def payload_generator(user_id: int, request_num: int) -> Dict[str, Any]:
            symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "NVDA"]
            return {
                "symbol": symbols[(user_id + request_num) % len(symbols)],
                "user_id": f"sustained_user_{user_id}"
            }
        
        start_time = time.time()
        
        results = await self.concurrent_requests(
            endpoint="/predict",
            method="POST",
            payload_generator=payload_generator,
            concurrent_users=30,
            requests_per_user=20,
            delay_between_requests=0.5  # Sustained, realistic rate
        )
        
        duration = time.time() - start_time
        return self.analyze_results(results, "Sustained Load", duration)
    
    async def run_comprehensive_tests(self) -> List[PerformanceTestResult]:
        """Run all performance tests"""
        logger.info("Starting comprehensive performance testing...")
        
        test_methods = [
            self.test_health_endpoints,
            self.test_prediction_endpoint,
            self.test_high_concurrency,
            self.test_burst_load,
            self.test_sustained_load
        ]
        
        results = []
        
        for test_method in test_methods:
            try:
                result = await test_method()
                results.append(result)
                result.print_summary()
                
                # Brief pause between tests
                await asyncio.sleep(2)
                
            except Exception as e:
                logger.error(f"Test {test_method.__name__} failed: {e}")
        
        self.test_results = results
        return results
    
    def generate_performance_report(self) -> str:
        """Generate comprehensive performance report"""
        if not self.test_results:
            return "No test results available"
        
        report = f"""
Performance Test Report
=======================
Generated: {datetime.now().isoformat()}
Base URL: {self.base_url}

Test Summary:
"""
        
        for result in self.test_results:
            report += f"""
{result.test_name}:
  Total Requests: {result.total_requests}
  Success Rate: {((result.successful_requests/result.total_requests)*100):.1f}%
  Avg Response Time: {result.avg_response_time_ms:.2f}ms
  95th Percentile: {result.p95_response_time_ms:.2f}ms
  Throughput: {result.requests_per_second:.2f} req/sec
  Error Rate: {result.error_rate:.2f}%
"""
        
        # Performance benchmarks
        report += f"""

Performance Benchmarks:
"""
        
        avg_response_times = [r.avg_response_time_ms for r in self.test_results]
        p95_response_times = [r.p95_response_time_ms for r in self.test_results]
        throughputs = [r.requests_per_second for r in self.test_results]
        error_rates = [r.error_rate for r in self.test_results]
        
        report += f"""
  Overall Average Response Time: {statistics.mean(avg_response_times):.2f}ms
  Overall 95th Percentile: {statistics.mean(p95_response_times):.2f}ms
  Peak Throughput: {max(throughputs):.2f} req/sec
  Average Error Rate: {statistics.mean(error_rates):.2f}%
  
  Performance Assessment:
"""
        
        # Performance assessment
        overall_avg_response = statistics.mean(avg_response_times)
        overall_error_rate = statistics.mean(error_rates)
        
        if overall_avg_response < 100:
            report += "  ✅ Excellent response times (< 100ms average)\n"
        elif overall_avg_response < 500:
            report += "  ✅ Good response times (< 500ms average)\n"
        elif overall_avg_response < 1000:
            report += "  ⚠️ Acceptable response times (< 1s average)\n"
        else:
            report += "  ❌ Poor response times (> 1s average)\n"
        
        if overall_error_rate < 1:
            report += "  ✅ Excellent reliability (< 1% error rate)\n"
        elif overall_error_rate < 5:
            report += "  ✅ Good reliability (< 5% error rate)\n"
        else:
            report += "  ❌ Poor reliability (> 5% error rate)\n"
        
        if max(throughputs) > 50:
            report += "  ✅ High throughput capability (> 50 req/sec)\n"
        elif max(throughputs) > 20:
            report += "  ✅ Good throughput capability (> 20 req/sec)\n"
        else:
            report += "  ⚠️ Limited throughput capability (< 20 req/sec)\n"
        
        return report
    
    def save_results_to_csv(self, filename: str = "performance_results.csv"):
        """Save test results to CSV for analysis"""
        if not self.test_results:
            logger.warning("No test results to save")
            return
        
        df = pd.DataFrame([result.to_dict() for result in self.test_results])
        df.to_csv(filename, index=False)
        logger.info(f"Performance results saved to {filename}")

async def run_performance_tests():
    """Main function to run performance tests"""
    
    # Initialize performance optimizations
    try:
        await initialize_performance_optimizations()
        logger.info("Performance optimizations initialized")
    except Exception as e:
        logger.warning(f"Performance optimization init failed: {e}")
    
    # Run tests
    async with PerformanceTester() as tester:
        try:
            # Check if API is accessible
            test_result = await tester.single_request("/")
            if not test_result["success"]:
                logger.error(f"API not accessible: {test_result}")
                print("❌ API is not accessible at http://localhost:8000")
                print("Please ensure the API is running with: docker-compose up -d")
                return
            
            logger.info("API is accessible, starting performance tests...")
            
            # Run comprehensive tests
            results = await tester.run_comprehensive_tests()
            
            # Generate and display report
            report = tester.generate_performance_report()
            print("\n" + "="*80)
            print("COMPREHENSIVE PERFORMANCE REPORT")
            print("="*80)
            print(report)
            
            # Save results
            tester.save_results_to_csv("performance_test_results.csv")
            
            # Performance summary
            if results:
                avg_response_time = statistics.mean([r.avg_response_time_ms for r in results])
                max_throughput = max([r.requests_per_second for r in results])
                avg_error_rate = statistics.mean([r.error_rate for r in results])
                
                print(f"\n{'='*80}")
                print("PERFORMANCE SUMMARY")
                print(f"{'='*80}")
                print(f"🚀 Average Response Time: {avg_response_time:.2f}ms")
                print(f"⚡ Peak Throughput: {max_throughput:.2f} requests/second")
                print(f"✅ Average Error Rate: {avg_error_rate:.2f}%")
                
                # Performance grade
                if avg_response_time < 200 and avg_error_rate < 2 and max_throughput > 30:
                    print(f"🏆 Overall Grade: EXCELLENT - Production Ready")
                elif avg_response_time < 500 and avg_error_rate < 5 and max_throughput > 15:
                    print(f"✅ Overall Grade: GOOD - Ready with monitoring")
                elif avg_response_time < 1000 and avg_error_rate < 10:
                    print(f"⚠️ Overall Grade: ACCEPTABLE - Needs optimization")
                else:
                    print(f"❌ Overall Grade: POOR - Requires significant improvement")
            
        except Exception as e:
            logger.error(f"Performance testing failed: {e}")
            import traceback
            traceback.print_exc()

def main():
    """Entry point for performance testing"""
    print("🔥 MLOps API Performance Testing Suite")
    print("="*50)
    
    try:
        asyncio.run(run_performance_tests())
    except KeyboardInterrupt:
        print("\n\n⏹️ Performance testing interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Performance testing failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()