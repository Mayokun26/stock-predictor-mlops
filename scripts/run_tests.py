#!/usr/bin/env python3
"""
Test Runner for Local Development and CI/CD
Handles test execution with proper environment setup
"""

import os
import sys
import subprocess
import time
import requests
from pathlib import Path

def check_services():
    """Check if required services are running"""
    services = {
        'MLflow': 'http://localhost:5001/health',
        'API': 'http://localhost:8000/health',
        'Redis': None,  # TODO: Add Redis health check
        'PostgreSQL': None  # TODO: Add PostgreSQL health check
    }
    
    print("🔍 Checking service availability...")
    
    for service, url in services.items():
        if url:
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    print(f"✅ {service}: Running")
                else:
                    print(f"⚠️  {service}: Responding but unhealthy (status: {response.status_code})")
            except requests.exceptions.RequestException:
                print(f"❌ {service}: Not accessible at {url}")
        else:
            print(f"ℹ️  {service}: Health check not implemented")

def run_tests(test_type="all", verbose=True):
    """Run tests with specified type"""
    
    print(f"🧪 Running {test_type} tests...")
    print("=" * 50)
    
    # Check services first
    check_services()
    print()
    
    # Base pytest command
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/",
        "--tb=short"
    ]
    
    if verbose:
        cmd.append("-v")
    
    # Add coverage for full test runs
    if test_type in ["all", "unit"]:
        cmd.extend([
            "--cov=src",
            "--cov-report=term-missing",
            "--cov-report=html:htmlcov"
        ])
    
    # Filter tests by type
    if test_type == "unit":
        cmd.extend(["-k", "not integration"])
    elif test_type == "integration":
        cmd.extend(["-k", "integration"])
    elif test_type == "api":
        cmd.extend(["-k", "test_api"])
    elif test_type == "mlflow":
        cmd.extend(["-k", "mlflow"])
    
    # Run tests
    try:
        result = subprocess.run(cmd, check=False)
        return result.returncode
    except KeyboardInterrupt:
        print("\n⚠️  Test execution interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Test execution failed: {e}")
        return 1

def main():
    """Main test runner"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run MLOps pipeline tests")
    parser.add_argument(
        "--type", 
        choices=["all", "unit", "integration", "api", "mlflow"],
        default="all",
        help="Type of tests to run"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Run tests in quiet mode"
    )
    
    args = parser.parse_args()
    
    # Set up environment
    os.environ["PYTHONPATH"] = str(Path(__file__).parent.parent / "src")
    
    # Run tests
    exit_code = run_tests(
        test_type=args.type,
        verbose=not args.quiet
    )
    
    # Print summary
    if exit_code == 0:
        print("\n🎉 All tests passed!")
    else:
        print(f"\n❌ Tests failed (exit code: {exit_code})")
    
    return exit_code

if __name__ == "__main__":
    sys.exit(main())