#!/usr/bin/env python3
"""
Simple system test script to verify the MLOps setup
"""
import sqlite3
import os
import json
from datetime import datetime

def test_database():
    """Test database connectivity and data"""
    print("🔍 Testing database...")
    
    if not os.path.exists('stocks.db'):
        print("❌ Database file not found")
        return False
    
    try:
        conn = sqlite3.connect('stocks.db')
        
        # Check tables exist
        tables = conn.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name NOT LIKE 'sqlite_%'
        """).fetchall()
        
        print(f"✅ Found {len(tables)} tables: {[t[0] for t in tables]}")
        
        # Check stock data
        stocks = conn.execute("SELECT COUNT(*) FROM stock_info").fetchone()[0]
        prices = conn.execute("SELECT COUNT(*) FROM stock_prices").fetchone()[0]
        
        print(f"✅ Stock info: {stocks} stocks")
        print(f"✅ Price data: {prices} records")
        
        # Check recent data
        recent = conn.execute("""
            SELECT symbol, MAX(date) as latest_date, COUNT(*) as count
            FROM stock_prices 
            GROUP BY symbol 
            ORDER BY latest_date DESC 
            LIMIT 5
        """).fetchall()
        
        print("📊 Recent data:")
        for symbol, date, count in recent:
            print(f"   {symbol}: {count} records, latest: {date}")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Database test failed: {e}")
        return False

def test_files():
    """Test that all required files exist"""
    print("\n🔍 Testing file structure...")
    
    required_files = [
        'api.py',
        'pipeline.py', 
        'monitoring.py',
        'collect_data.py',
        'train_with_mlflow.py',
        'requirements.txt',
        'Dockerfile',
        'docker-compose.yml',
        'Makefile',
        'README.md'
    ]
    
    missing_files = []
    for file in required_files:
        if os.path.exists(file):
            size = os.path.getsize(file)
            print(f"✅ {file} ({size} bytes)")
        else:
            missing_files.append(file)
            print(f"❌ {file} missing")
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    
    print("✅ All required files present")
    return True

def test_src_structure():
    """Test source code structure"""
    print("\n🔍 Testing source structure...")
    
    if not os.path.exists('src'):
        print("❌ src directory missing")
        return False
    
    src_dirs = ['data', 'database', 'models', 'api']
    for dir_name in src_dirs:
        path = f"src/{dir_name}"
        if os.path.exists(path):
            files = len([f for f in os.listdir(path) if f.endswith('.py')])
            print(f"✅ {path} ({files} Python files)")
        else:
            print(f"❌ {path} missing")
            return False
    
    return True

def test_config():
    """Test configuration files"""
    print("\n🔍 Testing configuration...")
    
    # Check if .env exists
    if os.path.exists('.env'):
        print("✅ .env file found")
    else:
        print("⚠️  .env file missing (optional)")
    
    # Check requirements.txt content
    try:
        with open('requirements.txt', 'r') as f:
            requirements = f.read()
            
        key_packages = ['fastapi', 'pandas', 'scikit-learn', 'mlflow']
        found_packages = []
        
        for package in key_packages:
            if package in requirements.lower():
                found_packages.append(package)
                print(f"✅ {package} in requirements")
            else:
                print(f"❌ {package} missing from requirements")
        
        return len(found_packages) == len(key_packages)
        
    except Exception as e:
        print(f"❌ Requirements check failed: {e}")
        return False

def generate_system_report():
    """Generate a comprehensive system report"""
    print("\n📊 Generating system report...")
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'system_status': 'unknown',
        'components': {}
    }
    
    # Test all components
    tests = [
        ('database', test_database),
        ('files', test_files),
        ('src_structure', test_src_structure),
        ('config', test_config)
    ]
    
    all_passed = True
    
    for test_name, test_func in tests:
        try:
            passed = test_func()
            report['components'][test_name] = {
                'status': 'pass' if passed else 'fail',
                'timestamp': datetime.now().isoformat()
            }
            
            if not passed:
                all_passed = False
                
        except Exception as e:
            print(f"❌ {test_name} test error: {e}")
            report['components'][test_name] = {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            all_passed = False
    
    report['system_status'] = 'healthy' if all_passed else 'issues_detected'
    
    # Save report
    os.makedirs('reports', exist_ok=True)
    report_file = f"reports/system_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📋 Report saved: {report_file}")
    
    return report

def print_deployment_instructions():
    """Print next steps for deployment"""
    print("\n🚀 Deployment Instructions")
    print("=" * 50)
    print()
    print("1. Install Dependencies:")
    print("   pip install -r requirements.txt")
    print()
    print("2. Run Data Collection:")
    print("   python3 collect_data.py")
    print()
    print("3. Train Models:")
    print("   python3 train_with_mlflow.py")
    print()
    print("4. Start API Server:")
    print("   python3 -m uvicorn api:app --host 0.0.0.0 --port 8000")
    print()
    print("5. Test API:")
    print('   curl -X POST "http://localhost:8000/predict" \\')
    print('     -H "Content-Type: application/json" \\')
    print('     -d \'{"symbol": "AAPL", "news_headlines": ["Apple earnings strong"]}\'')
    print()
    print("6. Docker Deployment:")
    print("   docker-compose up -d")
    print()
    print("7. Monitor System:")
    print("   python3 monitoring.py")

def main():
    """Main test function"""
    print("🧪 MLOps System Test")
    print("=" * 50)
    
    # Run system tests
    report = generate_system_report()
    
    # Print results
    print(f"\n🎯 System Status: {report['system_status'].upper()}")
    
    if report['system_status'] == 'healthy':
        print("\n🎉 All tests passed! System is ready for deployment.")
        print_deployment_instructions()
    else:
        print("\n⚠️  Some issues detected. Check the report for details.")
        
        # Show failed components
        failed = [name for name, result in report['components'].items() 
                 if result['status'] != 'pass']
        if failed:
            print(f"\n❌ Failed components: {', '.join(failed)}")
    
    return report['system_status'] == 'healthy'

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)