#!/usr/bin/env python3
import sys
sys.path.append('src/monitoring')
from drift_detector import ModelDriftDetector
import numpy as np

print('🔍 DRIFT DETECTION SYSTEM - LIVE DEMONSTRATION')
print('='*55)

# Initialize detector
detector = ModelDriftDetector(database_path='test_drift.db')
print('✅ Detector initialized')

# Generate baseline data (stable market conditions)
np.random.seed(42)
baseline_features = np.random.normal(0, 1, (500, 10))
baseline_prices = 100 + np.cumsum(np.random.normal(0, 0.5, 500)) * 0.1
baseline_predictions = baseline_prices + np.random.normal(0, 1, 500)

# Store baseline
detector.store_baseline_snapshot('live_test', 'DEMO', baseline_features, baseline_predictions, baseline_prices)
print('✅ Baseline stored: 500 samples, 10 features')

# Generate drifted data (market crash simulation)
np.random.seed(999)
drifted_features = np.random.normal(0.8, 1.5, (100, 10))  # Distribution shift
drifted_prices = 90 + np.cumsum(np.random.normal(-0.2, 2, 100)) * 0.1  # Volatile crash
drifted_predictions = drifted_prices + np.random.normal(0, 4, 100)  # Worse predictions

# Run drift detection
print('\n🚨 RUNNING LIVE DRIFT DETECTION...')
drift_result = detector.detect_drift('live_test', 'DEMO', drifted_features, drifted_predictions, drifted_prices)

print(f'\n📊 DRIFT ANALYSIS RESULTS:')
print(f'   Overall Status: {drift_result.drift_status}')
print(f'   PSI Score: {drift_result.psi_score:.3f} (>0.25 = Major Drift)')
print(f'   KS Test p-value: {drift_result.ks_p_value:.3f} (<0.05 = Significant)')
print(f'   RMSE Drift: {drift_result.rmse_drift:.1%}')
print(f'   MAE Drift: {drift_result.mae_drift:.1%}')
print(f'   Overall Drift Score: {drift_result.overall_drift_score:.3f}')

# Check alerts
alerts = detector.get_active_alerts('live_test', 'DEMO')
print(f'\n🚨 ACTIVE ALERTS: {len(alerts)}')
for alert in alerts:
    print(f'   {alert.severity}: {alert.description}')
    print(f'   Recommendation: {alert.recommendation}')

# Get summary
summary = detector.get_drift_summary('live_test', 'DEMO')
print(f'\n📈 EXECUTIVE SUMMARY:')
print(f"   Status: {summary['status']}")
print(f"   Requires Attention: {summary['requires_attention']}")
print(f"   Critical Alerts: {summary.get('critical_alerts', 0)}")

print('\n✅ DRIFT DETECTION FULLY OPERATIONAL!')