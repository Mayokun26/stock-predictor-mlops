#!/usr/bin/env python3
"""
Automated Model Drift Detection System

Enterprise-grade drift detection for production ML models.
Monitors statistical drift, performance drift, and concept drift.
"""

import asyncio
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import sqlite3
from pathlib import Path
import warnings
from scipy import stats

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DriftAlert:
    """Data structure for drift detection alerts"""
    alert_id: str
    timestamp: str
    model_id: str
    symbol: str
    drift_type: str  # 'statistical', 'performance', 'concept'
    severity: str    # 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'
    metric_name: str
    baseline_value: float
    current_value: float
    threshold: float
    confidence: float
    description: str
    recommendation: str
    requires_retraining: bool

@dataclass
class DriftMetrics:
    """Comprehensive drift detection metrics"""
    timestamp: str
    model_id: str
    symbol: str
    
    # Statistical drift metrics
    psi_score: float              # Population Stability Index
    ks_statistic: float           # Kolmogorov-Smirnov test
    ks_p_value: float
    jensen_shannon_divergence: float
    
    # Performance drift metrics
    rmse_baseline: float
    rmse_current: float
    rmse_drift: float
    mae_baseline: float
    mae_current: float
    mae_drift: float
    r2_baseline: float
    r2_current: float
    r2_drift: float
    
    # Concept drift indicators
    prediction_drift: float
    confidence_drift: float
    feature_importance_drift: float
    
    # Overall drift assessment
    overall_drift_score: float
    drift_status: str             # 'STABLE', 'MINOR_DRIFT', 'MAJOR_DRIFT', 'CRITICAL_DRIFT'

class ModelDriftDetector:
    """
    Production-grade model drift detection system
    
    Features:
    - Statistical drift detection (PSI, KS test, Jensen-Shannon divergence)
    - Performance drift monitoring (accuracy degradation over time)
    - Concept drift identification (prediction pattern changes)
    - Automated alerting and retraining triggers
    - Historical drift trend analysis
    """
    
    def __init__(self, database_path: str = "drift_monitoring.db", 
                 monitoring_window: int = 7):
        self.database_path = Path(database_path)
        self.monitoring_window = monitoring_window  # days
        self.drift_thresholds = self._get_drift_thresholds()
        self.baseline_data = {}
        self.current_predictions = {}
        
        # Initialize database
        self._init_database()
        
        logger.info("🔍 Model Drift Detector initialized")
        logger.info(f"📊 Monitoring window: {monitoring_window} days")
        logger.info(f"🗄️  Database: {database_path}")
    
    def _get_drift_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Define industry-standard drift detection thresholds"""
        return {
            'statistical': {
                'psi_minor': 0.1,      # PSI > 0.1: minor change
                'psi_major': 0.25,     # PSI > 0.25: major change  
                'ks_critical': 0.05,   # KS p-value < 0.05: significant drift
                'js_threshold': 0.1    # Jensen-Shannon divergence threshold
            },
            'performance': {
                'rmse_degradation': 0.15,    # 15% increase in RMSE
                'mae_degradation': 0.15,     # 15% increase in MAE
                'r2_degradation': 0.1        # 10% decrease in R²
            },
            'concept': {
                'prediction_drift': 0.2,     # 20% change in prediction patterns
                'confidence_drift': 0.15,    # 15% change in confidence scores
                'feature_importance': 0.25   # 25% change in feature importance
            }
        }
    
    def _init_database(self):
        """Initialize SQLite database for drift monitoring"""
        with sqlite3.connect(self.database_path) as conn:
            cursor = conn.cursor()
            
            # Drift metrics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS drift_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    psi_score REAL,
                    ks_statistic REAL,
                    ks_p_value REAL,
                    jensen_shannon_divergence REAL,
                    rmse_baseline REAL,
                    rmse_current REAL,
                    rmse_drift REAL,
                    mae_baseline REAL,
                    mae_current REAL,
                    mae_drift REAL,
                    r2_baseline REAL,
                    r2_current REAL,
                    r2_drift REAL,
                    prediction_drift REAL,
                    confidence_drift REAL,
                    feature_importance_drift REAL,
                    overall_drift_score REAL,
                    drift_status TEXT
                )
            """)
            
            # Drift alerts table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS drift_alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    alert_id TEXT UNIQUE NOT NULL,
                    timestamp TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    drift_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    baseline_value REAL,
                    current_value REAL,
                    threshold REAL,
                    confidence REAL,
                    description TEXT,
                    recommendation TEXT,
                    requires_retraining BOOLEAN,
                    resolved BOOLEAN DEFAULT FALSE
                )
            """)
            
            # Baseline snapshots table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS baseline_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    snapshot_date TEXT NOT NULL,
                    feature_statistics TEXT,  -- JSON blob
                    performance_metrics TEXT,  -- JSON blob
                    prediction_statistics TEXT  -- JSON blob
                )
            """)
            
            conn.commit()
        
        logger.info("✅ Drift monitoring database initialized")
    
    def calculate_psi(self, baseline: np.ndarray, current: np.ndarray, 
                      bins: int = 10) -> float:
        """
        Calculate Population Stability Index (PSI)
        
        PSI measures distribution shift between baseline and current data:
        - PSI < 0.1: No significant change
        - 0.1 <= PSI < 0.25: Minor change, monitor
        - PSI >= 0.25: Major change, investigate/retrain
        """
        try:
            # Create bins based on baseline data
            _, bin_edges = np.histogram(baseline, bins=bins)
            
            # Calculate frequencies for baseline and current
            baseline_freq, _ = np.histogram(baseline, bins=bin_edges)
            current_freq, _ = np.histogram(current, bins=bin_edges)
            
            # Convert to percentages (add small epsilon to avoid division by zero)
            epsilon = 1e-8
            baseline_pct = (baseline_freq + epsilon) / (np.sum(baseline_freq) + epsilon * bins)
            current_pct = (current_freq + epsilon) / (np.sum(current_freq) + epsilon * bins)
            
            # Calculate PSI
            psi = np.sum((current_pct - baseline_pct) * np.log(current_pct / baseline_pct))
            
            return float(psi)
            
        except Exception as e:
            logger.error(f"PSI calculation error: {e}")
            return 0.0
    
    def calculate_ks_test(self, baseline: np.ndarray, current: np.ndarray) -> Tuple[float, float]:
        """
        Kolmogorov-Smirnov test for distribution comparison
        
        Returns:
        - KS statistic: Maximum difference between CDFs
        - p-value: Statistical significance (< 0.05 indicates significant drift)
        """
        try:
            ks_stat, p_value = stats.ks_2samp(baseline, current)
            return float(ks_stat), float(p_value)
        except Exception as e:
            logger.error(f"KS test error: {e}")
            return 0.0, 1.0
    
    def calculate_jensen_shannon_divergence(self, baseline: np.ndarray, 
                                           current: np.ndarray, bins: int = 50) -> float:
        """
        Jensen-Shannon divergence for distribution comparison
        
        JS divergence is symmetric and bounded [0, 1]:
        - 0: Identical distributions
        - 1: Completely different distributions
        """
        try:
            # Create histograms
            min_val = min(baseline.min(), current.min())
            max_val = max(baseline.max(), current.max())
            bin_edges = np.linspace(min_val, max_val, bins + 1)
            
            hist_baseline, _ = np.histogram(baseline, bins=bin_edges, density=True)
            hist_current, _ = np.histogram(current, bins=bin_edges, density=True)
            
            # Normalize to probabilities
            epsilon = 1e-8
            p = hist_baseline + epsilon
            q = hist_current + epsilon
            p = p / np.sum(p)
            q = q / np.sum(q)
            
            # Calculate Jensen-Shannon divergence
            m = 0.5 * (p + q)
            divergence = 0.5 * stats.entropy(p, m) + 0.5 * stats.entropy(q, m)
            
            return float(divergence)
            
        except Exception as e:
            logger.error(f"JS divergence calculation error: {e}")
            return 0.0
    
    def store_baseline_snapshot(self, model_id: str, symbol: str, 
                               features: np.ndarray, predictions: np.ndarray,
                               actuals: np.ndarray):
        """Store baseline model performance and data statistics"""
        try:
            # Calculate feature statistics
            feature_stats = {
                'mean': features.mean(axis=0).tolist(),
                'std': features.std(axis=0).tolist(),
                'min': features.min(axis=0).tolist(),
                'max': features.max(axis=0).tolist(),
                'percentiles': {
                    '25': np.percentile(features, 25, axis=0).tolist(),
                    '50': np.percentile(features, 50, axis=0).tolist(),
                    '75': np.percentile(features, 75, axis=0).tolist()
                }
            }
            
            # Calculate performance metrics
            performance_metrics = {
                'rmse': float(np.sqrt(mean_squared_error(actuals, predictions))),
                'mae': float(mean_absolute_error(actuals, predictions)),
                'r2': float(r2_score(actuals, predictions)),
                'prediction_mean': float(predictions.mean()),
                'prediction_std': float(predictions.std()),
                'residuals_mean': float((actuals - predictions).mean()),
                'residuals_std': float((actuals - predictions).std())
            }
            
            # Calculate prediction statistics
            prediction_stats = {
                'distribution': np.histogram(predictions, bins=50)[0].tolist(),
                'confidence_mean': 0.5,  # Placeholder - would come from model
                'trend_distribution': {
                    'bullish': float(np.sum(predictions > actuals) / len(predictions)),
                    'bearish': float(np.sum(predictions < actuals) / len(predictions))
                }
            }
            
            # Store in database
            with sqlite3.connect(self.database_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO baseline_snapshots 
                    (model_id, symbol, snapshot_date, feature_statistics, 
                     performance_metrics, prediction_statistics)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    model_id, symbol, datetime.now().isoformat(),
                    json.dumps(feature_stats),
                    json.dumps(performance_metrics),
                    json.dumps(prediction_stats)
                ))
                conn.commit()
            
            # Cache in memory for quick access
            self.baseline_data[f"{model_id}_{symbol}"] = {
                'features': feature_stats,
                'performance': performance_metrics,
                'predictions': prediction_stats
            }
            
            logger.info(f"✅ Baseline snapshot stored for {model_id}_{symbol}")
            
        except Exception as e:
            logger.error(f"Error storing baseline snapshot: {e}")
    
    def detect_drift(self, model_id: str, symbol: str,
                    current_features: np.ndarray, current_predictions: np.ndarray,
                    current_actuals: np.ndarray = None) -> DriftMetrics:
        """
        Comprehensive drift detection analysis
        
        Returns drift metrics and triggers alerts if significant drift detected
        """
        timestamp = datetime.now().isoformat()
        
        # Get baseline data
        baseline_key = f"{model_id}_{symbol}"
        if baseline_key not in self.baseline_data:
            self._load_baseline_from_db(model_id, symbol)
        
        if baseline_key not in self.baseline_data:
            logger.warning(f"No baseline data for {model_id}_{symbol}")
            return self._create_empty_drift_metrics(timestamp, model_id, symbol)
        
        baseline = self.baseline_data[baseline_key]
        
        try:
            # 1. Statistical Drift Detection
            psi_scores = []
            ks_stats = []
            ks_p_values = []
            
            # Calculate drift for each feature
            baseline_features = np.array(baseline['features']['mean'])  # Simplified baseline
            
            for i in range(min(len(baseline_features), current_features.shape[1])):
                if current_features.shape[0] > 10:  # Need sufficient data
                    # Generate synthetic baseline data for PSI (simplified)
                    synthetic_baseline = np.random.normal(
                        baseline['features']['mean'][i],
                        baseline['features']['std'][i],
                        size=current_features.shape[0]
                    )
                    
                    psi = self.calculate_psi(synthetic_baseline, current_features[:, i])
                    ks_stat, ks_p = self.calculate_ks_test(synthetic_baseline, current_features[:, i])
                    
                    psi_scores.append(psi)
                    ks_stats.append(ks_stat)
                    ks_p_values.append(ks_p)
            
            # Aggregate statistical metrics
            psi_score = np.mean(psi_scores) if psi_scores else 0.0
            ks_statistic = np.mean(ks_stats) if ks_stats else 0.0
            ks_p_value = np.mean(ks_p_values) if ks_p_values else 1.0
            
            # Jensen-Shannon divergence for predictions
            if len(current_predictions) > 10:
                baseline_pred_mean = baseline['predictions']['distribution']
                current_pred_hist = np.histogram(current_predictions, bins=50)[0]
                
                # Normalize histograms
                baseline_hist = np.array(baseline_pred_mean, dtype=float)
                if baseline_hist.sum() > 0:
                    baseline_hist = baseline_hist / baseline_hist.sum()
                if current_pred_hist.sum() > 0:
                    current_pred_hist = current_pred_hist / current_pred_hist.sum()
                
                js_divergence = self.calculate_jensen_shannon_divergence(
                    baseline_hist, current_pred_hist
                ) if len(baseline_hist) == len(current_pred_hist) else 0.0
            else:
                js_divergence = 0.0
            
            # 2. Performance Drift Detection
            baseline_perf = baseline['performance']
            
            if current_actuals is not None and len(current_actuals) == len(current_predictions):
                # Calculate current performance
                current_rmse = float(np.sqrt(mean_squared_error(current_actuals, current_predictions)))
                current_mae = float(mean_absolute_error(current_actuals, current_predictions))
                current_r2 = float(r2_score(current_actuals, current_predictions))
                
                # Calculate drift percentages
                rmse_drift = (current_rmse - baseline_perf['rmse']) / baseline_perf['rmse']
                mae_drift = (current_mae - baseline_perf['mae']) / baseline_perf['mae']
                r2_drift = (baseline_perf['r2'] - current_r2) / abs(baseline_perf['r2']) if baseline_perf['r2'] != 0 else 0
            else:
                # Use predictions only (no actuals available)
                current_rmse = baseline_perf['rmse']
                current_mae = baseline_perf['mae'] 
                current_r2 = baseline_perf['r2']
                rmse_drift = 0.0
                mae_drift = 0.0
                r2_drift = 0.0
            
            # 3. Concept Drift Detection
            prediction_drift = abs(current_predictions.mean() - baseline_perf['prediction_mean']) / baseline_perf['prediction_std']
            confidence_drift = 0.0  # Placeholder - would need confidence scores
            feature_importance_drift = 0.0  # Placeholder - would need feature importance
            
            # 4. Overall Drift Assessment
            drift_components = {
                'statistical': max(psi_score / 0.25, (1 - ks_p_value)),
                'performance': max(abs(rmse_drift), abs(mae_drift), abs(r2_drift)),
                'concept': max(prediction_drift, confidence_drift, feature_importance_drift)
            }
            
            overall_drift_score = np.mean(list(drift_components.values()))
            
            # Determine drift status
            if overall_drift_score >= 0.8:
                drift_status = 'CRITICAL_DRIFT'
            elif overall_drift_score >= 0.5:
                drift_status = 'MAJOR_DRIFT'
            elif overall_drift_score >= 0.2:
                drift_status = 'MINOR_DRIFT'
            else:
                drift_status = 'STABLE'
            
            # Create drift metrics
            drift_metrics = DriftMetrics(
                timestamp=timestamp,
                model_id=model_id,
                symbol=symbol,
                psi_score=float(psi_score),
                ks_statistic=float(ks_statistic),
                ks_p_value=float(ks_p_value),
                jensen_shannon_divergence=float(js_divergence),
                rmse_baseline=float(baseline_perf['rmse']),
                rmse_current=float(current_rmse),
                rmse_drift=float(rmse_drift),
                mae_baseline=float(baseline_perf['mae']),
                mae_current=float(current_mae),
                mae_drift=float(mae_drift),
                r2_baseline=float(baseline_perf['r2']),
                r2_current=float(current_r2),
                r2_drift=float(r2_drift),
                prediction_drift=float(prediction_drift),
                confidence_drift=float(confidence_drift),
                feature_importance_drift=float(feature_importance_drift),
                overall_drift_score=float(overall_drift_score),
                drift_status=drift_status
            )
            
            # Store metrics
            self._store_drift_metrics(drift_metrics)
            
            # Generate alerts if necessary
            self._check_and_generate_alerts(drift_metrics)
            
            logger.info(f"🔍 Drift detection completed for {model_id}_{symbol}: {drift_status}")
            logger.info(f"   PSI: {psi_score:.3f}, KS p-value: {ks_p_value:.3f}, Overall: {overall_drift_score:.3f}")
            
            return drift_metrics
            
        except Exception as e:
            logger.error(f"Drift detection error: {e}")
            return self._create_empty_drift_metrics(timestamp, model_id, symbol)
    
    def _create_empty_drift_metrics(self, timestamp: str, model_id: str, symbol: str) -> DriftMetrics:
        """Create empty drift metrics when detection fails"""
        return DriftMetrics(
            timestamp=timestamp,
            model_id=model_id,
            symbol=symbol,
            psi_score=0.0,
            ks_statistic=0.0,
            ks_p_value=1.0,
            jensen_shannon_divergence=0.0,
            rmse_baseline=0.0,
            rmse_current=0.0,
            rmse_drift=0.0,
            mae_baseline=0.0,
            mae_current=0.0,
            mae_drift=0.0,
            r2_baseline=0.0,
            r2_current=0.0,
            r2_drift=0.0,
            prediction_drift=0.0,
            confidence_drift=0.0,
            feature_importance_drift=0.0,
            overall_drift_score=0.0,
            drift_status='UNKNOWN'
        )
    
    def _load_baseline_from_db(self, model_id: str, symbol: str):
        """Load baseline data from database"""
        try:
            with sqlite3.connect(self.database_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT feature_statistics, performance_metrics, prediction_statistics
                    FROM baseline_snapshots
                    WHERE model_id = ? AND symbol = ?
                    ORDER BY snapshot_date DESC
                    LIMIT 1
                """, (model_id, symbol))
                
                result = cursor.fetchone()
                if result:
                    feature_stats, perf_metrics, pred_stats = result
                    
                    baseline_key = f"{model_id}_{symbol}"
                    self.baseline_data[baseline_key] = {
                        'features': json.loads(feature_stats),
                        'performance': json.loads(perf_metrics),
                        'predictions': json.loads(pred_stats)
                    }
                    
                    logger.info(f"📊 Loaded baseline data for {model_id}_{symbol}")
        
        except Exception as e:
            logger.error(f"Error loading baseline from database: {e}")
    
    def _store_drift_metrics(self, metrics: DriftMetrics):
        """Store drift metrics in database"""
        try:
            with sqlite3.connect(self.database_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO drift_metrics (
                        timestamp, model_id, symbol, psi_score, ks_statistic, ks_p_value,
                        jensen_shannon_divergence, rmse_baseline, rmse_current, rmse_drift,
                        mae_baseline, mae_current, mae_drift, r2_baseline, r2_current, r2_drift,
                        prediction_drift, confidence_drift, feature_importance_drift,
                        overall_drift_score, drift_status
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    metrics.timestamp, metrics.model_id, metrics.symbol,
                    metrics.psi_score, metrics.ks_statistic, metrics.ks_p_value,
                    metrics.jensen_shannon_divergence,
                    metrics.rmse_baseline, metrics.rmse_current, metrics.rmse_drift,
                    metrics.mae_baseline, metrics.mae_current, metrics.mae_drift,
                    metrics.r2_baseline, metrics.r2_current, metrics.r2_drift,
                    metrics.prediction_drift, metrics.confidence_drift,
                    metrics.feature_importance_drift,
                    metrics.overall_drift_score, metrics.drift_status
                ))
                conn.commit()
        
        except Exception as e:
            logger.error(f"Error storing drift metrics: {e}")
    
    def _check_and_generate_alerts(self, metrics: DriftMetrics):
        """Check drift metrics against thresholds and generate alerts"""
        alerts = []
        
        # Statistical drift alerts
        if metrics.psi_score > self.drift_thresholds['statistical']['psi_major']:
            alerts.append(self._create_alert(
                metrics, 'statistical', 'HIGH', 'psi_score',
                0.0, metrics.psi_score, self.drift_thresholds['statistical']['psi_major'],
                "Major population stability shift detected",
                "Consider retraining model with recent data",
                True
            ))
        elif metrics.psi_score > self.drift_thresholds['statistical']['psi_minor']:
            alerts.append(self._create_alert(
                metrics, 'statistical', 'MEDIUM', 'psi_score', 
                0.0, metrics.psi_score, self.drift_thresholds['statistical']['psi_minor'],
                "Minor population stability shift detected",
                "Monitor closely, consider data quality checks",
                False
            ))
        
        # Performance drift alerts
        if abs(metrics.rmse_drift) > self.drift_thresholds['performance']['rmse_degradation']:
            severity = 'CRITICAL' if abs(metrics.rmse_drift) > 0.3 else 'HIGH'
            alerts.append(self._create_alert(
                metrics, 'performance', severity, 'rmse_drift',
                metrics.rmse_baseline, metrics.rmse_current, 
                self.drift_thresholds['performance']['rmse_degradation'],
                f"Model RMSE degradation: {metrics.rmse_drift:.1%}",
                "Model retraining recommended",
                severity == 'CRITICAL'
            ))
        
        # Overall drift status alerts
        if metrics.drift_status in ['MAJOR_DRIFT', 'CRITICAL_DRIFT']:
            severity = 'CRITICAL' if metrics.drift_status == 'CRITICAL_DRIFT' else 'HIGH'
            alerts.append(self._create_alert(
                metrics, 'concept', severity, 'overall_drift_score',
                0.0, metrics.overall_drift_score, 0.5,
                f"Overall model drift detected: {metrics.drift_status}",
                "Immediate model retraining required",
                True
            ))
        
        # Store alerts
        for alert in alerts:
            self._store_alert(alert)
            logger.warning(f"🚨 DRIFT ALERT: {alert.severity} - {alert.description}")
    
    def _create_alert(self, metrics: DriftMetrics, drift_type: str, severity: str,
                     metric_name: str, baseline_value: float, current_value: float,
                     threshold: float, description: str, recommendation: str,
                     requires_retraining: bool) -> DriftAlert:
        """Create a drift alert"""
        return DriftAlert(
            alert_id=f"{metrics.model_id}_{metrics.symbol}_{drift_type}_{int(datetime.now().timestamp())}",
            timestamp=metrics.timestamp,
            model_id=metrics.model_id,
            symbol=metrics.symbol,
            drift_type=drift_type,
            severity=severity,
            metric_name=metric_name,
            baseline_value=baseline_value,
            current_value=current_value,
            threshold=threshold,
            confidence=0.95,  # Placeholder
            description=description,
            recommendation=recommendation,
            requires_retraining=requires_retraining
        )
    
    def _store_alert(self, alert: DriftAlert):
        """Store drift alert in database"""
        try:
            with sqlite3.connect(self.database_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR IGNORE INTO drift_alerts (
                        alert_id, timestamp, model_id, symbol, drift_type, severity,
                        metric_name, baseline_value, current_value, threshold, confidence,
                        description, recommendation, requires_retraining
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    alert.alert_id, alert.timestamp, alert.model_id, alert.symbol,
                    alert.drift_type, alert.severity, alert.metric_name,
                    alert.baseline_value, alert.current_value, alert.threshold,
                    alert.confidence, alert.description, alert.recommendation,
                    alert.requires_retraining
                ))
                conn.commit()
        
        except Exception as e:
            logger.error(f"Error storing alert: {e}")
    
    def get_drift_history(self, model_id: str, symbol: str, 
                         days: int = 30) -> List[DriftMetrics]:
        """Get drift metrics history for analysis"""
        try:
            cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
            
            with sqlite3.connect(self.database_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM drift_metrics 
                    WHERE model_id = ? AND symbol = ? AND timestamp >= ?
                    ORDER BY timestamp DESC
                """, (model_id, symbol, cutoff_date))
                
                results = cursor.fetchall()
                columns = [description[0] for description in cursor.description]
                
                drift_history = []
                for row in results:
                    data = dict(zip(columns, row))
                    # Remove ID column and convert to DriftMetrics
                    data.pop('id', None)
                    drift_history.append(DriftMetrics(**data))
                
                return drift_history
        
        except Exception as e:
            logger.error(f"Error getting drift history: {e}")
            return []
    
    def get_active_alerts(self, model_id: str = None, symbol: str = None) -> List[DriftAlert]:
        """Get unresolved drift alerts"""
        try:
            with sqlite3.connect(self.database_path) as conn:
                cursor = conn.cursor()
                
                query = "SELECT * FROM drift_alerts WHERE resolved = FALSE"
                params = []
                
                if model_id:
                    query += " AND model_id = ?"
                    params.append(model_id)
                if symbol:
                    query += " AND symbol = ?"
                    params.append(symbol)
                
                query += " ORDER BY timestamp DESC"
                
                cursor.execute(query, params)
                results = cursor.fetchall()
                columns = [description[0] for description in cursor.description]
                
                alerts = []
                for row in results:
                    data = dict(zip(columns, row))
                    # Remove database-specific columns
                    data.pop('id', None)
                    data.pop('resolved', None)
                    alerts.append(DriftAlert(**data))
                
                return alerts
        
        except Exception as e:
            logger.error(f"Error getting active alerts: {e}")
            return []
    
    def resolve_alert(self, alert_id: str):
        """Mark drift alert as resolved"""
        try:
            with sqlite3.connect(self.database_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE drift_alerts SET resolved = TRUE 
                    WHERE alert_id = ?
                """, (alert_id,))
                conn.commit()
                
                logger.info(f"✅ Resolved alert: {alert_id}")
        
        except Exception as e:
            logger.error(f"Error resolving alert: {e}")
    
    def get_drift_summary(self, model_id: str, symbol: str) -> Dict[str, Any]:
        """Get comprehensive drift analysis summary"""
        try:
            # Get recent drift metrics
            recent_drift = self.get_drift_history(model_id, symbol, days=7)
            active_alerts = self.get_active_alerts(model_id, symbol)
            
            if not recent_drift:
                return {
                    'status': 'NO_DATA',
                    'message': 'No drift data available',
                    'requires_attention': False
                }
            
            latest_drift = recent_drift[0]
            
            # Calculate trends
            psi_trend = np.mean([d.psi_score for d in recent_drift[-3:]])
            performance_trend = np.mean([abs(d.rmse_drift) for d in recent_drift[-3:]])
            
            # Determine overall status
            critical_alerts = [a for a in active_alerts if a.severity == 'CRITICAL']
            high_alerts = [a for a in active_alerts if a.severity == 'HIGH']
            
            if critical_alerts or latest_drift.drift_status == 'CRITICAL_DRIFT':
                overall_status = 'CRITICAL'
                requires_attention = True
            elif high_alerts or latest_drift.drift_status == 'MAJOR_DRIFT':
                overall_status = 'WARNING'
                requires_attention = True
            elif latest_drift.drift_status == 'MINOR_DRIFT':
                overall_status = 'MONITOR'
                requires_attention = False
            else:
                overall_status = 'STABLE'
                requires_attention = False
            
            return {
                'status': overall_status,
                'requires_attention': requires_attention,
                'latest_drift': asdict(latest_drift),
                'trends': {
                    'psi_trend': float(psi_trend),
                    'performance_trend': float(performance_trend)
                },
                'active_alerts': len(active_alerts),
                'critical_alerts': len(critical_alerts),
                'high_alerts': len(high_alerts),
                'last_updated': latest_drift.timestamp,
                'monitoring_period_days': len(recent_drift)
            }
        
        except Exception as e:
            logger.error(f"Error generating drift summary: {e}")
            return {
                'status': 'ERROR', 
                'message': str(e),
                'requires_attention': True
            }

# Example usage and testing
async def demo_drift_detection():
    """Demonstration of drift detection capabilities"""
    print("🔍 Drift Detection System Demo")
    print("=" * 50)
    
    detector = ModelDriftDetector()
    
    # Generate synthetic data for demo
    np.random.seed(42)
    
    # Baseline data (stable period)
    baseline_features = np.random.normal(0, 1, (1000, 10))
    baseline_actuals = np.random.normal(100, 10, 1000)
    baseline_predictions = baseline_actuals + np.random.normal(0, 2, 1000)
    
    # Store baseline
    detector.store_baseline_snapshot(
        "demo_model", "AAPL", 
        baseline_features, baseline_predictions, baseline_actuals
    )
    
    print("✅ Baseline snapshot stored")
    
    # Current data with drift
    current_features = np.random.normal(0.5, 1.2, (200, 10))  # Mean shift + variance increase
    current_actuals = np.random.normal(105, 12, 200)  # Price increase + volatility
    current_predictions = current_actuals + np.random.normal(0, 4, 200)  # Worse predictions
    
    # Run drift detection
    print("\n🔍 Running drift detection...")
    drift_metrics = detector.detect_drift(
        "demo_model", "AAPL",
        current_features, current_predictions, current_actuals
    )
    
    print(f"\n📊 Drift Analysis Results:")
    print(f"   Status: {drift_metrics.drift_status}")
    print(f"   PSI Score: {drift_metrics.psi_score:.3f}")
    print(f"   KS p-value: {drift_metrics.ks_p_value:.3f}")
    print(f"   RMSE Drift: {drift_metrics.rmse_drift:.1%}")
    print(f"   Overall Score: {drift_metrics.overall_drift_score:.3f}")
    
    # Check alerts
    alerts = detector.get_active_alerts("demo_model", "AAPL")
    print(f"\n🚨 Active Alerts: {len(alerts)}")
    for alert in alerts:
        print(f"   {alert.severity}: {alert.description}")
    
    # Get summary
    summary = detector.get_drift_summary("demo_model", "AAPL")
    print(f"\n📈 Drift Summary:")
    print(f"   Status: {summary['status']}")
    print(f"   Requires Attention: {summary['requires_attention']}")
    print(f"   Critical Alerts: {summary['critical_alerts']}")

if __name__ == "__main__":
    asyncio.run(demo_drift_detection())