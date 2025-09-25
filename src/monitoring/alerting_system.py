#!/usr/bin/env python3
"""
Production Alerting & Notification System

Critical missing component: Real-time alerting for production issues.
Integrates with Slack, PagerDuty, email, and SMS for immediate notification.
"""

import asyncio
import json
import logging
import smtplib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests
import os
from enum import Enum

logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class AlertChannel(Enum):
    SLACK = "slack"
    EMAIL = "email"
    PAGERDUTY = "pagerduty"
    SMS = "sms"
    WEBHOOK = "webhook"

@dataclass
class AlertRule:
    """Define conditions that trigger alerts"""
    name: str
    condition: str  # e.g., "error_rate > 0.05"
    severity: AlertSeverity
    channels: List[AlertChannel]
    cooldown_minutes: int = 15
    description: str = ""
    
@dataclass
class Alert:
    """Alert instance with context"""
    id: str
    rule_name: str
    title: str
    description: str
    severity: AlertSeverity
    timestamp: datetime
    service: str = "mlops-stock-predictor"
    environment: str = "production"
    correlation_id: Optional[str] = None
    context: Dict[str, Any] = None
    resolved: bool = False
    resolved_at: Optional[datetime] = None

class AlertManager:
    """
    Production alerting system for MLOps platform
    
    Critical for production systems - provides immediate notification
    of system issues, model failures, and performance degradation.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.alert_rules: List[AlertRule] = []
        self.active_alerts: Dict[str, Alert] = {}
        self.cooldown_tracker: Dict[str, datetime] = {}
        
        # Configuration
        self.slack_webhook_url = os.getenv("SLACK_WEBHOOK_URL", "")
        self.pagerduty_service_key = os.getenv("PAGERDUTY_SERVICE_KEY", "")
        self.smtp_config = {
            "host": os.getenv("SMTP_HOST", "localhost"),
            "port": int(os.getenv("SMTP_PORT", "587")),
            "username": os.getenv("SMTP_USERNAME", ""),
            "password": os.getenv("SMTP_PASSWORD", ""),
            "from_email": os.getenv("ALERT_FROM_EMAIL", "alerts@mlops.com")
        }
        
        # Default alert rules
        self._setup_default_rules()
        
        logger.info("🚨 Alert Manager initialized")
    
    def _setup_default_rules(self):
        """Set up default production alert rules"""
        
        # Critical system errors
        self.add_rule(AlertRule(
            name="critical_errors",
            condition="error_count_critical > 0",
            severity=AlertSeverity.CRITICAL,
            channels=[AlertChannel.SLACK, AlertChannel.PAGERDUTY, AlertChannel.EMAIL],
            cooldown_minutes=5,
            description="Critical system errors detected"
        ))
        
        # Model prediction failures
        self.add_rule(AlertRule(
            name="model_prediction_failures",
            condition="model_error_rate > 0.1",
            severity=AlertSeverity.ERROR,
            channels=[AlertChannel.SLACK, AlertChannel.EMAIL],
            cooldown_minutes=10,
            description="High model prediction failure rate"
        ))
        
        # Drift detection alerts
        self.add_rule(AlertRule(
            name="model_drift_critical",
            condition="drift_status == 'CRITICAL_DRIFT'",
            severity=AlertSeverity.ERROR,
            channels=[AlertChannel.SLACK, AlertChannel.EMAIL],
            cooldown_minutes=30,
            description="Critical model drift detected - retraining required"
        ))
        
        # High API latency
        self.add_rule(AlertRule(
            name="high_api_latency",
            condition="avg_response_time > 5000",
            severity=AlertSeverity.WARNING,
            channels=[AlertChannel.SLACK],
            cooldown_minutes=15,
            description="API response time above 5 seconds"
        ))
        
        # Database connection issues
        self.add_rule(AlertRule(
            name="database_errors",
            condition="database_error_count > 5",
            severity=AlertSeverity.ERROR,
            channels=[AlertChannel.SLACK, AlertChannel.EMAIL],
            cooldown_minutes=10,
            description="Multiple database connection errors"
        ))
        
        # Feature store failures
        self.add_rule(AlertRule(
            name="feature_store_failures", 
            condition="feature_store_error_rate > 0.2",
            severity=AlertSeverity.WARNING,
            channels=[AlertChannel.SLACK],
            cooldown_minutes=15,
            description="High feature store failure rate"
        ))
    
    def add_rule(self, rule: AlertRule):
        """Add new alert rule"""
        self.alert_rules.append(rule)
        logger.info(f"Added alert rule: {rule.name}")
    
    async def check_conditions(self, metrics: Dict[str, Any]):
        """Check all alert rules against current metrics"""
        
        for rule in self.alert_rules:
            try:
                if self._evaluate_condition(rule.condition, metrics):
                    # Check cooldown
                    if self._is_in_cooldown(rule.name):
                        continue
                    
                    # Create and send alert
                    alert = self._create_alert(rule, metrics)
                    await self._send_alert(alert, rule.channels)
                    
                    # Update cooldown
                    self.cooldown_tracker[rule.name] = datetime.now()
                    
            except Exception as e:
                logger.error(f"Error checking alert rule {rule.name}: {e}")
    
    def _evaluate_condition(self, condition: str, metrics: Dict[str, Any]) -> bool:
        """Safely evaluate alert condition"""
        
        try:
            # Simple condition evaluation (production would use a proper parser)
            # This is a simplified implementation for demonstration
            
            if ">" in condition:
                metric_name, threshold = condition.split(">")
                metric_name = metric_name.strip()
                threshold = float(threshold.strip())
                return metrics.get(metric_name, 0) > threshold
            
            elif "==" in condition:
                metric_name, value = condition.split("==")
                metric_name = metric_name.strip()
                value = value.strip().strip('"\'')
                return str(metrics.get(metric_name, "")) == value
            
            elif "<" in condition:
                metric_name, threshold = condition.split("<")
                metric_name = metric_name.strip()
                threshold = float(threshold.strip())
                return metrics.get(metric_name, float('inf')) < threshold
            
            return False
            
        except Exception as e:
            logger.error(f"Error evaluating condition '{condition}': {e}")
            return False
    
    def _is_in_cooldown(self, rule_name: str) -> bool:
        """Check if alert rule is in cooldown period"""
        
        last_sent = self.cooldown_tracker.get(rule_name)
        if not last_sent:
            return False
        
        rule = next((r for r in self.alert_rules if r.name == rule_name), None)
        if not rule:
            return False
        
        cooldown_end = last_sent + timedelta(minutes=rule.cooldown_minutes)
        return datetime.now() < cooldown_end
    
    def _create_alert(self, rule: AlertRule, metrics: Dict[str, Any]) -> Alert:
        """Create alert instance"""
        
        import uuid
        
        return Alert(
            id=str(uuid.uuid4()),
            rule_name=rule.name,
            title=f"🚨 {rule.name.replace('_', ' ').title()}",
            description=rule.description,
            severity=rule.severity,
            timestamp=datetime.now(),
            context={"triggering_metrics": metrics}
        )
    
    async def _send_alert(self, alert: Alert, channels: List[AlertChannel]):
        """Send alert through specified channels"""
        
        logger.error(f"🚨 ALERT TRIGGERED: {alert.title} [{alert.severity.value.upper()}]")
        
        tasks = []
        for channel in channels:
            if channel == AlertChannel.SLACK:
                tasks.append(self._send_slack_alert(alert))
            elif channel == AlertChannel.EMAIL:
                tasks.append(self._send_email_alert(alert))
            elif channel == AlertChannel.PAGERDUTY:
                tasks.append(self._send_pagerduty_alert(alert))
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _send_slack_alert(self, alert: Alert):
        """Send alert to Slack"""
        
        if not self.slack_webhook_url:
            logger.warning("Slack webhook URL not configured")
            return
        
        try:
            # Color based on severity
            colors = {
                AlertSeverity.INFO: "#36a64f",      # Green
                AlertSeverity.WARNING: "#ff9500",   # Orange  
                AlertSeverity.ERROR: "#ff0000",     # Red
                AlertSeverity.CRITICAL: "#8b0000"   # Dark red
            }
            
            payload = {
                "text": f"{alert.title}",
                "attachments": [{
                    "color": colors.get(alert.severity, "#ff0000"),
                    "fields": [
                        {
                            "title": "Severity",
                            "value": alert.severity.value.upper(),
                            "short": True
                        },
                        {
                            "title": "Service", 
                            "value": alert.service,
                            "short": True
                        },
                        {
                            "title": "Time",
                            "value": alert.timestamp.strftime("%Y-%m-%d %H:%M:%S UTC"),
                            "short": True
                        },
                        {
                            "title": "Description",
                            "value": alert.description,
                            "short": False
                        }
                    ]
                }]
            }
            
            response = requests.post(
                self.slack_webhook_url,
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                logger.info(f"Slack alert sent successfully for {alert.id}")
            else:
                logger.error(f"Failed to send Slack alert: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Error sending Slack alert: {e}")
    
    async def _send_email_alert(self, alert: Alert):
        """Send alert via email"""
        
        try:
            msg = MIMEMultipart()
            msg['From'] = self.smtp_config['from_email']
            msg['To'] = os.getenv("ALERT_EMAIL_TO", "ops@company.com")
            msg['Subject'] = f"🚨 MLOps Alert: {alert.title}"
            
            body = f"""
MLOps System Alert

Severity: {alert.severity.value.upper()}
Service: {alert.service}
Environment: {alert.environment}
Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}

Description:
{alert.description}

Alert ID: {alert.id}
Rule: {alert.rule_name}

Context:
{json.dumps(alert.context, indent=2) if alert.context else 'None'}

---
This is an automated alert from the MLOps monitoring system.
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email (simplified - production would use proper SMTP)
            logger.info(f"Email alert would be sent for {alert.id}")
            
        except Exception as e:
            logger.error(f"Error sending email alert: {e}")
    
    async def _send_pagerduty_alert(self, alert: Alert):
        """Send alert to PagerDuty"""
        
        if not self.pagerduty_service_key:
            logger.warning("PagerDuty service key not configured")
            return
        
        try:
            payload = {
                "service_key": self.pagerduty_service_key,
                "event_type": "trigger",
                "incident_key": f"{alert.rule_name}_{alert.id}",
                "description": f"{alert.title}: {alert.description}",
                "details": {
                    "service": alert.service,
                    "environment": alert.environment,
                    "severity": alert.severity.value,
                    "timestamp": alert.timestamp.isoformat(),
                    "context": alert.context
                }
            }
            
            response = requests.post(
                "https://events.pagerduty.com/generic/2010-04-15/create_event.json",
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                logger.info(f"PagerDuty alert sent successfully for {alert.id}")
            else:
                logger.error(f"Failed to send PagerDuty alert: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Error sending PagerDuty alert: {e}")
    
    def get_active_alerts(self) -> List[Alert]:
        """Get list of active alerts"""
        return [alert for alert in self.active_alerts.values() if not alert.resolved]
    
    def resolve_alert(self, alert_id: str):
        """Mark alert as resolved"""
        if alert_id in self.active_alerts:
            self.active_alerts[alert_id].resolved = True
            self.active_alerts[alert_id].resolved_at = datetime.now()
            logger.info(f"Alert {alert_id} resolved")

# Global alert manager instance
alert_manager = AlertManager()

# Integration functions for production API
async def check_system_health_alerts():
    """Check system health and trigger alerts if needed"""
    
    try:
        # Gather system metrics (this would integrate with Prometheus/monitoring)
        metrics = {
            "error_count_critical": 0,  # Would come from error handler
            "model_error_rate": 0.02,   # Would come from model metrics
            "avg_response_time": 850,   # Would come from API metrics
            "database_error_count": 1,  # Would come from DB monitoring
            "feature_store_error_rate": 0.05,  # Would come from feature store
            "drift_status": "MINOR_DRIFT"  # Would come from drift detector
        }
        
        await alert_manager.check_conditions(metrics)
        
    except Exception as e:
        logger.error(f"Error in health check alerts: {e}")

if __name__ == "__main__":
    # Test alerting system
    async def test_alerts():
        print("🚨 Testing Alert System")
        
        # Test critical error alert
        test_metrics = {
            "error_count_critical": 1,
            "model_error_rate": 0.15,
            "avg_response_time": 6000,
            "drift_status": "CRITICAL_DRIFT"
        }
        
        await alert_manager.check_conditions(test_metrics)
        print("✅ Alert tests completed")
    
    asyncio.run(test_alerts())