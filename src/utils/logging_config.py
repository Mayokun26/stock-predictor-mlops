#!/usr/bin/env python3
"""
Production-Grade Logging Configuration
Structured logging with correlation IDs, performance tracking, and security monitoring
"""

import logging
import logging.handlers
import json
import time
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, Optional
import contextvars
import traceback
import sys
from pathlib import Path
import os

# Context variables for request correlation
correlation_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar('correlation_id', default=None)
user_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar('user_id', default=None)
request_start_time: contextvars.ContextVar[Optional[float]] = contextvars.ContextVar('request_start_time', default=None)

class StructuredFormatter(logging.Formatter):
    """
    Custom formatter for structured JSON logging with correlation IDs
    """
    
    def __init__(self, service_name: str = "mlops-stock-predictor", environment: str = "production"):
        super().__init__()
        self.service_name = service_name
        self.environment = environment
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON"""
        
        # Base log structure
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "service": self.service_name,
            "environment": self.environment,
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "thread": record.thread,
            "process": record.process
        }
        
        # Add correlation tracking
        corr_id = correlation_id.get()
        if corr_id:
            log_entry["correlation_id"] = corr_id
            
        user = user_id.get()
        if user:
            log_entry["user_id"] = user
        
        # Add request timing if available
        start_time = request_start_time.get()
        if start_time:
            log_entry["request_duration_ms"] = round((time.time() - start_time) * 1000, 2)
        
        # Add exception information
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": self.formatException(record.exc_info)
            }
        
        # Add extra fields from LogRecord
        extra_fields = {}
        for key, value in record.__dict__.items():
            if key not in {
                'name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 'filename',
                'module', 'exc_info', 'exc_text', 'stack_info', 'lineno', 'funcName',
                'created', 'msecs', 'relativeCreated', 'thread', 'threadName',
                'processName', 'process', 'getMessage', 'message'
            }:
                if isinstance(value, (str, int, float, bool, list, dict)):
                    extra_fields[key] = value
                else:
                    extra_fields[key] = str(value)
        
        if extra_fields:
            log_entry["extra"] = extra_fields
        
        return json.dumps(log_entry, ensure_ascii=False)


class SecurityAuditFormatter(StructuredFormatter):
    """
    Specialized formatter for security audit logs
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """Format security audit log with additional security context"""
        
        log_entry = json.loads(super().format(record))
        
        # Add security-specific fields
        log_entry["log_type"] = "security_audit"
        
        # Extract security context from record
        if hasattr(record, 'security_event'):
            log_entry["security_event"] = record.security_event
        if hasattr(record, 'client_ip'):
            log_entry["client_ip"] = record.client_ip
        if hasattr(record, 'user_agent'):
            log_entry["user_agent"] = record.user_agent
        if hasattr(record, 'endpoint'):
            log_entry["endpoint"] = record.endpoint
        if hasattr(record, 'http_method'):
            log_entry["http_method"] = record.http_method
        if hasattr(record, 'status_code'):
            log_entry["status_code"] = record.status_code
        
        return json.dumps(log_entry, ensure_ascii=False)


class PerformanceFormatter(StructuredFormatter):
    """
    Specialized formatter for performance monitoring logs
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """Format performance log with metrics"""
        
        log_entry = json.loads(super().format(record))
        
        # Add performance-specific fields
        log_entry["log_type"] = "performance"
        
        # Extract performance metrics from record
        if hasattr(record, 'operation'):
            log_entry["operation"] = record.operation
        if hasattr(record, 'duration_ms'):
            log_entry["duration_ms"] = record.duration_ms
        if hasattr(record, 'memory_usage_mb'):
            log_entry["memory_usage_mb"] = record.memory_usage_mb
        if hasattr(record, 'cpu_usage_percent'):
            log_entry["cpu_usage_percent"] = record.cpu_usage_percent
        if hasattr(record, 'model_version'):
            log_entry["model_version"] = record.model_version
        if hasattr(record, 'prediction_count'):
            log_entry["prediction_count"] = record.prediction_count
        
        return json.dumps(log_entry, ensure_ascii=False)


class BusinessMetricsFormatter(StructuredFormatter):
    """
    Specialized formatter for business metrics logs
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """Format business metrics log"""
        
        log_entry = json.loads(super().format(record))
        
        # Add business metrics specific fields
        log_entry["log_type"] = "business_metrics"
        
        # Extract business context from record
        if hasattr(record, 'metric_name'):
            log_entry["metric_name"] = record.metric_name
        if hasattr(record, 'metric_value'):
            log_entry["metric_value"] = record.metric_value
        if hasattr(record, 'symbol'):
            log_entry["symbol"] = record.symbol
        if hasattr(record, 'portfolio_value'):
            log_entry["portfolio_value"] = record.portfolio_value
        if hasattr(record, 'sharpe_ratio'):
            log_entry["sharpe_ratio"] = record.sharpe_ratio
        if hasattr(record, 'drawdown'):
            log_entry["drawdown"] = record.drawdown
        
        return json.dumps(log_entry, ensure_ascii=False)


def setup_logging(
    service_name: str = "mlops-stock-predictor",
    environment: str = "production", 
    log_level: str = "INFO",
    log_dir: str = "/var/log/mlops"
) -> Dict[str, logging.Logger]:
    """
    Set up comprehensive logging configuration
    
    Args:
        service_name: Name of the service
        environment: Environment (development, staging, production)
        log_level: Logging level
        log_dir: Directory for log files
        
    Returns:
        Dictionary of configured loggers
    """
    
    # Create log directory
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove default handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Main application logger
    app_logger = logging.getLogger("mlops.app")
    app_handler = logging.handlers.RotatingFileHandler(
        filename=f"{log_dir}/application.log",
        maxBytes=100 * 1024 * 1024,  # 100MB
        backupCount=10
    )
    app_handler.setFormatter(StructuredFormatter(service_name, environment))
    app_logger.addHandler(app_handler)
    
    # Console handler for development
    if environment == "development":
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(StructuredFormatter(service_name, environment))
        app_logger.addHandler(console_handler)
    
    # Security audit logger
    security_logger = logging.getLogger("mlops.security")
    security_handler = logging.handlers.RotatingFileHandler(
        filename=f"{log_dir}/security.log",
        maxBytes=50 * 1024 * 1024,  # 50MB
        backupCount=20
    )
    security_handler.setFormatter(SecurityAuditFormatter(service_name, environment))
    security_logger.addHandler(security_handler)
    security_logger.setLevel(logging.INFO)
    
    # Performance logger
    performance_logger = logging.getLogger("mlops.performance")
    performance_handler = logging.handlers.RotatingFileHandler(
        filename=f"{log_dir}/performance.log",
        maxBytes=100 * 1024 * 1024,  # 100MB
        backupCount=15
    )
    performance_handler.setFormatter(PerformanceFormatter(service_name, environment))
    performance_logger.addHandler(performance_handler)
    performance_logger.setLevel(logging.INFO)
    
    # Business metrics logger
    business_logger = logging.getLogger("mlops.business")
    business_handler = logging.handlers.RotatingFileHandler(
        filename=f"{log_dir}/business.log",
        maxBytes=50 * 1024 * 1024,  # 50MB
        backupCount=10
    )
    business_handler.setFormatter(BusinessMetricsFormatter(service_name, environment))
    business_logger.addHandler(business_handler)
    business_logger.setLevel(logging.INFO)
    
    # Error logger (for critical errors)
    error_logger = logging.getLogger("mlops.errors")
    error_handler = logging.handlers.RotatingFileHandler(
        filename=f"{log_dir}/errors.log",
        maxBytes=50 * 1024 * 1024,  # 50MB
        backupCount=30
    )
    error_handler.setFormatter(StructuredFormatter(service_name, environment))
    error_logger.addHandler(error_handler)
    error_logger.setLevel(logging.ERROR)
    
    # Model operations logger
    model_logger = logging.getLogger("mlops.models")
    model_handler = logging.handlers.RotatingFileHandler(
        filename=f"{log_dir}/models.log",
        maxBytes=100 * 1024 * 1024,  # 100MB
        backupCount=10
    )
    model_handler.setFormatter(StructuredFormatter(service_name, environment))
    model_logger.addHandler(model_handler)
    model_logger.setLevel(logging.INFO)
    
    # Set up syslog handler for centralized logging
    if environment == "production":
        try:
            syslog_handler = logging.handlers.SysLogHandler(address='/dev/log')
            syslog_handler.setFormatter(StructuredFormatter(service_name, environment))
            app_logger.addHandler(syslog_handler)
        except Exception:
            # Syslog not available, continue without it
            pass
    
    return {
        "app": app_logger,
        "security": security_logger,
        "performance": performance_logger,
        "business": business_logger,
        "errors": error_logger,
        "models": model_logger
    }


def get_correlation_id() -> str:
    """Get or generate correlation ID for request tracking"""
    corr_id = correlation_id.get()
    if not corr_id:
        corr_id = str(uuid.uuid4())
        correlation_id.set(corr_id)
    return corr_id


def set_user_context(user_id_value: str):
    """Set user context for logging"""
    user_id.set(user_id_value)


def start_request_timer():
    """Start request timing for performance logging"""
    request_start_time.set(time.time())


def log_security_event(
    event_type: str,
    message: str,
    client_ip: Optional[str] = None,
    user_agent: Optional[str] = None,
    endpoint: Optional[str] = None,
    http_method: Optional[str] = None,
    status_code: Optional[int] = None,
    **kwargs
):
    """Log security audit event"""
    security_logger = logging.getLogger("mlops.security")
    
    extra = {
        "security_event": event_type,
        "client_ip": client_ip,
        "user_agent": user_agent,
        "endpoint": endpoint,
        "http_method": http_method,
        "status_code": status_code,
        **kwargs
    }
    
    security_logger.info(message, extra=extra)


def log_performance_metric(
    operation: str,
    duration_ms: float,
    message: str = None,
    **kwargs
):
    """Log performance metric"""
    performance_logger = logging.getLogger("mlops.performance")
    
    extra = {
        "operation": operation,
        "duration_ms": duration_ms,
        **kwargs
    }
    
    msg = message or f"Operation {operation} completed in {duration_ms}ms"
    performance_logger.info(msg, extra=extra)


def log_business_metric(
    metric_name: str,
    metric_value: float,
    message: str = None,
    **kwargs
):
    """Log business metric"""
    business_logger = logging.getLogger("mlops.business")
    
    extra = {
        "metric_name": metric_name,
        "metric_value": metric_value,
        **kwargs
    }
    
    msg = message or f"Business metric {metric_name}: {metric_value}"
    business_logger.info(msg, extra=extra)


def log_model_event(
    event_type: str,
    model_version: str,
    message: str,
    **kwargs
):
    """Log model-related event"""
    model_logger = logging.getLogger("mlops.models")
    
    extra = {
        "event_type": event_type,
        "model_version": model_version,
        **kwargs
    }
    
    model_logger.info(message, extra=extra)


class LoggingMiddleware:
    """
    FastAPI middleware for automatic request logging
    """
    
    def __init__(self):
        self.app_logger = logging.getLogger("mlops.app")
        self.security_logger = logging.getLogger("mlops.security")
        self.performance_logger = logging.getLogger("mlops.performance")
    
    async def __call__(self, request, call_next):
        """Process request with logging"""
        
        # Generate correlation ID
        corr_id = str(uuid.uuid4())
        correlation_id.set(corr_id)
        
        # Start timing
        start_time = time.time()
        request_start_time.set(start_time)
        
        # Extract request information
        client_ip = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("user-agent", "unknown")
        method = request.method
        url = str(request.url)
        
        # Log request start
        self.app_logger.info(
            f"Request started: {method} {url}",
            extra={
                "client_ip": client_ip,
                "user_agent": user_agent,
                "http_method": method,
                "url": url,
                "correlation_id": corr_id
            }
        )
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate duration
            duration_ms = round((time.time() - start_time) * 1000, 2)
            
            # Log successful request
            self.app_logger.info(
                f"Request completed: {method} {url} - {response.status_code}",
                extra={
                    "client_ip": client_ip,
                    "user_agent": user_agent,
                    "http_method": method,
                    "url": url,
                    "status_code": response.status_code,
                    "duration_ms": duration_ms,
                    "correlation_id": corr_id
                }
            )
            
            # Log security events for certain conditions
            if response.status_code >= 400:
                log_security_event(
                    event_type="http_error",
                    message=f"HTTP error {response.status_code} for {method} {url}",
                    client_ip=client_ip,
                    user_agent=user_agent,
                    endpoint=url,
                    http_method=method,
                    status_code=response.status_code
                )
            
            # Log performance metrics
            log_performance_metric(
                operation=f"{method}_{url.split('/')[-1] or 'root'}",
                duration_ms=duration_ms,
                client_ip=client_ip,
                status_code=response.status_code
            )
            
            return response
            
        except Exception as e:
            # Calculate duration
            duration_ms = round((time.time() - start_time) * 1000, 2)
            
            # Log error
            self.app_logger.error(
                f"Request failed: {method} {url} - {str(e)}",
                extra={
                    "client_ip": client_ip,
                    "user_agent": user_agent,
                    "http_method": method,
                    "url": url,
                    "error": str(e),
                    "duration_ms": duration_ms,
                    "correlation_id": corr_id
                },
                exc_info=True
            )
            
            # Log security event for error
            log_security_event(
                event_type="request_exception",
                message=f"Request exception: {str(e)}",
                client_ip=client_ip,
                user_agent=user_agent,
                endpoint=url,
                http_method=method,
                error=str(e)
            )
            
            raise


# Performance monitoring decorators
def log_execution_time(operation_name: str = None):
    """Decorator to log function execution time"""
    
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            op_name = operation_name or f"{func.__module__}.{func.__name__}"
            
            try:
                result = func(*args, **kwargs)
                duration_ms = round((time.time() - start_time) * 1000, 2)
                
                log_performance_metric(
                    operation=op_name,
                    duration_ms=duration_ms,
                    message=f"Function {func.__name__} completed successfully"
                )
                
                return result
                
            except Exception as e:
                duration_ms = round((time.time() - start_time) * 1000, 2)
                
                error_logger = logging.getLogger("mlops.errors")
                error_logger.error(
                    f"Function {func.__name__} failed after {duration_ms}ms: {str(e)}",
                    extra={
                        "operation": op_name,
                        "duration_ms": duration_ms,
                        "function": func.__name__,
                        "error": str(e)
                    },
                    exc_info=True
                )
                
                raise
        
        return wrapper
    return decorator


if __name__ == "__main__":
    # Example usage
    loggers = setup_logging(environment="development")
    
    # Test logging
    app_logger = loggers["app"]
    app_logger.info("Application started", extra={"version": "2.0.0"})
    
    # Test security logging
    log_security_event(
        event_type="login_attempt",
        message="User login successful",
        client_ip="192.168.1.100",
        user_agent="Mozilla/5.0",
        endpoint="/auth/login"
    )
    
    # Test performance logging
    log_performance_metric(
        operation="model_prediction",
        duration_ms=45.2,
        model_version="xgboost_v2.1",
        prediction_count=1
    )
    
    # Test business metric logging
    log_business_metric(
        metric_name="daily_sharpe_ratio",
        metric_value=1.42,
        symbol="AAPL",
        portfolio_value=125000.50
    )