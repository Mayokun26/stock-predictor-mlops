#!/usr/bin/env python3
"""
Production-Grade Error Handling System
Comprehensive error handling, circuit breakers, retry logic, and fault tolerance
"""

import asyncio
import time
import logging
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any, Union
from enum import Enum
from dataclasses import dataclass, field
import functools
from contextlib import asynccontextmanager
import json

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse

class ErrorSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ErrorCategory(Enum):
    VALIDATION = "validation"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    BUSINESS_LOGIC = "business_logic"
    EXTERNAL_SERVICE = "external_service"
    DATABASE = "database"
    NETWORK = "network"
    SYSTEM = "system"
    MODEL_ERROR = "model_error"
    DATA_QUALITY = "data_quality"

@dataclass
class ErrorContext:
    """Context information for error tracking and debugging"""
    
    error_id: str
    timestamp: datetime
    severity: ErrorSeverity
    category: ErrorCategory
    service: str = "mlops-stock-predictor"
    correlation_id: Optional[str] = None
    user_id: Optional[str] = None
    request_path: Optional[str] = None
    model_version: Optional[str] = None
    symbol: Optional[str] = None
    additional_context: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "error_id": self.error_id,
            "timestamp": self.timestamp.isoformat(),
            "severity": self.severity.value,
            "category": self.category.value,
            "service": self.service,
            "correlation_id": self.correlation_id,
            "user_id": self.user_id,
            "request_path": self.request_path,
            "model_version": self.model_version,
            "symbol": self.symbol,
            "additional_context": self.additional_context
        }

class MLOpsError(Exception):
    """Base exception class for MLOps system errors"""
    
    def __init__(
        self,
        message: str,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        category: ErrorCategory = ErrorCategory.SYSTEM,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        super().__init__(message)
        self.message = message
        self.severity = severity
        self.category = category
        self.error_code = error_code
        self.context = context or {}
        self.cause = cause
        self.timestamp = datetime.utcnow()
        
        # Generate unique error ID
        import uuid
        self.error_id = str(uuid.uuid4())

class ModelPredictionError(MLOpsError):
    """Error during model prediction"""
    
    def __init__(self, message: str, model_version: str = None, **kwargs):
        super().__init__(
            message, 
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.MODEL_ERROR,
            **kwargs
        )
        self.model_version = model_version

class FeatureStoreError(MLOpsError):
    """Error accessing feature store"""
    
    def __init__(self, message: str, feature_group: str = None, **kwargs):
        super().__init__(
            message,
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.EXTERNAL_SERVICE,
            **kwargs
        )
        self.feature_group = feature_group

class DataValidationError(MLOpsError):
    """Data validation error"""
    
    def __init__(self, message: str, validation_rule: str = None, **kwargs):
        super().__init__(
            message,
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.VALIDATION,
            **kwargs
        )
        self.validation_rule = validation_rule

class BusinessLogicError(MLOpsError):
    """Business logic validation error"""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.BUSINESS_LOGIC,
            **kwargs
        )

class CircuitBreakerError(MLOpsError):
    """Circuit breaker is open, preventing operation"""
    
    def __init__(self, service_name: str, **kwargs):
        super().__init__(
            f"Circuit breaker open for service: {service_name}",
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.EXTERNAL_SERVICE,
            **kwargs
        )
        self.service_name = service_name

@dataclass
class CircuitBreakerState:
    """State tracking for circuit breaker"""
    
    failure_count: int = 0
    last_failure_time: Optional[datetime] = None
    state: str = "closed"  # closed, open, half_open
    success_count: int = 0
    
class CircuitBreaker:
    """Circuit breaker implementation for fault tolerance"""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: tuple = (Exception,),
        name: str = "default"
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.name = name
        self.state = CircuitBreakerState()
        self.logger = logging.getLogger(f"mlops.circuit_breaker.{name}")
    
    def __call__(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return self._call(func, *args, **kwargs)
        return wrapper
    
    def _call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        
        if self._is_open():
            if self._should_attempt_reset():
                self.state.state = "half_open"
                self.logger.info(f"Circuit breaker {self.name} entering half-open state")
            else:
                self.logger.warning(f"Circuit breaker {self.name} is open, rejecting call")
                raise CircuitBreakerError(self.name)
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
            
        except self.expected_exception as e:
            self._on_failure()
            raise
    
    def _is_open(self) -> bool:
        """Check if circuit breaker is open"""
        return self.state.state == "open"
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset"""
        if not self.state.last_failure_time:
            return False
        
        return (
            datetime.utcnow() - self.state.last_failure_time 
            >= timedelta(seconds=self.recovery_timeout)
        )
    
    def _on_success(self):
        """Handle successful operation"""
        if self.state.state == "half_open":
            self.state.success_count += 1
            if self.state.success_count >= 2:  # Require 2 successes to close
                self._reset()
        elif self.state.state == "closed":
            self.state.failure_count = 0
    
    def _on_failure(self):
        """Handle failed operation"""
        self.state.failure_count += 1
        self.state.last_failure_time = datetime.utcnow()
        
        if self.state.failure_count >= self.failure_threshold:
            self.state.state = "open"
            self.logger.error(
                f"Circuit breaker {self.name} opened after {self.state.failure_count} failures"
            )
    
    def _reset(self):
        """Reset circuit breaker to closed state"""
        self.state.state = "closed"
        self.state.failure_count = 0
        self.state.success_count = 0
        self.state.last_failure_time = None
        self.logger.info(f"Circuit breaker {self.name} reset to closed state")

def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_factor: float = 2.0,
    exceptions: tuple = (Exception,)
):
    """Decorator for exponential backoff retry logic"""
    
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            delay = base_delay
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                    
                except exceptions as e:
                    last_exception = e
                    
                    if attempt == max_retries:
                        logger = logging.getLogger("mlops.errors")
                        logger.error(
                            f"Function {func.__name__} failed after {max_retries + 1} attempts",
                            extra={
                                "function": func.__name__,
                                "attempt": attempt + 1,
                                "error": str(e)
                            },
                            exc_info=True
                        )
                        raise
                    
                    logger = logging.getLogger("mlops.app")
                    logger.warning(
                        f"Function {func.__name__} failed on attempt {attempt + 1}, retrying in {delay}s",
                        extra={
                            "function": func.__name__,
                            "attempt": attempt + 1,
                            "delay": delay,
                            "error": str(e)
                        }
                    )
                    
                    time.sleep(delay)
                    delay = min(delay * backoff_factor, max_delay)
            
            raise last_exception
        
        return wrapper
    return decorator

class ErrorHandler:
    """Centralized error handling and reporting"""
    
    def __init__(self):
        self.logger = logging.getLogger("mlops.errors")
        self.error_stats: Dict[str, int] = {}
        
    def handle_error(
        self,
        error: Exception,
        context: Optional[ErrorContext] = None,
        reraise: bool = True
    ) -> Optional[ErrorContext]:
        """Handle and log error with context"""
        
        # Create error context if not provided
        if context is None:
            if isinstance(error, MLOpsError):
                context = ErrorContext(
                    error_id=error.error_id,
                    timestamp=error.timestamp,
                    severity=error.severity,
                    category=error.category
                )
            else:
                import uuid
                context = ErrorContext(
                    error_id=str(uuid.uuid4()),
                    timestamp=datetime.utcnow(),
                    severity=ErrorSeverity.MEDIUM,
                    category=ErrorCategory.SYSTEM
                )
        
        # Log error with context
        self._log_error(error, context)
        
        # Update error statistics
        self._update_error_stats(context)
        
        # Send alerts for critical errors
        if context.severity == ErrorSeverity.CRITICAL:
            self._send_critical_alert(error, context)
        
        if reraise:
            raise error
        
        return context
    
    def _log_error(self, error: Exception, context: ErrorContext):
        """Log error with structured context"""
        
        error_info = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "error_context": context.to_dict(),
            "traceback": traceback.format_exc()
        }
        
        if isinstance(error, MLOpsError):
            error_info["mlops_error_code"] = error.error_code
            error_info["mlops_context"] = error.context
            error_info["cause"] = str(error.cause) if error.cause else None
        
        # Log at appropriate level based on severity
        if context.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(
                f"CRITICAL ERROR: {error_info['error_message']}",
                extra=error_info
            )
        elif context.severity == ErrorSeverity.HIGH:
            self.logger.error(
                f"HIGH SEVERITY ERROR: {error_info['error_message']}",
                extra=error_info
            )
        elif context.severity == ErrorSeverity.MEDIUM:
            self.logger.warning(
                f"ERROR: {error_info['error_message']}",
                extra=error_info
            )
        else:
            self.logger.info(
                f"LOW SEVERITY ERROR: {error_info['error_message']}",
                extra=error_info
            )
    
    def _update_error_stats(self, context: ErrorContext):
        """Update error statistics for monitoring"""
        
        stat_key = f"{context.category.value}_{context.severity.value}"
        self.error_stats[stat_key] = self.error_stats.get(stat_key, 0) + 1
        
        # Log business metric for error tracking
        from .logging_config import log_business_metric
        log_business_metric(
            metric_name=f"error_count_{stat_key}",
            metric_value=self.error_stats[stat_key],
            error_category=context.category.value,
            severity=context.severity.value
        )
    
    def _send_critical_alert(self, error: Exception, context: ErrorContext):
        """Send alert for critical errors"""
        
        # In production, this would integrate with alerting systems
        # like PagerDuty, Slack, or email notifications
        
        alert_data = {
            "alert_type": "critical_error",
            "service": context.service,
            "error_id": context.error_id,
            "error_message": str(error),
            "timestamp": context.timestamp.isoformat(),
            "context": context.to_dict()
        }
        
        self.logger.critical(
            "CRITICAL ALERT TRIGGERED",
            extra={"alert_data": alert_data}
        )

# Global error handler instance
error_handler = ErrorHandler()

# FastAPI exception handlers
async def mlops_error_handler(request: Request, exc: MLOpsError) -> JSONResponse:
    """Handle MLOps-specific errors"""
    
    error_handler.handle_error(exc, reraise=False)
    
    return JSONResponse(
        status_code=500 if exc.severity == ErrorSeverity.CRITICAL else 400,
        content={
            "error": {
                "id": exc.error_id,
                "message": exc.message,
                "type": type(exc).__name__,
                "severity": exc.severity.value,
                "category": exc.category.value,
                "timestamp": exc.timestamp.isoformat(),
                "code": exc.error_code
            }
        }
    )

async def validation_error_handler(request: Request, exc: ValueError) -> JSONResponse:
    """Handle validation errors"""
    
    validation_error = DataValidationError(
        str(exc),
        context={"request_path": str(request.url.path)}
    )
    
    error_handler.handle_error(validation_error, reraise=False)
    
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "id": validation_error.error_id,
                "message": "Validation failed",
                "details": str(exc),
                "type": "ValidationError",
                "timestamp": validation_error.timestamp.isoformat()
            }
        }
    )

async def general_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle general unexpected errors"""
    
    import uuid
    error_id = str(uuid.uuid4())
    
    context = ErrorContext(
        error_id=error_id,
        timestamp=datetime.utcnow(),
        severity=ErrorSeverity.HIGH,
        category=ErrorCategory.SYSTEM,
        request_path=str(request.url.path)
    )
    
    error_handler.handle_error(exc, context, reraise=False)
    
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "id": error_id,
                "message": "Internal server error",
                "type": "InternalError",
                "timestamp": context.timestamp.isoformat(),
                "correlation_id": context.correlation_id
            }
        }
    )

# Circuit breaker instances for external services
feature_store_circuit_breaker = CircuitBreaker(
    failure_threshold=3,
    recovery_timeout=30,
    name="feature_store"
)

model_registry_circuit_breaker = CircuitBreaker(
    failure_threshold=5,
    recovery_timeout=60,
    name="model_registry"
)

database_circuit_breaker = CircuitBreaker(
    failure_threshold=3,
    recovery_timeout=45,
    name="database"
)

# Health check with circuit breaker status
def get_circuit_breaker_status() -> Dict[str, Dict[str, Any]]:
    """Get status of all circuit breakers"""
    
    breakers = {
        "feature_store": feature_store_circuit_breaker,
        "model_registry": model_registry_circuit_breaker,
        "database": database_circuit_breaker
    }
    
    status = {}
    for name, breaker in breakers.items():
        status[name] = {
            "state": breaker.state.state,
            "failure_count": breaker.state.failure_count,
            "last_failure_time": (
                breaker.state.last_failure_time.isoformat() 
                if breaker.state.last_failure_time else None
            )
        }
    
    return status

# Context manager for error handling
@asynccontextmanager
async def error_context(operation_name: str, **context_kwargs):
    """Context manager for consistent error handling"""
    
    start_time = time.time()
    
    try:
        yield
        
        # Log successful operation
        duration_ms = round((time.time() - start_time) * 1000, 2)
        from .logging_config import log_performance_metric
        log_performance_metric(
            operation=operation_name,
            duration_ms=duration_ms,
            **context_kwargs
        )
        
    except Exception as e:
        # Create error context
        import uuid
        context = ErrorContext(
            error_id=str(uuid.uuid4()),
            timestamp=datetime.utcnow(),
            severity=ErrorSeverity.HIGH if isinstance(e, MLOpsError) else ErrorSeverity.MEDIUM,
            category=e.category if isinstance(e, MLOpsError) else ErrorCategory.SYSTEM,
            additional_context={
                "operation": operation_name,
                "duration_ms": round((time.time() - start_time) * 1000, 2),
                **context_kwargs
            }
        )
        
        # Handle error
        error_handler.handle_error(e, context)

if __name__ == "__main__":
    # Example usage and testing
    
    # Test circuit breaker
    @feature_store_circuit_breaker
    def test_function():
        import random
        if random.random() < 0.7:  # 70% failure rate
            raise Exception("Test failure")
        return "Success"
    
    # Test retry decorator
    @retry_with_backoff(max_retries=2, base_delay=0.1)
    def test_retry_function():
        import random
        if random.random() < 0.5:  # 50% failure rate
            raise Exception("Test retry failure")
        return "Retry success"
    
    # Test error handling
    try:
        raise ModelPredictionError(
            "Test model error",
            model_version="test_v1.0",
            context={"symbol": "AAPL", "feature_count": 15}
        )
    except MLOpsError as e:
        print(f"Caught MLOps error: {e.error_id}")
        
    print("Error handling tests completed")