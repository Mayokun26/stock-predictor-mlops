#!/usr/bin/env python3
"""
Production Performance Optimization and Monitoring
Caching strategies, connection pooling, async optimization, and performance profiling
"""

import asyncio
import time
import functools
import hashlib
import json
import pickle
from typing import Dict, List, Optional, Any, Callable, Union
from datetime import datetime, timedelta
import logging
import psutil
import threading
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
import gc

import redis
import aioredis
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool
import asyncpg

from .logging_config import log_performance_metric

logger = logging.getLogger("mlops.performance")

@dataclass
class PerformanceMetrics:
    """Performance metrics tracking"""
    
    operation: str
    start_time: float
    end_time: Optional[float] = None
    duration_ms: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    cpu_usage_percent: Optional[float] = None
    cache_hits: int = 0
    cache_misses: int = 0
    database_queries: int = 0
    api_calls: int = 0
    context: Dict[str, Any] = field(default_factory=dict)
    
    def complete(self):
        """Mark operation as complete and calculate metrics"""
        self.end_time = time.time()
        self.duration_ms = round((self.end_time - self.start_time) * 1000, 2)
        
        # Get memory usage
        process = psutil.Process()
        self.memory_usage_mb = round(process.memory_info().rss / 1024 / 1024, 2)
        self.cpu_usage_percent = round(process.cpu_percent(), 2)
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            "operation": self.operation,
            "duration_ms": self.duration_ms,
            "memory_usage_mb": self.memory_usage_mb,
            "cpu_usage_percent": self.cpu_usage_percent,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "database_queries": self.database_queries,
            "api_calls": self.api_calls,
            "context": self.context
        }

class PerformanceProfiler:
    """Performance profiling and monitoring"""
    
    def __init__(self):
        self.metrics: List[PerformanceMetrics] = []
        self.active_operations: Dict[str, PerformanceMetrics] = {}
        self.lock = threading.Lock()
        
    def start_operation(self, operation: str, **context) -> str:
        """Start tracking an operation"""
        operation_id = f"{operation}_{int(time.time() * 1000000)}"
        
        metrics = PerformanceMetrics(
            operation=operation,
            start_time=time.time(),
            context=context
        )
        
        with self.lock:
            self.active_operations[operation_id] = metrics
        
        return operation_id
    
    def end_operation(self, operation_id: str) -> Optional[PerformanceMetrics]:
        """End tracking an operation"""
        with self.lock:
            metrics = self.active_operations.pop(operation_id, None)
        
        if metrics:
            metrics.complete()
            self.metrics.append(metrics)
            
            # Log performance metric
            log_performance_metric(
                operation=metrics.operation,
                duration_ms=metrics.duration_ms,
                memory_usage_mb=metrics.memory_usage_mb,
                cpu_usage_percent=metrics.cpu_usage_percent,
                **metrics.context
            )
            
            return metrics
        
        return None
    
    def get_metrics_summary(self, operation: str = None, last_n_minutes: int = 60) -> Dict[str, Any]:
        """Get performance metrics summary"""
        cutoff_time = time.time() - (last_n_minutes * 60)
        
        relevant_metrics = [
            m for m in self.metrics 
            if (not operation or m.operation == operation) and 
               (m.end_time and m.end_time >= cutoff_time)
        ]
        
        if not relevant_metrics:
            return {"count": 0}
        
        durations = [m.duration_ms for m in relevant_metrics if m.duration_ms]
        memory_usage = [m.memory_usage_mb for m in relevant_metrics if m.memory_usage_mb]
        
        return {
            "count": len(relevant_metrics),
            "avg_duration_ms": round(sum(durations) / len(durations), 2) if durations else 0,
            "min_duration_ms": min(durations) if durations else 0,
            "max_duration_ms": max(durations) if durations else 0,
            "avg_memory_mb": round(sum(memory_usage) / len(memory_usage), 2) if memory_usage else 0,
            "total_cache_hits": sum(m.cache_hits for m in relevant_metrics),
            "total_cache_misses": sum(m.cache_misses for m in relevant_metrics),
            "cache_hit_rate": self._calculate_cache_hit_rate(relevant_metrics)
        }
    
    def _calculate_cache_hit_rate(self, metrics: List[PerformanceMetrics]) -> float:
        """Calculate cache hit rate"""
        total_hits = sum(m.cache_hits for m in metrics)
        total_misses = sum(m.cache_misses for m in metrics)
        total_requests = total_hits + total_misses
        
        if total_requests == 0:
            return 0.0
        
        return round((total_hits / total_requests) * 100, 2)

# Global profiler instance
profiler = PerformanceProfiler()

class AdvancedCache:
    """Advanced caching with TTL, versioning, and performance tracking"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url
        self.redis_client = None
        self.cache_stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "deletes": 0
        }
    
    async def initialize(self):
        """Initialize async Redis connection"""
        try:
            self.redis_client = aioredis.from_url(self.redis_url, decode_responses=True)
            await self.redis_client.ping()
            logger.info("Advanced cache initialized with Redis")
        except Exception as e:
            logger.warning(f"Redis connection failed, using memory cache: {e}")
            self.redis_client = {}  # Fallback to dict for development
    
    def _generate_key(self, namespace: str, key: str, version: str = "v1") -> str:
        """Generate cache key with namespace and version"""
        return f"{namespace}:{version}:{key}"
    
    async def get(self, namespace: str, key: str, version: str = "v1") -> Optional[Any]:
        """Get value from cache"""
        cache_key = self._generate_key(namespace, key, version)
        
        try:
            if isinstance(self.redis_client, dict):
                # Memory cache fallback
                value = self.redis_client.get(cache_key)
            else:
                value = await self.redis_client.get(cache_key)
            
            if value:
                self.cache_stats["hits"] += 1
                
                # Try to deserialize JSON first, then pickle
                try:
                    return json.loads(value)
                except (json.JSONDecodeError, TypeError):
                    return pickle.loads(value.encode()) if isinstance(value, str) else pickle.loads(value)
            
            self.cache_stats["misses"] += 1
            return None
            
        except Exception as e:
            logger.error(f"Cache get error for key {cache_key}: {e}")
            self.cache_stats["misses"] += 1
            return None
    
    async def set(self, namespace: str, key: str, value: Any, ttl: int = 3600, version: str = "v1"):
        """Set value in cache with TTL"""
        cache_key = self._generate_key(namespace, key, version)
        
        try:
            # Serialize value
            if isinstance(value, (dict, list)):
                serialized_value = json.dumps(value)
            else:
                serialized_value = pickle.dumps(value)
            
            if isinstance(self.redis_client, dict):
                # Memory cache fallback
                self.redis_client[cache_key] = serialized_value
            else:
                await self.redis_client.setex(cache_key, ttl, serialized_value)
            
            self.cache_stats["sets"] += 1
            
        except Exception as e:
            logger.error(f"Cache set error for key {cache_key}: {e}")
    
    async def delete(self, namespace: str, key: str, version: str = "v1"):
        """Delete value from cache"""
        cache_key = self._generate_key(namespace, key, version)
        
        try:
            if isinstance(self.redis_client, dict):
                self.redis_client.pop(cache_key, None)
            else:
                await self.redis_client.delete(cache_key)
            
            self.cache_stats["deletes"] += 1
            
        except Exception as e:
            logger.error(f"Cache delete error for key {cache_key}: {e}")
    
    async def clear_namespace(self, namespace: str, version: str = "v1"):
        """Clear all keys in a namespace"""
        pattern = f"{namespace}:{version}:*"
        
        try:
            if isinstance(self.redis_client, dict):
                keys_to_delete = [k for k in self.redis_client.keys() if k.startswith(f"{namespace}:{version}:")]
                for k in keys_to_delete:
                    del self.redis_client[k]
            else:
                keys = await self.redis_client.keys(pattern)
                if keys:
                    await self.redis_client.delete(*keys)
            
        except Exception as e:
            logger.error(f"Cache clear error for namespace {namespace}: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.cache_stats["hits"] + self.cache_stats["misses"]
        hit_rate = (self.cache_stats["hits"] / total_requests * 100) if total_requests > 0 else 0
        
        return {
            **self.cache_stats,
            "hit_rate_percent": round(hit_rate, 2),
            "total_requests": total_requests
        }

# Global cache instance
cache = AdvancedCache()

def cached(namespace: str, ttl: int = 3600, version: str = "v1", key_func: Optional[Callable] = None):
    """Decorator for caching function results"""
    
    def decorator(func: Callable):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                # Generate key from function name and arguments
                arg_str = str(args) + str(sorted(kwargs.items()))
                cache_key = f"{func.__name__}:{hashlib.md5(arg_str.encode()).hexdigest()}"
            
            # Try to get from cache
            cached_result = await cache.get(namespace, cache_key, version)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = await func(*args, **kwargs)
            await cache.set(namespace, cache_key, result, ttl, version)
            
            return result
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            # For sync functions, we need to handle caching differently
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                arg_str = str(args) + str(sorted(kwargs.items()))
                cache_key = f"{func.__name__}:{hashlib.md5(arg_str.encode()).hexdigest()}"
            
            # Simple synchronous cache check (fallback)
            # In production, this would use a sync Redis client or memory cache
            result = func(*args, **kwargs)
            return result
        
        # Return appropriate wrapper based on function type
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator

class ConnectionPoolManager:
    """Manages database connection pools for optimal performance"""
    
    def __init__(self):
        self.postgres_pool = None
        self.redis_pool = None
        
    async def initialize_postgres_pool(
        self,
        database_url: str,
        min_size: int = 10,
        max_size: int = 20,
        command_timeout: int = 60
    ):
        """Initialize PostgreSQL connection pool"""
        try:
            self.postgres_pool = await asyncpg.create_pool(
                database_url,
                min_size=min_size,
                max_size=max_size,
                command_timeout=command_timeout,
                server_settings={
                    'application_name': 'mlops_stock_predictor',
                    'timezone': 'UTC'
                }
            )
            logger.info(f"PostgreSQL connection pool initialized: {min_size}-{max_size} connections")
        except Exception as e:
            logger.error(f"Failed to initialize PostgreSQL pool: {e}")
            raise
    
    async def initialize_redis_pool(self, redis_url: str, max_connections: int = 20):
        """Initialize Redis connection pool"""
        try:
            self.redis_pool = aioredis.ConnectionPool.from_url(
                redis_url,
                max_connections=max_connections,
                decode_responses=True
            )
            logger.info(f"Redis connection pool initialized: {max_connections} max connections")
        except Exception as e:
            logger.error(f"Failed to initialize Redis pool: {e}")
            raise
    
    @asynccontextmanager
    async def get_postgres_connection(self):
        """Get PostgreSQL connection from pool"""
        if not self.postgres_pool:
            raise RuntimeError("PostgreSQL pool not initialized")
        
        async with self.postgres_pool.acquire() as connection:
            yield connection
    
    async def get_redis_connection(self):
        """Get Redis connection from pool"""
        if not self.redis_pool:
            raise RuntimeError("Redis pool not initialized")
        
        return aioredis.Redis(connection_pool=self.redis_pool)
    
    async def close_pools(self):
        """Close all connection pools"""
        if self.postgres_pool:
            await self.postgres_pool.close()
            logger.info("PostgreSQL pool closed")
        
        if self.redis_pool:
            await self.redis_pool.disconnect()
            logger.info("Redis pool closed")

# Global connection pool manager
pool_manager = ConnectionPoolManager()

class BatchProcessor:
    """Batch processing for improved performance"""
    
    def __init__(self, batch_size: int = 100, max_wait_time: float = 1.0):
        self.batch_size = batch_size
        self.max_wait_time = max_wait_time
        self.batches: Dict[str, List[Any]] = {}
        self.batch_timers: Dict[str, float] = {}
        self.processors: Dict[str, Callable] = {}
        
    def register_processor(self, batch_type: str, processor_func: Callable):
        """Register a batch processor function"""
        self.processors[batch_type] = processor_func
    
    async def add_to_batch(self, batch_type: str, item: Any):
        """Add item to batch for processing"""
        if batch_type not in self.batches:
            self.batches[batch_type] = []
            self.batch_timers[batch_type] = time.time()
        
        self.batches[batch_type].append(item)
        
        # Process batch if size threshold reached or max wait time exceeded
        should_process = (
            len(self.batches[batch_type]) >= self.batch_size or
            time.time() - self.batch_timers[batch_type] >= self.max_wait_time
        )
        
        if should_process:
            await self._process_batch(batch_type)
    
    async def _process_batch(self, batch_type: str):
        """Process a batch of items"""
        if batch_type not in self.batches or not self.batches[batch_type]:
            return
        
        batch = self.batches[batch_type]
        self.batches[batch_type] = []
        self.batch_timers[batch_type] = time.time()
        
        if batch_type in self.processors:
            try:
                await self.processors[batch_type](batch)
            except Exception as e:
                logger.error(f"Batch processing error for {batch_type}: {e}")
        else:
            logger.warning(f"No processor registered for batch type: {batch_type}")
    
    async def flush_all_batches(self):
        """Process all pending batches"""
        batch_types = list(self.batches.keys())
        for batch_type in batch_types:
            if self.batches[batch_type]:
                await self._process_batch(batch_type)

# Global batch processor
batch_processor = BatchProcessor()

def performance_monitor(operation_name: str = None):
    """Decorator to monitor function performance"""
    
    def decorator(func: Callable):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            op_name = operation_name or f"{func.__module__}.{func.__name__}"
            operation_id = profiler.start_operation(op_name, function=func.__name__)
            
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                profiler.end_operation(operation_id)
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            op_name = operation_name or f"{func.__module__}.{func.__name__}"
            operation_id = profiler.start_operation(op_name, function=func.__name__)
            
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                profiler.end_operation(operation_id)
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator

class ResourceMonitor:
    """Monitor system resources"""
    
    def __init__(self, check_interval: int = 30):
        self.check_interval = check_interval
        self.monitoring = False
        self.stats_history = []
        
    async def start_monitoring(self):
        """Start resource monitoring"""
        self.monitoring = True
        
        while self.monitoring:
            stats = self.get_current_stats()
            self.stats_history.append(stats)
            
            # Keep only last 100 entries
            if len(self.stats_history) > 100:
                self.stats_history.pop(0)
            
            # Log resource usage
            log_performance_metric(
                operation="resource_monitoring",
                duration_ms=0,
                memory_usage_mb=stats["memory_mb"],
                cpu_usage_percent=stats["cpu_percent"],
                disk_usage_percent=stats["disk_percent"],
                network_connections=stats["connections"]
            )
            
            await asyncio.sleep(self.check_interval)
    
    def stop_monitoring(self):
        """Stop resource monitoring"""
        self.monitoring = False
    
    def get_current_stats(self) -> Dict[str, Any]:
        """Get current system resource statistics"""
        process = psutil.Process()
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "cpu_percent": round(process.cpu_percent(), 2),
            "memory_mb": round(process.memory_info().rss / 1024 / 1024, 2),
            "memory_percent": round(process.memory_percent(), 2),
            "disk_usage_percent": round(psutil.disk_usage('/').percent, 2),
            "connections": len(process.connections()),
            "threads": process.num_threads(),
            "file_descriptors": process.num_fds() if hasattr(process, 'num_fds') else 0
        }
    
    def get_stats_summary(self, last_n_minutes: int = 60) -> Dict[str, Any]:
        """Get resource usage summary"""
        if not self.stats_history:
            return {}
        
        cutoff_time = datetime.utcnow() - timedelta(minutes=last_n_minutes)
        
        relevant_stats = [
            s for s in self.stats_history
            if datetime.fromisoformat(s["timestamp"]) >= cutoff_time
        ]
        
        if not relevant_stats:
            return {}
        
        cpu_values = [s["cpu_percent"] for s in relevant_stats]
        memory_values = [s["memory_mb"] for s in relevant_stats]
        
        return {
            "avg_cpu_percent": round(sum(cpu_values) / len(cpu_values), 2),
            "max_cpu_percent": max(cpu_values),
            "avg_memory_mb": round(sum(memory_values) / len(memory_values), 2),
            "max_memory_mb": max(memory_values),
            "sample_count": len(relevant_stats)
        }

# Global resource monitor
resource_monitor = ResourceMonitor()

async def optimize_garbage_collection():
    """Optimize garbage collection for better performance"""
    # Force garbage collection
    collected = gc.collect()
    
    # Log GC stats
    stats = gc.get_stats()
    log_performance_metric(
        operation="garbage_collection",
        duration_ms=0,
        objects_collected=collected,
        gc_stats=stats
    )
    
    logger.info(f"Garbage collection completed: {collected} objects collected")

async def warm_up_caches():
    """Warm up caches with commonly used data"""
    logger.info("Starting cache warm-up process")
    
    # This would be customized based on application needs
    # Example: pre-load common stock symbols, model features, etc.
    
    common_symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]
    
    for symbol in common_symbols:
        # Pre-load feature data into cache
        cache_key = f"features_{symbol}"
        mock_features = {
            "symbol": symbol,
            "rsi_14": 50.0,
            "sma_20": 100.0,
            "volatility": 0.02,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        await cache.set("features", cache_key, mock_features, ttl=1800)
    
    logger.info(f"Cache warm-up completed for {len(common_symbols)} symbols")

async def initialize_performance_optimizations():
    """Initialize all performance optimization components"""
    logger.info("Initializing performance optimization systems...")
    
    # Initialize cache
    await cache.initialize()
    
    # Initialize connection pools (if database URLs are provided)
    # await pool_manager.initialize_postgres_pool("postgresql://...")
    # await pool_manager.initialize_redis_pool("redis://...")
    
    # Start resource monitoring
    asyncio.create_task(resource_monitor.start_monitoring())
    
    # Warm up caches
    await warm_up_caches()
    
    # Initial garbage collection
    await optimize_garbage_collection()
    
    logger.info("Performance optimization systems initialized")

if __name__ == "__main__":
    # Example usage and testing
    
    @performance_monitor("test_operation")
    @cached("test", ttl=60)
    async def test_cached_function(value: int):
        """Test function with caching and performance monitoring"""
        await asyncio.sleep(0.1)  # Simulate work
        return value * 2
    
    async def main():
        await initialize_performance_optimizations()
        
        # Test caching
        result1 = await test_cached_function(5)
        result2 = await test_cached_function(5)  # Should be cached
        
        print(f"Results: {result1}, {result2}")
        print(f"Cache stats: {cache.get_stats()}")
        print(f"Performance summary: {profiler.get_metrics_summary()}")
        
        # Clean up
        resource_monitor.stop_monitoring()
    
    asyncio.run(main())