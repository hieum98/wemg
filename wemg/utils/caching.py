import json
import logging
import os
from typing import Any, Dict, List, Optional
from hashlib import sha256, md5


logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("LOGGING_LEVEL", "INFO"))
try:
    import redis
    from redis import ConnectionPool
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("Redis package not installed. Cache functionality will be disk-cached. Install with: pip install redis")


class RedisCacheManager:
    """
    Redis-based cache manager for LLM requests and responses.
    
    Provides efficient caching with:
    - Connection pooling for better performance
    - Configurable TTL (Time To Live) for cache entries
    - JSON serialization for complex objects
    - Thread-safe operations
    - Graceful fallback when Redis is unavailable
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        ttl: Optional[int] = None,
        prefix: str = "llm_cache",
        max_connections: int = 50,
        socket_timeout: int = 5,
        socket_connect_timeout: int = 5,
        decode_responses: bool = False,
        **redis_kwargs
    ):
        """
        Initialize Redis cache manager.
        
        Args:
            host: Redis server host
            port: Redis server port
            db: Redis database number
            password: Redis password (if required)
            ttl: Default time-to-live for cache entries in seconds (None = no expiration)
            prefix: Prefix for all cache keys to avoid collisions
            max_connections: Maximum number of connections in the pool
            socket_timeout: Socket timeout in seconds
            socket_connect_timeout: Socket connection timeout in seconds
            decode_responses: Whether to decode byte responses to strings
            **redis_kwargs: Additional arguments to pass to Redis connection
        """
        if not REDIS_AVAILABLE:
            self.enabled = False
            logger.warning("Redis not available. Caching is disabled.")
            return
        
        self.enabled = True
        self.ttl = ttl
        self.prefix = prefix
        
        try:
            # Create connection pool for better performance
            self.pool = ConnectionPool(
                host=host,
                port=port,
                db=db,
                password=password,
                max_connections=max_connections,
                socket_timeout=socket_timeout,
                socket_connect_timeout=socket_connect_timeout,
                decode_responses=decode_responses,
                **redis_kwargs
            )
            
            # Create Redis client
            self.client = redis.Redis(connection_pool=self.pool)
            
            # Test connection
            self.client.ping()
            logger.info(f"Redis cache initialized successfully at {host}:{port}")
            
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}. Caching disabled.")
            self.enabled = False
    
    def _make_key(self, request_data: Dict[str, Any]) -> str:
        """
        Generate a unique cache key from request data.
        
        Args:
            request_data: Dictionary containing messages and model parameters
            
        Returns:
            SHA256 hash of the serialized request data
        """
        # Create a deterministic string representation of the request
        # Sort keys to ensure consistent hashing
        serialized = json.dumps(request_data, sort_keys=True, default=str)
        key_hash = sha256(serialized.encode()).hexdigest()
        return f"{self.prefix}:{key_hash}"
    
    def get(self, request_data: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
        """
        Retrieve cached response for a request.
        
        Args:
            request_data: Request parameters to look up
            
        Returns:
            Cached response if found, None otherwise
        """
        if not self.enabled:
            return None
        
        try:
            key = self._make_key(request_data)
            cached = self.client.get(key)
            
            if cached:
                logger.debug(f"Cache HIT for key: {key[:16]}...")
                # Deserialize the cached response
                if isinstance(cached, bytes):
                    cached = cached.decode('utf-8')
                return json.loads(cached)
            else:
                logger.debug(f"Cache MISS for key: {key[:16]}...")
                return None
                
        except Exception as e:
            logger.warning(f"Error retrieving from cache: {e}")
            return None
    
    def set(
        self, 
        request_data: Dict[str, Any], 
        response_data: List[Dict[str, Any]],
        ttl: Optional[int] = None
    ) -> bool:
        """
        Store a response in cache.
        
        Args:
            request_data: Request parameters used as cache key
            response_data: Response to cache
            ttl: Time-to-live in seconds (overrides default)
            
        Returns:
            True if successfully cached, False otherwise
        """
        if not self.enabled:
            return False
        
        try:
            key = self._make_key(request_data)
            
            # Serialize response to JSON
            serialized = json.dumps(response_data, default=str)
            
            # Use provided TTL or default
            expiry = ttl if ttl is not None else self.ttl
            
            if expiry:
                self.client.setex(key, expiry, serialized)
            else:
                self.client.set(key, serialized)
            
            logger.debug(f"Cached response for key: {key[:16]}... (TTL: {expiry})")
            return True
            
        except Exception as e:
            logger.warning(f"Error caching response: {e}")
            return False
    
    def delete(self, request_data: Dict[str, Any]) -> bool:
        """
        Delete a cached entry.
        
        Args:
            request_data: Request parameters to delete
            
        Returns:
            True if deleted, False otherwise
        """
        if not self.enabled:
            return False
        
        try:
            key = self._make_key(request_data)
            deleted = self.client.delete(key)
            return deleted > 0
        except Exception as e:
            logger.warning(f"Error deleting from cache: {e}")
            return False
    
    def clear_all(self, pattern: Optional[str] = None) -> int:
        """
        Clear cache entries matching a pattern.
        
        Args:
            pattern: Pattern to match keys (default: all keys with prefix)
            
        Returns:
            Number of keys deleted
        """
        if not self.enabled:
            return 0
        
        try:
            if pattern is None:
                pattern = f"{self.prefix}:*"
            
            keys = self.client.keys(pattern)
            if keys:
                return self.client.delete(*keys)
            return 0
            
        except Exception as e:
            logger.warning(f"Error clearing cache: {e}")
            return 0
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache stats
        """
        if not self.enabled:
            return {"enabled": False}
        
        try:
            info = self.client.info("stats")
            keyspace = self.client.info("keyspace")
            
            # Count keys with our prefix
            our_keys = len(self.client.keys(f"{self.prefix}:*"))
            
            return {
                "enabled": True,
                "total_keys": our_keys,
                "hits": info.get("keyspace_hits", 0),
                "misses": info.get("keyspace_misses", 0),
                "hit_rate": info.get("keyspace_hits", 0) / max(info.get("keyspace_hits", 0) + info.get("keyspace_misses", 1), 1),
                "memory_used": info.get("used_memory_human", "N/A"),
                "connected_clients": info.get("connected_clients", 0),
            }
        except Exception as e:
            logger.warning(f"Error getting cache stats: {e}")
            return {"enabled": True, "error": str(e)}
    
    def close(self):
        """Close Redis connection pool."""
        if self.enabled and hasattr(self, 'pool'):
            try:
                self.pool.disconnect()
                logger.info("Redis connection pool closed")
            except Exception as e:
                logger.warning(f"Error closing Redis connection: {e}")