import json
import logging
from hashlib import sha256
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import redis
    from redis import ConnectionPool

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning(
        "Redis package not installed. Cache functionality will be disabled. "
        "Install with: pip install redis"
    )


class RedisCacheManager:

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
        **redis_kwargs,
    ):
        if not REDIS_AVAILABLE:
            self.enabled = False
            logger.warning("Redis not available. Caching is disabled.")
            return

        self.enabled = True
        self.ttl = ttl
        self.prefix = prefix

        try:
            self.pool = ConnectionPool(
                host=host,
                port=port,
                db=db,
                password=password,
                max_connections=max_connections,
                socket_timeout=socket_timeout,
                socket_connect_timeout=socket_connect_timeout,
                decode_responses=decode_responses,
                **redis_kwargs,
            )
            self.client = redis.Redis(connection_pool=self.pool)
            self.client.ping()
            logger.info(f"Redis cache initialized at {host}:{port}")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}. Caching disabled.")
            self.enabled = False

    def _make_key(self, request_data: Dict[str, Any]) -> str:
        serialized = json.dumps(request_data, sort_keys=True, default=str)
        key_hash = sha256(serialized.encode()).hexdigest()
        return f"{self.prefix}:{key_hash}"

    def get(self, request_data: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
        if not self.enabled:
            return None

        try:
            key = self._make_key(request_data)
            cached = self.client.get(key)

            if cached is None:
                logger.debug(f"Cache MISS for key: {key[:16]}...")
                return None

            logger.debug(f"Cache HIT for key: {key[:16]}...")
            if isinstance(cached, bytes):
                cached = cached.decode("utf-8")
            return json.loads(cached)
        except Exception as e:
            logger.warning(f"Error retrieving from cache: {e}")
            return None

    def set(
        self,
        request_data: Dict[str, Any],
        response_data: List[Dict[str, Any]],
        ttl: Optional[int] = None,
    ) -> bool:
        if not self.enabled:
            return False

        try:
            key = self._make_key(request_data)
            serialized = json.dumps(response_data, default=str)
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
        if not self.enabled:
            return False

        try:
            key = self._make_key(request_data)
            return self.client.delete(key) > 0
        except Exception as e:
            logger.warning(f"Error deleting from cache: {e}")
            return False

    def clear_all(self, pattern: Optional[str] = None) -> int:
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
        if not self.enabled:
            return {"enabled": False}

        try:
            info = self.client.info("stats")
            our_keys = len(self.client.keys(f"{self.prefix}:*"))

            hits = info.get("keyspace_hits", 0)
            misses = info.get("keyspace_misses", 0)
            total = hits + misses

            return {
                "enabled": True,
                "total_keys": our_keys,
                "hits": hits,
                "misses": misses,
                "hit_rate": hits / max(total, 1),
                "memory_used": info.get("used_memory_human", "N/A"),
                "connected_clients": info.get("connected_clients", 0),
            }
        except Exception as e:
            logger.warning(f"Error getting cache stats: {e}")
            return {"enabled": True, "error": str(e)}

    def close(self):
        if self.enabled and hasattr(self, "pool"):
            try:
                self.pool.disconnect()
                logger.info("Redis connection pool closed")
            except Exception as e:
                logger.warning(f"Error closing Redis connection: {e}")
