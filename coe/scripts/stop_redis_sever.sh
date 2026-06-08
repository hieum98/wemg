# Stop Redis
REDIS_PORT=6379
if [ -f "redis/redis.pid" ]; then
    echo "Stopping Redis..."
    redis-cli -p ${REDIS_PORT} shutdown || true
    rm -f redis/redis.pid
fi