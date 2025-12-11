# Create data directories
SCRIPT_DIR="$(pwd)/wemg/configs"
REDIS_PORT=6379
mkdir -p redis

echo "Starting Redis for caching..."
# Check if Redis is already running on the specified port
if ! lsof -i :${REDIS_PORT} > /dev/null 2>&1; then
    # Start Redis with config file and override specific settings via command line
    # Command line arguments take precedence over config file settings
    cd redis
    redis-server "${SCRIPT_DIR}/redis.conf" \
        --port ${REDIS_PORT} \
        --dir $(pwd) \
        --pidfile redis.pid \
        --logfile redis.log
    # Wait for Redis to start
    sleep 2
    echo "Redis started successfully"
else
    echo "Redis is already running on port ${REDIS_PORT}"
fi