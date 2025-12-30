#!/bin/bash
# Milvus Quick Start Script
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================"
echo "Starting Milvus with Docker Compose"
echo "========================================"

# Check if docker-compose is available
if command -v docker-compose &> /dev/null; then
    COMPOSE_CMD="docker-compose"
elif docker compose version &> /dev/null; then
    COMPOSE_CMD="docker compose"
else
    echo "ERROR: docker-compose not found"
    echo "Please install docker-compose first:"
    echo "  sudo apt-get install docker-compose"
    exit 1
fi

# Start Milvus
echo "Starting Milvus services..."
$COMPOSE_CMD -f docker-compose.milvus.yml up -d

echo ""
echo "Waiting for Milvus to be ready..."
sleep 10

# Check if Milvus is responding
for i in {1..30}; do
    if curl -s http://localhost:19530/healthz > /dev/null 2>&1; then
        echo ""
        echo "========================================"
        echo "Milvus is ready!"
        echo "========================================"
        echo "Endpoint: localhost:19530"
        echo ""
        echo "To stop Milvus:"
        echo "  $COMPOSE_CMD -f docker-compose.milvus.yml down"
        echo ""
        echo "To view logs:"
        echo "  $COMPOSE_CMD -f docker-compose.milvus.yml logs -f"
        echo "========================================"
        exit 0
    fi
    echo -n "."
    sleep 2
done

echo ""
echo "ERROR: Milvus failed to start"
echo "Check logs: $COMPOSE_CMD -f docker-compose.milvus.yml logs"
exit 1
