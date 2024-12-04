#!/bin/bash

# Exit on any error
set -e

# Helper function to wait for a condition
wait_for_condition() {
  local condition=$1
  local description=$2
  echo "Waiting for $description..."
  while ! eval "$condition"; do
    echo "$description not ready. Retrying in 5 seconds..."
    sleep 5
  done
  echo "$description is ready!"
}

# Step 1: Stop and rebuild all containers
echo "Stopping all running services..."
docker-compose stop

echo "Building all services..."
docker-compose build

# Step 2: Start the train service
echo "Starting 'train' service..."
docker-compose up -d train

# Step 3: Wait for train to complete
wait_for_condition "[ -f ./checkpoints/train_done.flag ]" "'train' service to complete"

# Step 4: Start the eval service
echo "Starting 'eval' service..."
docker-compose up -d eval

# Step 5: Start the server service
echo "Starting 'server' service..."
docker-compose up -d server

# Step 6: Wait for the server to be healthy
wait_for_condition "curl -s http://localhost:8080/health" "'server' service to be ready"

# Step 7: Start the client service
echo "Starting 'client' service..."
docker-compose up -d client

# Step 8: Show all running services
echo "All services are up and running:"
docker-compose ps

# Step 9: Stop and remove all containers after completion
echo "Stopping all services..."
docker-compose stop

echo "Removing all stopped containers..."
docker-compose rm -f

echo "Workflow complete!"
