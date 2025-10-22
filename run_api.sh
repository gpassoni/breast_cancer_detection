#!/bin/bash
# Script to run the FastAPI backend

echo "Starting FastAPI backend..."
echo "API will be available at http://localhost:8000"
echo "API documentation at http://localhost:8000/docs"
echo ""

cd "$(dirname "$0")"
python -m api.main
