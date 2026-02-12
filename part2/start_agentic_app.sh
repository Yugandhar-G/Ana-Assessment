#!/bin/bash

# Kill any existing processes on ports 8001 and 7862
lsof -ti:8001 | xargs kill -9 2>/dev/null
lsof -ti:7862 | xargs kill -9 2>/dev/null

echo "Starting Part 2 API..."
cd "$(dirname "$0")"
python3 -m uvicorn api:app --host 0.0.0.0 --port 8001 --reload &

echo "Waiting for API to start..."
sleep 5

echo "Starting Part 2 Gradio App..."
python3 conversational_gradio_app.py
