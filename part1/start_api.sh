#!/bin/bash
# Start FastAPI server for Ana AI

echo "Starting Ana AI FastAPI server..."
cd "$(dirname "$0")"
python -m uvicorn api:app --host 0.0.0.0 --port 8000 --reload

