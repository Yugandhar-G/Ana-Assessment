#!/bin/bash
# Start Gradio interface for Ana AI

echo "Starting Ana AI Gradio interface..."
echo "Make sure the FastAPI server is running on port 8000 first!"
cd "$(dirname "$0")"
python gradio_app.py

