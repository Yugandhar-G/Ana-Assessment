#!/bin/bash
# Start Conversational Gradio interface for Ana AI

echo "Starting Conversational Ana AI Restaurant Search..."
echo "Make sure the FastAPI server is running on port 8000 first!"
cd "$(dirname "$0")"
python conversational_gradio_app.py

