# Ana AI Restaurant Search - API & Gradio Interface

This directory contains a FastAPI backend with streaming support and a Gradio web interface for the Ana AI restaurant search system.

## Features

- **FastAPI Backend**: RESTful API with streaming support
- **Gradio Interface**: User-friendly web interface with real-time streaming
- **Token Streaming**: Responses stream token-by-token for better perceived latency
- **Video Links**: Automatically includes video links when available

## Setup

1. **Install dependencies**:
   ```bash
   pip install -r ../requirements.txt
   ```

2. **Set Gemini API Key**:
   ```bash
   export GEMINI_API_KEY="your-api-key-here"
   ```
   Get your API key from [Google AI Studio](https://makersuite.google.com/app/apikey)

## Running the Application

### Option 1: Run Both Services Separately

**Terminal 1 - Start FastAPI Server**:
```bash
cd part1
python -m uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

**Terminal 2 - Start Gradio Interface**:
```bash
cd part1
python gradio_app.py
```

### Option 2: Use Startup Scripts

**Terminal 1 - Start FastAPI Server**:
```bash
cd part1
./start_api.sh
```

**Terminal 2 - Start Gradio Interface**:
```bash
cd part1
./start_gradio.sh
```

## API Endpoints

### Streaming Search (Recommended)
```
POST /api/search/stream
Body: {"query": "your search query"}
Response: Server-Sent Events (SSE) stream
```

### Non-Streaming Search
```
POST /api/search
Body: {"query": "your search query"}
Response: Complete JSON response
```

### Health Check
```
GET /api/health
Response: {"status": "healthy", "initialized": true}
```

## Gradio Interface

Once both services are running:
- Open your browser to `http://localhost:7860`
- Enter your restaurant search query
- Results stream in real-time as they're generated

## Example Queries

- "What is ULUPALAKUA RANCH STORE & GRILL famous for?"
- "Find me a romantic Italian restaurant in Lahaina"
- "Best Japanese restaurant with outdoor seating"
- "Where can I get vegan food in Wailuku?"

## Architecture

```
User Query
    ↓
Gradio Interface (port 7860)
    ↓
FastAPI Backend (port 8000)
    ↓
AnaVibeSearch Pipeline
    ↓
StreamingResponseGenerator
    ↓
Gemini (streaming tokens)
    ↓
Stream back to user (token-by-token)
```

## Notes

- The FastAPI server must be running before starting Gradio
- Streaming provides better perceived latency - users see results immediately
- Video links are automatically included when available in video_metadata.json
- Restaurant name detection ensures exact matches are prioritized

