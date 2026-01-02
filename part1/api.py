"""FastAPI backend for Ana AI restaurant search - exact CLI replication."""
import sys
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
import json
import asyncio
from dotenv import load_dotenv

# Add parent directory to path
_current_dir = Path(__file__).parent
_parent_dir = _current_dir.parent
if str(_parent_dir) not in sys.path:
    sys.path.insert(0, str(_parent_dir))

try:
    from part1.src.pipeline import AnaVibeSearch
    from part1.src.streaming_response_generator import StreamingResponseGenerator
    from part1.structured_output_responses.api import search_structured as structured_search_func
except ImportError:
    from src.pipeline import AnaVibeSearch
    try:
        from src.streaming_response_generator import StreamingResponseGenerator
    except ImportError:
        StreamingResponseGenerator = None
    try:
        from structured_output_responses.api import search_structured as structured_search_func
    except ImportError:
        structured_search_func = None

# Load environment variables
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
else:
    load_dotenv()

# Global search instance
search_instance: Optional[AnaVibeSearch] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown."""
    global search_instance
    # Startup - use exact same initialization as CLI
    # Explicitly enable advanced fusion and LLM for full capabilities
    search_instance = AnaVibeSearch(
        persist_vectors=True, 
        use_llm=True,
        use_advanced_fusion=True  # Use AdvancedScoreFusion with rank_with_award_priority
    )
    await search_instance._ensure_initialized()
    yield
    # Shutdown
    pass


app = FastAPI(
    title="Ana AI Restaurant Search API",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for Gradio frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SearchRequest(BaseModel):
    query: str


@app.post("/api/search")
async def search(request: SearchRequest):
    """Search endpoint - uses exact same search() method as CLI."""
    if not search_instance:
        raise HTTPException(status_code=503, detail="Search service not initialized")
    
    # Check if query is restaurant-related
    quick_check = search_instance.query_parser._quick_restaurant_check(request.query)
    if quick_check is False or (quick_check is None and not await search_instance.query_parser.is_restaurant_query(request.query)):
        return {
            "success": False,
            "explanation": "I can only help you with restaurant-related questions. Please ask me about restaurants, dining, food, or specific restaurant names.",
            "confidence": "low",
            "caveats": ["Query is not restaurant-related"]
        }
    
    try:
        # Use the EXACT same search method as CLI
        response = await search_instance.search(request.query)
        return response.model_dump()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/search/structured")
async def search_structured_endpoint(request: SearchRequest):
    """Structured search endpoint - returns enriched restaurant data for conversational interface."""
    if not search_instance:
        raise HTTPException(status_code=503, detail="Search service not initialized")
    
    if not structured_search_func:
        raise HTTPException(status_code=503, detail="Structured search function not available")
    
    try:
        response = await structured_search_func(search_instance, request.query)
        return response.model_dump()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/search/stream")
async def search_stream_endpoint(request: SearchRequest):
    """Streaming search endpoint - uses StreamingResponseGenerator with full ranking strategies."""
    if not search_instance:
        raise HTTPException(status_code=503, detail="Search service not initialized")
    
    if not StreamingResponseGenerator:
        raise HTTPException(status_code=503, detail="StreamingResponseGenerator not available")
    
    # Check if query is restaurant-related
    quick_check = search_instance.query_parser._quick_restaurant_check(request.query)
    if quick_check is False or (quick_check is None and not await search_instance.query_parser.is_restaurant_query(request.query)):
        async def error_stream():
            yield f"data: {json.dumps({'type': 'error', 'message': 'I can only help you with restaurant-related questions. Please ask me about restaurants, dining, food, or specific restaurant names.'})}\n\n"
        return StreamingResponse(error_stream(), media_type="text/event-stream")
    
    async def generate_stream():
        """Generate streaming response using full pipeline with ranking strategies."""
        try:
            # Use search_for_streaming to get parsed query and ranked results
            # This uses rank_with_award_priority and AdvancedScoreFusion
            parsed_query, ranked_results = await search_instance.search_for_streaming(request.query)
            
            if not ranked_results:
                yield f"data: {json.dumps({'type': 'error', 'message': 'No restaurants found matching your criteria.'})}\n\n"
                return
            
            # Create streaming response generator
            streaming_generator = StreamingResponseGenerator(client=search_instance.client)
            
            # Stream the response
            async for chunk in streaming_generator.generate_stream(parsed_query, ranked_results):
                yield f"data: {json.dumps(chunk)}\n\n"
                
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': f'Error: {str(e)}'})}\n\n"
    
    return StreamingResponse(generate_stream(), media_type="text/event-stream")


@app.get("/api/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "initialized": search_instance is not None}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

