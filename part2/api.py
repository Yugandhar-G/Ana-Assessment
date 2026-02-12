import sys
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import json
from dotenv import load_dotenv

# Add current directory to path
_current_dir = Path(__file__).parent
if str(_current_dir) not in sys.path:
    sys.path.insert(0, str(_current_dir))

try:
    from src.agent_pipeline import AnaAgenticSearch
    from structured_output_responses.api import search_structured as structured_search_func
except ImportError:
    # Handle absolute imports if needed
    sys.path.append(str(_current_dir.parent))
    from part2.src.agent_pipeline import AnaAgenticSearch
    from part2.structured_output_responses.api import search_structured as structured_search_func

# Load environment variables
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path, override=True)
else:
    load_dotenv(override=True)

# Global search instance
search_instance: Optional[AnaAgenticSearch] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global search_instance
    search_instance = AnaAgenticSearch()
    await search_instance._ensure_initialized()
    yield

app = FastAPI(
    title="Ana AI Agentic Search API (Part 2)",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SearchRequest(BaseModel):
    query: str

@app.post("/api/search/structured")
async def search_structured_endpoint(request: SearchRequest):
    if not search_instance:
        raise HTTPException(status_code=503, detail="Search service not initialized")
    
    try:
        # Note: We use search_structured logic but with our agentic search instance
        # The agentic search already handles the complex logic inside search()
        response = await search_instance.search(request.query)
        return response.model_dump()
    except Exception as e:
        import traceback
        print(f"Error in agentic search: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001) # Use 8001 to avoid conflict with part1
