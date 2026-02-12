import json
import asyncio
import logging
import re
from pathlib import Path
from typing import List, Tuple, Optional
from .gemini_client import AsyncGeminiClient
from .schemas import Restaurant, RestaurantVibe, ParsedQuery, AnaResponse
from .query_parser import QueryParser
from .video_lookup import get_video_urls_for_restaurant
from .fusion import ScoredRestaurant
from .agent_response_generator import AgentResponseGenerator

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AnaAgenticSearch:
    """Agentic orchestrator for Ana AI search."""
    
    def __init__(
        self,
        client: AsyncGeminiClient | None = None,
    ):
        import os
        self.client = client or AsyncGeminiClient(
            api_key=os.getenv("GEMINI_API_KEY"),
            default_chat_model=os.getenv("GEMINI_CHAT_MODEL", "gemini-3-flash-preview"),
        )
        self.restaurants = self._load_restaurants()
        self._restaurant_lookup = {r.id: r for r in self.restaurants}
        self.query_parser = QueryParser(self.client)
        self.response_generator = AgentResponseGenerator(self.client, self.restaurants)
        self._initialized = True
    
    async def _ensure_initialized(self):
        """No-op for agentic search as it loads data synchronously for now."""
        pass

    def _load_restaurants(self) -> list[Restaurant]:
        """Load restaurant data from JSON file."""
        data_path = Path(__file__).parent.parent.parent / "data" / "restaurants.json"
        if not data_path.exists():
            return []
        
        with open(data_path) as f:
            data = json.load(f)
        
        restaurants = []
        for item in data:
            vibe_data = item.pop("vibe", {})
            if isinstance(vibe_data, dict):
                item["vibe"] = RestaurantVibe(**vibe_data)
            else:
                # Handle cases where vibe might be missing or in wrong format
                item["vibe"] = RestaurantVibe(
                    formality="casual",
                    noise_level="moderate",
                    atmosphere_tags=[],
                    best_for=[],
                    vibe_summary=item.get("highlights", "") or "A welcoming dining experience."
                )
            restaurants.append(Restaurant(**item))
        
        return restaurants

    async def search(self, query: str) -> AnaResponse:
        """Execute agentic search pipeline."""
        parsed_query = await self.query_parser.parse(query)
        
        # The agentic workflow is handled within the ResponseGenerator
        # 1. Gemini + Web Search brainstorms candidates
        # 2. Candidates are grounded in the local database
        # 3. Gemini synthesizes the final structured response
        return await self.response_generator.generate(parsed_query)

    async def search_for_streaming(self, query: str) -> Tuple[ParsedQuery, List[ScoredRestaurant]]:
        """Placeholder for streaming compatibility with existing API structure.
        
        In Agentic mode, we return the parsed query and an empty list of restaurants,
        as the restaurant selection happens during the generation process.
        """
        parsed_query = await self.query_parser.parse(query)
        # We'll return empty ranked results as selection happens inside the agent loop
        return parsed_query, []
