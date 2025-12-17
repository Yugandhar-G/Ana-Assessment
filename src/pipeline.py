import json
import asyncio
from pathlib import Path
from openai import AsyncOpenAI
from .schemas import Restaurant, RestaurantVibe, ParsedQuery, AnaResponse
from .query_parser import QueryParser
from .filters import HardFilter
from .scorers import VibeScorer, CuisineScorer, PriceScorer, FeatureScorer
from .fusion import ScoreFusion, ScoredRestaurant
from .response_generator import ResponseGenerator
from .vector_store import VectorStore, initialize_vector_store


class AnaVibeSearch:
    """Main orchestrator for Ana AI vibe-based restaurant search."""
    
    def __init__(self, client: AsyncOpenAI | None = None, persist_vectors: bool = False):
        self.client = client or AsyncOpenAI()
        
        # Load restaurant data first (needed for vector store init)
        self.restaurants = self._load_restaurants()
        self._restaurant_lookup = {r.id: r for r in self.restaurants}
        
        # Initialize vector store
        persist_dir = str(Path(__file__).parent.parent / "chroma_db") if persist_vectors else None
        self.vector_store = initialize_vector_store(persist_dir)
        
        # Initialize components
        self.query_parser = QueryParser(self.client)
        self.hard_filter = HardFilter()
        self.vibe_scorer = VibeScorer(self.client, self.vector_store)
        self.cuisine_scorer = CuisineScorer()
        self.price_scorer = PriceScorer()
        self.feature_scorer = FeatureScorer()
        self.fusion = ScoreFusion()
        self.response_generator = ResponseGenerator(self.client)
    
    def _load_restaurants(self) -> list[Restaurant]:
        """Load restaurant data from JSON file."""
        data_path = Path(__file__).parent.parent / "data" / "restaurants.json"
        if not data_path.exists():
            return []
        
        with open(data_path) as f:
            data = json.load(f)
        
        restaurants = []
        for item in data:
            vibe_data = item.pop("vibe", {})
            item["vibe"] = RestaurantVibe(**vibe_data)
            restaurants.append(Restaurant(**item))
        
        return restaurants
    
    async def search(self, query: str) -> AnaResponse:
        """Execute full search pipeline.
        
        Flow:
        1. Parse query → structured intent
        2. Vector search → top N semantically similar restaurants
        3. Hard filter → remove excluded restaurants
        4. Score remaining → cuisine, price, features
        5. Fuse scores → final ranking
        6. Generate response → natural language explanation
        """
        # Stage 1: Parse query
        parsed_query = await self.query_parser.parse(query)
        
        # Stage 2: Vector search for semantic similarity (FAST!)
        # This retrieves top candidates from ChromaDB instead of scoring all
        vector_results = self.vector_store.search(
            query=parsed_query.semantic_query,
            n_results=50,  # Get top 50 by vibe similarity
        )
        
        # Map vector results back to Restaurant objects with vibe scores
        candidates_with_vibe_scores = []
        for result in vector_results:
            restaurant = self._restaurant_lookup.get(result["id"])
            if restaurant:
                candidates_with_vibe_scores.append((restaurant, result["similarity"]))
        
        # Stage 3: Apply hard filters
        filtered_candidates = []
        for restaurant, vibe_score in candidates_with_vibe_scores:
            if self.hard_filter._passes_filters(restaurant, parsed_query):
                filtered_candidates.append((restaurant, vibe_score))
        
        if not filtered_candidates:
            return AnaResponse(
                success=False,
                explanation="No restaurants match your criteria after filtering. Try relaxing some constraints?",
                confidence="low",
                caveats=["All restaurants were filtered out by hard constraints"],
            )
        
        # Stage 4: Score remaining candidates (vibe score already from vector search)
        scored_results = await self._score_candidates(filtered_candidates, parsed_query)
        
        # Stage 5: Rank results
        ranked_results = self.fusion.rank(scored_results)
        
        # Stage 6: Generate response
        response = await self.response_generator.generate(parsed_query, ranked_results)
        
        return response
    
    async def _score_candidates(
        self,
        candidates_with_vibe: list[tuple[Restaurant, float]],
        parsed_query: ParsedQuery,
    ) -> list[ScoredRestaurant]:
        """Score candidates across all signals (vibe score already computed)."""
        scored_results = []
        
        for restaurant, vibe_score in candidates_with_vibe:
            # Run remaining scorers in parallel (vibe already done by vector search)
            cuisine_score, price_score, feature_score = await asyncio.gather(
                self.cuisine_scorer.score(restaurant, parsed_query),
                self.price_scorer.score(restaurant, parsed_query),
                self.feature_scorer.score(restaurant, parsed_query),
            )
            
            # Fuse scores
            scored = self.fusion.fuse(
                restaurant=restaurant,
                vibe_score=vibe_score,
                cuisine_score=cuisine_score,
                price_score=price_score,
                feature_score=feature_score,
                weights=parsed_query.weights,
            )
            scored_results.append(scored)
        
        return scored_results

