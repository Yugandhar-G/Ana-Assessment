import os
import numpy as np
from .base import SignalScorer
from ..schemas import Restaurant, ParsedQuery
from ..vector_store import VectorStore
from ..gemini_client import AsyncGeminiClient


class VibeScorer(SignalScorer):
    """Score restaurants based on semantic vibe similarity using embeddings."""
    
    def __init__(self, client: AsyncGeminiClient | None = None, vector_store: VectorStore | None = None, embedding_model: str | None = None):
        self.client = client or AsyncGeminiClient(
            api_key=os.getenv("GEMINI_API_KEY"),
            default_embedding_model=os.getenv("GEMINI_EMBEDDING_MODEL", "models/text-embedding-004"),
        )
        self.embedding_model = embedding_model or os.getenv("GEMINI_EMBEDDING_MODEL", "models/text-embedding-004")
        self.vector_store = vector_store
        self._embedding_cache: dict[str, list[float]] = {}
    
    async def _get_embedding(self, text: str) -> list[float]:
        """Get embedding for text, with caching."""
        if text in self._embedding_cache:
            return self._embedding_cache[text]
        
        response = await self.client.embeddings.create(
            model=self.embedding_model,
            input=text,
        )
        embedding = response.data[0].embedding
        self._embedding_cache[text] = embedding
        return embedding
    
    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        a_np = np.array(a)
        b_np = np.array(b)
        return float(np.dot(a_np, b_np) / (np.linalg.norm(a_np) * np.linalg.norm(b_np)))
    
    async def score(self, restaurant: Restaurant, parsed_query: ParsedQuery) -> float:
        """Score based on semantic similarity between query and restaurant vibe."""
        query_embedding = await self._get_embedding(parsed_query.semantic_query)
        vibe_embedding = await self._get_embedding(restaurant.vibe.vibe_summary)
        
        similarity = self._cosine_similarity(query_embedding, vibe_embedding)
        return (similarity + 1) / 2
    
    async def score_batch(self, restaurants: list[Restaurant], parsed_query: ParsedQuery) -> dict[str, float]:
        """Score multiple restaurants efficiently."""
        query_embedding = await self._get_embedding(parsed_query.semantic_query)
        
        scores = {}
        for restaurant in restaurants:
            vibe_embedding = await self._get_embedding(restaurant.vibe.vibe_summary)
            similarity = self._cosine_similarity(query_embedding, vibe_embedding)
            scores[restaurant.id] = (similarity + 1) / 2
        
        return scores
    
    async def search_similar(self, query: str, n_results: int = 10) -> list[dict]:
        """Use vector store for fast similarity search."""
        if not self.vector_store:
            raise ValueError("Vector store not initialized")
        return await self.vector_store.search(query, n_results=n_results)

