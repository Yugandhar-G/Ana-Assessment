import asyncio
from .base import SignalScorer
from ..schemas import Restaurant, ParsedQuery


PRICE_LEVELS = ["$", "$$", "$$$", "$$$$"]


class PriceScorer(SignalScorer):
    """Score restaurants based on price preference matching.
    
    This scorer is CPU-bound (no I/O), so the async method wraps a synchronous
    implementation for cleaner code and better optimization opportunities.
    """
    
    def _price_to_index(self, price: str) -> int:
        """Convert price level to numeric index."""
        try:
            return PRICE_LEVELS.index(price)
        except ValueError:
            return 1
    
    def _score_sync(self, restaurant: Restaurant, parsed_query: ParsedQuery) -> float:
        """Synchronous scoring logic (CPU-bound, no I/O)."""
        if not parsed_query.preferences.price:
            return 0.5
        
        restaurant_idx = self._price_to_index(restaurant.price_level)
        best_score = 0.0
        for preferred in parsed_query.preferences.price:
            preferred_idx = self._price_to_index(preferred)
            distance = abs(restaurant_idx - preferred_idx)
            
            if distance == 0:
                score = 1.0
            elif distance == 1:
                score = 0.7
            elif distance == 2:
                score = 0.4
            else:
                score = 0.2
            
            best_score = max(best_score, score)
        
        return best_score
    
    async def score(self, restaurant: Restaurant, parsed_query: ParsedQuery) -> float:
        """Async wrapper for synchronous scoring (for interface compatibility)."""
        # Run in thread pool to avoid blocking event loop (though overhead is minimal for fast CPU ops)
        # For very fast operations, direct execution is fine - asyncio handles it efficiently
        return self._score_sync(restaurant, parsed_query)

