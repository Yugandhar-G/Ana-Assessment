from .base import SignalScorer
from ..schemas import Restaurant, ParsedQuery


PRICE_LEVELS = ["$", "$$", "$$$", "$$$$"]


class PriceScorer(SignalScorer):
    """Score restaurants based on price preference matching."""
    
    def _price_to_index(self, price: str) -> int:
        """Convert price level to numeric index."""
        try:
            return PRICE_LEVELS.index(price)
        except ValueError:
            return 1  # Default to $$
    
    async def score(self, restaurant: Restaurant, parsed_query: ParsedQuery) -> float:
        """Score based on price preference matching."""
        if not parsed_query.preferences.price:
            return 0.5  # Neutral if no preference
        
        restaurant_idx = self._price_to_index(restaurant.price_level)
        
        # Find best match among preferences
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

