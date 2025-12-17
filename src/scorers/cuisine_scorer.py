from .base import SignalScorer
from ..schemas import Restaurant, ParsedQuery


# Cuisine similarity mapping
CUISINE_GROUPS = {
    "asian": ["thai", "japanese", "chinese", "vietnamese", "korean", "asian-fusion"],
    "hawaiian": ["hawaiian", "hawaiian-local", "hawaiian-fusion", "hawaiian-regional", "hawaiian-seafood", "hawaiian-american", "hawaiian-ranch", "hawaiian-diner"],
    "seafood": ["seafood", "hawaiian-seafood", "pacific-rim"],
    "american": ["american", "new american", "american-comfort", "hawaiian-american"],
    "italian": ["italian", "mediterranean"],
    "mediterranean": ["mediterranean", "mediterranean-indian", "italian"],
}


class CuisineScorer(SignalScorer):
    """Score restaurants based on cuisine preference matching."""
    
    def _normalize_cuisine(self, cuisine: str) -> str:
        """Normalize cuisine string for comparison."""
        return cuisine.lower().strip()
    
    def _get_cuisine_group(self, cuisine: str) -> set[str]:
        """Get all cuisines in the same group."""
        normalized = self._normalize_cuisine(cuisine)
        groups = set()
        for group_name, cuisines in CUISINE_GROUPS.items():
            if normalized in cuisines or group_name in normalized:
                groups.update(cuisines)
        return groups
    
    async def score(self, restaurant: Restaurant, parsed_query: ParsedQuery) -> float:
        """Score based on cuisine preference matching."""
        if not parsed_query.preferences.cuisine:
            return 0.5  # Neutral if no preference
        
        restaurant_cuisine = self._normalize_cuisine(restaurant.cuisine)
        preferred_cuisines = [self._normalize_cuisine(c) for c in parsed_query.preferences.cuisine]
        
        # Exact match
        for preferred in preferred_cuisines:
            if preferred in restaurant_cuisine or restaurant_cuisine in preferred:
                return 1.0
        
        # Group match (e.g., Thai matches Asian)
        restaurant_groups = self._get_cuisine_group(restaurant_cuisine)
        for preferred in preferred_cuisines:
            preferred_groups = self._get_cuisine_group(preferred)
            if restaurant_groups & preferred_groups:
                return 0.7
        
        # No match
        return 0.2

