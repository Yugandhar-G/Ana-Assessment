from abc import ABC, abstractmethod
from ..schemas import Restaurant, ParsedQuery


class SignalScorer(ABC):
    """Base class for scoring signals."""
    
    @abstractmethod
    async def score(self, restaurant: Restaurant, parsed_query: ParsedQuery) -> float:
        """Score a restaurant on this signal. Returns 0.0-1.0."""
        pass
    
    async def score_batch(self, restaurants: list[Restaurant], parsed_query: ParsedQuery) -> dict[str, float]:
        """Score multiple restaurants. Returns {restaurant_id: score}."""
        scores = {}
        for restaurant in restaurants:
            scores[restaurant.id] = await self.score(restaurant, parsed_query)
        return scores

