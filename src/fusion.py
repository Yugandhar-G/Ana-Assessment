from dataclasses import dataclass
from .schemas import Restaurant, ParsedQuery, SignalWeights


@dataclass
class ScoredRestaurant:
    """Restaurant with all signal scores and final fused score."""
    restaurant: Restaurant
    vibe_score: float
    cuisine_score: float
    price_score: float
    feature_score: float
    final_score: float


class ScoreFusion:
    """Combine multiple signal scores into a final ranking."""
    
    def fuse(
        self,
        restaurant: Restaurant,
        vibe_score: float,
        cuisine_score: float,
        price_score: float,
        feature_score: float,
        weights: SignalWeights,
    ) -> ScoredRestaurant:
        """Compute weighted fusion of all scores."""
        final_score = (
            weights.vibe * vibe_score +
            weights.cuisine * cuisine_score +
            weights.price * price_score +
            weights.features * feature_score
        )
        
        return ScoredRestaurant(
            restaurant=restaurant,
            vibe_score=vibe_score,
            cuisine_score=cuisine_score,
            price_score=price_score,
            feature_score=feature_score,
            final_score=final_score,
        )
    
    def rank(self, scored_restaurants: list[ScoredRestaurant]) -> list[ScoredRestaurant]:
        """Rank restaurants by final score, descending."""
        return sorted(scored_restaurants, key=lambda x: x.final_score, reverse=True)

