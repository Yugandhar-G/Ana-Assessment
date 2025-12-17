from .base import SignalScorer
from ..schemas import Restaurant, ParsedQuery


class FeatureScorer(SignalScorer):
    """Score restaurants based on feature matching."""
    
    async def score(self, restaurant: Restaurant, parsed_query: ParsedQuery) -> float:
        """Score based on percentage of desired features present."""
        desired_features = parsed_query.preferences.features
        desired_atmosphere = parsed_query.preferences.atmosphere
        
        if not desired_features and not desired_atmosphere:
            return 0.5  # Neutral if no preference
        
        matches = 0
        total = 0
        
        # Check boolean features
        for feature in desired_features:
            total += 1
            feature_key = feature.lower().replace(" ", "_").replace("-", "_")
            if restaurant.features.get(feature_key, False):
                matches += 1
            # Also check common variations
            elif restaurant.features.get(feature, False):
                matches += 1
        
        # Check atmosphere tags
        restaurant_tags = [t.lower() for t in restaurant.vibe.atmosphere_tags]
        for atmosphere in desired_atmosphere:
            total += 1
            if atmosphere.lower() in restaurant_tags:
                matches += 1
            # Partial match
            elif any(atmosphere.lower() in tag for tag in restaurant_tags):
                matches += 0.5
        
        # Check best_for
        restaurant_best_for = [b.lower() for b in restaurant.vibe.best_for]
        for atmosphere in desired_atmosphere:
            if atmosphere.lower() in restaurant_best_for:
                matches += 0.5
        
        if total == 0:
            return 0.5
        
        return min(1.0, matches / total)

