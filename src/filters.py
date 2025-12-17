from .schemas import Restaurant, ParsedQuery


class HardFilter:
    """Apply hard exclusion filters to eliminate restaurants that violate constraints."""
    
    def filter(self, restaurants: list[Restaurant], parsed_query: ParsedQuery) -> list[Restaurant]:
        """Filter out restaurants matching any exclusion criteria."""
        candidates = []
        
        for restaurant in restaurants:
            if self._passes_filters(restaurant, parsed_query):
                candidates.append(restaurant)
        
        return candidates
    
    def _passes_filters(self, restaurant: Restaurant, parsed_query: ParsedQuery) -> bool:
        """Check if restaurant passes all hard filters."""
        must_not = parsed_query.must_not
        
        # Check formality exclusions
        if must_not.formality:
            if restaurant.vibe.formality in must_not.formality:
                return False
        
        # Check price exclusions
        if must_not.price:
            if restaurant.price_level in must_not.price:
                return False
        
        # Check cuisine exclusions
        if must_not.cuisine:
            if restaurant.cuisine.lower() in [c.lower() for c in must_not.cuisine]:
                return False
        
        # Check feature exclusions (e.g., "not loud")
        if must_not.features:
            for feature in must_not.features:
                # Check noise level
                if feature.lower() == "loud" and restaurant.vibe.noise_level == "loud":
                    return False
                if feature.lower() == "quiet" and restaurant.vibe.noise_level == "quiet":
                    return False
                # Check boolean features
                if restaurant.features.get(feature, False):
                    return False
        
        # Check location filter
        if parsed_query.location:
            if parsed_query.location.lower() not in restaurant.region.lower():
                return False
        
        return True

