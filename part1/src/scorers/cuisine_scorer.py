from collections import defaultdict
from typing import Dict, Set, List
from .base import SignalScorer
from ..schemas import Restaurant, ParsedQuery


class CuisineScorer(SignalScorer):
    """Score restaurants based on cuisine preference matching.
    
    Learn cuisine relationships from actual restaurant data.
    Cuisines that frequently appear together in restaurants are considered related.
    """
    
    def __init__(self, restaurants: List[Restaurant] | None = None, min_cooccurrence: int = 2):
        """Initialize cuisine scorer with restaurant data to learn relationships.
        
        Args:
            restaurants: List of restaurants to learn cuisine relationships from.
                        If None, relationships will be learned when first needed.
            min_cooccurrence: Minimum number of times cuisines must appear together
                            to be considered related. Default: 2
        """
        self.min_cooccurrence = min_cooccurrence
        self._cuisine_similarity: Dict[str, Set[str]] = {}
        self._cuisine_cooccurrence_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        
        if restaurants:
            self._learn_cuisine_relationships(restaurants)
    
    def _normalize_cuisine(self, cuisine: str) -> str:
        return cuisine.lower().strip().replace('/', ' ').replace('-', ' ')
    
    def _learn_cuisine_relationships(self, restaurants: List[Restaurant]):
        """Learn cuisine relationships from restaurant data based on co-occurrence. """
        for restaurant in restaurants:
            if not restaurant.cuisine:
                continue
            
            cuisine_parts = [self._normalize_cuisine(c.strip()) for c in restaurant.cuisine.split(',')]
            
            for c1 in cuisine_parts:
                for c2 in cuisine_parts:
                    if c1 != c2:
                        self._cuisine_cooccurrence_counts[c1][c2] += 1
        self._cuisine_similarity = defaultdict(set)
        for c1, cooccurrences in self._cuisine_cooccurrence_counts.items():
            for c2, count in cooccurrences.items():
                if count >= self.min_cooccurrence:
                    # Make it bidirectional
                    self._cuisine_similarity[c1].add(c2)
                    self._cuisine_similarity[c2].add(c1)
    
    def _get_related_cuisines(self, cuisine: str) -> Set[str]:
        """Get cuisines that are related to the given cuisine (learned from data).
        
        Returns empty set if no relationships were learned or cuisine not found.
        """
        normalized = self._normalize_cuisine(cuisine)

        if not self._cuisine_similarity:
            return set()
        

        return self._cuisine_similarity.get(normalized, set())
    
    async def score(self, restaurant: Restaurant, parsed_query: ParsedQuery) -> float:
        """Score based on cuisine preference matching.
        
        Scoring (priority order):
        1. Exact match: 1.0 (e.g., "Italian" query matches "Italian, Pizza")
        2. Substring match: 0.8 (e.g., "Italian" query matches restaurant with "Italian" in cuisine string)
        3. Related cuisine (learned from data): 0.4 (e.g., "Italian" → "Pizza" if they co-occur frequently)
        4. No match: 0.1 (strongly penalize non-matching cuisines)
        """
        if not parsed_query.preferences.cuisine:
            return 0.5
        
        restaurant_cuisine_normalized = self._normalize_cuisine(restaurant.cuisine)
        preferred_cuisines = [self._normalize_cuisine(c) for c in parsed_query.preferences.cuisine]
        

        restaurant_cuisine_parts = [
            self._normalize_cuisine(part.strip()) 
            for part in restaurant.cuisine.split(',') 
            if part.strip()
        ]
        

        # Special handling for Hawaiian cuisine variations
        def expand_cuisine_variations(cuisine: str) -> set[str]:
            """Expand cuisine to include common variations."""
            normalized = self._normalize_cuisine(cuisine)
            variations = {normalized}
            
            # Hawaiian variations - handle "Hawaiian" matching "Hawaii Regional"
            if 'hawaiian' in normalized or 'hawaii' in normalized:
                variations.add('hawaiian')
                variations.add('hawaii regional')
                variations.add('hawaii')
            
            return variations
        
        for preferred in preferred_cuisines:
            preferred_variations = expand_cuisine_variations(preferred)
            
            # Check if any variation matches restaurant cuisine
            for variation in preferred_variations:
                # PRIORITY 1: Check individual cuisine parts FIRST (most precise matching)
                # This allows us to distinguish primary from secondary cuisines
                for i, part in enumerate(restaurant_cuisine_parts):
                    if variation in part or part in variation:
                        if i == 0:
                            return 1.0  # Primary cuisine - perfect match
                        else:
                            return 0.75  # Secondary cuisine - lower score to prioritize primary matches
                
                # PRIORITY 2: Check if preferred cuisine appears in restaurant cuisine string (fallback)
                # Only use this if we didn't find a match in individual parts
                if variation in restaurant_cuisine_normalized or restaurant_cuisine_normalized in variation:
                    # If it's a single cuisine (no comma), treat as primary
                    if len(restaurant_cuisine_parts) == 1:
                        return 1.0
                    # Otherwise, we couldn't determine primary/secondary from parts, so treat as secondary to be safe
                    return 0.75
        
        # 2. Check for related cuisines (learned from data co-occurrence - score 0.4)
        # Only check if we've learned relationships
        if self._cuisine_similarity:
            for preferred in preferred_cuisines:
                # Get cuisines related to the preferred cuisine
                related_to_preferred = self._get_related_cuisines(preferred)
                
                # Check if any restaurant cuisine part is related to preferred
                for part in restaurant_cuisine_parts:
                    if part in related_to_preferred:
                        # Related cuisine found - lower score than exact match
                        return 0.4
        
        # 3. No match - strongly penalize (score 0.1)
        return 0.1
