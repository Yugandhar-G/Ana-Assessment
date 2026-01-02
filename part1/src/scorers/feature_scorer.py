from .base import SignalScorer
from ..schemas import Restaurant, ParsedQuery


class FeatureScorer(SignalScorer):
    """Score restaurants based on feature matching.
    
    This scorer is CPU-bound (no I/O), so the async method wraps a synchronous
    implementation for cleaner code and better optimization opportunities.
    """
    
    def _score_sync(self, restaurant: Restaurant, parsed_query: ParsedQuery) -> float:
        """Score based on percentage of desired features present.
        
        Matches features like:
        - wheelchair_accessible, wheelchair accessible, wheelchair-access
        - serves_dinner, serves dinner, dinner service
        - outdoor_seating, outdoor seating
        etc.
        """
        desired_features = parsed_query.preferences.features
        desired_atmosphere = parsed_query.preferences.atmosphere
        
        if not desired_features and not desired_atmosphere:
            return 0.5
        
        matches = 0
        total = 0
        
        for feature in desired_features:
            total += 1
            feature_lower = feature.lower().strip()
            matched = False
            
            # Try exact match first
            if restaurant.features.get(feature_lower, False):
                matched = True
            
            # Try normalized key (with underscores, no spaces)
            normalized_key = feature_lower.replace(" ", "_").replace("-", "_")
            if not matched and restaurant.features.get(normalized_key, False):
                matched = True
            
            # Try fuzzy matching if exact matches fail
            # Check if feature name appears in any restaurant feature key or vice versa
            if not matched:
                for feature_key, feature_value in restaurant.features.items():
                    if feature_value:  # Only check features that are True
                        key_lower = feature_key.lower()
                        # Check substring matches (handles "wheelchair accessible" → "wheelchair_accessible")
                        if (feature_lower in key_lower or 
                            key_lower in feature_lower or
                            feature_lower.replace("_", " ").replace("-", " ") in key_lower.replace("_", " ").replace("-", " ") or
                            key_lower.replace("_", " ").replace("-", " ") in feature_lower.replace("_", " ").replace("-", " ")):
                            matched = True
                            break
                     # If still not matched, check text fields for semantic matches (e.g., "ocean view" in highlights)
            # This handles features that are described in text but not in the features dict
            if not matched:
                feature_words = set(feature_lower.replace("_", " ").replace("-", " ").split())
                # Remove common stop words
                feature_words = {w for w in feature_words if len(w) >= 3}
                
                # Check highlights, vibe_summary, and details for feature mentions
                text_fields = [
                    restaurant.highlights or "",
                    restaurant.vibe.vibe_summary or "",
                    restaurant.details or "",
                    restaurant.editorial_summary or "",
                ]
                
                for text_field in text_fields:
                    if not text_field:
                        continue
                    text_lower = text_field.lower()
                    # Check if feature words appear in text (e.g., "ocean view", "oceanfront", "ocean-view")
                    # Handle variations like "ocean view" vs "oceanfront" vs "ocean-view"
                    feature_variants = [
                        feature_lower,
                        feature_lower.replace(" ", "-"),
                        feature_lower.replace(" ", "_"),
                        feature_lower.replace("-", " "),
                        feature_lower.replace("_", " "),
                    ]
                    # Also check for compound terms like "oceanfront" when searching for "ocean view"
                    if "ocean" in feature_lower and "view" in feature_lower:
                        feature_variants.extend(["oceanfront", "ocean front", "ocean-side", "oceanside", "water view", "waterfront"])
                    if "oceanfront" in feature_lower or "ocean front" in feature_lower:
                        feature_variants.extend(["ocean view", "ocean-view", "water view", "waterfront", "oceanside"])
                    
                    for variant in feature_variants:
                        if variant in text_lower:
                            matched = True
                            break
                    
                    if matched:
                        break
            
            if matched:
                matches += 1
        
        restaurant_tags = [t.lower() for t in restaurant.vibe.atmosphere_tags]
        restaurant_best_for = [b.lower() for b in restaurant.vibe.best_for]
        
        for atmosphere in desired_atmosphere:
            total += 1
            atmosphere_lower = atmosphere.lower()
            matched = False
            
            # Check atmosphere_tags first (primary source)
            if atmosphere_lower in restaurant_tags:
                matches += 1
                matched = True
            elif any(atmosphere_lower in tag for tag in restaurant_tags):
                matches += 0.5
                matched = True
        
            # Check best_for as bonus/fallback (only if not already matched in atmosphere_tags)
            if not matched and atmosphere_lower in restaurant_best_for:
                matches += 0.5
        
        if total == 0:
            return 0.5
        
        return min(1.0, matches / total)
    
    async def score(self, restaurant: Restaurant, parsed_query: ParsedQuery) -> float:
        """Async wrapper for synchronous scoring (for interface compatibility)."""
        # Run synchronously - asyncio handles fast CPU-bound ops efficiently
        return self._score_sync(restaurant, parsed_query)

