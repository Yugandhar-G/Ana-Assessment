import re
import logging
from .schemas import Restaurant, ParsedQuery

logger = logging.getLogger(__name__)


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
        
        if restaurant.business_status and restaurant.business_status != "OPERATIONAL":
            return False
        
        if must_not.formality and restaurant.vibe.formality in must_not.formality:
            return False
        
        if must_not.price and restaurant.price_level in must_not.price:
            return False
        
        if must_not.cuisine and restaurant.cuisine.lower() in [c.lower() for c in must_not.cuisine]:
            return False
        
        # Apply cuisine hard filter only when:
        # 1. Cuisine weight is high (>= 0.5) - indicates explicit cuisine preference, OR
        # 2. Cuisine is explicitly mentioned in the raw query (e.g., "indian restaurant", "best italian food")
        # This prevents false positives from vague queries like "romantic dinner" where LLM might incorrectly infer cuisine
        cuisine_explicitly_mentioned = False
        if parsed_query.preferences.cuisine:
            query_lower = parsed_query.raw_query.lower()
            for cuisine in parsed_query.preferences.cuisine:
                # Check if cuisine name appears in raw query using word boundaries
                cuisine_normalized = cuisine.lower().replace('/', ' ').replace('-', ' ')
                # Use word boundary regex to match whole words only (prevents "indian" matching "indiana")
                cuisine_pattern = r'\b' + re.escape(cuisine_normalized) + r'\b'
                if re.search(cuisine_pattern, query_lower):
                    cuisine_explicitly_mentioned = True
                    break
                # Also check individual words for multi-word cuisines like "north indian"
                cuisine_words = [w for w in cuisine_normalized.split() if len(w) >= 3]
                if cuisine_words:
                    # Check if all significant words appear in query (with word boundaries)
                    all_words_found = all(
                        re.search(r'\b' + re.escape(w) + r'\b', query_lower) 
                        for w in cuisine_words
                    )
                    if all_words_found:
                        cuisine_explicitly_mentioned = True
                        break
        
        if parsed_query.preferences.cuisine and (parsed_query.weights.cuisine >= 0.5 or cuisine_explicitly_mentioned):
            # CRITICAL: Check if query is about menu items first
            # If restaurant has matching menu items, don't filter by cuisine
            query_lower = parsed_query.raw_query.lower()
            menu_item_indicators = ['fare', 'dish', 'item', 'menu', 'food', 'specialty', 'special', 'signature', 'inspired']
            is_menu_item_query = any(indicator in query_lower for indicator in menu_item_indicators)
            
            # Also check if cuisine preference might be a menu item (e.g., "Polynesian-inspired Fare")
            # by checking if it contains food-related terms or is unusually long
            for cuisine_pref in parsed_query.preferences.cuisine:
                cuisine_lower = cuisine_pref.lower()
                if any(term in cuisine_lower for term in ['fare', 'inspired', 'style', 'dish', 'specialty']):
                    is_menu_item_query = True
                    break
                # If cuisine preference is longer than typical cuisine names, might be a menu item
                if len(cuisine_pref.split()) > 2:
                    is_menu_item_query = True
                    break
            
            # Check if restaurant has menu items that match the query or cuisine preference
            menu_item_match = False
            if restaurant.top_menu_items:
                # Check against query text
                query_words = set(query_lower.split())
                # Remove common stop words
                stop_words = {'best', 'spots', 'for', 'the', 'a', 'an', 'and', 'or', 'in', 'on', 'at', 'to', 'of', 'with', 'cuisine', 'restaurant', 'restaurants'}
                query_words = {w for w in query_words if w not in stop_words and len(w) >= 3}
                
                # Also check cuisine preferences as potential menu items
                cuisine_words = set()
                for cuisine_pref in parsed_query.preferences.cuisine:
                    cuisine_lower = cuisine_pref.lower()
                    cuisine_words.update(cuisine_lower.split())
                cuisine_words = {w for w in cuisine_words if w not in stop_words and len(w) >= 3}
                
                # Combine query and cuisine words
                all_search_words = query_words | cuisine_words
                
                for menu_item in restaurant.top_menu_items:
                    menu_item_lower = menu_item.lower()
                    menu_item_words = set(menu_item_lower.split())
                    
                    # Check if significant words from query/cuisine appear in menu item
                    significant_overlap = {w for w in all_search_words & menu_item_words if len(w) >= 4}
                    if significant_overlap:
                        menu_item_match = True
                        logger.debug(f"✅ MENU ITEM MATCH: {restaurant.name} has menu item '{menu_item}' matching query/cuisine")
                        break
                    
                    # Also check if menu item contains key phrases from query/cuisine
                    for search_word in all_search_words:
                        if len(search_word) >= 5 and search_word in menu_item_lower:
                            menu_item_match = True
                            logger.debug(f"✅ MENU ITEM MATCH: {restaurant.name} has menu item '{menu_item}' containing '{search_word}'")
                            break
                    if menu_item_match:
                        break
            
            # If menu item matches, skip cuisine filtering
            if menu_item_match:
                logger.debug(f"✅ PASSED FILTER: {restaurant.name} (menu item match, skipping cuisine filter)")
            else:
                # Use same normalization as CuisineScorer (replace / and - with spaces)
                def normalize_cuisine(cuisine: str) -> str:
                    return cuisine.lower().strip().replace('/', ' ').replace('-', ' ')
                
                # Special mappings for common cuisine variations
                def expand_cuisine_variations(cuisine: str) -> set[str]:
                    """Expand cuisine to include common variations."""
                    normalized = normalize_cuisine(cuisine)
                    variations = {normalized}
                    
                    # Hawaiian variations
                    if 'hawaiian' in normalized or 'hawaii' in normalized:
                        variations.add('hawaiian')
                        variations.add('hawaii regional')
                        variations.add('hawaii')
                    
                    # Add base word variations (e.g., "hawaiian" -> "hawaii")
                    words = normalized.split()
                    for word in words:
                        if len(word) >= 4:
                            variations.add(word)
                    
                    return variations
                
                restaurant_cuisine_normalized = normalize_cuisine(restaurant.cuisine)
                restaurant_cuisine_parts = [normalize_cuisine(part.strip()) for part in restaurant.cuisine.split(',') if part.strip()]
                
                cuisine_match = False
                for preferred in parsed_query.preferences.cuisine:
                    preferred_normalized = normalize_cuisine(preferred)
                    preferred_variations = expand_cuisine_variations(preferred)
                    
                    # Check if any variation matches restaurant cuisine
                    for variation in preferred_variations:
                        # Check full cuisine string
                        if variation in restaurant_cuisine_normalized or restaurant_cuisine_normalized in variation:
                            cuisine_match = True
                            break
                        
                        # Check individual cuisine parts
                        for part in restaurant_cuisine_parts:
                            if variation in part or part in variation:
                                cuisine_match = True
                                break
                        
                        if cuisine_match:
                            break
                    
                    # Also check word overlap for multi-word cuisines
                    if not cuisine_match:
                        preferred_words = set(preferred_normalized.split())
                        for part in restaurant_cuisine_parts:
                            part_words = set(part.split())
                            meaningful_preferred = {w for w in preferred_words if len(w) >= 4}
                            meaningful_part = {w for w in part_words if len(w) >= 4}
                            if meaningful_preferred and meaningful_part and meaningful_preferred & meaningful_part:
                                cuisine_match = True
                                break
                    
                    if cuisine_match:
                        break
                
                if not cuisine_match:
                    logger.debug(f"❌ FILTERED OUT: {restaurant.name} (cuisine mismatch: '{restaurant.cuisine}' vs requested '{parsed_query.preferences.cuisine}')")
                    return False
                else:
                    logger.debug(f"✅ PASSED FILTER: {restaurant.name} (cuisine match: '{restaurant.cuisine}' matches '{parsed_query.preferences.cuisine}')")
        
        # NOTE: For non-explicit cuisine preferences (low cuisine weight < 0.5),
        # preferences.cuisine should NOT be used for hard filtering.
        # Those are soft filters that only affect scoring via CuisineScorer.
        
        if must_not.features:
            for feature in must_not.features:
                if feature.lower() == "loud" and restaurant.vibe.noise_level == "loud":
                    return False
                if feature.lower() == "quiet" and restaurant.vibe.noise_level == "quiet":
                    return False
                if restaurant.features.get(feature, False):
                    return False
        
        if parsed_query.location:
            query_loc = parsed_query.location.lower().strip()
            
            # Island names - don't filter by island since all restaurants are on Maui
            island_names = {'maui', 'hawaii', 'hawaiian islands', 'oahu', 'kauai', 'big island', 'molokai', 'lanai'}
            if query_loc in island_names:
                # Skip location filtering for island-level queries - all restaurants are on Maui
                pass
            else:
                # For specific locations (cities, regions), apply filtering BUT be more lenient
                # Only filter if location is explicitly important (e.g., "in Lahaina" vs "Maui")
                region_match = query_loc in restaurant.region.lower() if restaurant.region else False
                city = getattr(restaurant, "city", None)
                city_match = query_loc in city.lower() if city else False
                
                # Check if location is explicitly mentioned in query (more strict filtering)
                # vs just inferred (more lenient filtering)
                query_lower = parsed_query.raw_query.lower()
                location_explicitly_mentioned = (
                    f"in {query_loc}" in query_lower or
                    f"at {query_loc}" in query_lower or
                    f"{query_loc} restaurant" in query_lower or
                    f"restaurant in {query_loc}" in query_lower
                )
                
                # Only filter out if location is explicitly mentioned AND no match
                # For general queries (e.g., "what desserts are famous in Maui"), be lenient
                if not (region_match or city_match):
                    if location_explicitly_mentioned:
                        # Location explicitly requested - filter strictly
                        return False
                    else:
                        # Location inferred but not explicit - be lenient, don't filter
                        # This allows showing related restaurants from nearby areas
                        pass
        
        return True

