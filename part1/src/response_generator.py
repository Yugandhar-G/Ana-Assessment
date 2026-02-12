import os
from pathlib import Path
from .gemini_client import AsyncGeminiClient
from .schemas import ParsedQuery, AnaResponse, RestaurantMatch, MatchReason
from .fusion import ScoredRestaurant
from .video_lookup import get_video_urls_for_restaurant


class ResponseGenerator:
    """Generate natural language responses using LLM."""
    
    def __init__(self, client: AsyncGeminiClient | None = None, model: str | None = None):
        self.client = client or AsyncGeminiClient(
            api_key=os.getenv("GEMINI_API_KEY"),
            default_chat_model=os.getenv("GEMINI_CHAT_MODEL", "gemini-3-flash-preview"),
        )
        self.model = model or os.getenv("GEMINI_CHAT_MODEL", "gemini-3-flash-preview")
        self.system_prompt = self._load_prompt()
    
    def _load_prompt(self) -> str:
        prompt_path = Path(__file__).parent.parent / "prompts" / "response_generator.txt"
        if prompt_path.exists():
            return prompt_path.read_text()
        return "Generate a helpful restaurant recommendation response."
    
    def _scored_to_match(self, scored: ScoredRestaurant) -> RestaurantMatch:
        """Convert ScoredRestaurant to RestaurantMatch."""
        # Get video URLs for this restaurant from video metadata
        video_urls = get_video_urls_for_restaurant(
            restaurant_id=scored.restaurant.id,
            restaurant_name=scored.restaurant.name,
            google_place_id=getattr(scored.restaurant, "google_place_id", None)
        )
        
        return RestaurantMatch(
            id=scored.restaurant.id,
            name=scored.restaurant.name,
            cuisine=scored.restaurant.cuisine,
            price_level=scored.restaurant.price_level,
            price_level_curated=scored.restaurant.price_level_curated,
            region=scored.restaurant.region,
            city=getattr(scored.restaurant, "city", None),
            formatted_address=getattr(scored.restaurant, "formatted_address", None),
            location_raw=getattr(scored.restaurant, "location_raw", None),
            state=getattr(scored.restaurant, "state", None),
            zipcode=getattr(scored.restaurant, "zipcode", None),
            country=getattr(scored.restaurant, "country", None),
            latitude=getattr(scored.restaurant, "latitude", None),
            longitude=getattr(scored.restaurant, "longitude", None),
            rating=scored.restaurant.rating,
            features=scored.restaurant.features,
            business_status=scored.restaurant.business_status,
            national_phone=scored.restaurant.national_phone,
            international_phone=scored.restaurant.international_phone,
            website_uri=scored.restaurant.website_uri,
            google_maps_uri=scored.restaurant.google_maps_uri,
            opening_hours_text=scored.restaurant.opening_hours_text,
            is_open_now=scored.restaurant.is_open_now,
            serves_meal_times=scored.restaurant.serves_meal_times,
            photos_urls=scored.restaurant.photos_urls,
            restaurant_photos_urls=scored.restaurant.restaurant_photos_urls,
            video_urls=video_urls,
            reviews=scored.restaurant.reviews,
            payment_options=scored.restaurant.payment_options,
            parking_options=scored.restaurant.parking_options,
            restroom=scored.restaurant.restroom,
            match_status=scored.restaurant.match_status,
            match_confidence=scored.restaurant.match_confidence,
            name_similarity=scored.restaurant.name_similarity,
            data_completeness_score=scored.restaurant.data_completeness_score,
            google_matched_name=scored.restaurant.google_matched_name,
            created_at=scored.restaurant.created_at,
            updated_at=scored.restaurant.updated_at,
            live_music_curated=scored.restaurant.live_music_curated,
            live_music_google=scored.restaurant.live_music_google,
            vibe_summary=scored.restaurant.vibe.vibe_summary,
            final_score=scored.final_score,
            vibe_score=scored.vibe_score,
            cuisine_score=scored.cuisine_score,
            price_score=scored.price_score,
            feature_score=scored.feature_score,
        )
    
    def _generate_match_reasons(self, scored: ScoredRestaurant, parsed_query: ParsedQuery) -> list[MatchReason]:
        """Generate match reasons based on scores."""
        reasons = []
        scores = [
            ("vibe", scored.vibe_score, parsed_query.weights.vibe),
            ("cuisine", scored.cuisine_score, parsed_query.weights.cuisine),
            ("price", scored.price_score, parsed_query.weights.price),
            ("features", scored.feature_score, parsed_query.weights.features),
        ]
        scores.sort(key=lambda x: x[1] * x[2], reverse=True)
        
        for i, (signal, score, weight) in enumerate(scores):
            if score < 0.3:
                continue
                
            importance = "primary" if i == 0 else ("secondary" if i == 1 else "minor")
            
            if signal == "vibe":
                reasons.append(MatchReason(
                    signal=signal,
                    importance=importance,
                    query_wanted=parsed_query.semantic_query[:100],
                    restaurant_has=scored.restaurant.vibe.vibe_summary[:100],
                    score=score,
                ))
            elif signal == "cuisine":
                reasons.append(MatchReason(
                    signal=signal,
                    importance=importance,
                    query_wanted=", ".join(parsed_query.preferences.cuisine) or "any",
                    restaurant_has=scored.restaurant.cuisine,
                    score=score,
                ))
            elif signal == "price":
                reasons.append(MatchReason(
                    signal=signal,
                    importance=importance,
                    query_wanted=", ".join(parsed_query.preferences.price) or "any",
                    restaurant_has=scored.restaurant.price_level,
                    score=score,
                ))
            elif signal == "features":
                reasons.append(MatchReason(
                    signal=signal,
                    importance=importance,
                    query_wanted=", ".join(parsed_query.preferences.features + parsed_query.preferences.atmosphere) or "any",
                    restaurant_has=", ".join(scored.restaurant.vibe.atmosphere_tags[:3]),
                    score=score,
                ))
        
        return reasons
    
    def _select_relevant_alternatives(
        self,
        ranked_results: list[ScoredRestaurant],
        top_match: ScoredRestaurant,
        parsed_query: ParsedQuery,
        max_total_results: int = 6,
        base_min_score: float = 0.3,  # Lowered from 0.4 to include more restaurants
    ) -> list[ScoredRestaurant]:
        """
        Select alternatives purely based on relevance and ranking.

        - If query is about a SPECIFIC restaurant, return [] (only show that restaurant).
        - If query is GENERAL (cuisine/item/vibe), return up to max_total_results - 1 alternatives.
        - Returns up to 5 alternatives (for 6 total restaurants including top match).
        """
        # Check if query is about a SPECIFIC restaurant (not general food culture queries)
        # SIMPLIFIED LOGIC: If a restaurant name appears in the query, it's restaurant-specific
        query_lower = parsed_query.raw_query.lower().strip()
        is_restaurant_specific = False
        
        import re
        
        # Normalize query (remove punctuation, extra spaces)
        query_normalized = re.sub(r'[^\w\s]', ' ', query_lower)
        query_normalized = ' '.join(query_normalized.split())
        
        # Check if top match restaurant name appears in query
        # This is the PRIMARY indicator - if user mentions a restaurant name, they want that restaurant
        if ranked_results:
            top_restaurant_name = (ranked_results[0].restaurant.name or "").lower()
            if top_restaurant_name:
                top_name_normalized = re.sub(r'[^\w\s]', ' ', top_restaurant_name)
                top_name_normalized = ' '.join(top_name_normalized.split())
                
                # Remove common words for better matching
                common_words = {'the', 'a', 'an', 'at', 'for', 'with', 'about', 'restaurant', 'restaurants', 
                               'what', 'where', 'when', 'how', 'is', 'are', 'was', 'were', 'can', 'you', 
                               'tell', 'me', 'in', 'maui', 'hawaii', 'on', 'of', 'to', 'and', 'or', 'but'}
                
                # Method 1: Full name match (most reliable)
                if top_name_normalized in query_normalized:
                    print(f"[DEBUG] ✅ Restaurant-specific: Top match name '{top_name_normalized}' found in query")
                    is_restaurant_specific = True
                else:
                    # Method 2: Check for significant word matches (at least 2 words for multi-word names)
                    top_words = [w for w in top_name_normalized.split() if w not in common_words and len(w) >= 3]
                    query_words = [w for w in query_normalized.split() if w not in common_words and len(w) >= 3]
                    
                    if len(top_words) >= 2:  # Multi-word restaurant names
                        matching_words = [w for w in top_words if w in query_words]
                        if len(matching_words) >= 2:  # At least 2 words match
                            match_ratio = len(matching_words) / len(top_words)
                            if match_ratio >= 0.5:  # At least 50% of words match (lowered threshold)
                                print(f"[DEBUG] ✅ Restaurant-specific: {len(matching_words)}/{len(top_words)} words of '{top_name_normalized}' match query")
                                is_restaurant_specific = True
                    elif len(top_words) == 1:  # Single-word restaurant name
                        if top_words[0] in query_words:
                            print(f"[DEBUG] ✅ Restaurant-specific: Single-word name '{top_words[0]}' found in query")
                            is_restaurant_specific = True
                
                # EXCEPTION: Only override if query is CLEARLY a food culture query
                # Examples: "best dishes at Mama's Fish House" → general (show alternatives)
                # But: "Mama's Fish House menu" → restaurant-specific (no alternatives)
                # But: "tell me about Mama's Fish House" → restaurant-specific (no alternatives)
                if is_restaurant_specific:
                    # Only treat as general if there's a food culture keyword BEFORE the restaurant name
                    # AND the query is asking about food/dishes (not about the restaurant itself)
                    food_culture_patterns = [
                        r'\bbest\s+(dish|dishes|food|dessert|desserts)\s+(at|in|from)',
                        r'\bfamous\s+(dish|dishes|food|dessert|desserts)\s+(at|in|from)',
                        r'\bwhat\s+(dish|dishes|food|dessert|desserts|to\s+eat|to\s+try)\s+(at|in|from)',
                        r'\bwhere\s+to\s+find\s+(dish|dishes|food|dessert|desserts)',
                    ]
                    
                    # Check if query matches food culture patterns
                    is_food_culture_pattern = any(re.search(pattern, query_lower) for pattern in food_culture_patterns)
                    
                    if is_food_culture_pattern:
                        # Query like "best dishes at Mama's" → general query, show alternatives
                        print(f"[DEBUG] ⚠️  Food culture pattern detected - treating as general query")
                        is_restaurant_specific = False
                    else:
                        # Query like "Mama's Fish House menu" or "tell me about Mama's" → restaurant-specific
                        print(f"[DEBUG] ✅ Restaurant-specific query confirmed - no food culture pattern")
        
        # If it's a restaurant-specific query, return exactly 0 alternatives (only top match)
        if is_restaurant_specific:
            print(f"[DEBUG] ⚠️  Restaurant-specific query detected - returning 0 alternatives")
            print(f"[DEBUG]   Query: '{parsed_query.raw_query}'")
            print(f"[DEBUG]   Top match: '{ranked_results[0].restaurant.name if ranked_results else 'N/A'}'")
            return []
        
        print(f"[DEBUG] ✅ Not restaurant-specific - will select alternatives")

        if not ranked_results or len(ranked_results) <= 1:
            return []

        # Return up to 5 alternatives (for 6 total restaurants: 1 top + 5 alternatives)
        # For food culture queries, try to show maximum alternatives
        max_alternatives = min(max(0, max_total_results - 1), len(ranked_results) - 1)  # Don't exceed available results
        
        # Determine query type BEFORE using it
        is_cuisine_query = bool(parsed_query.preferences.cuisine) and parsed_query.weights.cuisine >= 0.3
        is_vibe_query = parsed_query.weights.vibe >= 0.5
        
        # Detect general food culture queries (e.g., "best dessert", "famous dishes", "what to eat")
        query_lower = parsed_query.raw_query.lower()
        food_culture_indicators = [
            'best', 'famous', 'popular', 'favorite', 'must try', 'what', 'where to find',
            'dessert', 'desserts', 'dish', 'dishes', 'food', 'eat', 'try', 'specialty',
            'specialties', 'known for', 'famous for', 'recommend', 'suggest',
            'scenic', 'view', 'views', 'oceanfront', 'beachfront', 'with'  # Add scenic/view keywords
        ]
        is_food_culture_query = any(indicator in query_lower for indicator in food_culture_indicators)
        
        # For food culture queries, ensure we're trying to fill all slots
        if is_food_culture_query:
            # Try to get as many alternatives as possible (up to max)
            print(f"[DEBUG] Food culture query detected - maximizing alternatives (target: {max_alternatives})")

        top_score = top_match.final_score
        # More lenient scoring for general queries - include more restaurants
        # For perfect matches (score 1.0), use a much lower threshold to include more options
        # Make thresholds even more lenient to show more related restaurants
        if top_score >= 0.95:
            dynamic_min_score = max(0.2, top_score * 0.3)  # Very lenient for perfect matches
        elif top_score >= 0.8:
            dynamic_min_score = max(0.25, top_score * 0.35)  # Lenient for high scores
        else:
            dynamic_min_score = max(base_min_score, top_score * 0.3)  # More lenient threshold
        
        is_general_query = is_cuisine_query or is_vibe_query or is_food_culture_query or not is_restaurant_specific
        

        alternatives: list[ScoredRestaurant] = []

        top_name = (top_match.restaurant.name or "").lower()
        
        # DEBUG: Check what we're iterating over
        restaurants_to_check = ranked_results[1:]
        print(f"[DEBUG] Alternative selection: Checking {len(restaurants_to_check)} restaurants from {len(ranked_results)} total")
        print(f"[DEBUG] Query type: cuisine={is_cuisine_query}, vibe={is_vibe_query}, food_culture={is_food_culture_query}, general={is_general_query}")
        print(f"[DEBUG] Score thresholds: base_min={base_min_score}, dynamic_min={dynamic_min_score}, top_score={top_score:.3f}")

        for scored in restaurants_to_check:
            # For food culture queries, be more aggressive about filling slots
            # Only stop early if we've filled all slots AND it's not a food culture query
            if len(alternatives) >= max_alternatives and not is_food_culture_query:
                break
            # For food culture queries, allow some overflow to find better matches
            if len(alternatives) >= max_alternatives * 1.5:  # Cap at 1.5x to prevent infinite loop
                break

            restaurant = scored.restaurant
            restaurant_name = (restaurant.name or "").lower()
            

            # Skip if same restaurant as top (by name) - avoid duplicates
            if restaurant_name == top_name:
                # print(f"DEBUG: Skipping {restaurant.name} (duplicate)")
                continue

            # Skip non-operational (only skip if status is explicitly set and not OPERATIONAL)
            # If business_status is None, assume it's operational (don't skip)
            if restaurant.business_status is not None and restaurant.business_status != "OPERATIONAL":
                # print(f"DEBUG: Skipping {restaurant.name} (not operational: {restaurant.business_status})")
                continue

            restaurant_cuisine = (restaurant.cuisine or "").lower()
            restaurant_price = restaurant.price_level
            restaurant_region = (restaurant.region or "").lower() if restaurant.region else ""

            # More lenient filtering - prioritize showing related restaurants
            # For food culture queries (e.g., "best dessert"), be VERY lenient
            if is_food_culture_query:
                # For food culture queries, include if ANY score is decent OR final_score is reasonable
                # This ensures we show diverse options for "best X" type queries
                # Be EXTREMELY lenient - show restaurants even with lower scores
                # For dessert queries, we want to show MANY options
                has_decent_score = (
                    scored.final_score >= 0.1 or  # Very low threshold (was 0.15)
                    scored.vibe_score >= 0.3 or  # Lowered from 0.4
                    scored.cuisine_score >= 0.3 or  # Lowered from 0.4
                    scored.feature_score >= 0.3  # Lowered from 0.4
                )
                if has_decent_score:
                    alternatives.append(scored)
                    print(f"[DEBUG] Food culture query: Added {restaurant.name} (final_score={scored.final_score:.3f}, vibe={scored.vibe_score:.3f}, feature={scored.feature_score:.3f})")
                    continue
                else:
                    print(f"[DEBUG] Food culture query: Skipped {restaurant.name} (final={scored.final_score:.3f}, vibe={scored.vibe_score:.3f}, feature={scored.feature_score:.3f}) - scores too low")
            
            # For cuisine queries: include if cuisine_score is good OR final_score is reasonable
            if is_cuisine_query:
                # For cuisine queries, include if cuisine_score >= 0.6 OR final_score is reasonable
                # Lowered threshold from 0.7 to 0.6 to include more restaurants
                cuisine_condition = scored.cuisine_score >= 0.6
                score_condition = scored.final_score >= max(0.2, dynamic_min_score * 0.8)  # More lenient
                if cuisine_condition or score_condition:
                    alternatives.append(scored)
                continue
            else:
                # For non-cuisine queries, be more lenient with score threshold
                # Lower threshold to include more related restaurants
                lenient_score_threshold = max(0.2, dynamic_min_score * 0.7)
                if scored.final_score < lenient_score_threshold:
                    # Still check if it has good individual scores (vibe, cuisine, etc.)
                    has_good_individual_score = (
                        scored.vibe_score >= 0.5 or  # Lowered from 0.6
                        scored.cuisine_score >= 0.5 or  # Lowered from 0.6
                        scored.feature_score >= 0.5  # Lowered from 0.6
                    )
                    if not has_good_individual_score:
                        continue
                
                # Apply filters but make them less strict
                is_relevant = True
                
                # Price relevance - only filter if price is VERY important (weight >= 0.7)
                # Lowered threshold from 0.5 to 0.7 to be less strict
                if parsed_query.preferences.price and parsed_query.weights.price >= 0.7:
                    if restaurant_price not in parsed_query.preferences.price:
                        is_relevant = False
                
                # Location relevance - be more lenient, only filter for very specific locations
                # Skip location filtering for island names (already filtered by HardFilter)
                if is_relevant and parsed_query.location:
                    query_loc = parsed_query.location.lower().strip()
                    island_names = {
                        "maui", "hawaii", "hawaiian islands", "oahu", "kauai",
                        "big island", "molokai", "lanai",
                    }
                    # Skip location filtering for island-level queries
                    if query_loc not in island_names:
                        # Only filter if location weight is high AND no match at all
                        # Be more lenient - allow if region OR city matches OR if location weight is low
                        region_match = query_loc in restaurant_region if restaurant_region else False
                        city = getattr(restaurant, "city", None)
                        city_match = query_loc in city.lower() if city else False
                        # Only filter out if location is very important (weight >= 0.6) AND no match
                        location_weight = getattr(parsed_query.weights, 'location', 0.0) if hasattr(parsed_query.weights, 'location') else 0.0
                        if not (region_match or city_match) and location_weight >= 0.6:
                            is_relevant = False
                        # If location weight is lower, don't filter - show restaurants from nearby areas
                
                # Feature relevance - only filter if features are VERY important (weight >= 0.7)
                # Lowered threshold from 0.5 to 0.7 to be less strict
                if (
                    is_relevant
                    and parsed_query.preferences.features
                    and parsed_query.weights.features >= 0.7
                ):
                    has_feature = any(
                        restaurant.features.get(feature.lower().replace(" ", "_"), False)
                        or restaurant.features.get(feature.lower(), False)
                        for feature in parsed_query.preferences.features
                    )
                    # Only filter if no feature match AND feature score is very low
                    if not has_feature and scored.feature_score < 0.2:
                        is_relevant = False

                if is_relevant:
                    alternatives.append(scored)

        # No extra sorting: keep fusion ranking order
        print(f"[DEBUG] Selected {len(alternatives)} alternatives (max allowed: {max_alternatives})")
        if len(alternatives) < max_alternatives and len(ranked_results) > len(alternatives) + 1:
            print(f"[DEBUG] WARNING: Only {len(alternatives)} alternatives selected but {len(ranked_results)-1} available. Consider relaxing filters.")
        
        return alternatives[:max_alternatives]
    
    def _format_all_restaurants_for_gemini(self, ranked_results: list[ScoredRestaurant], query: str = "") -> str:
        """Format restaurants for Gemini to analyze and select from.
        
        Smart selection: Shows award winners + top matches + diverse options (up to 150)
        This balances giving Gemini enough options while maintaining fast latency.
        """
        formatted = []
        query_lower = query.lower() if query else ""
        
        # Smart selection strategy for low latency:
        # 1. All award winners (they're already prioritized and sorted first)
        # 2. Top scoring restaurants (up to 100 total)
        # 3. Diverse options for variety (up to 150 total)
        
        from .fusion import has_award
        award_winners = [r for r in ranked_results if has_award(r.restaurant)]
        non_award_winners = [r for r in ranked_results if not has_award(r.restaurant)]
        
        # For dessert queries, prioritize dessert-related restaurants
        dessert_keywords = ['bakery', 'dessert', 'ice cream', 'shave ice', 'pie', 'sweet']
        is_dessert_query = any(keyword in query_lower for keyword in dessert_keywords)
        
        restaurants_to_show = []
        
        # Step 1: Add all award winners (they're already sorted by award level)
        restaurants_to_show.extend(award_winners)
        
        # Step 2: Add top non-award restaurants (sorted by score)
        # For dessert queries, prioritize dessert-related ones
        if is_dessert_query:
            dessert_related = []
            other_restaurants = []
            for r in non_award_winners:
                restaurant = r.restaurant
                cuisine_lower = (restaurant.cuisine or "").lower()
                name_lower = (restaurant.name or "").lower()
                features_str = " ".join([k.lower() for k, v in restaurant.features.items() if v])
                
                if any(kw in cuisine_lower or kw in name_lower or kw in features_str for kw in dessert_keywords):
                    dessert_related.append(r)
                else:
                    other_restaurants.append(r)
            
            # Add dessert-related first, then others
            restaurants_to_show.extend(dessert_related[:50])
            restaurants_to_show.extend(other_restaurants[:50])
        else:
            # For non-dessert queries, just add top scoring ones
            restaurants_to_show.extend(non_award_winners[:100])
        
        # Limit to top 150 total for optimal latency (enough options, not too slow)
        restaurants_to_show = restaurants_to_show[:150]
        
        print(f"[DEBUG] Formatting {len(restaurants_to_show)} restaurants for Gemini (out of {len(ranked_results)} total)")
        print(f"[DEBUG]   - Award winners: {len(award_winners)}")
        print(f"[DEBUG]   - Total shown: {len(restaurants_to_show)}")
        
        for i, scored in enumerate(restaurants_to_show, 1):
            restaurant = scored.restaurant
            features_list = [k.replace("_", " ").title() for k, v in restaurant.features.items() if v]
            features_str = ", ".join(features_list) if features_list else "none specified"
            
            # Get video URLs for this restaurant
            video_urls = get_video_urls_for_restaurant(
                restaurant_id=restaurant.id,
                restaurant_name=restaurant.name,
                google_place_id=getattr(restaurant, "google_place_id", None)
            )
            videos_str = ", ".join(video_urls) if video_urls else "none"
            
            # Add award information prominently
            from .fusion import has_award, get_award_level
            award_badge = ""
            if has_award(restaurant):
                award_level = get_award_level(restaurant)
                if award_level >= 1.0:
                    award_badge = " 🏆 GOLD AWARD WINNER"
                elif award_level >= 0.8:
                    award_badge = " 🏆 SILVER AWARD WINNER"
                elif award_level >= 0.5:
                    award_badge = " 🏆 HONORABLE MENTION"
                else:
                    award_badge = " 🏆 AWARD WINNER"
            
            # Ultra-compact format to fit all restaurants without timeout
            formatted.append(f"[{i}] {restaurant.name}{award_badge} | {restaurant.cuisine} | {restaurant.price_level_curated or restaurant.price_level} | {restaurant.region} | {restaurant.rating}⭐ | Features: {features_str[:80]} | Vibe: {(restaurant.vibe.vibe_summary[:80] if restaurant.vibe.vibe_summary else (restaurant.highlights[:80] if restaurant.highlights else 'N/A'))} | Menu: {', '.join(getattr(restaurant, 'top_menu_items', [])[:2]) if hasattr(restaurant, 'top_menu_items') and getattr(restaurant, 'top_menu_items', []) else 'N/A'} | Videos: {videos_str}\n")
        
        return "\n".join(formatted)
    
    def _determine_confidence(self, top_score: float, num_results: int) -> str:
        """Determine confidence level based on match quality."""
        if top_score >= 0.8 and num_results >= 1:
            return "high"
        elif top_score >= 0.6:
            return "medium"
        else:
            return "low"

    def _determine_boosted_confidence(
        self,
        top_score: float,
        num_results: int,
        parsed_query: ParsedQuery,
    ) -> str:
        """
        Confidence heuristic for low-latency (no-LLM) mode.
        
        Slightly boosts confidence when:
        - the user gave strong constraints (location, features, atmosphere)
        - and we still found at least one match.
        """
        base = self._determine_confidence(top_score, num_results)
        if not num_results:
            return base
        has_location = bool(parsed_query.location)
        has_prefs = bool(
            parsed_query.preferences.cuisine
            or parsed_query.preferences.features
            or parsed_query.preferences.atmosphere
        )
        if base == "medium" and (has_location or has_prefs):
            return "high"
        return base
    
    async def _llm_reason_about_results(
        self,
        parsed_query: ParsedQuery,
        ranked_results: list[ScoredRestaurant],
    ) -> tuple[list[ScoredRestaurant], str]:
        """Use LLM to reason about query and results, potentially improving ranking.
        
        This step leverages LLM's general knowledge to:
        - Understand the query intent deeply
        - Validate results make sense given the query
        - Suggest improvements or identify missing context
        - Use domain knowledge to enhance recommendations
        
        Returns:
            Tuple of (potentially reranked results, reasoning insights)
        """
        if len(ranked_results) == 0:
            return ranked_results, ""
        
        # Prepare restaurant summaries for LLM reasoning
        restaurant_summaries = []
        for i, scored in enumerate(ranked_results[:10]):  # Top 10 for reasoning
            rest = scored.restaurant
            summary = f"""
Restaurant {i+1}: {rest.name}
- Cuisine: {rest.cuisine}
- Price: {rest.price_level_curated or rest.price_level}
- Location: {rest.region}{f", {getattr(rest, 'city', None)}" if getattr(rest, 'city', None) else ""}
- Rating: {rest.rating}/5.0
- Vibe: {rest.vibe.vibe_summary[:200] if rest.vibe.vibe_summary else "Not specified"}
- Features: {', '.join([k.replace('_', ' ').title() for k, v in rest.features.items() if v][:5])}
- Match Scores: vibe={scored.vibe_score:.2f}, cuisine={scored.cuisine_score:.2f}, price={scored.price_score:.2f}, final={scored.final_score:.2f}
"""
            restaurant_summaries.append(summary.strip())
        
        reasoning_prompt = f"""You are an expert food and dining consultant for Maui, Hawaii. Your task is to reason about a user's query and the restaurant results retrieved by the system.

**USER QUERY:** {parsed_query.raw_query}

**QUERY ANALYSIS:**
- Semantic intent: {parsed_query.semantic_query}
- Cuisine preferences: {', '.join(parsed_query.preferences.cuisine) if parsed_query.preferences.cuisine else 'Any'}
- Price preferences: {', '.join(parsed_query.preferences.price) if parsed_query.preferences.price else 'Any'}
- Features wanted: {', '.join(parsed_query.preferences.features) if parsed_query.preferences.features else 'Any'}
- Location: {parsed_query.location or 'Anywhere in Maui'}

**RETRIEVED RESTAURANTS (ranked by system):**
{chr(10).join(restaurant_summaries)}

**YOUR TASK:**
1. **Understand the query deeply**: What is the user REALLY looking for? Consider cultural context, dining experiences, and what makes sense for Maui/Hawaii.
2. **Validate the results**: Do these restaurants make sense for this query? Are there any that don't fit well?
3. **Use your knowledge**: Based on your expertise about Maui, Hawaiian cuisine, and dining culture, are there any restaurants that should be ranked higher or lower?
4. **Identify gaps**: Is anything missing from the results that would be helpful for this query?
5. **Provide insights**: What should the response emphasize? What cultural or contextual information would help the user?

**RESPOND WITH:**
- A brief reasoning summary (2-3 sentences) about the query and results
- Any important insights about Maui/Hawaiian dining culture relevant to this query
- Whether the ranking seems appropriate or if any restaurants should be prioritized differently
- Any missing context that would improve the recommendation

Keep it concise but insightful."""
        
        try:
            reasoning_response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert food and dining consultant with deep knowledge of Maui, Hawaii, food culture, and dining experiences. Use your expertise to reason about queries and validate results."},
                    {"role": "user", "content": reasoning_prompt},
                ],
                temperature=0.3,  # Lower temperature for more focused reasoning
                max_tokens=500,
            )
            
            reasoning_text = reasoning_response.choices[0].message["content"].strip()
            print(f"[DEBUG] LLM Reasoning: {reasoning_text[:200]}...")
            
            # Optional: LLM-based reranking based on reasoning
            # Check if reasoning suggests reordering
            reranking_prompt = f"""Based on this reasoning analysis:
{reasoning_text}

And these top restaurants:
{chr(10).join([f"{i+1}. {scored.restaurant.name} (score: {scored.final_score:.2f})" for i, scored in enumerate(ranked_results[:5])])}

Should any restaurants be reordered based on your knowledge of Maui/Hawaii dining? Respond with just the restaurant numbers in order (e.g., "2,1,3,4,5" if restaurant 2 should be first), or "no change" if the order is good."""
            
            try:
                rerank_response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are an expert on Maui dining. Suggest restaurant reordering if your knowledge indicates a better fit."},
                        {"role": "user", "content": reranking_prompt},
                    ],
                    temperature=0.2,
                    max_tokens=50,
                )
                
                rerank_text = rerank_response.choices[0].message["content"].strip().lower()
                
                # Parse reranking suggestion
                if "no change" not in rerank_text and "," in rerank_text:
                    try:
                        suggested_order = [int(x.strip()) - 1 for x in rerank_text.split(",") if x.strip().isdigit()]
                        if len(suggested_order) == len(ranked_results[:5]) and all(0 <= i < len(ranked_results) for i in suggested_order):
                            # Reorder top 5 based on LLM suggestion
                            reranked = [ranked_results[i] for i in suggested_order] + ranked_results[5:]
                            print(f"[DEBUG] LLM suggested reranking: {suggested_order}")
                            return reranked, reasoning_text
                    except:
                        pass  # If parsing fails, use original order
                
            except Exception as e:
                print(f"[DEBUG] Reranking failed, using original order: {e}")
            
            # Return original ranking with reasoning
            return ranked_results, reasoning_text
            
        except Exception as e:
            print(f"[WARNING] LLM reasoning failed: {e}")
            return ranked_results, ""
    
    def _generate_dynamic_thinking_steps(
        self,
        parsed_query: ParsedQuery,
        needs_web: bool,
        web_results: list[dict],
        num_restaurants: int,
        num_alternatives: int,
    ) -> list[str]:
        """Generate dynamic thinking steps based on query type and actual processing."""
        steps = []
        query_lower = parsed_query.raw_query.lower()
        
        # Detect query type
        has_restaurant_name = any(
            word in query_lower for word in ['mama', "merriman", "morimoto", "hula", "mala"]
        ) or len([r for r in parsed_query.raw_query.split() if len(r) > 3]) <= 3
        
        is_cuisine_query = bool(parsed_query.preferences.cuisine) and parsed_query.weights.cuisine >= 0.3
        is_vibe_query = parsed_query.weights.vibe >= 0.5
        is_location_query = bool(parsed_query.location) and parsed_query.location.lower() not in ['maui', 'hawaii']
        
        # Step 1: Initial search description (varies by query type)
        if has_restaurant_name:
            steps.append("🔍 **Looking up specific restaurant information...**")
        elif is_cuisine_query:
            cuisine_types = ", ".join(parsed_query.preferences.cuisine[:2])
            steps.append(f"🔍 **Searching for {cuisine_types} restaurants in Maui...**")
        elif is_vibe_query:
            steps.append("🔍 **Searching restaurants by atmosphere and vibe...**")
        elif is_location_query:
            steps.append(f"🔍 **Searching restaurants in {parsed_query.location}...**")
        elif "best" in query_lower or "famous" in query_lower or "popular" in query_lower:
            if "scenic" in query_lower or "view" in query_lower:
                steps.append("🔍 **Searching for restaurants with scenic views...**")
            elif "dessert" in query_lower:
                steps.append("🔍 **Searching for dessert spots and bakeries...**")
            else:
                steps.append("🔍 **Searching restaurant database for top recommendations...**")
        else:
            steps.append("🔍 **Searching restaurant database...**")
        
        # Step 2: Web search (only if actually performed)
        if needs_web:
            if "recent" in query_lower or "trending" in query_lower:
                steps.append("🌐 **Checking recent reviews and trending discussions...**")
            elif "culture" in query_lower or "traditional" in query_lower:
                steps.append("🌐 **Researching cultural context and traditions...**")
            elif "best" in query_lower or "famous" in query_lower:
                steps.append("🌐 **Searching web for popular recommendations and reviews...**")
            else:
                steps.append("🌐 **Gathering additional context from web sources...**")
            
            if web_results:
                source_types = set(r["source"] for r in web_results)
                sources_desc = []
                if "reddit" in source_types:
                    sources_desc.append("Reddit discussions")
                if "blog" in source_types:
                    # Check which blogs were actually found
                    blog_results = [r for r in web_results if r["source"] == "blog"]
                    blog_domains = set()
                    for r in blog_results:
                        url = r.get("url", "").lower()
                        if "yelp.com" in url:
                            blog_domains.add("Yelp")
                        if "tripadvisor.com" in url:
                            blog_domains.add("TripAdvisor")
                        if "eater.com" in url:
                            blog_domains.add("Eater")
                        if "timeout.com" in url:
                            blog_domains.add("Timeout")
                    
                    if blog_domains:
                        sources_desc.append(f"{'/'.join(blog_domains)} reviews")
                    else:
                        sources_desc.append("food blogs")
                if "google" in source_types:
                    sources_desc.append("web reviews")
                
                if sources_desc:
                    steps.append(f"✅ Found insights from {', '.join(sources_desc)}")
                else:
                    steps.append(f"✅ Found {len(web_results)} relevant sources")
            else:
                # Web search was attempted but returned no results
                steps.append("⚠️ **No web results found - using restaurant database only**")
        
        # Step 3: Analysis step (varies by results)
        if num_restaurants == 0:
            steps.append("⚠️ **No matching restaurants found - expanding search criteria...**")
        elif num_restaurants == 1:
            steps.append("🧠 **Analyzing restaurant details...**")
        elif num_alternatives > 0:
            if num_alternatives >= 5:
                steps.append(f"🧠 **Analyzing {num_restaurants} restaurants and ranking top options...**")
            else:
                steps.append(f"🧠 **Comparing {num_restaurants} restaurants to find the best matches...**")
        else:
            steps.append("🧠 **Analyzing results and cultural context...**")
        
        return steps
    
    def _needs_web_search(self, query: str, parsed_query: ParsedQuery) -> bool:
        """Determine if web search is needed."""
        web_keywords = [
            "best", "famous", "popular", "trending", "recent",
            "culture", "traditional", "authentic", "what is",
            "tell me about", "explain", "famous"
        ]
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in web_keywords)
    
    async def _perform_web_search(self, query: str) -> list[dict]:
        """Perform web search using multi-source search."""
        if not hasattr(self.client, 'web_search') or not self.client.web_search:
            print(f"[DEBUG] Web search not available (client.web_search = {getattr(self.client, 'web_search', None)})")
            return []
        
        try:
            from .multi_source_search import SearchSource
            print(f"[DEBUG] Performing web search for: '{query}'")
            results = await self.client.web_search.search(
                query=f"{query} Maui Hawaii restaurant",
                sources=[SearchSource.GOOGLE, SearchSource.REDDIT, SearchSource.BLOGS],
                num_results_per_source=2,
            )
            print(f"[DEBUG] Web search returned {len(results)} results")
            for i, r in enumerate(results[:5], 1):
                print(f"[DEBUG]   [{i}] {r.source}: {r.title[:60]}...")
            
            return [
                {
                    "title": r.title,
                    "url": r.url,
                    "snippet": r.snippet,
                    "source": r.source,
                }
                for r in results
            ]
        except Exception as e:
            print(f"[DEBUG] Web search error: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    async def generate(
        self,
        parsed_query: ParsedQuery,
        ranked_results: list[ScoredRestaurant],
    ) -> AnaResponse:
        """Generate complete response using Gemini-first architecture.
        
        Flow:
        1. Gemini parses/enhances the query
        2. Gemini generates answer AND selects restaurants in one unified step
        3. Web search is integrated and summarized by Gemini
        4. No ranking - Gemini selects from all restaurants based on query relevance
        """
        if not ranked_results:
            return AnaResponse(
                success=False,
                explanation="I couldn't find any restaurants matching your criteria. Try broadening your search?",
                confidence="low",
                caveats=["No matching restaurants found"],
            )
        
        # STEP 1: Use Gemini to parse/enhance the query first
        print(f"[DEBUG] Step 1: Gemini parsing/enhancing query: '{parsed_query.raw_query}'")
        enhanced_query_prompt = f"""You are an expert query parser for a restaurant search system in Maui, Hawaii.

**USER QUERY:** {parsed_query.raw_query}

**YOUR TASK:**
1. Parse and understand the user's query deeply
2. Enhance it with context about what the user is REALLY looking for
3. Identify if this is about a SPECIFIC restaurant (mention restaurant name) or a GENERAL query (cuisine, vibe, food culture, etc.)
4. Extract key requirements: cuisine, price, location, features, atmosphere, etc.
5. Provide cultural context and implicit requirements

**RESPOND WITH:**
- Enhanced query understanding (what the user really wants)
- Query type: "restaurant-specific" or "general"
- If restaurant-specific: restaurant name
- Key requirements extracted
- Cultural context and implicit needs

Keep it concise but insightful."""
        
        try:
            query_parse_response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert query parser for restaurant search. Parse queries deeply and enhance them with context."},
                    {"role": "user", "content": enhanced_query_prompt},
                ],
                temperature=0.3,
                max_tokens=500,
            )
            enhanced_query_analysis = query_parse_response.choices[0].message["content"].strip()
            print(f"[DEBUG] Enhanced query analysis: {enhanced_query_analysis[:200]}...")
        except Exception as e:
            print(f"[DEBUG] Query enhancement failed: {e}")
            enhanced_query_analysis = f"Query: {parsed_query.raw_query}"
        
        # STEP 2: Perform web search (if needed) - Gemini will summarize it
        needs_web = self._needs_web_search(parsed_query.raw_query, parsed_query)
        web_results = []
        
        if needs_web:
            print(f"[DEBUG] Step 2: Performing web search for: '{parsed_query.raw_query}'")
            web_results = await self._perform_web_search(parsed_query.raw_query)
            print(f"[DEBUG] Web search completed: {len(web_results)} results")
            if web_results:
                print(f"[DEBUG] Web result sources: {set(r['source'] for r in web_results)}")
        else:
            print(f"[DEBUG] Web search NOT needed for query: '{parsed_query.raw_query}'")
        
        # STEP 3: Format all restaurants for Gemini (no ranking, just all restaurants)
        print(f"[DEBUG] Step 3: Formatting {len(ranked_results)} restaurants for Gemini (no ranking)")
        all_restaurants_formatted = self._format_all_restaurants_for_gemini(ranked_results, parsed_query.raw_query)
        
        # STEP 4: Build unified prompt for Gemini to generate answer AND select restaurants
        # Build web search summary for Gemini
        web_summary = ""
        if web_results:
            web_summary = "\n\n**WEB SEARCH RESULTS (Summarize and integrate these into your response):**\n"
            for result in web_results:
                source_emoji = "🔗" if result["source"] == "google" else "💬" if result["source"] == "reddit" else "📝"
                web_summary += f"{source_emoji} **{result['title']}** ({result['source']}): {result['snippet']}\n"
            web_summary += "\n**INSTRUCTIONS:** Summarize and integrate these web search results naturally into your response. Use them to provide context, recent reviews, cultural information, and trends."
        
        # Build thinking steps
        thinking_steps = []
        thinking_steps.append("🔍 **Analyzing your query and understanding what you're looking for...**")
        if web_results:
            thinking_steps.append(f"🌐 **Searching web for recent reviews and cultural context...** ✅ Found {len(web_results)} relevant sources")
        thinking_steps.append(f"🧠 **Reviewing {len(ranked_results)} restaurants and selecting the best matches for your query...**")
        thinking_section = "\n".join(thinking_steps)
        
        # STEP 5: Unified Gemini call - generates answer AND selects restaurants
        print(f"[DEBUG] Step 4: Unified Gemini call - generating answer AND selecting restaurants")
        
        user_prompt = f"""
**THINKING PROCESS (Show this to user at the start of your response):**
{thinking_section}

**ENHANCED QUERY ANALYSIS:**
{enhanced_query_analysis}

**USER QUERY:** {parsed_query.raw_query}

{web_summary}

**RESTAURANT DATABASE (You have access to ALL restaurants - analyze and select the best matches):**
You have access to {len(ranked_results)} restaurants in the database. Your task is to:

1. **Understand the query**: Based on the enhanced query analysis above, what is the user REALLY looking for?
2. **CRITICAL - RESTAURANT-SPECIFIC DETECTION**: 
   - If the query is about a SPECIFIC restaurant (restaurant name mentioned), select ONLY that restaurant - NO alternatives
   - If the query is GENERAL (cuisine, vibe, food culture, "best X", etc.), select 1 top match + up to 5 alternatives
3. **Select restaurants** that best match the query:
   - For "best dessert" → prioritize dessert shops, bakeries, shave ice places
   - For "luau" → select luau restaurants
   - For "scenic view" → select restaurants with ocean/mountain views
   - For specific restaurant → select ONLY that restaurant
   - Prioritize award-winning restaurants (🏆 GOLD/SILVER AWARD WINNER)
4. **Generate your answer** using your knowledge + web search results + restaurant data
5. **Summarize web search results** naturally in your response

**ALL AVAILABLE RESTAURANTS ({len(ranked_results)} total):**
{all_restaurants_formatted}

**RESPONSE FORMAT - STRUCTURED (Same Structure, More Conversational Content):**

**1. OPENING (Natural Context - Optional):**
You can start with a conversational opening that shows understanding:
- "Maui offers some of the most [X] in Hawaii..."
- "Since you're looking for [X], here are the best options..."
- "I've found [X] restaurants that match what you're looking for..."

**2. ORGANIZATION (Categorize when you have multiple restaurants - Optional):**
You can group restaurants by type/vibe when appropriate:
- "Iconic Oceanfront Fine Dining"
- "Casual Beachside"
- "Panoramic Upcountry Views"
- "Local Favorites"
- "Upscale Special Occasion"
- "Dessert Shops & Bakeries"
- etc.

**3. FOR EACH RESTAURANT (STRUCTURED FORMAT - Same as Before):**

## Restaurant Name

[2-4 sentences in conversational style explaining why this restaurant matches. Include:
- What makes it special or notable
- Specific dishes/items to try ("Must-Try: X" or "The Y is a standout...")
- Why it fits the query
- Any cultural or contextual significance
- Write conversationally, like Gemini, but keep it concise]

**Good for:**
[On a SINGLE LINE, list menu items, dishes, or specialties this restaurant is known for. If menu items are provided in the restaurant data (in "Popular menu items" field), use those exactly. If not, mention the cuisine type and what that cuisine typically offers. Format as a simple comma-separated list on one line, e.g., "Authentic Curries, Vegetarian Options, Tandoori Dishes" or "The Polynesian Black Pearl, Fresh Catch of the Day, Hawaiian Specialties". Do NOT use bullet points or line breaks.]

**Vibe at this restaurant:**
[CRITICAL: Use the EXACT vibe_summary text from the restaurant data (restaurants.json). Do NOT paraphrase or summarize - use the vibe_summary field directly as it contains the authentic description of the restaurant's atmosphere and vibe. If vibe_summary is provided in the restaurant data, use it verbatim or with minimal editing to maintain its authentic character. DO NOT mention photos or images - they will be displayed automatically after this section.]

**Features:**
[On a SINGLE LINE, list the features from the restaurant data as a comma-separated list (e.g., "Outdoor Seating, Vegetarian Options, Wheelchair Accessible"). If no features, say "Not specified". Do NOT use bullet points.]

**Videos:**
[List video URLs if available, one per line with markdown links: "- [Video Title](url)". If no videos, say "No videos available"]

**4. CLOSING (Helpful Follow-up - Optional):**
You can end with a natural question or offer additional help:
- "Are you looking for a reservation for a specific date?"
- "Would you like recommendations for [related topic]?"
- "Are you staying in a specific part of Maui?"

**CRITICAL REMINDER**: 
- Select and include **up to 6 restaurants total** (1 top match + up to 5 alternatives)
- For queries about food culture (like "best dessert", "famous dishes"), showing multiple diverse options is essential
- **SELECT DIVERSE RESTAURANTS** - Don't pick the same restaurants for every query. Vary based on query intent.
- **DO NOT include the same restaurant twice** - if you select a restaurant as top match, don't include it in alternatives

**GEMINI-STYLE CONTENT GUIDELINES:**

1. **CONVERSATIONAL FLOW**: Write naturally, like you're talking to a friend. Use transitions: "While famous for X, Y also offers...", "If you're looking for something different...", "For a different vibe..."

2. **KNOWLEDGE-FIRST APPROACH**: 
   - Lead with what you know about the topic (Maui/Hawaii food culture, famous spots, cultural context)
   - Use restaurant data to validate and provide specific details
   - Use LLM reasoning and web search results to enhance your knowledge-based response

3. **NATURAL OPENING**: Start with context that shows understanding: "Maui offers some of the most [X]...", "Since you're looking for [X]..."

4. **CATEGORIZATION**: When you have multiple restaurants, group them by type/vibe (e.g., "Iconic Oceanfront Fine Dining", "Casual Beachside", "Panoramic Upcountry Views")

5. **SPECIFIC RECOMMENDATIONS**: 
   - Use "Must-Try: [specific dish]" format
   - Mention specific dishes, experiences, cultural significance
   - Use "Popular menu items" from data if available, otherwise use your knowledge

6. **USE WEB SEARCH & REASONING**: 
   - Incorporate web search results naturally: "Recent reviews mention..." or "According to recent discussions..."
   - Use LLM reasoning insights to understand cultural context and what really matters

7. **ACCURACY**: Only claim features/menu items that are in the restaurant data, but use your knowledge for context and recommendations

8. **CRITICAL - SELECT AND INCLUDE RESTAURANTS**: You MUST select and include up to 6 restaurants total (1 top match + up to 5 alternatives), each in the conversational format above. **DO NOT include the same restaurant twice - if you select a restaurant as top match, don't include it in alternatives.**

9. **VIBE SUMMARY**: CRITICAL - Use the EXACT vibe_summary text from restaurants.json verbatim. Do NOT paraphrase. This contains the authentic description of the restaurant's atmosphere and vibe.

10. **FEATURES**: List only features marked as True in the Features field, formatted as comma-separated list. If none, say "Not specified".

11. **VIDEOS**: If video URLs are provided, list them one per line with markdown links. If none, say "No videos available".

12. **PHOTOS**: DO NOT mention photos or images - they will be displayed automatically by the formatter.

13. **HELPFUL CLOSING**: End with a natural question or offer additional help: "Are you looking for X?" or "Would you like recommendations for Y?"

14. **WRITE LIKE GEMINI**: Be conversational, show your thinking naturally, organize by categories when appropriate, provide specific recommendations, and end with helpful follow-ups.
"""
        
        # Note: We use manual web search (Google Custom Search API) which is already integrated.
        # Web search results are passed to Gemini in the prompt (web_context).
        # Gemini's native web search tool is not available in the current SDK format.
        
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            # Higher temperature for more natural, conversational, human-like tone
            temperature=0.4,
            # Allow much longer, richer answers (top + up to 9 alternatives)
            max_tokens=10000,  # Increased to prevent cutting midway
        )

        explanation = response.choices[0].message["content"].strip()
        
        # Gemini has generated the answer AND selected restaurants in the explanation
        # The formatter will parse the explanation to extract restaurant names and inject images
        # Return a dummy top_match (formatter will extract actual restaurants from explanation)
        dummy_top_match = self._scored_to_match(ranked_results[0]) if ranked_results else None
        
        return AnaResponse(
            success=True,
            top_match=dummy_top_match,
            alternatives=[],  # Gemini selects restaurants in the explanation - formatter will handle image injection
            match_reasons=[],  # No ranking, so no match reasons
            explanation=explanation,
            confidence="high",  # Gemini handles selection, so confidence is high
        )

