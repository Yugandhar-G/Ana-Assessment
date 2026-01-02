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
            default_chat_model=os.getenv("GEMINI_CHAT_MODEL", "gemini-2.0-flash"),
        )
        self.model = model or os.getenv("GEMINI_CHAT_MODEL", "gemini-2.0-flash")
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
        # CRITICAL: Check for food culture queries FIRST - these should NEVER be treated as restaurant-specific
        query_lower = parsed_query.raw_query.lower()
        is_restaurant_specific = False
        
        # Food culture patterns that indicate general queries (should show alternatives)
        food_culture_patterns = [
            r'\bbest\b', r'\bfamous\b', r'\bpopular\b', r'\bfavorite\b', r'\bmust try\b',
            r'\bwhat\b.*\bis\b', r'\bwhat\b.*\bare\b', r'\bwhere to find\b',
            r'\bdessert\b', r'\bdesserts\b', r'\bdish\b', r'\bdishes\b', r'\bfood\b',
            r'\beat\b', r'\btry\b', r'\bspecialty\b', r'\bspecialties\b',
            r'\bknown for\b', r'\bfamous for\b', r'\brecommend\b', r'\bsuggest\b', r'\btop\b'
        ]
        
        import re
        is_food_culture_query = any(re.search(pattern, query_lower) for pattern in food_culture_patterns)
        
        # If it's a food culture query, NEVER treat as restaurant-specific
        if is_food_culture_query:
            print(f"[DEBUG] ✅ Food culture query detected - will show alternatives (NOT restaurant-specific)")
            is_restaurant_specific = False
        else:
            # Only check for restaurant names if it's NOT a food culture query
            # Normalize query and restaurant names for better matching (remove punctuation, extra spaces)
            query_normalized = re.sub(r'[^\w\s]', ' ', query_lower)
            query_normalized = ' '.join(query_normalized.split())
            
            # Check all restaurants in ranked results to see if any name appears in query
            # But require STRONG match (full name or most words) to avoid false positives
            for scored in ranked_results[:10]:  # Check top 10 to catch exact matches
                restaurant_name = (scored.restaurant.name or "").lower()
                if not restaurant_name:
                    continue
                
                # Normalize restaurant name
                name_normalized = re.sub(r'[^\w\s]', ' ', restaurant_name)
                name_normalized = ' '.join(name_normalized.split())
                
                # Method 1: Check if FULL restaurant name appears in query (most reliable)
                # This catches cases like "ULUPALAKUA RANCH STORE & GRILL" in query
                if name_normalized in query_normalized:
                    print(f"[DEBUG] Restaurant name match found: '{name_normalized}' in '{query_normalized}'")
                    is_restaurant_specific = True
                    break
                
                # Method 2: For multi-word names, require STRONG match (most words match)
                # This prevents false positives from single word matches
                common_words = {'the', 'a', 'an', 'at', 'for', 'with', 'about', 'what', 'where', 'when', 'how', 'is', 'are', 'was', 'were', 'restaurant', 'restaurants', 'can', 'you', 'tell', 'me', 'famous', 'for', 'and', 'or', 'but', 'best', 'in', 'maui', 'hawaii'}
                name_words = set(word for word in name_normalized.split() if word not in common_words and len(word) >= 4)  # Require 4+ chars to avoid common words
                query_words = set(word for word in query_normalized.split() if word not in common_words and len(word) >= 4)
                
                if name_words and len(name_words) >= 2:  # Only check multi-word names
                    matching_words = name_words & query_words
                    # Require at least 2 matching words AND 70%+ match for multi-word names
                    match_ratio = len(matching_words) / len(name_words) if name_words else 0
                    if len(matching_words) >= 2 and match_ratio >= 0.7:
                        print(f"[DEBUG] Strong restaurant name match: {len(matching_words)}/{len(name_words)} words match")
                        is_restaurant_specific = True
                        break
        
        # If it's a restaurant-specific query, return exactly 0 alternatives (only top match)
        if is_restaurant_specific:
            print(f"[DEBUG] ⚠️  Restaurant-specific query detected - returning 0 alternatives")
            print(f"[DEBUG]   Query: '{parsed_query.raw_query}'")
            return []
        
        print(f"[DEBUG] ✅ Not restaurant-specific - will select alternatives")

        if not ranked_results or len(ranked_results) <= 1:
            return []

        # Return up to 5 alternatives (for 6 total restaurants: 1 top + 5 alternatives)
        # For food culture queries, try to show maximum alternatives
        max_alternatives = min(max(0, max_total_results - 1), len(ranked_results) - 1)  # Don't exceed available results
        
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

        # Determine query type BEFORE the loop
        is_cuisine_query = bool(parsed_query.preferences.cuisine) and parsed_query.weights.cuisine >= 0.3
        is_vibe_query = parsed_query.weights.vibe >= 0.5
        
        # Detect general food culture queries (e.g., "best dessert", "famous dishes", "what to eat")
        query_lower = parsed_query.raw_query.lower()
        food_culture_indicators = [
            'best', 'famous', 'popular', 'favorite', 'must try', 'what', 'where to find',
            'dessert', 'desserts', 'dish', 'dishes', 'food', 'eat', 'try', 'specialty',
            'specialties', 'known for', 'famous for', 'recommend', 'suggest'
        ]
        is_food_culture_query = any(indicator in query_lower for indicator in food_culture_indicators)
        
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
    
    async def generate(
        self,
        parsed_query: ParsedQuery,
        ranked_results: list[ScoredRestaurant],
    ) -> AnaResponse:
        """Generate complete response with explanation."""
        if not ranked_results:
            return AnaResponse(
                success=False,
                explanation="I couldn't find any restaurants matching your criteria. Try broadening your search?",
                confidence="low",
                caveats=["No matching restaurants found"],
            )
        
        # STEP 1: Use LLM to reason about the query and results
        # This leverages LLM's general knowledge to validate and enhance recommendations
        ranked_results, llm_reasoning = await self._llm_reason_about_results(parsed_query, ranked_results)
        
        top_match = ranked_results[0]
        
        # For food culture queries, request more alternatives to show diversity
        query_lower = parsed_query.raw_query.lower()
        food_culture_indicators = [
            'best', 'famous', 'popular', 'favorite', 'must try', 'what', 'where to find',
            'dessert', 'desserts', 'dish', 'dishes', 'food', 'eat', 'try', 'specialty',
            'specialties', 'known for', 'famous for', 'recommend', 'suggest', 'top'
        ]
        is_food_culture_query = any(indicator in query_lower for indicator in food_culture_indicators)
        max_total = 6  # Show top 6 restaurants (1 top + 5 alternatives) - ranked properly
        
        alternatives = self._select_relevant_alternatives(ranked_results, top_match, parsed_query, max_total_results=max_total)
        print(f"[DEBUG] ResponseGenerator: Selected {len(alternatives)} alternatives out of {len(ranked_results)} ranked results")
        if alternatives:
            print(f"[DEBUG] Alternative restaurants: {[alt.restaurant.name for alt in alternatives[:5]]}")
        else:
            print(f"[DEBUG] WARNING: No alternatives selected! This might be why only one restaurant is shown.")
        restaurant = top_match.restaurant
        
        features_list = []
        for feature_key, feature_value in restaurant.features.items():
            if feature_value:
                features_list.append(feature_key.replace("_", " ").title())
        features_str = ", ".join(features_list) if features_list else "none specified"
        
        price_display = restaurant.price_level_curated or restaurant.price_level
        open_status = (
            "yes" if restaurant.is_open_now is True
            else "no" if restaurant.is_open_now is False
            else "unknown"
        )
        meal_times = ", ".join(restaurant.serves_meal_times) if restaurant.serves_meal_times else "not specified"
        phones = ", ".join(filter(None, [restaurant.national_phone, restaurant.international_phone])) or "not provided"
        
        # Get video URLs for this restaurant
        video_urls = get_video_urls_for_restaurant(
            restaurant_id=restaurant.id,
            restaurant_name=restaurant.name,
            google_place_id=getattr(restaurant, "google_place_id", None)
        )
        
        # Generate match reasons to provide context about WHY this restaurant matches
        match_reasons = self._generate_match_reasons(top_match, parsed_query)
        
        # Build match context string to help LLM understand what matched well
        match_context_lines = []
        for reason in match_reasons[:3]:  # Top 3 reasons
            importance_label = "🔹 Primary match" if reason.importance == "primary" else "🔸 Secondary match" if reason.importance == "secondary" else "• Minor match"
            match_context_lines.append(f"{importance_label}: {reason.signal.upper()} - User wanted: {reason.query_wanted[:80]} | Restaurant has: {reason.restaurant_has[:80]} (score: {reason.score:.2f})")
        match_context = "\n".join(match_context_lines) if match_context_lines else "General match across multiple criteria"
        
        # Get parking information if available
        parking_info = ""
        if hasattr(restaurant, "parking_options") and restaurant.parking_options:
            try:
                import ast
                parking_data = ast.literal_eval(restaurant.parking_options) if isinstance(restaurant.parking_options, str) else restaurant.parking_options
                parking_details = []
                if parking_data.get("freeParkingLot"):
                    parking_details.append("free parking lot")
                if parking_data.get("paidParkingLot"):
                    parking_details.append("paid parking lot")
                if parking_data.get("freeGarageParking"):
                    parking_details.append("free garage parking")
                if parking_data.get("freeStreetParking"):
                    parking_details.append("free street parking")
                if parking_data.get("valetParking"):
                    parking_details.append("valet parking")
                if parking_details:
                    parking_info = f"Parking: {', '.join(parking_details)}"
            except:
                parking_info = f"Parking: {restaurant.parking_options}" if restaurant.parking_options else ""
        
        # Check for awards
        from .fusion import has_award, get_award_level
        award_info = ""
        if has_award(restaurant):
            award_level = get_award_level(restaurant)
            if award_level >= 1.0:
                award_info = "🏆 AWARD-WINNING: Gold Award Winner"
            elif award_level >= 0.8:
                award_info = "🏆 AWARD-WINNING: Silver Award Winner"
            elif award_level >= 0.5:
                award_info = "🏆 AWARD-WINNING: Honorable Mention"
            else:
                award_info = "🏆 AWARD-WINNING: Award Recipient"
            # Extract award details from highlights
            if restaurant.highlights:
                award_info += f" - {restaurant.highlights}"
        
        # Clean restaurant info - only include user-relevant details (no internal metadata)
        restaurant_info = f"""
Restaurant: {restaurant.name}
Cuisine: {restaurant.cuisine}
Price: {price_display}
Location: {restaurant.region}{f", {getattr(restaurant, 'city', None)}" if getattr(restaurant, 'city', None) else ""}
Address: {getattr(restaurant, "formatted_address", None) or "not provided"}
Rating: {restaurant.rating}/5.0 ⭐ (out of 5.0)
{award_info if award_info else ""}
Features: {features_str}
Status: {restaurant.business_status or "unknown"} | Open now: {open_status}
Hours: {restaurant.opening_hours_text or "not provided"}
Meals served: {meal_times}
{f"Phone: {phones}" if phones != "not provided" else ""}
{f"Website: {restaurant.website_uri}" if restaurant.website_uri else ""}
{f"Map: {restaurant.google_maps_uri}" if restaurant.google_maps_uri else ""}
{parking_info}

Description: {restaurant.highlights or restaurant.details or restaurant.vibe.vibe_summary}
Vibe Summary (USE THIS FOR "Vibe at this restaurant" section): {restaurant.vibe.vibe_summary or restaurant.highlights or restaurant.details or "Not specified"}
{f"Popular menu items: {', '.join(getattr(restaurant, 'top_menu_items', [])[:5])}" if hasattr(restaurant, 'top_menu_items') and getattr(restaurant, 'top_menu_items', []) else ""}
Atmosphere: {restaurant.vibe.formality} | Noise: {restaurant.vibe.noise_level} | Tags: {', '.join(restaurant.vibe.atmosphere_tags) if restaurant.vibe.atmosphere_tags else 'not specified'}

{f"Photos: {', '.join(restaurant.restaurant_photos_urls[:5])}" if restaurant.restaurant_photos_urls else (f"Photos: {', '.join(restaurant.photos_urls[:5])}" if restaurant.photos_urls else "")}
{f"Videos: {', '.join(video_urls)}" if video_urls else ""}
"""
        
        # Photos are handled by the formatter - don't mention them in the LLM response
        # The formatter will display images after the vibe section automatically
        
        # Build video note if videos are available
        video_note = ""
        if video_urls:
            video_list = "\n".join([f"  - {url}" for url in video_urls])
            video_note = f"\nIMPORTANT: This restaurant has {len(video_urls)} video(s) available:\n{video_list}\nYou should mention these videos in your response and include the video links."
        
        # Build alternatives information
        alternatives_info = ""
        print(f"[DEBUG] Building alternatives_info: {len(alternatives)} alternatives available")
        if alternatives:
            alternatives_info = "\n\n**Also Consider (Alternative Options):**\n"
            for i, alt in enumerate(alternatives, 1):
                alt_restaurant = alt.restaurant
                alt_features = ", ".join([k.replace("_", " ").title() for k, v in alt_restaurant.features.items() if v]) or "various features"
                alt_video_urls = get_video_urls_for_restaurant(
                    restaurant_id=alt_restaurant.id,
                    restaurant_name=alt_restaurant.name,
                    google_place_id=getattr(alt_restaurant, "google_place_id", None)
                )
                # Check for awards in alternatives
                from .fusion import has_award, get_award_level
                alt_award_info = ""
                if has_award(alt_restaurant):
                    alt_award_level = get_award_level(alt_restaurant)
                    if alt_award_level >= 1.0:
                        alt_award_info = " | 🏆 Gold Award Winner"
                    elif alt_award_level >= 0.8:
                        alt_award_info = " | 🏆 Silver Award Winner"
                    elif alt_award_level >= 0.5:
                        alt_award_info = " | 🏆 Honorable Mention"
                    else:
                        alt_award_info = " | 🏆 Award Recipient"
                
                alt_photo_urls = alt_restaurant.restaurant_photos_urls[:3] if alt_restaurant.restaurant_photos_urls else (alt_restaurant.photos_urls[:3] if alt_restaurant.photos_urls else [])
                alternatives_info += f"""
{i}. **{alt_restaurant.name}**
   - Cuisine: {alt_restaurant.cuisine} | Price: {alt_restaurant.price_level_curated or alt_restaurant.price_level}
   - Location: {alt_restaurant.region}{f", {getattr(alt_restaurant, 'city', None)}" if getattr(alt_restaurant, 'city', None) else ""}
   - Rating: {alt_restaurant.rating}/5.0 ⭐{alt_award_info}
   - Features: {alt_features}
   - Description: {alt_restaurant.highlights or alt_restaurant.details or alt_restaurant.vibe.vibe_summary[:150]}...
   - Vibe Summary (USE THIS FOR "Vibe at this restaurant" section): {alt_restaurant.vibe.vibe_summary or alt_restaurant.highlights or alt_restaurant.details or "Not specified"}
   {f"- Photos: {', '.join(alt_photo_urls)}" if alt_photo_urls else ""}
   {f"- Videos: {', '.join(alt_video_urls)}" if alt_video_urls else ""}
"""
        
        # CRITICAL: Log if alternatives are being included
        print(f"[DEBUG] LLM Prompt: Including {len(alternatives)} alternatives in prompt")
        print(f"[DEBUG] alternatives_info length: {len(alternatives_info)} characters")
        if alternatives_info:
            print(f"[DEBUG] ✅ alternatives_info preview: {alternatives_info[:200]}...")
        if not alternatives:
            print(f"[DEBUG] ⚠️  WARNING: No alternatives to include! This will result in only one restaurant being shown.")
        
        user_prompt = f"""
User Query: {parsed_query.raw_query}

**LLM REASONING & CONTEXT (Use this to inform your response):**
{llm_reasoning if llm_reasoning else "No additional reasoning available."}

**RESTAURANT DATA (Use as supporting examples, but lead with your knowledge):**
Top Match: {restaurant.name}
{restaurant_info}

Why this restaurant matches (for reference):
{match_context}
{video_note}
{alternatives_info}

**CRITICAL: YOU HAVE {len(alternatives)} ALTERNATIVE RESTAURANTS LISTED ABOVE. YOU MUST INCLUDE ALL {len(alternatives)} OF THEM IN YOUR RESPONSE. DO NOT SKIP ANY.**

**IMPORTANT INSTRUCTIONS - STRUCTURED RESPONSE FORMAT:**
**CRITICAL: You MUST format your response using this exact structure for EACH restaurant:**

For the TOP MATCH, start with:
## {restaurant.name}

[2-3 sentences explaining why this restaurant matches the query]

**Good for:**
[On a SINGLE LINE, list menu items from "Popular menu items" field if available, or mention cuisine specialties. Format as comma-separated list on one line, e.g., "Authentic Curries, Vegetarian Options, Tandoori Dishes". Do NOT use bullet points.]

**Vibe at this restaurant:**
[CRITICAL: Use the EXACT vibe_summary text from the restaurant data (restaurants.json). The vibe_summary field contains the authentic description of the restaurant's atmosphere, vibe, and dining experience. Use it directly - do NOT paraphrase or create your own description. If vibe_summary is provided, use it verbatim or with minimal editing to preserve its authentic character.]

**Features:**
[On a SINGLE LINE, list features from the Features field as comma-separated list. If none, say "Not specified".]

**Videos:**
[List video URLs if available, one per line: "- [url](url)"]

Then, for EACH alternative restaurant, use the SAME structure:
## Alternative Restaurant Name

[2-3 sentences about this alternative]

**Good for:**
[On a SINGLE LINE, menu items or cuisine specialties as comma-separated list]

**Vibe at this restaurant:**
[CRITICAL: Use the EXACT vibe_summary text from the restaurant data (restaurants.json). Use it directly - do NOT paraphrase.]

**Features:**
[On a SINGLE LINE, features list as comma-separated]

**Videos:**
[Video links if available]

**CRITICAL REMINDER**: You have {len(alternatives)} alternative restaurants listed above. You MUST include ALL {len(alternatives)} of them in your response, each in the structured format shown above. Do NOT skip any alternatives. For queries about food culture (like "best dessert", "famous dishes"), showing multiple diverse options is essential - users want to see variety.

**CONTENT GUIDELINES:**
1. **USE LLM REASONING**: The "LLM REASONING & CONTEXT" section above contains insights from analyzing this query. Use these insights to provide a more informed, contextually aware response. If the reasoning suggests certain restaurants are better fits or identifies cultural context, incorporate that into your response.
2. **START WITH YOUR EXPERTISE**: For general questions, begin with your knowledge about the topic, then connect to restaurants using the structured format above. Use the LLM reasoning to enhance your knowledge-based response.
3. **COMBINE KNOWLEDGE + DATA**: Use your general knowledge about Maui/Hawaii food culture AND the restaurant data together. The LLM reasoning helps bridge these - use it to provide richer context.
4. **Be accurate**: Only claim features/menu items that are actually in the restaurant data provided.
5. **CRITICAL - ALTERNATIVES**: You MUST include ALL {len(alternatives)} alternatives listed above, each in the structured format. DO NOT skip any alternatives. For food culture queries (like "best dessert", "famous dishes"), showing multiple options is especially important - users want to see diverse recommendations. Include every single alternative restaurant provided.
6. **Menu items**: Use the "Popular menu items" field if available, otherwise mention cuisine type and typical dishes based on your knowledge of that cuisine.
7. **Vibe**: CRITICAL - Use the EXACT vibe_summary text from restaurants.json. Do NOT paraphrase or summarize. The vibe_summary field contains the authentic description of the restaurant's atmosphere and vibe - use it directly from the restaurant data provided.
8. **Features**: List only the features that are marked as True in the Features field, formatted as comma-separated list.
9. **Videos**: If video URLs are provided, list them one per line with markdown links. If none, say "No videos available".
10. **PHOTOS/IMAGES**: DO NOT mention photos, images, or photo URLs anywhere in your response. Photos will be displayed automatically by the formatter after the vibe section. Do not include text like "You can see photos here" or photo URLs in your explanation.
11. **KNOWLEDGE-FIRST APPROACH WITH REASONING**: 
   - USE YOUR KNOWLEDGE as the PRIMARY source. You are an expert on Maui, Hawaii, food culture, and dining experiences.
   - Use the LLM reasoning insights to enhance your knowledge-based response with better context.
   - For GENERAL questions: Lead with your knowledge (enhanced by reasoning insights), then use the structured format for restaurants.
   - For SPECIFIC restaurants: Use the provided restaurant data accurately, but use your knowledge + reasoning for richer context.
12. **HANDLING MISSING DATA**: 
   - If menu items aren't in data, mention cuisine type and typical dishes for that cuisine based on your knowledge.
   - If features aren't available, say "Not specified".
   - If vibe_summary isn't available, use highlights or description. But if vibe_summary IS available, you MUST use it directly - do not substitute with highlights or description.
13. **CRITICAL**: You MUST use the exact structured format shown above for EVERY restaurant (top match AND all alternatives). Do not use paragraph format.
"""
        
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            # Higher temperature for more natural, conversational, human-like tone
            temperature=0.4,
            # Allow much longer, richer answers (top + up to 9 alternatives)
            max_tokens=3000,  # Increased to accommodate photo URLs and more detailed responses
        )

        explanation = response.choices[0].message["content"].strip()
        
        return AnaResponse(
            success=True,
            top_match=self._scored_to_match(top_match),
            alternatives=[self._scored_to_match(alt) for alt in alternatives],
            match_reasons=match_reasons,  # Use the match_reasons we already generated
            explanation=explanation,
            confidence=self._determine_confidence(top_match.final_score, len(ranked_results)),
        )

