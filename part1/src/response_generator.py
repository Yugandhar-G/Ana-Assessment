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
        max_total_results: int = 10,
        base_min_score: float = 0.3,  # Lowered from 0.4 to include more restaurants
    ) -> list[ScoredRestaurant]:
        """
        Select alternatives purely based on relevance and ranking.

        - If query is about a SPECIFIC restaurant, return [] (only show that restaurant).
        - If query is GENERAL (cuisine/item/vibe), return up to max_total_results - 1 alternatives.
        - Returns up to 9 alternatives (for 10 total restaurants including top match).
        """
        # Check if query is about a specific restaurant
        
        # Check if query is about a specific restaurant
        # Check if ANY restaurant name from ranked results appears in the query
        query_lower = parsed_query.raw_query.lower()
        is_restaurant_specific = False
        
        # Normalize query and restaurant names for better matching (remove punctuation, extra spaces)
        import re
        query_normalized = re.sub(r'[^\w\s]', ' ', query_lower)
        query_normalized = ' '.join(query_normalized.split())
        
        # Check all restaurants in ranked results to see if any name appears in query
        for scored in ranked_results[:10]:  # Check top 10 to catch exact matches
            restaurant_name = (scored.restaurant.name or "").lower()
            if not restaurant_name:
                continue
            
            # Normalize restaurant name
            name_normalized = re.sub(r'[^\w\s]', ' ', restaurant_name)
            name_normalized = ' '.join(name_normalized.split())
            
            # Method 1: Check if restaurant name appears as substring in query (most reliable)
            # This catches cases like "ULUPALAKUA RANCH STORE & GRILL" in query
            if name_normalized in query_normalized:
                print(f"DEBUG: Restaurant name match found: '{name_normalized}' in '{query_normalized}'")
                is_restaurant_specific = True
                break
            
            # Method 2: Check if significant unique words from restaurant name appear in query
            # Remove only very common words, keep restaurant-specific words like "store", "grill", etc.
            common_words = {'the', 'a', 'an', 'at', 'for', 'with', 'about', 'what', 'where', 'when', 'how', 'is', 'are', 'was', 'were', 'restaurant', 'restaurants', 'can', 'you', 'tell', 'me', 'famous', 'for', 'and', 'or', 'but'}
            name_words = set(word for word in name_normalized.split() if word not in common_words and len(word) >= 3)
            query_words = set(word for word in query_normalized.split() if word not in common_words and len(word) >= 3)
            
            if name_words:
                matching_words = name_words & query_words
                # If 3+ words match, or if 60%+ of name words match, it's restaurant-specific
                # For short names (2-3 unique words), require most/all to match
                # CRITICAL: Require at least 1 matching word (0 matches should never trigger)
                if len(name_words) <= 3:
                    # For 1-word names, require exact match. For 2-3 words, allow 1 word difference
                    min_matches = max(1, len(name_words) - 1) if len(name_words) > 1 else 1
                    if len(matching_words) >= min_matches:
                        is_restaurant_specific = True
                        break
                else:
                    if len(matching_words) >= min(3, max(2, int(len(name_words) * 0.6))):
                        is_restaurant_specific = True
                        break
        
        # If it's a restaurant-specific query, return exactly 0 alternatives (only top match)
        if is_restaurant_specific:
            return []

        if not ranked_results or len(ranked_results) <= 1:
            return []

        # Return up to 9 alternatives (for 10 total restaurants: 1 top + 9 alternatives)
        max_alternatives = min(max(0, max_total_results - 1), len(ranked_results) - 1)  # Don't exceed available results

        top_score = top_match.final_score
        # More lenient scoring for general queries - include more restaurants
        # For perfect matches (score 1.0), use a much lower threshold to include more options
        if top_score >= 0.95:
            dynamic_min_score = max(base_min_score, top_score * 0.5)  # Very lenient for perfect matches
        else:
            dynamic_min_score = max(base_min_score, top_score * 0.4)  # More lenient threshold

        # Determine query type BEFORE the loop
        is_cuisine_query = bool(parsed_query.preferences.cuisine) and parsed_query.weights.cuisine >= 0.3
        is_vibe_query = parsed_query.weights.vibe >= 0.5
        is_general_query = is_cuisine_query or is_vibe_query or not is_restaurant_specific
        

        alternatives: list[ScoredRestaurant] = []

        top_name = (top_match.restaurant.name or "").lower()
        
        # DEBUG: Check what we're iterating over
        restaurants_to_check = ranked_results[1:]
        # print(f"DEBUG: Checking {len(restaurants_to_check)} restaurants (from {len(ranked_results)} total, skipping first)")

        for scored in restaurants_to_check:
            if len(alternatives) >= max_alternatives:
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

            # For cuisine queries: include if cuisine_score is good (simplified - no other filters)
            # For other queries: check score threshold and apply filters
            if is_cuisine_query:
                # For cuisine queries, include if cuisine_score >= 0.7 OR final_score is good
                # This ensures we show all relevant cuisine matches up to the limit
                # IMPORTANT: For cuisine/vibe/food item queries, show up to 10 restaurants total
                cuisine_condition = scored.cuisine_score >= 0.7
                score_condition = scored.final_score >= dynamic_min_score
                if cuisine_condition or score_condition:
                    alternatives.append(scored)
                # Always continue for cuisine queries (don't fall through to else block)
                continue
            else:
                # For non-cuisine queries, check score threshold and apply filters
                if scored.final_score < dynamic_min_score:
                    continue
                
                # Apply additional filters for non-cuisine queries
                is_relevant = True
                
                # Price relevance - only filter if price is explicitly important
                if parsed_query.preferences.price and parsed_query.weights.price >= 0.5:
                    if restaurant_price not in parsed_query.preferences.price:
                        is_relevant = False
                
                # Location relevance - skip for island names (already filtered by HardFilter)
                if is_relevant and parsed_query.location:
                    query_loc = parsed_query.location.lower().strip()
                    island_names = {
                        "maui", "hawaii", "hawaiian islands", "oahu", "kauai",
                        "big island", "molokai", "lanai",
                    }
                    # Skip location filtering for island-level queries
                    if query_loc not in island_names:
                        region_match = query_loc in restaurant_region if restaurant_region else False
                        city = getattr(restaurant, "city", None)
                        city_match = query_loc in city.lower() if city else False
                        if not (region_match or city_match):
                            is_relevant = False
                
                # Feature relevance - only filter if features are very important
                if (
                    is_relevant
                    and parsed_query.preferences.features
                    and parsed_query.weights.features >= 0.5
                ):
                    has_feature = any(
                        restaurant.features.get(feature.lower().replace(" ", "_"), False)
                        or restaurant.features.get(feature.lower(), False)
                        for feature in parsed_query.preferences.features
                    )
                    if not has_feature and scored.feature_score < 0.3:
                        is_relevant = False

                if is_relevant:
                    alternatives.append(scored)

        # No extra sorting: keep fusion ranking order
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
        alternatives = self._select_relevant_alternatives(ranked_results, top_match, parsed_query)
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
[Vibe description]

**Features:**
[On a SINGLE LINE, features list as comma-separated]

**Videos:**
[Video links if available]

**CONTENT GUIDELINES:**
1. **USE LLM REASONING**: The "LLM REASONING & CONTEXT" section above contains insights from analyzing this query. Use these insights to provide a more informed, contextually aware response. If the reasoning suggests certain restaurants are better fits or identifies cultural context, incorporate that into your response.
2. **START WITH YOUR EXPERTISE**: For general questions, begin with your knowledge about the topic, then connect to restaurants using the structured format above. Use the LLM reasoning to enhance your knowledge-based response.
3. **COMBINE KNOWLEDGE + DATA**: Use your general knowledge about Maui/Hawaii food culture AND the restaurant data together. The LLM reasoning helps bridge these - use it to provide richer context.
4. **Be accurate**: Only claim features/menu items that are actually in the restaurant data provided.
5. **For alternatives**: You MUST include ALL {len(alternatives)} alternatives listed above, each in the structured format.
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

