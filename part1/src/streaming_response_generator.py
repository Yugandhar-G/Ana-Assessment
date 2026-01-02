"""Streaming response generator for Part 1 - streams tokens for better latency."""
import os
from pathlib import Path
from typing import AsyncGenerator
from .gemini_client import AsyncGeminiClient
from .schemas import ParsedQuery, RestaurantMatch, MatchReason
from .fusion import ScoredRestaurant
from .video_lookup import get_video_urls_for_restaurant


class StreamingResponseGenerator:
    """Generate streaming responses with token-by-token output."""
    
    def __init__(self, client: AsyncGeminiClient | None = None, model: str | None = None):
        self.client = client or AsyncGeminiClient(
            api_key=os.getenv("GEMINI_API_KEY"),
            default_chat_model=os.getenv("GEMINI_CHAT_MODEL", "gemini-2.0-flash"),
        )
        self.model = model or os.getenv("GEMINI_CHAT_MODEL", "gemini-2.0-flash")
        self.system_prompt = self._load_prompt()
    
    def _load_prompt(self) -> str:
        """Load system prompt."""
        prompt_path = Path(__file__).parent.parent / "prompts" / "response_generator.txt"
        if prompt_path.exists():
            return prompt_path.read_text()
        return "Generate a helpful restaurant recommendation response."
    
    def _select_relevant_alternatives(
        self,
        ranked_results: list[ScoredRestaurant],
        top_match: ScoredRestaurant,
        parsed_query: ParsedQuery,
        max_total_results: int = 10,
        base_min_score: float = 0.4,
    ) -> list[ScoredRestaurant]:
        """
        Select alternatives purely based on relevance and ranking.

        - Do NOT force diversity across cuisine/price/region.
        - If query is about a SPECIFIC restaurant, return [] (only show that restaurant).
        - If query is GENERAL (cuisine/item/vibe), return all relevant alternatives.
        - If only 1 restaurant is relevant, return [] (only top match is shown).
        - If N relevant (N > 1), return up to max_total_results - 1 alternatives,
          in the same order as ranked_results.
        """
        
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
                if len(name_words) <= 3:
                    if len(matching_words) >= len(name_words) - 1:  # Allow 1 word difference for typos
                        is_restaurant_specific = True
                        break
                else:
                    if len(matching_words) >= min(3, max(2, int(len(name_words) * 0.6))):
                        is_restaurant_specific = True
                        break
        
        # If it's a restaurant-specific query, don't show alternatives
        if is_restaurant_specific:
            return []

        if not ranked_results or len(ranked_results) <= 1:
            return []

        max_alternatives = max(0, max_total_results - 1)

        top_score = top_match.final_score
        # Dynamic minimum score: don't include alternatives that are much worse than top
        # For perfect matches (score 1.0), use a lower threshold to include more options
        if top_score >= 0.95:
            dynamic_min_score = max(base_min_score, top_score * 0.7)  # More lenient for perfect matches
        else:
            dynamic_min_score = max(base_min_score, top_score * 0.6)

        alternatives: list[ScoredRestaurant] = []

        top_name = (top_match.restaurant.name or "").lower()

        for scored in ranked_results[1:]:
            if len(alternatives) >= max_alternatives:
                break

            # Hard floor on score
            if scored.final_score < dynamic_min_score:
                continue

            restaurant = scored.restaurant

            # Skip non-operational
            if restaurant.business_status and restaurant.business_status != "OPERATIONAL":
                continue

            restaurant_name = (restaurant.name or "").lower()
            restaurant_cuisine = (restaurant.cuisine or "").lower()
            restaurant_price = restaurant.price_level
            restaurant_region = (restaurant.region or "").lower() if restaurant.region else ""

            # Skip if same restaurant as top
            if restaurant_name == top_name:
                continue

            # --- Relevance checks (no diversity heuristics) ---
            is_relevant = True

            # 1) Cuisine relevance if user explicitly asked for cuisine
            if parsed_query.preferences.cuisine and parsed_query.weights.cuisine >= 0.3:
                # Use similar normalization/variation rules as cuisine scorer & hard filter
                def _normalize_cuisine(value: str) -> str:
                    return value.lower().strip().replace("/", " ").replace("-", " ")

                def _expand_cuisine_variations(cuisine: str) -> set[str]:
                    """
                    Expand cuisine to include common variations.

                    Critical for Hawaiian queries:
                    - Treat "Hawaiian" and "Hawaii Regional" as compatible.
                    """
                    normalized = _normalize_cuisine(cuisine)
                    variations: set[str] = {normalized}

                    # Hawaiian variations - handle "Hawaiian" matching "Hawaii Regional"
                    if "hawaiian" in normalized or "hawaii" in normalized:
                        variations.add("hawaiian")
                        variations.add("hawaii regional")
                        variations.add("hawaii")

                    return variations

                restaurant_cuisine_normalized = _normalize_cuisine(restaurant.cuisine or "")
                restaurant_cuisine_parts = [
                    _normalize_cuisine(part.strip())
                    for part in (restaurant.cuisine or "").split(",")
                    if part.strip()
                ]

                cuisine_match = False
                for preferred in parsed_query.preferences.cuisine:
                    preferred_variations = _expand_cuisine_variations(preferred)

                    for variation in preferred_variations:
                        # Check full cuisine string
                        if (
                            variation in restaurant_cuisine_normalized
                            or restaurant_cuisine_normalized in variation
                        ):
                            cuisine_match = True
                            break

                        # Check individual cuisine parts
                        for part in restaurant_cuisine_parts:
                            if variation in part or part in variation:
                                cuisine_match = True
                                break

                        if cuisine_match:
                            break

                    if cuisine_match:
                        break

                if not cuisine_match:
                    is_relevant = False

            # 2) Price relevance if user specified price with some importance
            if is_relevant and parsed_query.preferences.price:
                if restaurant_price not in parsed_query.preferences.price:
                    if parsed_query.weights.price >= 0.3:
                        is_relevant = False

            # 3) Location relevance if user specified a location
            if is_relevant and parsed_query.location:
                query_loc = parsed_query.location.lower().strip()

                # Island names - treat like global Maui search (no extra filtering here),
                # since HardFilter already handled island-level logic.
                island_names = {
                    "maui",
                    "hawaii",
                    "hawaiian islands",
                    "oahu",
                    "kauai",
                    "big island",
                    "molokai",
                    "lanai",
                }
                if query_loc not in island_names:
                    region_match = query_loc in restaurant_region if restaurant_region else False
                    city = getattr(restaurant, "city", None)
                    city_match = query_loc in city.lower() if city else False
                    if not (region_match or city_match):
                        is_relevant = False

            # 4) Feature relevance if user specified features with some importance
            if (
                is_relevant
                and parsed_query.preferences.features
                and parsed_query.weights.features >= 0.3
            ):
                has_feature = any(
                    restaurant.features.get(feature.lower().replace(" ", "_"), False)
                    or restaurant.features.get(feature.lower(), False)
                    for feature in parsed_query.preferences.features
                )
                if not has_feature and scored.feature_score < 0.3:
                    is_relevant = False

            if not is_relevant:
                continue

            # If it passed all checks, keep it in the same order as ranking
            alternatives.append(scored)

        # No extra sorting: keep fusion ranking order
        return alternatives[:max_alternatives]
    
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
    
    def _scored_to_match(self, scored: ScoredRestaurant) -> RestaurantMatch:
        """Convert ScoredRestaurant to RestaurantMatch."""
        from .schemas.response import RestaurantMatch
        
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
    
    async def generate_stream(
        self,
        parsed_query: ParsedQuery,
        ranked_results: list[ScoredRestaurant],
    ) -> AsyncGenerator[dict, None]:
        """Generate streaming response with token-by-token output.
        
        Yields:
            - First: metadata dict with restaurant info
            - Then: token chunks as they're generated
            - Finally: completion dict
        """
        if not ranked_results:
            yield {
                "type": "error",
                "message": "No restaurants found matching your criteria."
            }
            return
        
        top_match = ranked_results[0]
        alternatives = self._select_relevant_alternatives(ranked_results, top_match, parsed_query)
        restaurant = top_match.restaurant
        
        # Build restaurant info
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
        
        # Get video URLs
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
        
        # Yield metadata first
        yield {
            "type": "metadata",
            "top_match": self._scored_to_match(top_match).model_dump(),
            "alternatives": [self._scored_to_match(alt).model_dump() for alt in alternatives],
            "confidence": self._determine_confidence(top_match.final_score, len(ranked_results)),
        }
        
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
                
                alternatives_info += f"""
{i}. **{alt_restaurant.name}**
   - Cuisine: {alt_restaurant.cuisine} | Price: {alt_restaurant.price_level_curated or alt_restaurant.price_level}
   - Location: {alt_restaurant.region}{f", {getattr(alt_restaurant, 'city', None)}" if getattr(alt_restaurant, 'city', None) else ""}
   - Rating: {alt_restaurant.rating}/5.0 ⭐{alt_award_info}
   - Features: {alt_features}
   - Description: {alt_restaurant.highlights or alt_restaurant.details or alt_restaurant.vibe.vibe_summary[:150]}...
   - Vibe Summary (USE THIS FOR "Vibe at this restaurant" section): {alt_restaurant.vibe.vibe_summary or alt_restaurant.highlights or alt_restaurant.details or "Not specified"}
   {f"- Photos: {', '.join(alt_restaurant.restaurant_photos_urls[:3])}" if alt_restaurant.restaurant_photos_urls else (f"- Photos: {', '.join(alt_restaurant.photos_urls[:3])}" if alt_restaurant.photos_urls else "")}
   {f"- Videos: {', '.join(alt_video_urls)}" if alt_video_urls else ""}
"""
        
        user_prompt = f"""
User Query: {parsed_query.raw_query}

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
[CRITICAL: Use the EXACT vibe_summary text from the restaurant data (restaurants.json). Use it directly - do NOT paraphrase.]

**Features:**
[On a SINGLE LINE, features list as comma-separated]

**Videos:**
[Video links if available]

**CONTENT GUIDELINES:**
1. **START WITH YOUR EXPERTISE**: For general questions, begin with your knowledge about the topic, then connect to restaurants using the structured format above.
2. **Be accurate**: Only claim features/menu items that are actually in the restaurant data provided.
3. **For alternatives**: You MUST include ALL {len(alternatives)} alternatives listed above, each in the structured format.
4. **Menu items**: Use the "Popular menu items" field if available, otherwise mention cuisine type and typical dishes.
5. **Vibe**: CRITICAL - Use the EXACT vibe_summary text from restaurants.json. Do NOT paraphrase or summarize. The vibe_summary field contains the authentic description of the restaurant's atmosphere and vibe - use it directly from the restaurant data provided.
6. **Features**: List only the features that are marked as True in the Features field, formatted as comma-separated list.
7. **Videos**: If video URLs are provided, list them one per line with markdown links. If none, say "No videos available".
8. **PHOTOS/IMAGES**: DO NOT mention photos, images, or photo URLs anywhere in your response. Photos will be displayed automatically by the formatter after the vibe section. Do not include text like "You can see photos here" or photo URLs in your explanation.
9. **KNOWLEDGE-FIRST APPROACH**: 
   - USE YOUR KNOWLEDGE as the PRIMARY source. You are an expert on Maui, Hawaii, food culture, and dining experiences.
   - For GENERAL questions: Lead with your knowledge, then use the structured format for restaurants.
   - For SPECIFIC restaurants: Use the provided restaurant data accurately, but use your knowledge for context.
10. **HANDLING MISSING DATA**: 
   - If menu items aren't in data, mention cuisine type and typical dishes for that cuisine.
   - If features aren't available, say "Not specified".
   - If vibe_summary isn't available, use highlights or description. But if vibe_summary IS available, you MUST use it directly - do not substitute with highlights or description.
11. **CRITICAL**: You MUST use the exact structured format shown above for EVERY restaurant (top match AND all alternatives). Do not use paragraph format.
"""
        
        # Stream response from LLM
        try:
            async for chunk in self.client.chat.completions.create_stream(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.4,  # Higher temperature for more natural, conversational, human-like tone
                # Allow much longer, richer answers for streaming as well
                max_tokens=3000,  # Increased to accommodate photo URLs and more detailed responses
            ):
                yield {
                    "type": "token",
                    "content": chunk.get("content", ""),
                    "done": chunk.get("done", False)
                }
        except Exception as e:
            yield {
                "type": "error",
                "message": f"Error generating response: {str(e)}"
            }
        
        # Yield completion
        yield {
            "type": "done"
        }
    
    def _determine_confidence(self, top_score: float, num_results: int) -> str:
        """Determine confidence level based on match quality."""
        if top_score >= 0.8 and num_results >= 1:
            return "high"
        elif top_score >= 0.6:
            return "medium"
        else:
            return "low"

