import json
import os
import re
from pathlib import Path
from typing import Optional, List
from .gemini_client import AsyncGeminiClient
from .schemas import ParsedQuery, MustNotFilters, Preferences, SignalWeights


class QueryParser:
    """Parse natural language queries into structured search intent using LLM."""
    
    def __init__(self, client: AsyncGeminiClient | None = None, model: str | None = None):
        self.client = client or AsyncGeminiClient(
            api_key=os.getenv("GEMINI_API_KEY"),
            default_chat_model=os.getenv("GEMINI_CHAT_MODEL", "gemini-2.0-flash"),
        )
        self.model = model or os.getenv("GEMINI_CHAT_MODEL", "gemini-2.0-flash")
        self.system_prompt = self._load_prompt()
        self._restaurant_names: Optional[List[str]] = None
        # Cache for query classification and parsing to reduce latency
        self._query_classification_cache: dict[str, bool] = {}
        self._query_parse_cache: dict[str, ParsedQuery] = {}
    
    def _load_prompt(self) -> str:
        prompt_path = Path(__file__).parent.parent / "prompts" / "query_parser.txt"
        if prompt_path.exists():
            return prompt_path.read_text()
        return "Parse the user's restaurant query into structured search intent."
    
    def _normalize_common_typos(self, query: str) -> str:
        """Normalize common typos in cuisine names and other terms before sending to LLM.
        
        This helps the LLM understand user intent even with spelling mistakes.
        """
        # Common misspellings → correct spelling (using word boundaries to avoid partial matches)
        typo_corrections = [
            (r'\bhawaain\b', 'hawaiian'),
            (r'\bhawaiin\b', 'hawaiian'),
            (r'\bhawaiain\b', 'hawaiian'),
            (r'\bhawai\b', 'hawaiian'),  # If standalone "hawai" likely means "hawaiian"
            (r'\bindain\b', 'indian'),
            (r'\bindian\b', 'indian'),  # Already correct, but ensure consistency
            (r'\bitalain\b', 'italian'),
            (r'\bjapaneese\b', 'japanese'),
            (r'\bjapanease\b', 'japanese'),
            (r'\bmexicain\b', 'mexican'),
            (r'\bpeacefull\b', 'peaceful'),
            (r'\bromantic\b', 'romantic'),  # Already correct, but keep for consistency
            (r'\btranquil\b', 'tranquil'),
            (r'\bthai\b', 'thai'),  # Already correct
        ]
        
        normalized = query
        for typo_pattern, correction in typo_corrections:
            normalized = re.sub(typo_pattern, correction, normalized, flags=re.IGNORECASE)
        
        return normalized
    
    def _load_restaurant_names(self) -> List[str]:
        """Load restaurant names from restaurants.json for name detection."""
        if self._restaurant_names is not None:
            return self._restaurant_names
        
        data_path = Path(__file__).parent.parent.parent / "data" / "restaurants.json"
        if not data_path.exists():
            self._restaurant_names = []
            return self._restaurant_names
        
        try:
            with open(data_path) as f:
                restaurants = json.load(f)
            self._restaurant_names = [r.get("name", "") for r in restaurants if r.get("name")]
            return self._restaurant_names
        except (json.JSONDecodeError, FileNotFoundError):
            self._restaurant_names = []
            return self._restaurant_names
    
    def _detect_restaurant_name(self, query: str) -> Optional[str]:
        """Detect if query contains a restaurant name.
        
        Returns the best matching restaurant name, prioritizing exact matches and higher word overlap.
        """
        import re
        
        query_lower = query.lower()
        # Normalize apostrophes and special characters for better matching
        query_normalized = re.sub(r"['']", "", query_lower)
        restaurant_names = self._load_restaurant_names()
        
        best_match = None
        best_score = 0
        
        # Try to find restaurant name in query
        for name in restaurant_names:
            if not name:
                continue
            name_lower = name.lower()
            name_normalized = re.sub(r"['']", "", name_lower)
            
            # Exact match (highest priority)
            if name_lower in query_lower or name_normalized in query_normalized:
                return name
            
            # Word-based matching with normalization
            # Split and normalize words (remove apostrophes, punctuation)
            name_words_raw = re.findall(r"\b\w+\b", name_normalized)
            query_words_raw = re.findall(r"\b\w+\b", query_normalized)
            
            # Remove common words
            common_words = {
                'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                'of', 'with', 'is', 'are', 'was', 'were', 'what', 'where', 'when', 
                'who', 'which', 'can', 'tell', 'me', 'about', 'restaurant', 'restaurants', 
                'famous', 'for', '&', 'store', 'grill', 'cafe', 'bar', 'lounge'
            }
            
            name_words = {w.lower() for w in name_words_raw if w.lower() not in common_words}
            query_words = {w.lower() for w in query_words_raw if w.lower() not in common_words}
            
            if len(name_words) == 0:
                continue
            
            # Match words exactly first
            matching_words = name_words & query_words
            match_count = len(matching_words)
            
            # Also check for partial matches (e.g., "mama" matches "mamas", "mama's")
            # This handles cases where user says "mama fish" instead of "mama's fish"
            for name_word in name_words:
                if name_word in matching_words:
                    continue  # Already matched exactly
                # Check if any query word is a prefix/suffix of name word or vice versa
                for query_word in query_words:
                    if query_word in matching_words:
                        continue  # Already matched
                    # Check if words match when considering common variations
                    # Remove trailing 's' for possessive matching
                    name_base = name_word.rstrip('s')
                    query_base = query_word.rstrip('s')
                    if name_base == query_base and len(name_base) >= 3:  # At least 3 chars to avoid false matches
                        matching_words.add(name_word)
                        match_count += 1
                        break
            
            if match_count == 0:
                continue
            
            # Calculate match score: ratio of matching words to total name words
            match_ratio = match_count / len(name_words)
            
            # For short names (1-2 words), require at least 1 word match
            if len(name_words) <= 2:
                if match_count >= 1:
                    score = match_ratio + (match_count * 0.1)  # Bonus for more matches
                    if score > best_score:
                        best_score = score
                        best_match = name
            else:
                # For longer names, require at least 50% word match (lowered from 60% for better recall)
                # But also consider absolute match count
                required_matches = max(2, int(len(name_words) * 0.5))
                if match_count >= required_matches:
                    score = match_ratio + (match_count * 0.05)  # Bonus for more matches
                    if score > best_score:
                        best_score = score
                        best_match = name
        
        return best_match
    
    def _normalize_list_fields(self, data: dict) -> dict:
        """Normalize list fields to ensure they are always lists.
        
        LLM sometimes returns single strings instead of lists for list fields.
        This ensures compatibility with Pydantic schema expectations.
        """
        list_fields = ['formality', 'price', 'cuisine', 'features', 'atmosphere']
        normalized = data.copy() if data else {}
        
        for field in list_fields:
            if field in normalized:
                value = normalized[field]
                if value is None:
                    normalized[field] = []
                elif isinstance(value, str):
                    normalized[field] = [value] if value else []
                elif isinstance(value, list):
                    normalized[field] = value
                else:
                    normalized[field] = [value]
        
        return normalized
    
    async def _enrich_query_with_context(self, query: str) -> str:
        """Use LLM to enrich query with context before RAG retrieval.
        
        This helps RAG understand the query better by adding:
        - Cultural context (e.g., "desserts in Maui" → "Hawaiian desserts, local specialties, tropical treats")
        - Implicit requirements (e.g., "romantic" → "intimate atmosphere, candlelit, quiet")
        - Domain knowledge (e.g., what's famous in Maui, what makes sense culturally)
        
        Returns:
            Enriched query string with better context for vector search
        """
        enrichment_prompt = f"""You are an expert on Maui, Hawaii, food culture, and dining experiences.

**USER QUERY:** {query}

**YOUR TASK:**
Enrich this query with context that will help a RAG system find better restaurant matches. Add:
1. **Cultural context**: What does this query mean in the context of Maui/Hawaii? (e.g., "desserts" → "Hawaiian desserts, shave ice, malasadas, tropical fruits, local specialties")
2. **Implicit requirements**: What else might the user want? (e.g., "romantic" → "intimate atmosphere, candlelit, quiet, cozy")
3. **Domain knowledge**: What's famous or typical for this type of query in Maui? (e.g., "best breakfast" → "local favorites, Hawaiian breakfast, fresh fruit, acai bowls")
4. **Related terms**: Synonyms and related concepts that restaurants might use in their descriptions

**IMPORTANT:**
- Keep the original query intent intact
- Add context that helps semantic search find relevant restaurants
- Focus on terms restaurants would actually use in their vibe_summary, highlights, or descriptions
- Don't add unrelated information
- Be concise but comprehensive

**RESPOND WITH:**
Just the enriched query text (no explanations, no JSON). The enriched query should be a natural expansion that includes the original query plus relevant context.

Example:
Query: "best desserts in Maui"
Enriched: "best desserts in Maui, Hawaiian desserts, shave ice, malasadas, tropical fruit desserts, local specialties, famous sweets, popular treats"

Query: "romantic dinner spot"
Enriched: "romantic dinner spot, intimate atmosphere, candlelit dining, quiet ambiance, cozy setting, special occasion restaurant"

Now enrich this query: {query}"""
        
        try:
            enrichment_response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert on Maui dining. Enrich queries with cultural context and domain knowledge to help RAG retrieval."},
                    {"role": "user", "content": enrichment_prompt},
                ],
                temperature=0.3,  # Slight creativity for context expansion
                max_tokens=200,
            )
            
            enriched_query = enrichment_response.choices[0].message["content"].strip()
            # Remove quotes if LLM added them
            if enriched_query.startswith('"') and enriched_query.endswith('"'):
                enriched_query = enriched_query[1:-1]
            
            print(f"[DEBUG] Query enrichment: '{query}' → '{enriched_query[:150]}...'")
            return enriched_query
            
        except Exception as e:
            print(f"[WARNING] Query enrichment failed: {e}, using original query")
            return query
    
    async def parse(self, query: str) -> ParsedQuery:
        """Parse a natural language query into structured form."""
        # Normalize common typos first to help LLM understand intent
        normalized_query = self._normalize_common_typos(query)
        
        # Cache parsed queries to avoid redundant API calls
        # Use original query for cache key to handle same query consistently
        query_key = query.strip().lower()
        if query_key in self._query_parse_cache:
            return self._query_parse_cache[query_key]
        
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": normalized_query}
            ],
            response_format={"type": "json_object"},
            # Use deterministic decoding to reduce variability between runs
            temperature=0.0,
            max_tokens=300,  # Optimized for speed - JSON responses don't need much
        )
        
        message_content = response.choices[0].message["content"]
        
        # Log raw LLM response for debugging
        import logging
        logger = logging.getLogger(__name__)
        logger.debug(f"Raw LLM response for query '{query}': {message_content[:200]}...")
        
        # Be robust to bad JSON from the model
        try:
            parsed_json = json.loads(message_content)
        except json.JSONDecodeError:
            parsed_json = {}

        semantic_query = parsed_json.get("semantic_query", query)
        
        # ENHANCEMENT: Enrich semantic_query with LLM context BEFORE RAG retrieval
        # This gives RAG better context to find relevant restaurants
        semantic_query = await self._enrich_query_with_context(semantic_query)
        
        # CRITICAL FIX: If semantic_query contains a restaurant name and cuisine is explicitly requested,
        # remove the restaurant name from semantic_query to prevent false matches
        # (We'll let the exact match detection in pipeline handle restaurant name queries properly)
        if parsed_json.get("preferences", {}).get("cuisine") and parsed_json.get("weights", {}).get("cuisine", 0) >= 0.3:
            detected_restaurant_name_str = self._detect_restaurant_name(semantic_query)
            if detected_restaurant_name_str:
                # Load restaurant data to check cuisine
                data_path = Path(__file__).parent.parent.parent / "data" / "restaurants.json"
                restaurant_cuisine = None
                try:
                    with open(data_path) as f:
                        restaurants = json.load(f)
                    for r in restaurants:
                        if r.get("name", "").lower() == detected_restaurant_name_str.lower():
                            restaurant_cuisine = r.get("cuisine", "").lower()
                            break
                except:
                    pass
                
                # Check if cuisine matches
                if restaurant_cuisine:
                    preferred_cuisines = [c.lower() for c in parsed_json.get("preferences", {}).get("cuisine", [])]
                    cuisine_match = False
                    for preferred in preferred_cuisines:
                        preferred_normalized = preferred.replace('/', ' ').replace('-', ' ')
                        restaurant_normalized = restaurant_cuisine.replace('/', ' ').replace('-', ' ')
                        if preferred_normalized in restaurant_normalized or restaurant_normalized in preferred_normalized:
                            cuisine_match = True
                            break
                    
                    if not cuisine_match:
                        logger.warning(f"⚠️  LLM included '{detected_restaurant_name_str}' (cuisine: {restaurant_cuisine}) in semantic_query but requested cuisine is {preferred_cuisines} - removing restaurant name")
                        # Remove restaurant name from semantic query (handle various patterns)
                        semantic_query_cleaned = semantic_query
                        # Remove if at start: "RESTAURANT_NAME query text"
                        if semantic_query_cleaned.lower().startswith(detected_restaurant_name_str.lower()):
                            semantic_query_cleaned = semantic_query_cleaned[len(detected_restaurant_name_str):].strip()
                        # Remove if anywhere: "text RESTAURANT_NAME text"
                        semantic_query_cleaned = semantic_query_cleaned.replace(detected_restaurant_name_str, "").strip()
                        # Clean up double spaces
                        semantic_query_cleaned = re.sub(r'\s+', ' ', semantic_query_cleaned)
                        if semantic_query_cleaned:
                            semantic_query = semantic_query_cleaned
                        else:
                            # If removal left empty, fall back to original query
                            semantic_query = query
                else:
                    # If we can't verify cuisine, be safe and remove restaurant name when cuisine is explicitly requested
                    logger.warning(f"⚠️  Cannot verify cuisine for '{detected_restaurant_name_str}' - removing from semantic_query to prevent false matches")
                    semantic_query_cleaned = semantic_query.replace(detected_restaurant_name_str, "").strip()
                    semantic_query_cleaned = re.sub(r'\s+', ' ', semantic_query_cleaned)
                    if semantic_query_cleaned:
                        semantic_query = semantic_query_cleaned
                    else:
                        semantic_query = query
        
        # Boost restaurant name in semantic query if detected
        # BUT: Only if cuisine is not explicitly requested, or if restaurant matches the cuisine
        detected_restaurant_name = self._detect_restaurant_name(query)
        if detected_restaurant_name:
            # Check if cuisine is explicitly requested
            preferred_cuisine = parsed_json.get("preferences", {}).get("cuisine", [])
            cuisine_weight = parsed_json.get("weights", {}).get("cuisine", 0.0)
            
            # Only boost restaurant name if:
            # 1. No cuisine preference, OR
            # 2. Cuisine weight is low (< 0.3), OR  
            # 3. We can't verify cuisine match (will be checked later in pipeline)
            should_boost_restaurant = (
                not preferred_cuisine or 
                cuisine_weight < 0.3
            )
            
            if should_boost_restaurant:
                # Prepend restaurant name to semantic query to boost it in vector search
                semantic_query = f"{detected_restaurant_name} {semantic_query}"
            else:
                # Don't boost restaurant name when cuisine is explicitly requested
                # This prevents wrong restaurants from ranking high due to name match
                import logging
                logger = logging.getLogger(__name__)
                logger.debug(f"Not boosting '{detected_restaurant_name}' in semantic query - cuisine explicitly requested: {preferred_cuisine}")
        
        weights_data = parsed_json.get("weights", {})
        weights = SignalWeights(**weights_data)
        
        # Normalize weights FIRST, then check if cuisine weight is high
        total_weight = weights.vibe + weights.cuisine + weights.price + weights.features
        if total_weight > 0:
            weights.vibe /= total_weight
            weights.cuisine /= total_weight
            weights.price /= total_weight
            weights.features /= total_weight
        else:
            # Fallback to defaults if all weights are 0
            weights = SignalWeights()
        
        # Normalize weights to sum to 1.0 for correct score fusion
        total_weight = weights.vibe + weights.cuisine + weights.price + weights.features
        if total_weight > 0:
            weights.vibe /= total_weight
            weights.cuisine /= total_weight
            weights.price /= total_weight
            weights.features /= total_weight
        else:
            # Fallback to defaults if all weights are 0
            weights = SignalWeights()
        
        parsed = ParsedQuery(
            raw_query=query,
            semantic_query=semantic_query,
            must_not=MustNotFilters(**self._normalize_list_fields(parsed_json.get("must_not", {}))),
            preferences=Preferences(**self._normalize_list_fields(parsed_json.get("preferences", {}))),
            weights=weights,
            location=parsed_json.get("location"),
        )
        # Deterministic fallback enrichment so we don't depend solely on LLM JSON
        self._enrich_from_raw(parsed, query)
        
        # After enrichment, boost cuisine in semantic_query if cuisine is explicitly requested
        # This ensures cuisine queries return matching restaurants in vector search results
        # PRIORITY 2 FIX: Lower threshold from 0.5 to 0.3 for elderly users
        if parsed.preferences.cuisine and parsed.weights.cuisine >= 0.3:
            cuisine_str = ", ".join(parsed.preferences.cuisine)
            if cuisine_str.lower() not in parsed.semantic_query.lower():
                parsed.semantic_query = f"{cuisine_str} cuisine {parsed.semantic_query}"
        
        # Cache the parsed query
        self._query_parse_cache[query_key] = parsed
        return parsed

    async def is_restaurant_query(self, query: str) -> bool:
        """Check if query is restaurant-related using optimized hybrid approach."""
        query_normalized = query.strip().lower()
        if query_normalized in self._query_classification_cache:
            return self._query_classification_cache[query_normalized]
        
        if self._detect_restaurant_name(query):
            self._query_classification_cache[query_normalized] = True
            return True
        
        quick_result = self._quick_restaurant_check(query)
        if quick_result is not None:
            self._query_classification_cache[query_normalized] = quick_result
            return quick_result
        
        try:
            classification_prompt = """Classify if this query is about restaurants/dining/food. Respond ONLY with JSON: {{"is_restaurant_query": true/false, "confidence": "high/medium/low"}}

Restaurant examples: "Find Italian restaurant", "Best sushi place", "Where to eat pizza"
Non-restaurant examples: "Explain machine learning", "What is RAG", "Weather forecast"

Query: {query}"""

            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Respond with valid JSON only."},
                    {"role": "user", "content": classification_prompt.format(query=query)}
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
                max_tokens=100,  # Very short response for speed (optimized)
            )
            
            message_content = response.choices[0].message["content"]
            
            try:
                result = json.loads(message_content)
                is_restaurant = result.get("is_restaurant_query", False)
                confidence = result.get("confidence", "medium")
                
                if confidence == "low" and not is_restaurant:
                    result_bool = False
                elif confidence == "low" and is_restaurant:
                    result_bool = self._fallback_restaurant_check(query)
                else:
                    result_bool = bool(is_restaurant)
                
                self._query_classification_cache[query_normalized] = result_bool
                return result_bool
            except json.JSONDecodeError:
                result_bool = self._fallback_restaurant_check(query)
                self._query_classification_cache[query_normalized] = result_bool
                return result_bool
        
        except Exception as e:
            result_bool = self._fallback_restaurant_check(query)
            self._query_classification_cache[query_normalized] = result_bool
            return result_bool
    
    def _quick_restaurant_check(self, query: str) -> Optional[bool]:
        """Fast keyword-based check that returns None if uncertain."""
        query_lower = query.lower()
        
        strong_non_restaurant = {
            'retrieval augmented generation', 'rag', 'machine learning', 'neural network',
            'deep learning', 'transformer', 'llm', 'language model', 'gpt', 'chatgpt',
            'python programming', 'code', 'algorithm', 'data structure',
            'weather', 'temperature', 'forecast', 'rain', 'snow',
            'sports', 'football', 'basketball', 'soccer',
            'movie', 'film', 'actor', 'cinema',
            'music', 'song', 'artist', 'album',
            'book', 'novel', 'author',
            'science', 'physics', 'chemistry', 'biology', 'mathematics'
        }
        
        for topic in strong_non_restaurant:
            if topic in query_lower:
                return False
        
        strong_restaurant = {
            'restaurant', 'restaurants', 'dining', 'dine', 'cuisine', 'menu',
            'eat', 'eating', 'meal', 'lunch', 'dinner', 'breakfast', 'brunch',
            'cafe', 'café', 'bistro', 'grill', 'bar', 'lounge',
            'reservation', 'reserve', 'table', 'waiter', 'chef',
            'where to eat', 'find restaurant', 'best restaurant', 'good restaurant'
        }
        
        for indicator in strong_restaurant:
            if indicator in query_lower:
                return True
        
        return None
    
    def _fallback_restaurant_check(self, query: str) -> bool:
        """Fallback keyword-based check if LLM classification fails."""
        query_lower = query.lower()
        
        # Check for obvious non-restaurant topics
        non_restaurant_topics = {
            'machine learning', 'ai', 'artificial intelligence', 'neural network',
            'deep learning', 'programming', 'code', 'algorithm', 'python',
            'weather', 'temperature', 'forecast', 'sports', 'movie', 'music',
            'book', 'science', 'physics', 'chemistry', 'history', 'politics'
        }
        
        for topic in non_restaurant_topics:
            if topic in query_lower:
                return False
        
        # Check for restaurant indicators
        restaurant_indicators = ['restaurant', 'dining', 'food', 'eat', 'meal', 'menu', 'cuisine', 
                                'dinner', 'lunch', 'breakfast', 'cafe', 'grill', 'bar', 'bistro']
        return any(indicator in query_lower for indicator in restaurant_indicators)

    def _enrich_from_raw(self, parsed: ParsedQuery, raw_query: str) -> None:
        """
        Generic, non hard-coded enrichments based purely on raw text.
        
        - Infer location from patterns like "in Lahaina" if LLM omitted it.
        - Infer important features/atmosphere like live music / romantic.
        - Fallback cuisine extraction for queries like "best place for X food"
        """
        q = raw_query.lower()

        # 1) Location fallback: look for "in <place>" if location is missing
        if not parsed.location:
            match = re.search(r"\bin\s+([a-zA-Z][a-zA-Z\s'-]{1,40})", raw_query)
            if match:
                # Take first chunk, strip trailing punctuation
                loc = match.group(1).strip().rstrip(",.?!")
                parsed.location = loc

        # 2) Feature/atmosphere hints from raw text
        prefs = parsed.preferences

        if "live music" in q and "live_music" not in [f.lower() for f in prefs.features]:
            prefs.features.append("live_music")

        if "romantic" in q and "romantic" not in [a.lower() for a in prefs.atmosphere]:
            prefs.atmosphere.append("romantic")
        
        # 3) Enhanced cuisine extraction with fuzzy matching for typos
        # Common cuisine types with common misspellings mapped to correct names
        cuisine_map = {
            # Exact matches
            'hawaiian': 'Hawaiian',
            'hawaii': 'Hawaiian',
            'hawaii regional': 'Hawaiian',
            # Common typos for Hawaiian
            'hawaain': 'Hawaiian',
            'hawaiin': 'Hawaiian',
            'hawaiain': 'Hawaiian',
            'hawai': 'Hawaiian',  # If standalone, likely means Hawaiian
            # Other cuisines
            'indian': 'Indian',
            'indain': 'Indian',  # Common typo
            'italian': 'Italian',
            'italain': 'Italian',  # Common typo
            'japanese': 'Japanese',
            'japaneese': 'Japanese',  # Common typo
            'japanease': 'Japanese',
            'chinese': 'Chinese',
            'thai': 'Thai',
            'tai': 'Thai',  # Common typo
            'mexican': 'Mexican',
            'mexicain': 'Mexican',  # Common typo
            'french': 'French',
            'greek': 'Greek',
            'mediterranean': 'Mediterranean',
            'american': 'American',
            'korean': 'Korean',
            'vietnamese': 'Vietnamese',
            'middle eastern': 'Middle Eastern',
            'south indian': 'South Indian',
            'north indian': 'North Indian',
        }
        
        if not prefs.cuisine:
            # First try fuzzy match in cuisine_map (handles typos)
            for typo, correct_cuisine in cuisine_map.items():
                # Use word boundaries to avoid partial matches
                pattern = rf'\b{re.escape(typo)}\b'
                if re.search(pattern, q, re.IGNORECASE):
                    if correct_cuisine not in prefs.cuisine:
                        prefs.cuisine.append(correct_cuisine)
                        # Boost cuisine weight significantly when detected (even with typo)
                        if parsed.weights.cuisine < 0.5:
                            parsed.weights.cuisine = 0.7  # Higher weight for explicit cuisine mention
                            parsed.weights.vibe = 0.2
                            parsed.weights.price = 0.05
                            parsed.weights.features = 0.05
                            # Re-normalize to ensure they sum to 1.0
                            total = parsed.weights.vibe + parsed.weights.cuisine + parsed.weights.price + parsed.weights.features
                            if total > 0:
                                parsed.weights.vibe /= total
                                parsed.weights.cuisine /= total
                                parsed.weights.price /= total
                                parsed.weights.features /= total
                    break
            
            # Fallback to original pattern matching if no fuzzy match found
            if not prefs.cuisine:
                cuisine_types = [
                    'indian', 'italian', 'chinese', 'japanese', 'thai', 'mexican', 'french', 
                    'greek', 'mediterranean', 'american', 'hawaiian', 'korean', 'vietnamese',
                    'middle eastern', 'south indian', 'north indian'
                ]
                
                # Look for cuisine mentions in the query with various patterns
                for cuisine in cuisine_types:
                    cuisine_lower = cuisine.lower()
                    # Patterns to match: "for X food", "X restaurant", "X cuisine", "X place", "X food", etc.
                    patterns = [
                        rf'\bfor\s+{re.escape(cuisine_lower)}\s+food\b',
                        rf'\b{re.escape(cuisine_lower)}\s+food\b',
                        rf'\b{re.escape(cuisine_lower)}\s+restaurant',
                        rf'\b{re.escape(cuisine_lower)}\s+cuisine',
                        rf'\b{re.escape(cuisine_lower)}\s+place',
                        rf'\b{re.escape(cuisine_lower)}\s+spots\b',  # "hawaiian spots"
                        rf'\b{re.escape(cuisine_lower)}\b',  # Standalone cuisine mention
                    ]
                    for pattern in patterns:
                        if re.search(pattern, q, re.IGNORECASE):
                            # Use title case for consistency
                            cuisine_title = cuisine.title()
                            if cuisine_title not in prefs.cuisine:
                                prefs.cuisine.append(cuisine_title)
                                # Boost cuisine weight if we detected it from query
                                if parsed.weights.cuisine < 0.5:
                                    parsed.weights.cuisine = 0.6
                                    parsed.weights.vibe = 0.25
                                    parsed.weights.price = 0.1
                                    parsed.weights.features = 0.05
                                    # Re-normalize to ensure they sum to 1.0
                                    total = parsed.weights.vibe + parsed.weights.cuisine + parsed.weights.price + parsed.weights.features
                                    if total > 0:
                                        parsed.weights.vibe /= total
                                        parsed.weights.cuisine /= total
                                        parsed.weights.price /= total
                                        parsed.weights.features /= total
                            break
                    if prefs.cuisine:
                        break

