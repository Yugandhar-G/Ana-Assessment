import json
import asyncio
import logging
import re
from pathlib import Path
from .gemini_client import AsyncGeminiClient
from .schemas import Restaurant, RestaurantVibe, ParsedQuery, AnaResponse
from .query_parser import QueryParser
from .filters import HardFilter
from .scorers import VibeScorer, CuisineScorer, PriceScorer, FeatureScorer
from .fusion import ScoreFusion, AdvancedScoreFusion, ScoredRestaurant
from .response_generator import ResponseGenerator
from .vector_store import VectorStore, initialize_vector_store

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AnaVibeSearch:
    """Main orchestrator for Ana AI vibe-based restaurant search."""
    
    def __init__(
        self,
        client: AsyncGeminiClient | None = None,
        persist_vectors: bool = False,
        use_llm: bool = True,
        use_advanced_fusion: bool = True,
    ):
        import os
        # CRITICAL: Always use text-embedding-004 for consistency with vector store
        # The vector store was built with text-embedding-004, so queries must use the same model
        embedding_model = os.getenv("GEMINI_EMBEDDING_MODEL", "models/text-embedding-004")
        # Override if environment variable is set to something else - we need text-embedding-004
        if embedding_model != "models/text-embedding-004":
            logger.warning(f"⚠️  GEMINI_EMBEDDING_MODEL is set to {embedding_model}, but vector store was built with models/text-embedding-004")
            logger.warning(f"   Using models/text-embedding-004 for consistency. Set GEMINI_EMBEDDING_MODEL='models/text-embedding-004' to avoid this warning.")
            embedding_model = "models/text-embedding-004"
        
        self.client = client or AsyncGeminiClient(
            api_key=os.getenv("GEMINI_API_KEY"),
            default_chat_model=os.getenv("GEMINI_CHAT_MODEL", "gemini-2.0-flash"),
            default_embedding_model=embedding_model,
        )
        self.use_llm = use_llm
        self.restaurants = self._load_restaurants()
        self._restaurant_lookup = {r.id: r for r in self.restaurants}
        # Use absolute path to ensure consistency regardless of working directory
        if persist_vectors:
            # Get absolute path to part1/chroma_db
            base_dir = Path(__file__).parent.parent.absolute()
            self.persist_dir = str(base_dir / "chroma_db")
        else:
            self.persist_dir = None
        self.vector_store: VectorStore | None = None
        
        self.query_parser = QueryParser(self.client)
        self.hard_filter = HardFilter()
        self.vibe_scorer = VibeScorer(self.client, None)
        self.cuisine_scorer = CuisineScorer(restaurants=self.restaurants)
        self.price_scorer = PriceScorer()
        self.feature_scorer = FeatureScorer()
        self.fusion = AdvancedScoreFusion() if use_advanced_fusion else ScoreFusion()
        self.response_generator = ResponseGenerator(self.client)
        self._initialized = False
    
    async def _ensure_initialized(self):
        """Initialize vector store if not already done."""
        if not self._initialized:
            try:
                logger.info(f"Initializing vector store (persist_dir: {self.persist_dir})")
                count_before = 0
                if self.persist_dir and Path(self.persist_dir).exists():
                    temp_store = VectorStore(self.persist_dir, gemini_client=self.client)
                    count_before = temp_store.get_count()
                    logger.info(f"Vector DB exists with {count_before} restaurants")
                
                # CRITICAL: Pass the embedding model explicitly to ensure consistency
                # The vector store must use the same embedding model that was used to build it
                # Default to text-embedding-004 which is what the vector store was built with
                embedding_model = self.client.default_embedding_model
                logger.info(f"Initializing vector store with embedding model: {embedding_model}")
                self.vector_store = await initialize_vector_store(self.persist_dir, gemini_client=self.client)
                # Ensure the vector store uses the client's embedding model
                if self.vector_store.embedding_model != embedding_model:
                    logger.warning(f"⚠️  Vector store embedding model mismatch! Store: {self.vector_store.embedding_model}, Client: {embedding_model}")
                    logger.warning(f"   This will cause poor search results. Setting vector store to use client's model.")
                    self.vector_store.embedding_model = embedding_model
                count_after = self.vector_store.get_count()
                self.vibe_scorer.vector_store = self.vector_store
                self._initialized = True
                
                if count_before == 0 and count_after > 0:
                    logger.info(f"✅ Vector DB rebuilt with {count_after} restaurants")
                elif count_after > 0:
                    logger.info(f"✅ Vector DB loaded with {count_after} restaurants")
                else:
                    logger.warning(f"⚠️  Vector DB is empty ({count_after} restaurants)")
            except ConnectionError as e:
                logger.error(f"Failed to initialize vector store: {e}")
                raise ConnectionError(
                    f"{str(e)}\n\n"
                    f"Make sure GEMINI_API_KEY is set correctly.\n"
                    f"Get your API key from https://makersuite.google.com/app/apikey"
                ) from e
    
    def _load_restaurants(self) -> list[Restaurant]:
        """Load restaurant data from JSON file."""
        data_path = Path(__file__).parent.parent.parent / "data" / "restaurants.json"
        if not data_path.exists():
            return []
        
        with open(data_path) as f:
            data = json.load(f)
        
        restaurants = []
        for item in data:
            vibe_data = item.pop("vibe", {})
            item["vibe"] = RestaurantVibe(**vibe_data)
            restaurants.append(Restaurant(**item))
        
        return restaurants
    
    def _find_restaurant_by_name_in_query(self, query: str) -> Restaurant | None:
        """Find restaurant by name if it appears in the query.
        
        Returns the best matching restaurant, prioritizing exact matches and higher word overlap.
        """
        import re
        
        query_lower = query.lower()
        # Normalize apostrophes and special characters for better matching
        query_normalized = re.sub(r"['']", "", query_lower)
        
        best_match = None
        best_score = 0
        
        for restaurant in self.restaurants:
            name_lower = restaurant.name.lower()
            name_normalized = re.sub(r"['']", "", name_lower)
            
            # Exact match (highest priority)
            if name_lower in query_lower or name_normalized in query_normalized:
                return restaurant
            
            # Word-based matching with normalization
            # Split and normalize words (remove apostrophes, punctuation)
            name_words_raw = re.findall(r"\b\w+\b", name_normalized)
            query_words_raw = re.findall(r"\b\w+\b", query_normalized)
            
            common_words = {
                'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                'of', 'with', 'is', 'are', 'was', 'were', 'what', 'where', 'when', 
                'who', 'which', 'can', 'tell', 'me', 'about', 'restaurant', 'restaurants', 
                'famous', '&', 'store', 'grill', 'cafe', 'bar', 'lounge'
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
                        best_match = restaurant
            else:
                # For longer names, require at least 50% word match (lowered from 60% for better recall)
                # But also consider absolute match count
                required_matches = max(2, int(len(name_words) * 0.5))
                if match_count >= required_matches:
                    score = match_ratio + (match_count * 0.05)  # Bonus for more matches
                    if score > best_score:
                        best_score = score
                        best_match = restaurant
        
        return best_match
    
    def _enhance_semantic_query_for_cuisine(self, parsed_query: ParsedQuery) -> str:
        """Enhance semantic query to better capture cuisine in embeddings.
        
        When cuisine is explicitly requested, we make the semantic query much more
        explicit and repetitive about the cuisine type to improve embedding quality.
        This helps the vector search return relevant restaurants even without metadata filtering.
        ChromaDB doesn't support $contains for metadata filtering, so we rely on better embeddings.
        """
        if not parsed_query.preferences.cuisine or parsed_query.weights.cuisine < 0.3:
            return parsed_query.semantic_query
        
        # Check if cuisine is explicitly mentioned in raw query
        cuisine_explicitly_mentioned = False
        query_lower = parsed_query.raw_query.lower()
        for cuisine in parsed_query.preferences.cuisine:
            cuisine_normalized = cuisine.lower().replace('/', ' ').replace('-', ' ')
            cuisine_pattern = r'\b' + re.escape(cuisine_normalized) + r'\b'
            if re.search(cuisine_pattern, query_lower):
                cuisine_explicitly_mentioned = True
                break
            # Also check individual words for multi-word cuisines
            cuisine_words = [w for w in cuisine_normalized.split() if len(w) >= 3]
            if cuisine_words:
                all_words_found = all(
                    re.search(r'\b' + re.escape(w) + r'\b', query_lower) 
                    for w in cuisine_words
                )
                if all_words_found:
                    cuisine_explicitly_mentioned = True
                    break
        
        if not (cuisine_explicitly_mentioned or parsed_query.weights.cuisine >= 0.5):
            return parsed_query.semantic_query
        
        # Build a highly explicit cuisine-focused semantic query
        # Repetition and explicit cuisine terms help embeddings capture cuisine better
        cuisine_terms = []
        for cuisine in parsed_query.preferences.cuisine:
            cuisine_normalized = cuisine.lower().replace('/', ' ').replace('-', ' ').strip()
            cuisine_terms.append(cuisine_normalized)
            # Add individual words for multi-word cuisines
            cuisine_words = [w for w in cuisine_normalized.split() if len(w) >= 3]
            cuisine_terms.extend(cuisine_words)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_terms = []
        for term in cuisine_terms:
            if term not in seen:
                seen.add(term)
                unique_terms.append(term)
        
        # Create a highly explicit query that emphasizes cuisine
        # Repeat cuisine terms multiple times to boost them in embeddings
        cuisine_emphasis = " ".join(unique_terms * 3)  # Repeat 3 times for emphasis
        enhanced_query = f"{cuisine_emphasis} {parsed_query.semantic_query} {cuisine_emphasis} cuisine restaurant food dining"
        
        return enhanced_query
    
    async def search(self, query: str) -> AnaResponse:
        """Execute full search pipeline."""
        await self._ensure_initialized()
        parsed_query = await self.query_parser.parse(query)
        
        # Check if query contains a specific restaurant name
        # BUT: Don't treat as exact match if cuisine is explicitly requested and restaurant doesn't match
        exact_restaurant = self._find_restaurant_by_name_in_query(query)
        
        # If cuisine is explicitly requested, verify the exact match actually matches the cuisine
        if exact_restaurant and parsed_query.preferences.cuisine and parsed_query.weights.cuisine >= 0.3:
            # Check if the exact match restaurant matches the requested cuisine
            restaurant_cuisine_lower = exact_restaurant.cuisine.lower()
            preferred_cuisines_lower = [c.lower() for c in parsed_query.preferences.cuisine]
            
            cuisine_match = False
            for preferred in preferred_cuisines_lower:
                # Normalize for matching
                preferred_normalized = preferred.replace('/', ' ').replace('-', ' ')
                restaurant_normalized = restaurant_cuisine_lower.replace('/', ' ').replace('-', ' ')
                
                # Check if preferred cuisine appears in restaurant cuisine
                if preferred_normalized in restaurant_normalized or restaurant_normalized in preferred_normalized:
                    cuisine_match = True
                    break
                
                # Check word overlap for multi-word cuisines
                preferred_words = set(preferred_normalized.split())
                restaurant_words = set(restaurant_normalized.split())
                meaningful_overlap = {w for w in preferred_words & restaurant_words if len(w) >= 4}
                if meaningful_overlap:
                    cuisine_match = True
                    break
            
            if not cuisine_match:
                logger.info(f"⚠️  Exact match '{exact_restaurant.name}' found but cuisine mismatch - treating as regular candidate")
                logger.info(f"   Requested: {parsed_query.preferences.cuisine}, Restaurant: {exact_restaurant.cuisine}")
                exact_restaurant = None  # Don't treat as exact match - let it go through normal filtering

        # PRIORITY 3 FIX: Dynamic n_results based on feature count
        # Elderly users with specific needs (wheelchair, parking, quiet, 6am breakfast)
        # need MORE candidates because rare features may not match vibe similarity
        feature_count = (
            len(parsed_query.preferences.features) +
            len(parsed_query.preferences.atmosphere)
        )

        # Increase n_results when cuisine is explicitly requested to get more candidates
        # Since we can't use metadata filtering (ChromaDB doesn't support $contains),
        # we need more candidates to ensure we find matching restaurants after filtering
        cuisine_explicitly_requested = (
            parsed_query.preferences.cuisine and 
            parsed_query.weights.cuisine >= 0.3
        )
        
        # More features = need more candidates to find rare feature matches
        # Cuisine queries also need more candidates since we rely on embeddings + filtering
        if feature_count >= 2:
            n_results = 40 if cuisine_explicitly_requested else 30
        else:
            n_results = 30 if cuisine_explicitly_requested else 20

        # Enhance semantic query to better capture cuisine in embeddings
        # This is critical since ChromaDB doesn't support $contains for metadata filtering
        enhanced_semantic_query = self._enhance_semantic_query_for_cuisine(parsed_query)
        if enhanced_semantic_query != parsed_query.semantic_query:
            logger.info(f"   Enhanced semantic query for cuisine: '{enhanced_semantic_query[:100]}...'")
        
        # Get candidates from vector search (vibe-based, with enhanced cuisine emphasis)
        logger.info(f"🔍 Vector Search: query='{enhanced_semantic_query}', n_results={n_results}")
        logger.info(f"   Using embedding model: {self.vector_store.embedding_model}")
        logger.info(f"   Vector store collection: {self.vector_store.collection.name}, count: {self.vector_store.collection.count()}")
        vector_results = await self.vector_store.search(
            query=enhanced_semantic_query,
            n_results=n_results,
        )
        logger.info(f"   Found {len(vector_results)} vector search results")
        if vector_results:
            logger.info(f"   Top 3 vector matches:")
            for i, result in enumerate(vector_results[:3]):
                restaurant = self._restaurant_lookup.get(result["id"])
                if restaurant:
                    logger.info(f"     [{i+1}] {restaurant.name} (similarity: {result.get('similarity', 0):.3f})")
        
        candidates_with_vibe_scores = []
        
        if exact_restaurant:
            # CRITICAL: When exact restaurant name is found, prioritize it heavily
            # Add it first with very high vibe score to ensure it ranks highest
            found_in_results = False
            for result in vector_results:
                if result["id"] == exact_restaurant.id:
                    found_in_results = True
                    # Use normalized similarity if found, but boost it significantly
                    similarity = result.get("similarity", 0.9)
                    normalized_vibe_score = max(0.0, min(1.0, (similarity + 1) / 2))
                    # Boost exact match to ensure it's always top result
                    normalized_vibe_score = max(0.98, normalized_vibe_score)
                    candidates_with_vibe_scores.append((exact_restaurant, normalized_vibe_score))
                    break
            
            if not found_in_results:
                # If not in vector results, still add with very high score
                candidates_with_vibe_scores.append((exact_restaurant, 0.98))
        
        for result in vector_results:
            restaurant = self._restaurant_lookup.get(result["id"])
            if restaurant:
                if exact_restaurant and restaurant.id == exact_restaurant.id:
                    continue
                # Normalize similarity from [-1, 1] to [0, 1] to match vibe_score constraint
                similarity = result["similarity"]
                normalized_vibe_score = max(0.0, min(1.0, (similarity + 1) / 2))
                candidates_with_vibe_scores.append((restaurant, normalized_vibe_score))
        
        logger.info(f"\n🔍 Hard Filtering: {len(candidates_with_vibe_scores)} candidates before filtering")
        filtered_candidates = []
        filtered_out = []
        for restaurant, vibe_score in candidates_with_vibe_scores:
            # If this is an exact restaurant match, bypass hard filters (except business status)
            # Exact matches should always be included regardless of location/cuisine filters
            is_exact_match = (exact_restaurant and restaurant.id == exact_restaurant.id)
            
            if is_exact_match:
                # Only check business status for exact matches, skip other filters
                if not restaurant.business_status or restaurant.business_status == "OPERATIONAL":
                    filtered_candidates.append((restaurant, vibe_score))
                    logger.debug(f"✅ EXACT MATCH (bypass filter): {restaurant.name}")
            elif self.hard_filter._passes_filters(restaurant, parsed_query):
                filtered_candidates.append((restaurant, vibe_score))
            else:
                filtered_out.append(restaurant.name)
        
        logger.info(f"   ✅ Passed filter: {len(filtered_candidates)} restaurants")
        if filtered_out:
            logger.info(f"   ❌ Filtered out: {len(filtered_out)} restaurants")
            if len(filtered_out) <= 5:
                logger.info(f"      {', '.join(filtered_out)}")
        
        if not filtered_candidates:
            # Provide detailed error message with debugging info
            error_details = []
            error_details.append(f"Query: '{parsed_query.raw_query}'")
            error_details.append(f"Vector search returned {len(vector_results)} results")
            if vector_results:
                similarities = [f"{r.get('similarity', 0):.3f}" for r in vector_results[:3]]
                error_details.append(f"Top 3 vector matches had similarities: {', '.join(similarities)}")
            error_details.append(f"After filtering, {len(filtered_candidates)} restaurants passed")
            if filtered_out:
                error_details.append(f"Filtered out: {', '.join(filtered_out[:10])}")
            if parsed_query.preferences.cuisine:
                error_details.append(f"Requested cuisine: {parsed_query.preferences.cuisine}")
            if parsed_query.location:
                error_details.append(f"Requested location: {parsed_query.location}")
            
            logger.error(f"❌ NO RESULTS: {' | '.join(error_details)}")
            
            return AnaResponse(
                success=False,
                explanation="No restaurants match your criteria after filtering. Try relaxing some constraints?",
                confidence="low",
                caveats=["All restaurants were filtered out by hard constraints"] + error_details,
            )
        
        scored_results = await self._score_candidates(filtered_candidates, parsed_query)
        
        # Log all scores before ranking
        logger.info(f"\n{'='*80}")
        logger.info(f"SCORING RESULTS (before ranking) - Query: '{parsed_query.raw_query}'")
        logger.info(f"Query Weights: vibe={parsed_query.weights.vibe:.2f}, cuisine={parsed_query.weights.cuisine:.2f}, price={parsed_query.weights.price:.2f}, features={parsed_query.weights.features:.2f}")
        logger.info(f"Preferred Cuisine: {parsed_query.preferences.cuisine}")
        logger.info(f"{'='*80}")
        
        for i, scored in enumerate(scored_results[:10]):  # Log top 10
            logger.info(f"\n[{i+1}] {scored.restaurant.name}")
            logger.info(f"    Cuisine: {scored.restaurant.cuisine}")
            logger.info(f"    Scores: vibe={scored.vibe_score:.3f}, cuisine={scored.cuisine_score:.3f}, price={scored.price_score:.3f}, feature={scored.feature_score:.3f}")
            logger.info(f"    Final Score: {scored.final_score:.3f}")
        
        # CRITICAL: Boost exact restaurant name matches to ensure they rank highest
        if exact_restaurant:
            for scored in scored_results:
                if scored.restaurant.id == exact_restaurant.id:
                    logger.info(f"\n🎯 Exact match boost applied to: {scored.restaurant.name}")
                    old_score = scored.final_score
                    # Massive boost for exact matches - ensure it's always #1
                    scored.final_score = min(1.0, scored.final_score + 0.5)
                    scored.vibe_score = min(1.0, scored.vibe_score + 0.3)
                    logger.info(f"    Score: {old_score:.3f} → {scored.final_score:.3f}")
                    break
        
        # Use award-priority ranking for top 10 results (with primary cuisine prioritization)
        ranked_results = self.fusion.rank_with_award_priority(scored_results, top_n=10, parsed_query=parsed_query)
        
        # Log top ranked results
        logger.info(f"\n{'='*80}")
        logger.info(f"TOP RANKED RESULTS (after fusion and ranking)")
        logger.info(f"{'='*80}")
        for i, scored in enumerate(ranked_results[:5]):  # Log top 5
            logger.info(f"\n🏆 [{i+1}] {scored.restaurant.name} (ID: {scored.restaurant.id})")
            logger.info(f"    Cuisine: {scored.restaurant.cuisine} | Price: {scored.restaurant.price_level}")
            logger.info(f"    Individual Scores:")
            logger.info(f"      • Vibe Score:     {scored.vibe_score:.4f} (weight: {parsed_query.weights.vibe:.2f}) → contribution: {scored.vibe_score * parsed_query.weights.vibe:.4f}")
            logger.info(f"      • Cuisine Score:  {scored.cuisine_score:.4f} (weight: {parsed_query.weights.cuisine:.2f}) → contribution: {scored.cuisine_score * parsed_query.weights.cuisine:.4f}")
            logger.info(f"      • Price Score:    {scored.price_score:.4f} (weight: {parsed_query.weights.price:.2f}) → contribution: {scored.price_score * parsed_query.weights.price:.4f}")
            logger.info(f"      • Feature Score:  {scored.feature_score:.4f} (weight: {parsed_query.weights.features:.2f}) → contribution: {scored.feature_score * parsed_query.weights.features:.4f}")
            logger.info(f"    ⭐ Final Score: {scored.final_score:.4f}")
            logger.info(f"    Vibe Summary: {scored.restaurant.vibe.vibe_summary[:100]}...")
            
            # Show why this restaurant ranked high
            if i == 0 and ranked_results:
                top = scored
                logger.info(f"\n📈 WHY THIS IS TOP RESULT:")
                if top.vibe_score > 0.8:
                    logger.info(f"   ✓ High vibe score ({top.vibe_score:.3f}) - strong semantic match")
                if top.cuisine_score > 0.7:
                    logger.info(f"   ✓ High cuisine score ({top.cuisine_score:.3f}) - cuisine matches")
                elif parsed_query.preferences.cuisine and top.cuisine_score < 0.3:
                    logger.info(f"   ⚠️  LOW cuisine score ({top.cuisine_score:.3f}) but still ranked #1 - check penalty!")
                    logger.info(f"      Requested: {parsed_query.preferences.cuisine}, Got: {top.restaurant.cuisine}")
                if top.price_score > 0.8:
                    logger.info(f"   ✓ High price score ({top.price_score:.3f}) - price matches")
                if top.feature_score > 0.8:
                    logger.info(f"   ✓ High feature score ({top.feature_score:.3f}) - features match")
        logger.info(f"{'='*80}\n")
        
        # Fast path: skip LLM to reduce latency (no natural-language explanation)
        if not self.use_llm:
            top_match = ranked_results[0]
            alternatives = self.response_generator._select_relevant_alternatives(ranked_results, top_match, parsed_query)
            return AnaResponse(
                success=True,
                top_match=self.response_generator._scored_to_match(top_match),
                alternatives=[self.response_generator._scored_to_match(a) for a in alternatives],
                match_reasons=self.response_generator._generate_match_reasons(
                    top_match, parsed_query
                ),
                explanation="Low-latency mode: returned top matches without LLM-generated narrative.",
                confidence=self.response_generator._determine_boosted_confidence(
                    top_match.final_score, len(ranked_results), parsed_query
                ),
            )

        return await self.response_generator.generate(parsed_query, ranked_results)
    
    async def search_for_streaming(self, query: str) -> tuple[ParsedQuery, list[ScoredRestaurant]]:
        """Execute search pipeline and return parsed query and ranked results for streaming.
        
        This method performs the same search logic as search() but returns intermediate results
        instead of generating the full response, allowing the API to stream the response.
        """
        await self._ensure_initialized()
        parsed_query = await self.query_parser.parse(query)
        
        # Check if query contains a specific restaurant name
        # BUT: Don't treat as exact match if cuisine is explicitly requested and restaurant doesn't match
        exact_restaurant = self._find_restaurant_by_name_in_query(query)
        
        # If cuisine is explicitly requested, verify the exact match actually matches the cuisine
        if exact_restaurant and parsed_query.preferences.cuisine and parsed_query.weights.cuisine >= 0.3:
            # Check if the exact match restaurant matches the requested cuisine
            restaurant_cuisine_lower = exact_restaurant.cuisine.lower()
            preferred_cuisines_lower = [c.lower() for c in parsed_query.preferences.cuisine]
            
            cuisine_match = False
            for preferred in preferred_cuisines_lower:
                # Normalize for matching
                preferred_normalized = preferred.replace('/', ' ').replace('-', ' ')
                restaurant_normalized = restaurant_cuisine_lower.replace('/', ' ').replace('-', ' ')
                
                # Check if preferred cuisine appears in restaurant cuisine
                if preferred_normalized in restaurant_normalized or restaurant_normalized in preferred_normalized:
                    cuisine_match = True
                    break
                
                # Check word overlap for multi-word cuisines
                preferred_words = set(preferred_normalized.split())
                restaurant_words = set(restaurant_normalized.split())
                meaningful_overlap = {w for w in preferred_words & restaurant_words if len(w) >= 4}
                if meaningful_overlap:
                    cuisine_match = True
                    break
            
            if not cuisine_match:
                logger.info(f"⚠️  Exact match '{exact_restaurant.name}' found but cuisine mismatch - treating as regular candidate")
                logger.info(f"   Requested: {parsed_query.preferences.cuisine}, Restaurant: {exact_restaurant.cuisine}")
                exact_restaurant = None  # Don't treat as exact match - let it go through normal filtering

        feature_count = (
            len(parsed_query.preferences.features) +
            len(parsed_query.preferences.atmosphere)
        )

        # Increase n_results when cuisine is explicitly requested
        cuisine_explicitly_requested = (
            parsed_query.preferences.cuisine and 
            parsed_query.weights.cuisine >= 0.3
        )
        
        if feature_count >= 2:
            n_results = 40 if cuisine_explicitly_requested else 30
        else:
            n_results = 30 if cuisine_explicitly_requested else 20

        # Enhance semantic query to better capture cuisine in embeddings
        enhanced_semantic_query = self._enhance_semantic_query_for_cuisine(parsed_query)
        if enhanced_semantic_query != parsed_query.semantic_query:
            logger.info(f"   Enhanced semantic query for cuisine: '{enhanced_semantic_query[:100]}...'")

        # Get candidates from vector search (vibe-based, with enhanced cuisine emphasis)
        logger.info(f"🔍 Vector Search: query='{enhanced_semantic_query}', n_results={n_results}")
        vector_results = await self.vector_store.search(
            query=enhanced_semantic_query,
            n_results=n_results,
        )
        logger.info(f"   Found {len(vector_results)} vector search results")
        if vector_results:
            logger.info(f"   Top 3 vector matches:")
            for i, result in enumerate(vector_results[:3]):
                restaurant = self._restaurant_lookup.get(result["id"])
                if restaurant:
                    logger.info(f"     [{i+1}] {restaurant.name} (similarity: {result.get('similarity', 0):.3f})")
        
        candidates_with_vibe_scores = []
        
        if exact_restaurant:
            # CRITICAL: When exact restaurant name is found, prioritize it heavily
            # Add it first with very high vibe score to ensure it ranks highest
            found_in_results = False
            for result in vector_results:
                if result["id"] == exact_restaurant.id:
                    found_in_results = True
                    # Use normalized similarity if found, but boost it significantly
                    similarity = result.get("similarity", 0.9)
                    normalized_vibe_score = max(0.0, min(1.0, (similarity + 1) / 2))
                    # Boost exact match to ensure it's always top result
                    normalized_vibe_score = max(0.98, normalized_vibe_score)
                    candidates_with_vibe_scores.append((exact_restaurant, normalized_vibe_score))
                    break
            
            if not found_in_results:
                # If not in vector results, still add with very high score
                candidates_with_vibe_scores.append((exact_restaurant, 0.98))
        
        for result in vector_results:
            restaurant = self._restaurant_lookup.get(result["id"])
            if restaurant:
                if exact_restaurant and restaurant.id == exact_restaurant.id:
                    continue
                # Normalize similarity from [-1, 1] to [0, 1] to match vibe_score constraint
                similarity = result["similarity"]
                normalized_vibe_score = max(0.0, min(1.0, (similarity + 1) / 2))
                candidates_with_vibe_scores.append((restaurant, normalized_vibe_score))
        
        logger.info(f"\n🔍 Hard Filtering: {len(candidates_with_vibe_scores)} candidates before filtering")
        filtered_candidates = []
        filtered_out = []
        for restaurant, vibe_score in candidates_with_vibe_scores:
            # If this is an exact restaurant match, bypass hard filters (except business status)
            # Exact matches should always be included regardless of location/cuisine filters
            is_exact_match = (exact_restaurant and restaurant.id == exact_restaurant.id)
            
            if is_exact_match:
                # Only check business status for exact matches, skip other filters
                if not restaurant.business_status or restaurant.business_status == "OPERATIONAL":
                    filtered_candidates.append((restaurant, vibe_score))
                    logger.debug(f"✅ EXACT MATCH (bypass filter): {restaurant.name}")
            elif self.hard_filter._passes_filters(restaurant, parsed_query):
                filtered_candidates.append((restaurant, vibe_score))
            else:
                filtered_out.append(restaurant.name)
        
        logger.info(f"   ✅ Passed filter: {len(filtered_candidates)} restaurants")
        if filtered_out:
            logger.info(f"   ❌ Filtered out: {len(filtered_out)} restaurants")
            if len(filtered_out) <= 5:
                logger.info(f"      {', '.join(filtered_out)}")
        
        if not filtered_candidates:
            logger.warning(f"⚠️  No restaurants passed filtering for query: '{parsed_query.raw_query}'")
            return parsed_query, []
        
        scored_results = await self._score_candidates(filtered_candidates, parsed_query)
        
        # Log scores for streaming endpoint too
        logger.info(f"\n{'='*80}")
        logger.info(f"STREAMING SEARCH - Query: '{parsed_query.raw_query}'")
        logger.info(f"Query Weights: vibe={parsed_query.weights.vibe:.2f}, cuisine={parsed_query.weights.cuisine:.2f}")
        logger.info(f"Preferred Cuisine: {parsed_query.preferences.cuisine}")
        logger.info(f"Top 5 Results:")
        for i, scored in enumerate(scored_results[:5]):
            logger.info(f"  [{i+1}] {scored.restaurant.name}: final={scored.final_score:.3f}, vibe={scored.vibe_score:.3f}, cuisine={scored.cuisine_score:.3f}")
        logger.info(f"{'='*80}\n")
        
        if exact_restaurant:
            for scored in scored_results:
                if scored.restaurant.id == exact_restaurant.id:
                    logger.info(f"🎯 Exact match boost: {scored.restaurant.name}")
                    scored.final_score = min(1.0, scored.final_score + 0.5)
                    scored.vibe_score = min(1.0, scored.vibe_score + 0.3)
                    break
        
        # Use award-priority ranking for top 10 results
        ranked_results = self.fusion.rank_with_award_priority(scored_results, top_n=10, parsed_query=parsed_query)
        return parsed_query, ranked_results
    
    async def _score_candidates(
        self,
        candidates_with_vibe: list[tuple[Restaurant, float]],
        parsed_query: ParsedQuery,
    ) -> list[ScoredRestaurant]:
        """Score candidates across all signals - OPTIMIZED: all restaurants scored in parallel."""
        
        async def score_one_restaurant(restaurant: Restaurant, vibe_score: float) -> ScoredRestaurant:
            """Score a single restaurant across all signals."""
            cuisine_score, price_score, feature_score = await asyncio.gather(
                self.cuisine_scorer.score(restaurant, parsed_query),
                self.price_scorer.score(restaurant, parsed_query),
                self.feature_scorer.score(restaurant, parsed_query),
            )
            
            return self.fusion.fuse(
                restaurant=restaurant,
                vibe_score=vibe_score,
                cuisine_score=cuisine_score,
                price_score=price_score,
                feature_score=feature_score,
                weights=parsed_query.weights,
                parsed_query=parsed_query if isinstance(self.fusion, AdvancedScoreFusion) else None,
            )
        
        # Score ALL restaurants in parallel (not sequentially)
        tasks = [
            score_one_restaurant(restaurant, vibe_score)
            for restaurant, vibe_score in candidates_with_vibe
        ]
        scored_results = await asyncio.gather(*tasks)
        return list(scored_results)

