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
            default_chat_model=os.getenv("GEMINI_CHAT_MODEL", "gemini-3-flash-preview"),
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
        
        # STEP 0: Enhance structured query with LLM general knowledge
        # This adds implicit requirements, cultural context, and domain knowledge
        # BEFORE RAG retrieval, so RAG uses the enriched structured query
        # Flow: User Query -> Query Parser -> LLM Enhancement -> RAG -> LLM Response
        from .query_enhancer import QueryEnhancer
        query_enhancer = QueryEnhancer(client=self.client)
        parsed_query = await query_enhancer.enhance_query(parsed_query)
        logger.info(f"✅ Query enhanced with LLM knowledge - ready for RAG retrieval")
        
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
        # General queries (vibe, food culture, etc.) also need more candidates to show diverse options
        is_general_query = (
            parsed_query.weights.vibe >= 0.5 or  # Vibe-focused queries
            not parsed_query.preferences.cuisine or  # No specific cuisine
            parsed_query.weights.cuisine < 0.3  # Low cuisine weight (general food queries)
        )
        
        if feature_count >= 2:
            n_results = 50 if cuisine_explicitly_requested else (45 if is_general_query else 35)
        else:
            n_results = 40 if cuisine_explicitly_requested else (35 if is_general_query else 25)
        
        logger.info(f"   Query type: cuisine_explicit={cuisine_explicitly_requested}, general={is_general_query}, features={feature_count}")
        logger.info(f"   Retrieving {n_results} candidates from vector search")

        # NO RANKING: Pass all restaurant data directly to Gemini for complete analysis
        logger.info(f"🤖 GEMINI-FIRST ARCHITECTURE: No ranking, Gemini handles query parsing, answer generation, and restaurant selection")
        logger.info(f"   Total restaurants in database: {len(self.restaurants)}")
        
        # Apply ONLY basic filters (business status, must_not constraints) - no ranking/scoring
        filtered_restaurants = []
        for restaurant in self.restaurants:
            # Only filter out closed restaurants and must_not constraints
            if restaurant.business_status and restaurant.business_status != "OPERATIONAL":
                continue
            # Check must_not constraints
            if parsed_query.must_not.formality and restaurant.vibe and restaurant.vibe.formality:
                if restaurant.vibe.formality.lower() in [f.lower() for f in parsed_query.must_not.formality]:
                    continue
            if parsed_query.must_not.price and restaurant.price_level:
                if restaurant.price_level in parsed_query.must_not.price:
                    continue
            if parsed_query.must_not.cuisine and restaurant.cuisine:
                restaurant_cuisine_lower = restaurant.cuisine.lower()
                for excluded_cuisine in parsed_query.must_not.cuisine:
                    if excluded_cuisine.lower() in restaurant_cuisine_lower:
                        continue
            filtered_restaurants.append(restaurant)
        
        logger.info(f"   After basic filtering: {len(filtered_restaurants)} restaurants available for Gemini")
        
        # Create simple list of restaurants (no scoring/ranking) - Gemini will do everything
        from .fusion import ScoredRestaurant
        # Just wrap restaurants in ScoredRestaurant for compatibility, but with neutral scores
        scored_results = [
            ScoredRestaurant(
                restaurant=restaurant,
                vibe_score=0.5,
                cuisine_score=0.5,
                price_score=0.5,
                feature_score=0.5,
                final_score=0.5,
            )
            for restaurant in filtered_restaurants
        ]
        
        # Log award winners for reference (but no ranking)
        from .fusion import has_award, get_award_level
        award_winners = [r for r in scored_results if has_award(r.restaurant)]
        if award_winners:
            logger.info(f"🏆 Found {len(award_winners)} award-winning restaurants (Gemini will prioritize):")
            for r in award_winners[:10]:
                award_level = get_award_level(r.restaurant)
                logger.info(f"   - {r.restaurant.name} (award_level={award_level:.2f})")
        
        # NO RANKING - Pass all restaurants to Gemini for selection
        ranked_results = scored_results  # Just use all restaurants, no ranking
        logger.info(f"✅ Passing {len(ranked_results)} restaurants to Gemini (no ranking/scoring)")

        return await self.response_generator.generate(parsed_query, ranked_results)
    
    async def search_for_streaming(self, query: str) -> tuple[ParsedQuery, list[ScoredRestaurant]]:
        """Execute search pipeline and return parsed query and ranked results for streaming.
        
        This method performs the same search logic as search() but returns intermediate results
        instead of generating the full response, allowing the API to stream the response.
        """
        await self._ensure_initialized()
        parsed_query = await self.query_parser.parse(query)
        
        # Enhance structured query with LLM general knowledge (same as regular search)
        from .query_enhancer import QueryEnhancer
        query_enhancer = QueryEnhancer(client=self.client)
        parsed_query = await query_enhancer.enhance_query(parsed_query)
        
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

        # SKIP RAG: Pass all restaurant data directly to Gemini for analysis
        logger.info(f"🤖 GEMINI DIRECT ACCESS (Streaming): Skipping RAG, passing all restaurant data to Gemini")
        logger.info(f"   Total restaurants in database: {len(self.restaurants)}")
        
        # Apply basic filters (business status, must_not constraints) but keep all others
        filtered_restaurants = []
        for restaurant in self.restaurants:
            # Only filter out closed restaurants and must_not constraints
            if restaurant.business_status and restaurant.business_status != "OPERATIONAL":
                continue
            # Check must_not constraints
            if parsed_query.must_not.formality and restaurant.vibe and restaurant.vibe.formality:
                if restaurant.vibe.formality.lower() in [f.lower() for f in parsed_query.must_not.formality]:
                    continue
            if parsed_query.must_not.price and restaurant.price_level:
                if restaurant.price_level in parsed_query.must_not.price:
                    continue
            if parsed_query.must_not.cuisine and restaurant.cuisine:
                restaurant_cuisine_lower = restaurant.cuisine.lower()
                for excluded_cuisine in parsed_query.must_not.cuisine:
                    if excluded_cuisine.lower() in restaurant_cuisine_lower:
                        continue
            filtered_restaurants.append(restaurant)
        
        logger.info(f"   After basic filtering: {len(filtered_restaurants)} restaurants available for Gemini")
        
        # Create dummy scored results - Gemini will do the actual selection
        # BUT: Prioritize award-winning restaurants by boosting their scores
        from .fusion import ScoredRestaurant, has_award, get_award_level
        dummy_scored_results = []
        for restaurant in filtered_restaurants:
            # Base neutral scores
            base_score = 0.5
            
            # BOOST award-winning restaurants significantly
            if has_award(restaurant):
                award_level = get_award_level(restaurant)
                # Award boost: Gold = +0.4, Silver = +0.3, Honorable = +0.2, Other = +0.1
                award_boost = 0.4 * award_level
                base_score = min(1.0, 0.5 + award_boost)
            
            # Create a ScoredRestaurant with boosted scores for award winners
            dummy_scored = ScoredRestaurant(
                restaurant=restaurant,
                vibe_score=base_score,
                cuisine_score=base_score,
                price_score=base_score,
                feature_score=base_score,
                final_score=base_score,  # Award winners get higher scores
            )
            dummy_scored_results.append(dummy_scored)
        
        # Sort by final_score (award winners will be at the top) before passing to Gemini
        scored_results = sorted(dummy_scored_results, key=lambda x: x.final_score, reverse=True)
        
        # Log award winners for debugging
        award_winners = [r for r in scored_results if has_award(r.restaurant)]
        if award_winners:
            logger.info(f"🏆 Found {len(award_winners)} award-winning restaurants (prioritized):")
            for r in award_winners[:10]:
                award_level = get_award_level(r.restaurant)
                logger.info(f"   - {r.restaurant.name} (award_level={award_level:.2f}, final_score={r.final_score:.3f})")
        
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

