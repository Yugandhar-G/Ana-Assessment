
┌─────────────────────────────────────────────────────────────────────────┐
│                         USER QUERY INPUT                                 │
│                    "best dessert in Maui"                                │
└──────────────────────────────┬──────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    STEP 1: QUERY PARSING                                 │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ QueryParser.parse(query)                                         │  │
│  │ - Extracts: cuisine, price, location, features, atmosphere      │  │
│  │ - Creates ParsedQuery object                                    │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────┬──────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│              STEP 2: QUERY ENHANCEMENT (LLM)                            │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ QueryEnhancer.enhance_query(parsed_query)                       │  │
│  │ - Adds implicit requirements (e.g., "dessert" → bakery features)│  │
│  │ - Adds cultural context and domain knowledge                    │  │
│  │ - Adjusts signal weights                                        │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────┬──────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│              STEP 3A: VECTOR SEARCH (Semantic Filtering)                 │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ VectorStore.search(enhanced_query, n_results=150)               │  │
│  │ - Uses ChromaDB with text-embedding-004                          │  │
│  │ - Cosine similarity search on restaurant embeddings              │  │
│  │ - Semantic matching: cuisine, vibe, menu items, features         │  │
│  │                                                                  │  │
│  │ Result: ~150 most semantically relevant restaurants             │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────┬──────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│              STEP 3B: BASIC FILTERING (Hard Filters)                    │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ Filter restaurants:                                             │  │
│  │ ✓ Business status = OPERATIONAL                                 │  │
│  │ ✓ Exclude must_not constraints (formality, price, cuisine)       │  │
│  │                                                                  │  │
│  │ Result: ~140-150 restaurants (filtered from vector search)      │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────┬──────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│        STEP 3C: MULTI-SIGNAL SCORING (Parallel Execution)               │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ Four parallel scorers compute individual relevance scores:       │  │
│  │                                                                  │  │
│  │ 1. VibeScorer: Cosine similarity of embeddings                  │  │
│  │    - Query embedding vs restaurant vibe embedding               │  │
│  │    - Captures semantic similarity (atmosphere, experience)      │  │
│  │                                                                  │  │
│  │ 2. CuisineScorer: Fuzzy string matching                        │  │
│  │    - Requested cuisine vs restaurant cuisine                    │  │
│  │    - Handles variations (e.g., "Hawaiian" = "Hawaii Regional")  │  │
│  │                                                                  │  │
│  │ 3. PriceScorer: Euclidean distance in price space                │  │
│  │    - Requested price range vs restaurant price level            │  │
│  │                                                                  │  │
│  │ 4. FeatureScorer: Jaccard similarity                            │  │
│  │    - Requested features vs restaurant features                  │  │
│  │    - Set intersection over union                                │  │
│  │                                                                  │  │
│  │ Result: Each restaurant has 4 individual scores                │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────┬──────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│        STEP 3D: ADVANCED SCORE FUSION (Multi-Signal Combination)        │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ AdvancedScoreFusion combines scores intelligently:              │  │
│  │                                                                  │  │
│  │ 1. Non-linear Transform: sqrt(score) to boost high scores       │  │
│  │    - 0.90 → 0.95, 0.50 → 0.71, 0.10 → 0.32                     │  │
│  │                                                                  │  │
│  │ 2. Weighted Sum: Dynamic weights from query                     │  │
│  │    - vibe_weight × vibe_score + cuisine_weight × cuisine_score  │  │
│  │                                                                  │  │
│  │ 3. Interaction Bonuses: When multiple signals align            │  │
│  │    - Cuisine + Features both >0.6: +0.10                        │  │
│  │    - Vibe + Cuisine both >0.6: +0.08                           │  │
│  │    - Triple alignment: +0.12                                    │  │
│  │                                                                  │  │
│  │ 4. Perfect Match Boosts: Exponential for high scores           │  │
│  │    - Cuisine ≥0.95: +0.08                                       │  │
│  │    - Features ≥0.95: +0.08                                      │  │
│  │    - Vibe ≥0.95: +0.05                                          │  │
│  │                                                                  │  │
│  │ 5. Award Recognition: Boost award-winning restaurants          │  │
│  │    - Gold Award: +0.15                                          │  │
│  │    - Silver Award: +0.12                                        │  │
│  │    - Honorable Mention: +0.08                                   │  │
│  │                                                                  │  │
│  │ 6. Safety Penalties: Accessibility and preferences             │  │
│  │    - Missing wheelchair access: -0.25                           │  │
│  │    - Noise mismatch (quiet requested, loud restaurant): -0.40   │  │
│  │    - Cuisine mismatch: -0.30                                     │  │
│  │                                                                  │  │
│  │ Result: Final score for each restaurant (0.0 to 1.0)            │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────┬──────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│        STEP 3E: AWARD-PRIORITY RERANKING (Intelligent Reordering)      │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ rank_with_award_priority() applies sophisticated reranking:     │  │
│  │                                                                  │  │
│  │ 1. Initial Sort: All restaurants by final_score (descending)    │  │
│  │                                                                  │  │
│  │ 2. Separate Categories:                                         │  │
│  │    - Primary cuisine matches (cuisine_score ≥ 0.95)            │  │
│  │    - Award winners + secondary cuisine                          │  │
│  │    - Others                                                      │  │
│  │                                                                  │  │
│  │ 3. Re-rank Primary Cuisine:                                     │  │
│  │    - Traditional restaurants first (not ghost kitchens)         │  │
│  │    - "Best [cuisine]" mentions boost                            │  │
│  │    - final_score (descending)                                   │  │
│  │    - rating (descending)                                        │  │
│  │    - award level (descending)                                   │  │
│  │                                                                  │  │
│  │ 4. Re-rank Award Winners:                                       │  │
│  │    - Award level (Gold > Silver > Honorable)                    │  │
│  │    - final_score (descending)                                   │  │
│  │    - rating (descending)                                        │  │
│  │                                                                  │  │
│  │ 5. Re-rank Others:                                              │  │
│  │    - rating (descending)                                        │  │
│  │    - final_score (descending)                                   │  │
│  │                                                                  │  │
│  │ 6. Combine: Primary cuisine → Award winners → Others            │  │
│  │                                                                  │  │
│  │ Result: Intelligently reranked list prioritizing quality        │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────┬──────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│        STEP 3F: LLM-BASED RERANKING (Optional Knowledge Enhancement)     │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ _llm_reason_about_results() uses Gemini to validate ranking:     │  │
│  │                                                                  │  │
│  │ 1. LLM Reasoning: Analyze query and top results                  │  │
│  │    - Understands cultural context and Maui dining scene          │  │
│  │    - Validates if ranking makes sense                            │  │
│  │    - Identifies missing context or gaps                          │  │
│  │                                                                  │  │
│  │ 2. Reranking Suggestion: LLM suggests reordering if needed      │  │
│  │    - "2,1,3,4,5" if restaurant 2 should be first                │  │
│  │    - "no change" if ranking is good                               │  │
│  │                                                                  │  │
│  │ 3. Apply Reranking: Reorder top 5 based on LLM knowledge         │  │
│  │                                                                  │  │
│  │ Result: Final ranked list enhanced by LLM domain knowledge       │  │
│  │                                                                  │  │
│  │ Note: Currently bypassed in Gemini-first mode, but available    │  │
│  │       for hybrid approaches or quality validation               │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────┬──────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│              STEP 4: WEB SEARCH (If Needed)                             │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ MultiSourceWebSearch.search()                                   │  │
│  │ - Google Custom Search                                          │  │
│  │ - Reddit (site:reddit.com)                                      │  │
│  │ - Blogs (Yelp, TripAdvisor, Eater, Timeout)                    │  │
│  │                                                                  │  │
│  │ Result: Recent reviews, cultural context, trends                 │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────┬──────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│         STEP 5: GEMINI QUERY PARSING & ENHANCEMENT                      │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ Gemini analyzes query:                                           │  │
│  │ - Deep understanding of user intent                             │  │
│  │ - Identifies if restaurant-specific or general                  │  │
│  │ - Extracts key requirements and cultural context                │  │
│  │                                                                  │  │
│  │ Output: Enhanced query analysis                                  │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────┬──────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│     STEP 6: UNIFIED GEMINI CALL (Answer + Restaurant Selection)         │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ Input to Gemini:                                                 │  │
│  │ • Enhanced query analysis                                        │  │
│  │ • All ~350 restaurants (formatted compactly)                    │  │
│  │ • Web search results (to summarize)                              │  │
│  │                                                                  │  │
│  │ Gemini Tasks:                                                    │  │
│  │ 1. Generate natural language answer                              │  │
│  │ 2. Select restaurants (1 top + up to 5 alternatives)            │  │
│  │    - OR: Select ONLY 1 if restaurant-specific query             │  │
│  │ 3. Summarize web search results naturally                        │  │
│  │ 4. Format response in structured format                          │  │
│  │                                                                  │  │
│  │ Output: Complete markdown response with selected restaurants     │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────┬──────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│              STEP 7: RESPONSE FORMATTING                                 │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ Formatter.inject_images_after_vibe()                             │  │
│  │ - Parses markdown to find restaurant names                       │  │
│  │ - Looks up restaurant photos from restaurants.json                │  │
│  │ - Injects images after "Vibe at this restaurant:" sections       │  │
│  │                                                                  │  │
│  │ Output: Final formatted response with images                     │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────┬──────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         FINAL RESPONSE                                   │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ Structured Format:                                               │  │
│  │ • Thinking Process (shown to user)                               │  │
│  │ • Natural language answer (Gemini-generated)                     │  │
│  │ • Restaurant 1:                                                  │  │
│  │   - Description                                                  │  │
│  │   - Good for: [menu items]                                       │  │
│  │   - Vibe at this restaurant: [vibe_summary]                      │  │
│  │   - Images (injected)                                            │  │
│  │   - Features                                                     │  │
│  │   - Videos                                                       │  │
│  │ • Restaurant 2-6 (if not restaurant-specific)                    │  │
│  │ • Helpful closing                                                │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```



### Hybrid Approach 
- ✅ Vector search + scoring + reranking (as above)
- ✅ Pass top 20-30 reranked restaurants to Gemini
- ✅ Gemini validates ranking and generates answer
- **Benefits**: Optimized performance + Gemini intelligence


