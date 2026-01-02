"""LLM-based query enhancement to enrich structured queries with general knowledge."""
import json
import logging
from typing import Dict, Any
from .schemas import ParsedQuery, Preferences, SignalWeights
from .gemini_client import AsyncGeminiClient
import os

logger = logging.getLogger(__name__)


class QueryEnhancer:
    """Use LLM general knowledge to enrich structured queries before RAG retrieval."""
    
    def __init__(self, client: AsyncGeminiClient | None = None, model: str | None = None):
        self.client = client or AsyncGeminiClient(
            api_key=os.getenv("GEMINI_API_KEY"),
            default_chat_model=os.getenv("GEMINI_CHAT_MODEL", "gemini-2.0-flash"),
        )
        self.model = model or os.getenv("GEMINI_CHAT_MODEL", "gemini-2.0-flash")
    
    async def enhance_query(self, parsed_query: ParsedQuery) -> ParsedQuery:
        """Enrich parsed query with LLM general knowledge about Maui/Hawaii dining.
        
        This adds:
        - Implicit preferences based on query intent (e.g., "dessert" → dessert-related features)
        - Cultural context (e.g., what "best dessert" means in Maui)
        - Domain knowledge (e.g., famous dessert types, typical features)
        - Adjusted weights based on what matters for this query type
        
        Returns:
            Enhanced ParsedQuery with LLM-informed preferences and weights
        """
        enhancement_prompt = f"""You are an expert on Maui, Hawaii, food culture, and dining experiences.

**USER QUERY:** {parsed_query.raw_query}

**CURRENT STRUCTURED QUERY:**
- Semantic Query: {parsed_query.semantic_query}
- Preferences: Cuisine={parsed_query.preferences.cuisine}, Price={parsed_query.preferences.price}, Features={parsed_query.preferences.features}, Atmosphere={parsed_query.preferences.atmosphere}
- Weights: Vibe={parsed_query.weights.vibe:.2f}, Cuisine={parsed_query.weights.cuisine:.2f}, Price={parsed_query.weights.price:.2f}, Features={parsed_query.weights.features:.2f}
- Location: {parsed_query.location or "Not specified"}

**YOUR TASK:**
Enrich this structured query with your knowledge about Maui/Hawaii dining to help RAG find better matches. Consider:

1. **Implicit Requirements**: What features/atmosphere does this query imply?
   - Example: "best dessert" → should look for restaurants with dessert features, bakeries, dessert shops
   - Example: "romantic dinner" → intimate atmosphere, candlelit, quiet
   - Example: "family-friendly breakfast" → kid-friendly features, casual atmosphere

2. **Cultural Context**: What does this query mean in Maui/Hawaii context?
   - Example: "dessert" → shave ice, malasadas, Hawaiian desserts, tropical fruits
   - Example: "breakfast" → local favorites, Hawaiian breakfast, fresh fruit, acai bowls
   - Example: "romantic" → beachfront, sunset views, intimate settings

3. **Domain Knowledge**: What features/atmosphere are typical for this query type?
   - Example: Dessert queries → look for dessert shops, bakeries, places with dessert menus
   - Example: Breakfast queries → early hours, casual atmosphere, local favorites

4. **Weight Adjustment**: What should matter most for this query?
   - Example: "best dessert" → vibe and features matter more than cuisine
   - Example: "Italian restaurant" → cuisine matters most
   - Example: "romantic dinner" → vibe and atmosphere matter most

**AVAILABLE FEATURES** (use exact names):
- serves_dessert, live_music, outdoor_seating, wheelchair_accessible, parking, takeout, reservations, etc.
- Use features that restaurants actually have in their data

**AVAILABLE ATMOSPHERE TAGS**:
- romantic, casual, intimate, lively, quiet, trendy, upscale, family-friendly, etc.

**RESPOND WITH JSON:**
{{
  "enhanced_preferences": {{
    "cuisine": ["list", "of", "cuisines"],  // Add if relevant, keep existing
    "price": ["$", "$$"],  // Add if relevant, keep existing  
    "features": ["serves_dessert", "feature2"],  // Add implicit features using exact feature names
    "atmosphere": ["intimate", "casual"]  // Add implicit atmosphere tags
  }},
  "enhanced_weights": {{
    "vibe": 0.0-1.0,  // Adjust based on what matters for this query type
    "cuisine": 0.0-1.0,
    "price": 0.0-1.0,
    "features": 0.0-1.0
    // Must sum to ~1.0
  }},
  "enhanced_semantic_query": "enriched query text with cultural context and related terms",
  "reasoning": "Brief explanation of what you added and why"
}}

**IMPORTANT:**
- Only add preferences/features that make sense for the query
- Don't remove existing preferences unless they conflict
- Weights must sum to approximately 1.0
- Enhanced semantic query should include cultural context and related terms
- Be specific: for "dessert" queries, add dessert-related features like "serves_dessert", "bakery", "dessert_shop"
"""
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert on Maui dining. Enrich structured queries with cultural context and domain knowledge. Respond with valid JSON only."},
                    {"role": "user", "content": enhancement_prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.3,  # Some creativity for context understanding
                max_tokens=500,
            )
            
            enhancement_json = json.loads(response.choices[0].message["content"])
            
            # Create enhanced preferences (merge with existing)
            enhanced_prefs = enhancement_json.get("enhanced_preferences", {})
            new_preferences = Preferences(
                cuisine=enhanced_prefs.get("cuisine", parsed_query.preferences.cuisine) or parsed_query.preferences.cuisine,
                price=enhanced_prefs.get("price", parsed_query.preferences.price) or parsed_query.preferences.price,
                features=list(set(parsed_query.preferences.features + enhanced_prefs.get("features", []))),
                atmosphere=list(set(parsed_query.preferences.atmosphere + enhanced_prefs.get("atmosphere", []))),
            )
            
            # Create enhanced weights (use LLM's adjusted weights)
            enhanced_weights = enhancement_json.get("enhanced_weights", {})
            if enhanced_weights:
                # Normalize weights to sum to 1.0
                total = sum(enhanced_weights.values())
                if total > 0:
                    new_weights = SignalWeights(
                        vibe=enhanced_weights.get("vibe", parsed_query.weights.vibe) / total,
                        cuisine=enhanced_weights.get("cuisine", parsed_query.weights.cuisine) / total,
                        price=enhanced_weights.get("price", parsed_query.weights.price) / total,
                        features=enhanced_weights.get("features", parsed_query.weights.features) / total,
                    )
                else:
                    new_weights = parsed_query.weights
            else:
                new_weights = parsed_query.weights
            
            # Enhanced semantic query
            enhanced_semantic = enhancement_json.get("enhanced_semantic_query", parsed_query.semantic_query)
            reasoning = enhancement_json.get("reasoning", "")
            
            logger.info(f"🔍 LLM Query Enhancement:")
            logger.info(f"   Reasoning: {reasoning}")
            logger.info(f"   Added features: {set(new_preferences.features) - set(parsed_query.preferences.features)}")
            logger.info(f"   Added atmosphere: {set(new_preferences.atmosphere) - set(parsed_query.preferences.atmosphere)}")
            logger.info(f"   Weight changes: vibe={parsed_query.weights.vibe:.2f}→{new_weights.vibe:.2f}, cuisine={parsed_query.weights.cuisine:.2f}→{new_weights.cuisine:.2f}")
            logger.info(f"   Enhanced semantic query: '{enhanced_semantic[:100]}...'")
            
            # Create enhanced ParsedQuery
            enhanced_query = ParsedQuery(
                raw_query=parsed_query.raw_query,
                semantic_query=enhanced_semantic,
                must_not=parsed_query.must_not,
                preferences=new_preferences,
                weights=new_weights,
                location=parsed_query.location,
            )
            
            return enhanced_query
            
        except Exception as e:
            logger.warning(f"⚠️  Query enhancement failed: {e}, using original parsed query")
            return parsed_query

