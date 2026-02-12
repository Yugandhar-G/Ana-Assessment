import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any
from .gemini_client import AsyncGeminiClient
from .schemas import ParsedQuery, AnaResponse, RestaurantMatch, MatchReason, Restaurant
from .video_lookup import get_video_urls_for_restaurant

logger = logging.getLogger(__name__)

class AgentResponseGenerator:
    """Generate responses using an Agentic Workflow: Brainstorm -> Ground -> Synthesize."""
    
    def __init__(self, client: AsyncGeminiClient, restaurants: List[Restaurant]):
        self.client = client
        self.model = os.getenv("GEMINI_CHAT_MODEL", "gemini-3-flash-preview")
        self.all_restaurants = restaurants
        # Build a lookup for faster grounding
        self.restaurant_name_map = {r.name.lower(): r for r in restaurants}
        
    async def generate(self, parsed_query: ParsedQuery) -> AnaResponse:
        """Main entry point for agentic response generation."""
        query = parsed_query.raw_query
        thinking_process = []
        logger.info(f"🚀 Starting Agentic Workflow for query: '{query}'")
        
        # STAGE 1: Brainstorm (Gemini + Web)
        thinking_process.append(f"🔍 **Brainstorming**: Using Gemini and Web Search to find Maui's best spots for '{query}'...")
        candidates_from_ai = await self._stage_brainstorm(parsed_query)
        thinking_process.append(f"✅ Found {len(candidates_from_ai)} potential candidates from real-time search.")
        
        # STAGE 2: Grounding (Database Lookup)
        thinking_process.append("🧬 **Grounding**: Verifying candidates against audited local database and looking up video metadata...")
        grounded_restaurants = await self._stage_grounding(candidates_from_ai)
        
        for rest in grounded_restaurants:
            thinking_process.append(f"   • Verified: **{rest.name}**")
            
        if not grounded_restaurants:
            thinking_process.append("❌ Could not verify any candidates in the local database.")
            return AnaResponse(
                success=False,
                explanation="I found some interesting options online, but I couldn't verify them in my Maui database. Try searching for something else?",
                thinking_process=thinking_process,
                confidence="low",
                caveats=["No grounded matches found in local database"]
            )
            
        # STAGE 3: Synthesis (Final Structured Response)
        thinking_process.append(f"✍️ **Synthesis**: Generating final recommendation for {len(grounded_restaurants)} verified restaurants...")
        final_response = await self._stage_synthesis(parsed_query, grounded_restaurants)
        final_response.thinking_process = thinking_process
        
        return final_response

    async def _stage_brainstorm(self, parsed_query: ParsedQuery) -> List[Dict[str, str]]:
        """Stage 1: Gemini + Web Search to find potential restaurant names and details."""
        query = parsed_query.raw_query
        
        # 1. Perform Web Search
        web_results = []
        if self.client.web_search:
            from .multi_source_search import SearchSource
            try:
                logger.info(f"🌐 Stage 1: Searching web for context on '{query}'...")
                results = await self.client.web_search.search(
                    query=f"{query} Maui Hawaii best restaurants",
                    sources=[SearchSource.GOOGLE, SearchSource.REDDIT, SearchSource.BLOGS],
                    num_results_per_source=3
                )
                web_results = [
                    {"title": r.title, "snippet": r.snippet, "source": r.source} 
                    for r in results
                ]
                logger.info(f"🌐 Web search returned {len(web_results)} snippets from Google, Reddit, and Blogs.")
            except Exception as e:
                logger.error(f"Web search failed in brainstorm stage: {e}")

        # 2. Ask Gemini to identify candidates
        web_context = "\n".join([f"- [{r['source']}] {r['title']}: {r['snippet']}" for r in web_results])
        
        prompt = f"""You are an expert Maui dining consultant. Based on the user query and web search results, suggest up to 8 specific restaurant names that would be the best fit.

USER QUERY: {query}

WEB SEARCH CONTEXT:
{web_context}

YOUR TASK:
1. Identify the best restaurants for this query using your knowledge and the web context.
2. For each restaurant, provide the exact name and a 1-sentence reason why it fits.
3. Prioritize actual Maui restaurants that likely exist in a high-quality database.

RESPOND IN JSON FORMAT:
{{
  "candidates": [
    {{"name": "Restaurant Name", "reason": "Why it fits the query"}}
  ]
}}
"""
        try:
            logger.info("🧠 Stage 1: Asking Gemini to brainstorm candidates...")
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.3
            )
            data = json.loads(response.choices[0].message["content"])
            candidates = data.get("candidates", [])
            logger.info(f"🧠 Gemini brainstormed {len(candidates)} candidates.")
            for c in candidates:
                logger.info(f"   • {c['name']}: {c['reason']}")
            return candidates
        except Exception as e:
            logger.error(f"Brainstorm stage failed: {e}")
            return []

    async def _stage_grounding(self, candidates: List[Dict[str, str]]) -> List[Restaurant]:
        """Stage 2: Match brainstormed candidates against the local database."""
        grounded = []
        seen_ids = set()
        
        logger.info(f"🧬 Stage 2: Grounding {len(candidates)} candidates in local database...")
        for candidate in candidates:
            name = candidate.get("name", "").lower().strip()
            if not name:
                continue
                
            # 1. Try exact/substring match in name map
            match = self.restaurant_name_map.get(name)
            if not match:
                # Try substring match
                for db_name, rest in self.restaurant_name_map.items():
                    if name in db_name or db_name in name:
                        match = rest
                        break
            
            if match and match.id not in seen_ids:
                grounded.append(match)
                seen_ids.add(match.id)
                logger.info(f"✅ Grounded: '{candidate['name']}' matches database entry '{match.name}' ({match.id})")
            else:
                logger.warning(f"❌ Could not ground candidate: '{candidate['name']}' - no match in restaurants.json")
                
        # Fallback: If we couldn't ground many candidates, do a quick semantic search for top 3
        if len(grounded) < 3:
            logger.info("⚠️ Low grounding count, attempting semantic fallback...")
            # This is a simplified fallback - in a full implementation, you'd use the vector store here
            pass
            
        return grounded[:6]  # Limit to top 6 grounded matches

    async def _stage_synthesis(self, parsed_query: ParsedQuery, grounded_restaurants: List[Restaurant]) -> AnaResponse:
        """Stage 3: Generate the final structured response using only grounded data."""
        
        # Prepare grounded data context
        grounded_data_text = ""
        for i, rest in enumerate(grounded_restaurants):
            video_urls = get_video_urls_for_restaurant(
                restaurant_id=rest.id,
                restaurant_name=rest.name,
                google_place_id=getattr(rest, "google_place_id", None)
            )
            features = ", ".join([k.replace("_", " ").title() for k, v in rest.features.items() if v])
            
            grounded_data_text += f"""
RESTAURANT {i+1}: {rest.name}
ID: {rest.id}
Cuisine: {rest.cuisine}
Price: {rest.price_level_curated or rest.price_level}
Region: {rest.region}
Rating: {rest.rating}
Vibe Summary (VERBATIM): {rest.vibe.vibe_summary}
Features: {features}
Menu: {", ".join(rest.top_menu_items[:5]) if rest.top_menu_items else "Not specified"}
Videos: {", ".join(video_urls) if video_urls else "None"}
"""

        prompt = f"""You are Ana, an expert Maui dining AI. Using ONLY the grounded restaurant data below, write a high-quality recommendation response.

       USER QUERY: {parsed_query.raw_query}

       GROUNDED RESTAURANT DATA:
       {grounded_data_text}

       **RESPONSE STRUCTURE (CRITICAL):**

       **1. OPENING:**
       Start with a natural, conversational opening about the user's request.

       **2. FOR EACH RESTAURANT (Strict Format):**

       ## Restaurant Name

       [2-4 sentences in a conversational style explaining why this restaurant matches the user's specific request.]

       **Good for:**
       [A SINGLE LINE comma-separated list of standout dishes or specialties.]

       **Vibe at this restaurant:**
       [COPY THE VIBE SUMMARY FROM THE DATA VERBATIM. DO NOT CHANGE A SINGLE WORD.]

       **Features:**
       [A SINGLE LINE comma-separated list of active features.]

       **Videos:**
       [List video URLs provided in the data, one per line using markdown: "- [Video Link](url)". If no videos, say "No videos available".]

       **3. CLOSING:**
       A helpful follow-up sentence or question.

       **RULES:**
       - Only include restaurants from the grounded data provided.
       - CRITICAL: Use '## 🏆 Restaurant Name' for the #1 best match.
       - CRITICAL: Use '## Restaurant Name' for all other restaurants.
       - DO NOT mention photos/images (they are injected automatically).
       - Write conversationally like Gemini, but follow the structure exactly.
       """

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
                max_tokens=10000
            )
            explanation = response.choices[0].message["content"].strip()
            
            # Prepare top match and alternatives for the response object
            all_matches = []
            for rest in grounded_restaurants:
                video_urls = get_video_urls_for_restaurant(rest.id, rest.name)
                match_obj = RestaurantMatch(
                    id=rest.id,
                    name=rest.name,
                    cuisine=rest.cuisine,
                    price_level=rest.price_level,
                    price_level_curated=rest.price_level_curated,
                    region=rest.region,
                    rating=rest.rating,
                    features=rest.features,
                    video_urls=video_urls,
                    vibe_summary=rest.vibe.vibe_summary,
                    photos_urls=getattr(rest, "photos_urls", []),
                    restaurant_photos_urls=getattr(rest, "restaurant_photos_urls", []),
                    final_score=1.0,
                    vibe_score=1.0,
                    cuisine_score=1.0,
                    price_score=1.0,
                    feature_score=1.0
                )
                all_matches.append(match_obj)

            return AnaResponse(
                success=True,
                top_match=all_matches[0],
                alternatives=all_matches[1:],
                explanation=explanation,
                confidence="high"
            )
        except Exception as e:
            logger.error(f"Synthesis stage failed: {e}")
            return AnaResponse(
                success=False,
                explanation=f"Error generating response: {str(e)}",
                confidence="low"
            )
