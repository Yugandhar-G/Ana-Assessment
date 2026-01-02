"""API endpoint for structured search responses."""
import sys
from pathlib import Path
from fastapi import HTTPException
from typing import Optional

# Add parent directory to path
_current_dir = Path(__file__).parent
_parent_dir = _current_dir.parent
if str(_parent_dir) not in sys.path:
    sys.path.insert(0, str(_parent_dir))

try:
    from part1.src.pipeline import AnaVibeSearch
    from part1.src.response_generator import ResponseGenerator
except ImportError:
    from src.pipeline import AnaVibeSearch
    from src.response_generator import ResponseGenerator

from .schemas import StructuredSearchResponse, MatchReason
from .enrichment import enrich_from_scored_restaurant
from .formatter import extract_vibe_only


async def search_structured(
    search_instance: AnaVibeSearch,
    query: str
) -> StructuredSearchResponse:
    """Perform structured search and return enriched restaurant data.
    
    This endpoint uses search_for_streaming to get scored restaurants with
    original restaurant objects, then enriches them with all needed fields.
    """
    # Check if query is restaurant-related
    quick_check = search_instance.query_parser._quick_restaurant_check(query)
    if quick_check is False or (quick_check is None and not await search_instance.query_parser.is_restaurant_query(query)):
        return StructuredSearchResponse(
            success=False,
            caveats=["Query is not restaurant-related"]
        )
    
    try:
        # Check if query is about a specific restaurant
        detected_restaurant = search_instance.query_parser._detect_restaurant_name(query)
        single_restaurant_query = detected_restaurant is not None
        
        # Use search_for_streaming to get scored restaurants with original objects
        parsed_query, ranked_results_full = await search_instance.search_for_streaming(query)
        
        if not ranked_results_full:
            return StructuredSearchResponse(
                success=False,
                caveats=["No restaurants found matching the query"]
            )
        
        # For single restaurant queries, verify the top match is actually the requested restaurant
        # and filter out other restaurants for display
        ranked_results_for_display = ranked_results_full.copy()
        if single_restaurant_query and detected_restaurant:
            # Find the exact restaurant in results
            exact_match_found = False
            for i, scored in enumerate(ranked_results_for_display):
                if scored.restaurant.name.lower() == detected_restaurant.lower():
                    # Move exact match to top
                    ranked_results_for_display.insert(0, ranked_results_for_display.pop(i))
                    exact_match_found = True
                    break
            
            # For single restaurant queries, only keep the exact match (or top result if exact match not found)
            if exact_match_found:
                ranked_results_for_display = [ranked_results_for_display[0]]  # Only keep the exact match
            else:
                # If exact match not found but restaurant was detected, keep only top result
                ranked_results_for_display = [ranked_results_for_display[0]]
        
        # Enrich the top match
        top_match_enriched = enrich_from_scored_restaurant(ranked_results_for_display[0])
        
        # Extract just the vibe from vibe_summary for top match
        if top_match_enriched.vibe_summary:
            try:
                extracted_vibe = await extract_vibe_only(top_match_enriched.vibe_summary)
                top_match_enriched.vibe_summary = extracted_vibe
            except Exception as e:
                # Fallback: use original vibe_summary if extraction fails
                print(f"Error extracting vibe for top match: {e}")
        
        # Only include alternatives if not a single restaurant query
        alternatives_enriched = []
        if not single_restaurant_query:
            # Enrich alternatives (up to 9 more, for 10 total)
            # Use ranked_results_for_display which has all results for general queries
            for scored in ranked_results_for_display[1:10]:
                alt_enriched = enrich_from_scored_restaurant(scored)
                # Extract vibe for alternatives too
                if alt_enriched.vibe_summary:
                    try:
                        extracted_vibe = await extract_vibe_only(alt_enriched.vibe_summary)
                        alt_enriched.vibe_summary = extracted_vibe
                    except Exception as e:
                        print(f"Error extracting vibe for alternative: {e}")
                alternatives_enriched.append(alt_enriched)
        
        # Generate LLM explanation using ResponseGenerator
        # This uses the LLM's knowledge to generate natural language responses
        response_generator = ResponseGenerator(client=search_instance.client)
        
        # For LLM generation, use the FULL ranked_results (not filtered)
        # This ensures the LLM can see all options and select appropriate alternatives
        # Generate full LLM response with explanation
        llm_response = await response_generator.generate(parsed_query, ranked_results_full)
        
        # Extract match reasons from LLM response
        match_reasons_list = llm_response.match_reasons
        
        # Convert MatchReason objects from src.schemas.response to our MatchReason schema
        # (they have the same structure, so we can convert by creating new instances)
        match_reasons_converted = [
            MatchReason(
                signal=reason.signal,
                importance=reason.importance,
                query_wanted=reason.query_wanted,
                restaurant_has=reason.restaurant_has,
                score=reason.score,
            )
            for reason in match_reasons_list
        ]
        
        # Use confidence from LLM response
        confidence = llm_response.confidence
        
        return StructuredSearchResponse(
            success=True,
            top_match=top_match_enriched,
            alternatives=alternatives_enriched,
            confidence=confidence,
            caveats=llm_response.caveats,
            match_reasons=match_reasons_converted,
            query=query,
            explanation=llm_response.explanation,  # LLM-generated natural language explanation
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

