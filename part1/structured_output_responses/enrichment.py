"""Enrichment utilities for converting restaurant data to enriched format."""
import sys
from pathlib import Path
from typing import Optional

# Add parent directory to path for imports
_current_dir = Path(__file__).parent
_parent_dir = _current_dir.parent
if str(_parent_dir) not in sys.path:
    sys.path.insert(0, str(_parent_dir))

try:
    from part1.src.video_lookup import get_video_urls_for_restaurant
    from part1.src.fusion import ScoredRestaurant
except ImportError:
    from src.video_lookup import get_video_urls_for_restaurant
    from src.fusion import ScoredRestaurant

from .schemas import EnrichedRestaurantMatch


def enrich_from_scored_restaurant(scored: ScoredRestaurant) -> EnrichedRestaurantMatch:
    """Convert ScoredRestaurant to EnrichedRestaurantMatch with all fields."""
    restaurant = scored.restaurant
    
    # Get video URLs
    video_urls = get_video_urls_for_restaurant(
        restaurant_id=restaurant.id,
        restaurant_name=restaurant.name,
        google_place_id=getattr(restaurant, 'google_place_id', None)
    )
    
    # Get all enriched fields
    highlights = getattr(restaurant, 'highlights', '') or ''
    details = getattr(restaurant, 'details', '') or ''
    editorial_summary = getattr(restaurant, 'editorial_summary', '') or ''
    top_menu_items = getattr(restaurant, 'top_menu_items', []) or []
    photos_urls = getattr(restaurant, 'photos_urls', []) or []
    restaurant_photos_urls = getattr(restaurant, 'restaurant_photos_urls', []) or []
    
    # Get vibe fields
    vibe_summary = ''
    vibe_formality = None
    vibe_noise_level = None
    vibe_atmosphere_tags = []
    vibe_best_for = []
    
    if hasattr(restaurant, 'vibe') and restaurant.vibe:
        vibe_summary = restaurant.vibe.vibe_summary or ''
        vibe_formality = restaurant.vibe.formality
        vibe_noise_level = restaurant.vibe.noise_level
        vibe_atmosphere_tags = restaurant.vibe.atmosphere_tags or []
        vibe_best_for = restaurant.vibe.best_for or []
    
    if not vibe_summary:
        vibe_summary = f"{restaurant.name} offers a welcoming dining experience."
    
    return EnrichedRestaurantMatch(
        id=restaurant.id,
        name=restaurant.name,
        cuisine=restaurant.cuisine,
        price_level=restaurant.price_level,
        price_level_curated=getattr(restaurant, 'price_level_curated', None),
        region=restaurant.region,
        city=getattr(restaurant, 'city', None),
        formatted_address=getattr(restaurant, 'formatted_address', None),
        rating=restaurant.rating,
        features=restaurant.features,
        vibe_summary=vibe_summary,
        vibe_formality=vibe_formality,
        vibe_noise_level=vibe_noise_level,
        vibe_atmosphere_tags=vibe_atmosphere_tags,
        vibe_best_for=vibe_best_for,
        highlights=highlights,
        details=details,
        editorial_summary=editorial_summary,
        top_menu_items=top_menu_items,
        photos_urls=photos_urls,
        restaurant_photos_urls=restaurant_photos_urls,
        video_urls=video_urls,
        final_score=scored.final_score,
        vibe_score=scored.vibe_score,
        cuisine_score=scored.cuisine_score,
        price_score=scored.price_score,
        feature_score=scored.feature_score,
    )

