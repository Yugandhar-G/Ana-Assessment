"""Helper to lookup video URLs for restaurants from video metadata."""
import json
from pathlib import Path
from typing import List, Optional, Dict


# Cache for video metadata to avoid reloading
_video_metadata_cache: Optional[List[Dict]] = None


def _load_video_metadata() -> List[Dict]:
    """Load video metadata from JSON file (with caching)."""
    global _video_metadata_cache
    
    if _video_metadata_cache is not None:
        return _video_metadata_cache
    
    # Try to find video_metadata.json in data directory
    data_path = Path(__file__).parent.parent.parent / "data" / "video_metadata.json"
    
    if not data_path.exists():
        _video_metadata_cache = []
        return _video_metadata_cache
    
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            _video_metadata_cache = json.load(f)
        return _video_metadata_cache
    except (json.JSONDecodeError, FileNotFoundError):
        _video_metadata_cache = []
        return _video_metadata_cache


def _normalize_restaurant_id(restaurant_id: str) -> str:
    """Normalize restaurant ID for matching.
    
    restaurants.json uses "rest_261" format
    video_metadata.json uses "261" format
    This function strips "rest_" prefix if present.
    """
    if restaurant_id.startswith("rest_"):
        return restaurant_id[5:]  # Remove "rest_" prefix
    return restaurant_id


def get_video_urls_for_restaurant(
    restaurant_id: Optional[str] = None,
    restaurant_name: Optional[str] = None,
    google_place_id: Optional[str] = None
) -> List[str]:
    """Get video URLs associated with a restaurant.
    
    Matches by:
    1. Restaurant ID (primary - handles "rest_261" vs "261" format)
    2. Google Place ID (most reliable for cross-reference)
    3. Restaurant name (exact or flexible match - fallback)
    
    Args:
        restaurant_id: Restaurant ID from restaurants.json (e.g., "rest_261")
        restaurant_name: Restaurant name
        google_place_id: Google Place ID
    
    Returns:
        List of video URLs (empty list if none found)
    """
    video_metadata = _load_video_metadata()
    
    if not video_metadata:
        return []
    
    video_urls = []
    
    # Normalize restaurant ID for matching
    normalized_restaurant_id = None
    if restaurant_id:
        normalized_restaurant_id = _normalize_restaurant_id(restaurant_id)
    
    for video in video_metadata:
        # Skip if no video URL
        if not video.get('video_url'):
            continue
        
        # PRIMARY: Match by restaurant ID (handles "rest_261" vs "261")
        if normalized_restaurant_id and video.get('restaurant_id'):
            video_restaurant_id = str(video.get('restaurant_id'))
            # Try direct match
            if video_restaurant_id == normalized_restaurant_id:
                video_urls.append(video['video_url'])
                continue
            # Also try matching the original ID format
            if video_restaurant_id == restaurant_id:
                video_urls.append(video['video_url'])
                continue
        
        # SECONDARY: Match by Google Place ID (most reliable for cross-reference)
        if google_place_id and video.get('restaurant_google_place_id') == google_place_id:
            video_urls.append(video['video_url'])
            continue
        
        # TERTIARY: Match by restaurant name (exact or case-insensitive) - fallback
        if restaurant_name:
            video_restaurant_name = video.get('restaurant_name')
            if video_restaurant_name:
                # Exact match
                if video_restaurant_name == restaurant_name:
                    video_urls.append(video['video_url'])
                    continue
                # Case-insensitive match
                if video_restaurant_name.lower() == restaurant_name.lower():
                    video_urls.append(video['video_url'])
                    continue
                # Partial match (restaurant name contains video name or vice versa)
                if restaurant_name.lower() in video_restaurant_name.lower() or \
                   video_restaurant_name.lower() in restaurant_name.lower():
                    video_urls.append(video['video_url'])
                    continue
    
    # Remove duplicates and return
    return list(dict.fromkeys(video_urls))  # Preserves order while removing duplicates

