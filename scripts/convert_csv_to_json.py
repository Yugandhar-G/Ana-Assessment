#!/usr/bin/env python3
"""
Convert CSV restaurant data to restaurants.json format.
"""
import csv
import json
import sys
import unicodedata
from pathlib import Path
from typing import Dict, Any, List, Optional

def parse_features_from_row(row: Dict[str, str]) -> Dict[str, bool]:
    """Build features dictionary from individual CSV columns."""
    features = {}
    
    # Map CSV column names to feature names
    feature_mapping = {
        "allows_dogs": "allows_dogs",
        "delivery": "delivery",
        "dine_in": "dine_in",
        "takeout": "takeout",
        "serves_breakfast": "serves_breakfast",
        "serves_lunch": "serves_lunch",
        "serves_dinner": "serves_dinner",
        "serves_beer": "serves_beer",
        "serves_wine": "serves_wine",
        "serves_vegetarian_food": "serves_vegetarian",
        "outdoor_seating": "outdoor_seating",
        "live_music_google": "live_music",
        "menu_for_children": "menu_for_children",
        "serves_cocktails": "full_bar",  # Map cocktails to full_bar
        "serves_dessert": "serves_dessert",
        "serves_coffee": "serves_coffee",
        "good_for_children": "good_for_children",
        "good_for_groups": "good_for_groups",
        "good_for_watching_sports": "good_for_watching_sports",
        "wheelchair_accessible_entrance": "wheelchair_accessible",
        "wheelchair_accessible_parking": "wheelchair_accessible_parking",
        "wheelchair_accessible_restroom": "wheelchair_accessible_restroom",
        "wheelchair_accessible_seating": "wheelchair_accessible_seating",
    }
    
    for csv_col, feature_name in feature_mapping.items():
        value = row.get(csv_col, "").strip().lower()
        # Handle boolean values: "true", "1", "yes", etc.
        features[feature_name] = value in ("true", "1", "yes", "t", "y")
    
    # Set defaults for missing features
    default_features = {
        "reservations": False,
        "full_bar": features.get("full_bar", False),  # Already set from serves_cocktails
    }
    features.update(default_features)
    
    return features

def infer_formality(price_level: str, theme: str = "") -> str:
    """Infer formality level from price and theme."""
    price_level = price_level.strip()
    theme_lower = theme.lower()
    
    # Check for hole-in-the-wall in theme first (overrides price)
    if "hole-in-the-wall" in theme_lower or "hole in the wall" in theme_lower:
        return "very_casual"
    
    # Handle price ranges (e.g., "$$$-$$$$" -> use the higher end)
    if "-" in price_level:
        # Take the higher price level from the range
        parts = price_level.split("-")
        price_level = parts[-1].strip()
    
    # Infer from price level
    if price_level == "$$$$":
        return "fine_dining"
    elif price_level == "$$$":
        return "upscale"
    elif price_level == "$$":
        return "smart_casual"
    elif price_level == "$":
        return "very_casual"
    
    # If price level is unknown/empty, return None (will be handled by caller)
    return None

def infer_noise_level(live_music: bool, highlights: str = "", details: str = "", editorial_summary: str = "") -> str:
    """Infer noise level from features and text content."""
    # Priority 1: live_music feature → "loud"
    if live_music:
        return "loud"
    
    # Priority 2: Keywords in text
    combined_text = f"{highlights} {details} {editorial_summary}".lower()
    
    # Check for loud keywords
    loud_keywords = ["loud", "bustling", "energetic", "vibrant", "rowdy", "noisy"]
    if any(keyword in combined_text for keyword in loud_keywords):
        return "loud"
    
    # Check for quiet keywords
    quiet_keywords = ["quiet", "peaceful", "tranquil", "serene", "calm", "relaxed", "intimate"]
    if any(keyword in combined_text for keyword in quiet_keywords):
        return "quiet"
    
    # Check for lively (moderate but energetic)
    if "lively" in combined_text:
        return "lively"
    
    # If no signals found, return None (will be handled by caller)
    return None

def infer_atmosphere_tags(theme: str = "", price_level: str = "", features: Dict[str, bool] = None, 
                         highlights: str = "", details: str = "", editorial_summary: str = "") -> List[str]:
    """Infer atmosphere tags from multiple sources."""
    tags = []
    theme_lower = theme.lower()
    combined_text = f"{highlights} {details} {editorial_summary}".lower()
    features = features or {}
    
    # From Theme
    if "local favorite" in theme_lower or "local" in theme_lower:
        tags.append("local-favorite")
    if "hole-in-the-wall" in theme_lower or "hole in the wall" in theme_lower:
        tags.append("hole-in-the-wall")
    if "vegan" in theme_lower:
        tags.append("vegan-friendly")
    
    # From Price
    price_level = price_level.strip()
    if price_level == "$":
        tags.append("budget-friendly")
    elif price_level == "$$$$":
        tags.append("upscale")
    
    # From Features
    if features.get("live_music", False):
        tags.append("lively")
    if features.get("good_for_children", False):
        tags.append("family-friendly")
    if features.get("good_for_groups", False):
        tags.append("group-friendly")
    
    # From Text Analysis (highlights/details/editorial_summary)
    if "romantic" in combined_text or "intimate" in combined_text:
        tags.append("romantic")
    if "casual" in combined_text or "relaxed" in combined_text:
        if "casual" not in tags:  # Avoid duplicates
            tags.append("casual")
    if "trendy" in combined_text or "hip" in combined_text:
        tags.append("trendy")
    if "peaceful" in combined_text or "tranquil" in combined_text or "serene" in combined_text:
        tags.append("peaceful")
    if "ocean" in combined_text or "beach" in combined_text or "waterfront" in combined_text:
        tags.append("ocean-view")
    
    # Remove duplicates while preserving order
    seen = set()
    unique_tags = []
    for tag in tags:
        if tag not in seen:
            seen.add(tag)
            unique_tags.append(tag)
    
    return unique_tags

def infer_best_for(features: Dict[str, bool] = None, atmosphere_tags: List[str] = None) -> List[str]:
    """Infer best_for from features and atmosphere tags."""
    best_for = []
    features = features or {}
    atmosphere_tags = atmosphere_tags or []
    
    # From atmosphere tags
    if "romantic" in atmosphere_tags:
        best_for.append("date night")
    
    # From features
    if features.get("good_for_groups", False):
        best_for.append("groups")
    if features.get("live_music", False):
        best_for.append("nightlife")
    if features.get("good_for_children", False):
        best_for.append("families")
    if features.get("serves_breakfast", False):
        best_for.append("breakfast")
    if features.get("serves_lunch", False):
        best_for.append("lunch")
    if features.get("serves_dinner", False):
        best_for.append("dinner")
    
    return best_for

def parse_vibe(vibe_str: str, restaurant_name: str = "", cuisine: str = "", details: str = "", 
               highlights: str = "", price_level: str = "", theme: str = "", 
               features: Dict[str, bool] = None, editorial_summary: str = "") -> Dict[str, Any]:
    """Parse vibe string to dictionary. Infer all fields from data if vibe_str is empty."""
    
    # If vibe_str exists and is valid JSON, try to parse it first
    if vibe_str and vibe_str.strip():
        try:
            if vibe_str.strip().startswith('{'):
                parsed = json.loads(vibe_str.replace("'", '"'))
            else:
                import ast
                parsed = ast.literal_eval(vibe_str)
            
            # Use parsed values if available, otherwise infer
            formality = parsed.get("formality") or infer_formality(price_level, theme)
            noise_level = parsed.get("noise_level") or infer_noise_level(
                features.get("live_music", False) if features else False,
                highlights, details, editorial_summary
            )
            atmosphere_tags = parsed.get("atmosphere_tags", [])
            if not atmosphere_tags:
                atmosphere_tags = infer_atmosphere_tags(theme, price_level, features, highlights, details, editorial_summary)
            best_for = parsed.get("best_for", [])
            if not best_for:
                best_for = infer_best_for(features, atmosphere_tags)
            
            vibe_summary = parsed.get("vibe_summary", "")
            if not vibe_summary or not vibe_summary.strip():
                vibe_summary = highlights or details or editorial_summary or f"{restaurant_name} serves {cuisine} cuisine."
            
            return {
                "formality": formality or "casual",  # Fallback only if inference fails
                "noise_level": noise_level or "moderate",  # Fallback only if inference fails
                "atmosphere_tags": atmosphere_tags,
                "best_for": best_for,
                "vibe_summary": vibe_summary[:500]
            }
        except:
            pass  # Fall through to inference logic
    
    # No valid vibe_str, infer everything from data
    formality = infer_formality(price_level, theme)
    noise_level = infer_noise_level(
        features.get("live_music", False) if features else False,
        highlights, details, editorial_summary
    )
    atmosphere_tags = infer_atmosphere_tags(theme, price_level, features, highlights, details, editorial_summary)
    best_for = infer_best_for(features, atmosphere_tags)
    
    # Generate vibe_summary
    vibe_summary = highlights or details or editorial_summary or f"{restaurant_name} serves {cuisine} cuisine."
    
    return {
        "formality": formality or "casual",  # Fallback only if inference fails
        "noise_level": noise_level or "moderate",  # Fallback only if inference fails
        "atmosphere_tags": atmosphere_tags,
        "best_for": best_for,
        "vibe_summary": vibe_summary[:500]
    }

def parse_list_field(field_str: str) -> List[str]:
    """Parse list field (e.g., serves_meal_times, photos_urls)."""
    if not field_str or field_str.strip() == '':
        return []
    
    # Handle JSON array first
    if field_str.strip().startswith('['):
        try:
            return json.loads(field_str.replace("'", '"'))
        except:
            pass
    
    # Try pipe-separated (common in CSV exports)
    # Split by ' | ' (space-pipe-space) first, then fallback to '|'
    if ' | ' in field_str:
        items = [item.strip() for item in field_str.split(' | ') if item.strip()]
        # Only return if we actually got multiple items (not just one)
        if len(items) > 1:
            return items
        # If only one item but it still contains '|', try splitting by '|' without spaces
        elif len(items) == 1 and '|' in items[0]:
            items = [item.strip() for item in items[0].split('|') if item.strip()]
            return items if len(items) > 1 else [field_str.strip()]
        else:
            return items
    
    # Fallback: split by pipe (no spaces)
    if '|' in field_str:
        items = [item.strip() for item in field_str.split('|') if item.strip()]
        return items if len(items) > 1 else [field_str.strip()]
    
    # Try comma-separated
    if ',' in field_str:
        return [item.strip() for item in field_str.split(',') if item.strip()]
    
    return [field_str.strip()] if field_str.strip() else []

def convert_row_to_restaurant(row: Dict[str, str]) -> Dict[str, Any]:
    """Convert CSV row to restaurant JSON object."""
    # Get restaurant ID - add "rest_" prefix if not present
    restaurant_id = row.get("id", "").strip()
    if restaurant_id and not restaurant_id.startswith("rest_"):
        restaurant_id = f"rest_{restaurant_id}"
    elif not restaurant_id:
        restaurant_id = f"rest_unknown_{hash(row.get('restaurant_name_curated', ''))}"
    
    # Get name - prefer curated name, fallback to google_name
    name = row.get("restaurant_name_curated", "").strip() or row.get("google_name", "").strip()
    
    # Get price level - prefer curated, fallback to google
    price_level = row.get("price_level_curated", "").strip() or row.get("price_level_google", "").strip() or "$"
    
    # Get region - use maui_region
    region = row.get("maui_region", "").strip() or "UNKNOWN"
    
    # Parse boolean fields
    def parse_bool(value: str) -> Optional[bool]:
        if not value or value.strip() == '':
            return None
        return value.strip().lower() in ("true", "1", "yes", "t", "y")
    
    # Parse float fields
    def parse_float(value: str) -> Optional[float]:
        if not value or value.strip() == '':
            return None
        try:
            return float(value.strip())
        except:
            return None
    
    restaurant = {
        "id": restaurant_id,
        "name": name,
        "google_place_id": row.get("google_place_id", "").strip() or None,
        "cuisine": row.get("cuisine", "").strip() or "Unknown",
        "price_level": price_level,
        "price_level_curated": row.get("price_level_curated", "").strip() or price_level,
        "region": region,
        "city": row.get("city", "").strip() or None,
        "formatted_address": row.get("formatted_address", "").strip() or None,
        "location_raw": row.get("location_raw", "").strip() or None,
        "state": row.get("state", "").strip() or None,
        "zipcode": row.get("zipcode", "").strip() or None,
        "country": row.get("country", "").strip() or "US",
        "latitude": parse_float(row.get("latitude", "")),
        "longitude": parse_float(row.get("longitude", "")),
        "rating": parse_float(row.get("rating", "")) or 0.0,
        "features": parse_features_from_row(row),
        "highlights": row.get("highlights", "").strip() or "",
        "details": row.get("details", "").strip() or "",
        "editorial_summary": row.get("editorial_summary", "").strip() or "",
        "top_menu_items": parse_list_field(row.get("top_menu_items", "")),  # Parse menu items if available
        "business_status": row.get("business_status", "").strip() or "OPERATIONAL",
        "national_phone": row.get("national_phone", "").strip() or None,
        "international_phone": row.get("international_phone", "").strip() or None,
        "website_uri": row.get("website_uri", "").strip() or None,
        "google_maps_uri": row.get("google_maps_uri", "").strip() or None,
        "opening_hours_text": row.get("opening_hours_text", "").strip() or None,
        "is_open_now": parse_bool(row.get("is_open_now", "")),
        "serves_meal_times": parse_list_field(row.get("serves_meal_times", "")),
        "photos_urls": parse_list_field(row.get("photos_urls", "")),
        "restaurant_photos_urls": parse_list_field(row.get("restaurant_photos_urls", "")),
        "reviews": row.get("reviews", "").strip() or "",
        "payment_options": row.get("payment_options", "").strip() or None,
        "parking_options": row.get("parking_options", "").strip() or None,
        "restroom": row.get("restroom", "").strip() or None,
        "match_status": row.get("match_status", "").strip() or None,
        "match_confidence": parse_float(row.get("match_confidence", "")),
        "name_similarity": parse_float(row.get("name_similarity", "")),
        "data_completeness_score": parse_float(row.get("data_completeness_score", "")),
        "google_matched_name": row.get("google_matched_name", "").strip() or None,
        "created_at": row.get("created_at", "").strip() or None,
        "updated_at": row.get("updated_at", "").strip() or None,
        "live_music_curated": parse_bool(row.get("live_music_curated", "")),
        "live_music_google": parse_bool(row.get("live_music_google", "")),
        "vibe": parse_vibe(
            row.get("vibe", ""),
            restaurant_name=name,
            cuisine=row.get("cuisine", "").strip() or "Unknown",
            details=row.get("details", "").strip() or "",
            highlights=row.get("highlights", "").strip() or "",
            price_level=price_level,
            theme=row.get("theme", "").strip() or "",
            features=parse_features_from_row(row),
            editorial_summary=row.get("editorial_summary", "").strip() or ""
        )
    }
    
    return restaurant

def main():
    csv_path = Path(__file__).parent.parent / "data" / "Gopu-Assignment-Data-All-Maui-DB-Restaurants - kaana_restaurants_export_20251227_011007.csv"
    output_path = Path(__file__).parent.parent / "data" / "restaurants.json"
    
    if not csv_path.exists():
        print(f"Error: CSV file not found at {csv_path}")
        sys.exit(1)
    
    restaurants = []
    
    print(f"Reading CSV from {csv_path}...")
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, 1):
            try:
                restaurant = convert_row_to_restaurant(row)
                restaurants.append(restaurant)
                if i % 50 == 0:
                    print(f"  Processed {i} restaurants...")
            except Exception as e:
                print(f"  Warning: Error processing row {i}: {e}")
                continue
    
    print(f"\nTotal restaurants processed: {len(restaurants)}")
    
    # Sort by ID for consistency
    restaurants.sort(key=lambda x: x.get("id", ""))
    
    print(f"Writing to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Normalize Unicode to avoid ambiguous characters
    # Replace smart quotes with ASCII quotes, but preserve Hawaiian okina (ʻ)
    def normalize_unicode(obj):
        """Recursively normalize Unicode strings in the data structure."""
        if isinstance(obj, str):
            # First normalize to NFC (Canonical Composition)
            normalized = unicodedata.normalize('NFC', obj)
            
            # Replace ambiguous Unicode characters with ASCII equivalents
            # Smart quotes → regular quotes/apostrophes
            normalized = normalized.replace('\u2018', "'")  # LEFT SINGLE QUOTATION MARK → '
            normalized = normalized.replace('\u2019', "'")  # RIGHT SINGLE QUOTATION MARK → '
            normalized = normalized.replace('\u201C', '"')  # LEFT DOUBLE QUOTATION MARK → "
            normalized = normalized.replace('\u201D', '"')  # RIGHT DOUBLE QUOTATION MARK → "
            normalized = normalized.replace('\u2013', '-')  # EN DASH → -
            normalized = normalized.replace('\u2014', '--')  # EM DASH → --
            
            # Preserve Hawaiian okina (ʻ - U+02BB) - this is legitimate and should stay
            # All other characters are preserved as-is
            
            return normalized
        elif isinstance(obj, dict):
            return {k: normalize_unicode(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [normalize_unicode(item) for item in obj]
        else:
            return obj
    
    normalized_restaurants = normalize_unicode(restaurants)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(normalized_restaurants, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Successfully converted {len(restaurants)} restaurants to {output_path}")
    print(f"   File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")

if __name__ == "__main__":
    main()

