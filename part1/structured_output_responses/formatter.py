"""Formatter for structured restaurant output."""
from typing import Dict, Optional
import os
import json
from pathlib import Path

# Cache for restaurant data loaded from JSON
_restaurant_data_cache = None

def _load_restaurant_data() -> list[Dict]:
    """Load restaurant data from JSON file (cached)."""
    global _restaurant_data_cache
    if _restaurant_data_cache is None:
        data_path = Path(__file__).parent.parent.parent / "data" / "restaurants.json"
        if data_path.exists():
            with open(data_path, 'r') as f:
                _restaurant_data_cache = json.load(f)
        else:
            _restaurant_data_cache = []
    return _restaurant_data_cache

def _find_restaurant_in_json(name: str) -> Dict | None:
    """Find restaurant by name in JSON data."""
    restaurants = _load_restaurant_data()
    name_lower = name.lower().strip()
    
    # Try exact match
    for r in restaurants:
        r_name = r.get('name', '').lower().strip()
        if r_name == name_lower:
            return r
    
    # Try substring match (either direction)
    for r in restaurants:
        r_name = r.get('name', '').lower().strip()
        if r_name and (r_name in name_lower or name_lower in r_name):
            return r
    
    # Try word-based matching (more lenient)
    # Include single-letter words but filter out common single letters
    name_words = set(word for word in name_lower.split() if len(word) >= 1)
    # Remove common words (but keep meaningful single letters like 'n' if they're part of the name)
    common_words = {'the', 'restaurant', 'bar', 'grill', 'cafe', 'café', 'and', '&', 'name', 'restaurant', 'of', 'on', 'at', 'in', 'for', 'to'}
    name_words_clean = name_words - common_words
    
    if name_words_clean:
        best_match = None
        best_score = 0
        for r in restaurants:
            r_name = r.get('name', '').lower().strip()
            r_words = set(word for word in r_name.split() if len(word) >= 1)
            r_words_clean = r_words - common_words
            
            if r_words_clean:
                overlap = name_words_clean & r_words_clean
                if overlap:
                    # Score based on overlap ratio - require at least 2 matching words OR high ratio
                    score = len(overlap) / max(len(name_words_clean), len(r_words_clean))
                    min_words = min(len(name_words_clean), len(r_words_clean))
                    # Require at least 50% match OR at least 2 words match (for short names)
                    if score > best_score and (score >= 0.5 or (len(overlap) >= 2 and min_words <= 4)):
                        best_score = score
                        best_match = r
        
        if best_match:
            return best_match
    
    return None


def _get_price_description(price_level: str, price_level_curated: Optional[str] = None) -> str:
    """Convert price level to natural language description."""
    price = price_level_curated or price_level
    price_lower = price.lower().strip()
    
    if price_lower == '$' or 'inexpensive' in price_lower or 'budget' in price_lower:
        return "an inexpensive"
    elif price_lower == '$$' or 'moderate' in price_lower:
        return "a moderate"
    elif price_lower == '$$$' or 'upscale' in price_lower or 'expensive' in price_lower:
        return "an upscale"
    elif price_lower == '$$$$' or 'fine dining' in price_lower or 'very expensive' in price_lower:
        return "a fine dining"
    else:
        return "a"


def _build_relevance_explanation(restaurant: Dict, query: str, match_reasons: list) -> str:
    """Build a natural, conversational explanation of why this restaurant matches the query."""
    name = restaurant.get('name', 'N/A')
    cuisine = restaurant.get('cuisine', '')
    city = restaurant.get('city', '')
    region = restaurant.get('region', '')
    price_level = restaurant.get('price_level', '$$')
    price_level_curated = restaurant.get('price_level_curated')
    rating = restaurant.get('rating', 0)
    top_menu_items = restaurant.get('top_menu_items', [])
    vibe_summary = restaurant.get('vibe_summary', '')
    
    # Start building the explanation
    sentences = []
    
    # First sentence: Name + location + "seems like a great fit"
    location = city or region
    if location:
        first_sentence = f"{name} in {location} seems like a great fit."
    else:
        first_sentence = f"{name} seems like a great fit."
    sentences.append(first_sentence)
    
    # Second sentence: Price + cuisine + menu items + vibe
    second_parts = []
    
    # Add price description
    price_desc = _get_price_description(price_level, price_level_curated)
    second_parts.append(f"It's {price_desc} spot")
    
    # Add cuisine specialization
    if cuisine and cuisine.lower() not in ['restaurant', 'food', 'dining']:
        cuisine_lower = cuisine.lower()
        second_parts.append(f"that specializes in {cuisine_lower}")
        
        # Add additional menu items if available (beyond the main cuisine)
        if top_menu_items and len(top_menu_items) > 0:
            # Filter out items that are clearly part of the cuisine name
            additional_items = []
            for item in top_menu_items[:3]:  # Take up to 3 items
                item_lower = item.lower()
                # Skip if the item is too generic or matches the cuisine
                if item_lower not in cuisine_lower and item_lower not in ['food', 'cuisine', 'dishes']:
                    additional_items.append(item)
            
            if additional_items:
                if len(additional_items) == 1:
                    second_parts.append(f"and also offers {additional_items[0].lower()}")
                elif len(additional_items) == 2:
                    second_parts.append(f"and also offers {additional_items[0].lower()} and {additional_items[1].lower()}")
                else:
                    items_str = ", ".join([item.lower() for item in additional_items[:-1]])
                    second_parts.append(f"and also offers {items_str}, and {additional_items[-1].lower()}")
    
    # Add vibe description (skip if vibe_summary looks like highlights/awards)
    if vibe_summary and len(vibe_summary) > 20:
        vibe_clean = vibe_summary.strip().lower()
        # Skip if it looks like highlights/awards text (contains "award", "serving", "offers")
        if not any(word in vibe_clean for word in ['award-winning', 'serving fresh', 'offers', 'serves']):
            # Extract key vibe phrases - look for descriptive words
            vibe_keywords = []
            for word in ['casual', 'upscale', 'relaxed', 'elegant', 'intimate', 'lively', 'romantic', 'trendy', 'cozy', 'polished', 'refined', 'unique']:
                if word in vibe_clean:
                    vibe_keywords.append(word)
                    break  # Just take one main vibe word
            
            if vibe_keywords:
                vibe_desc = vibe_keywords[0]
                second_parts.append(f", making it a {vibe_desc} dining experience")
            else:
                # Use a generic but nice description
                second_parts.append(", making it a great dining experience")
        else:
            # vibe_summary looks like highlights, skip it
            second_parts.append(", making it a great dining experience")
    
    # Combine second sentence parts into one sentence
    # Join with space, but handle commas properly (they're already in the strings)
    second_sentence = " ".join(second_parts).replace(" ,", ",") + "."
    sentences.append(second_sentence.capitalize())
    
    # Third sentence: Rating with context
    if rating >= 4.5:
        sentences.append(f"With a stellar rating of {rating:.1f} out of 5 stars, it's clear that customers love what they're doing.")
    elif rating >= 4.0:
        sentences.append(f"With a strong rating of {rating:.1f} out of 5 stars, it's a popular choice among diners.")
    elif rating >= 3.5:
        sentences.append(f"It has a solid rating of {rating:.1f} out of 5 stars.")
    
    # Combine all sentences
    return " ".join(sentences)


async def format_restaurant_details_async(restaurant: Dict, query: str = "", match_reasons: list = None) -> str:
    """Async version that extracts vibe information."""
    name = restaurant.get('name', 'N/A')
    cuisine = restaurant.get('cuisine', '')
    
    # 1. Relevance explanation (2-3 lines)
    relevance_text = _build_relevance_explanation(restaurant, query, match_reasons)
    formatted = f"{relevance_text}\n\n"
    
    # 2. "Good for" section with top menu items - make it natural like the reference
    top_menu_items = restaurant.get('top_menu_items', [])
    formatted += "**Good for:**\n"
    
    if top_menu_items:
        # Create natural description similar to reference
        if len(top_menu_items) == 1:
            formatted += f"{top_menu_items[0]}.\n\n"
        elif len(top_menu_items) == 2:
            formatted += f"{top_menu_items[0]} and {top_menu_items[1]}.\n\n"
        elif len(top_menu_items) == 3:
            formatted += f"{top_menu_items[0]}, {top_menu_items[1]}, and {top_menu_items[2]}.\n\n"
        else:
            # For 4+ items, list first few
            items_str = ", ".join(top_menu_items[:3])
            formatted += f"{items_str}, and more.\n\n"
    else:
        # Fallback with natural language
        if cuisine:
            formatted += f"Authentic {cuisine} cuisine and local favorites.\n\n"
        else:
            formatted += f"Quality dining and local favorites.\n\n"
    
    # 3. "Vibe at this restaurant" section - extract just vibe from vibe_summary
    vibe_summary = restaurant.get('vibe_summary', '')
    photos_urls = restaurant.get('restaurant_photos_urls', []) or restaurant.get('photos_urls', [])
    
    formatted += "**Vibe at this restaurant:**\n"
    
    # Extract just the vibe information
    if vibe_summary and vibe_summary.strip():
        extracted_vibe = await extract_vibe_only(vibe_summary)
        formatted += f"{extracted_vibe}\n"
    else:
        # Fallback if vibe_summary is missing
        formatted += "A welcoming dining experience.\n"
    
    # Add images
    if photos_urls:
        formatted += "\n"
        # Include first 3 images
        for i, photo_url in enumerate(photos_urls[:3], 1):
            formatted += f"![Image {i}]({photo_url})\n"
    formatted += "\n"
    
    # 4. Features section
    features = restaurant.get('features', {})
    if features:
        # Get all True features and format them nicely
        true_features = [key.replace('_', ' ').title() for key, value in features.items() if value]
        if true_features:
            formatted += "**Features:**\n"
            # Format as a friendly list
            features_str = ", ".join(true_features)
            formatted += f"{features_str}\n\n"
    
    # 5. Video links
    video_urls = restaurant.get('video_urls', [])
    if video_urls:
        formatted += "**Videos:**\n"
        for video_url in video_urls:
            formatted += f"- [{video_url}]({video_url})\n"
        formatted += "\n"
    
    return formatted


def format_restaurant_details(restaurant: Dict, query: str = "", match_reasons: list = None) -> str:
    """Format a single restaurant according to the specified structure.
    
    Structure:
    1. 2-3 lines explaining why this restaurant is relevant to the query
    2. "Good for" section with top menu items
    3. "Vibe at this restaurant" section with vibe description and images
    4. "Features" section listing features
    5. Video links if available
    
    Args:
        restaurant: Restaurant dictionary
        query: Original user query (for context)
        match_reasons: List of match reason dicts explaining why this restaurant matches
    
    Note: This is a synchronous wrapper. For async vibe extraction, use format_restaurant_details_async.
    """
    name = restaurant.get('name', 'N/A')
    cuisine = restaurant.get('cuisine', '')
    
    # 1. Relevance explanation (2-3 lines)
    relevance_text = _build_relevance_explanation(restaurant, query, match_reasons)
    formatted = f"{relevance_text}\n\n"
    
    # 2. "Good for" section with top menu items - make it natural like the reference
    top_menu_items = restaurant.get('top_menu_items', [])
    formatted += "**Good for:**\n"
    
    if top_menu_items:
        # Create natural description similar to reference
        if len(top_menu_items) == 1:
            formatted += f"{top_menu_items[0]}.\n\n"
        elif len(top_menu_items) == 2:
            formatted += f"{top_menu_items[0]} and {top_menu_items[1]}.\n\n"
        elif len(top_menu_items) == 3:
            formatted += f"{top_menu_items[0]}, {top_menu_items[1]}, and {top_menu_items[2]}.\n\n"
        else:
            # For 4+ items, list first few
            items_str = ", ".join(top_menu_items[:3])
            formatted += f"{items_str}, and more.\n\n"
    else:
        # Fallback with natural language
        if cuisine:
            formatted += f"Authentic {cuisine} cuisine and local favorites.\n\n"
        else:
            formatted += f"Quality dining and local favorites.\n\n"
    
    # 3. "Vibe at this restaurant" section - extract just vibe from vibe_summary
    # For sync version, we'll do a simple extraction (first 200 chars or until sentence end)
    vibe_summary = restaurant.get('vibe_summary', '')
    photos_urls = restaurant.get('restaurant_photos_urls', []) or restaurant.get('photos_urls', [])
    
    formatted += "**Vibe at this restaurant:**\n"
    
    # Simple vibe extraction for sync version (fallback)
    if vibe_summary and vibe_summary.strip():
        # Try to extract just vibe by looking for atmosphere keywords
        vibe_clean = vibe_summary.strip().lower()
        # If it contains vibe keywords, try to extract that part
        vibe_keywords = ['atmosphere', 'ambiance', 'vibe', 'feels', 'experience', 'setting', 'decor', 'mood', 'tone']
        if any(keyword in vibe_clean for keyword in vibe_keywords):
            # Try to find sentences with vibe keywords
            sentences = vibe_summary.split('.')
            vibe_sentences = []
            for sentence in sentences:
                sentence_lower = sentence.lower()
                if any(keyword in sentence_lower for keyword in vibe_keywords):
                    vibe_sentences.append(sentence.strip())
            if vibe_sentences:
                formatted += ". ".join(vibe_sentences[:3]) + ".\n"
            else:
                formatted += f"{vibe_summary.strip()[:200]}\n"
        else:
            # No clear vibe keywords, just use first 200 chars
            formatted += f"{vibe_summary.strip()[:200]}\n"
    else:
        # Fallback if vibe_summary is missing
        formatted += "A welcoming dining experience.\n"
    
    # Add images
    if photos_urls:
        formatted += "\n"
        # Include first 3 images
        for i, photo_url in enumerate(photos_urls[:3], 1):
            formatted += f"![Image {i}]({photo_url})\n"
    formatted += "\n"
    
    # 4. Features section
    features = restaurant.get('features', {})
    if features:
        # Get all True features and format them nicely
        true_features = [key.replace('_', ' ').title() for key, value in features.items() if value]
        if true_features:
            formatted += "**Features:**\n"
            # Format as a friendly list
            features_str = ", ".join(true_features)
            formatted += f"{features_str}\n\n"
    
    # 5. Video links
    video_urls = restaurant.get('video_urls', [])
    if video_urls:
        formatted += "**Videos:**\n"
        for video_url in video_urls:
            formatted += f"- [{video_url}]({video_url})\n"
        formatted += "\n"
    
    return formatted


async def extract_vibe_only(vibe_summary: str) -> str:
    """Extract just the vibe/atmosphere information from vibe_summary using LLM.
    
    The vibe_summary may contain awards, menu items, and other details.
    This function extracts only the atmosphere/vibe description.
    """
    if not vibe_summary or not vibe_summary.strip():
        return "A welcoming dining experience."
    
    # If vibe_summary is short, assume it's already just vibe
    if len(vibe_summary.strip()) < 100:
        return vibe_summary.strip()
    
    try:
        # Import here to avoid circular dependencies
        import sys
        from pathlib import Path
        _current_dir = Path(__file__).parent
        _parent_dir = _current_dir.parent
        if str(_parent_dir) not in sys.path:
            sys.path.insert(0, str(_parent_dir))
        
        try:
            from part1.src.gemini_client import AsyncGeminiClient
        except ImportError:
            from src.gemini_client import AsyncGeminiClient
        
        client = AsyncGeminiClient(
            api_key=os.getenv("GEMINI_API_KEY"),
            default_chat_model=os.getenv("GEMINI_CHAT_MODEL", "gemini-2.0-flash"),
        )
        
        system_prompt = """You are extracting only the atmosphere/vibe description from restaurant information.

The input may contain:
- Awards and accolades
- Menu items and cuisine details
- Business information
- Atmosphere/vibe descriptions

Your task: Extract ONLY the atmosphere, vibe, ambiance, and dining experience description. Ignore awards, menu items, and other details.

Respond with ONLY the vibe description, nothing else. Keep it concise (2-3 sentences max)."""
        
        user_prompt = f"""Extract only the vibe/atmosphere description from this restaurant information:

{vibe_summary}

Extract only the vibe/atmosphere description:"""
        
        response = await client.chat.completions.create(
            model=client.default_chat_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=200,
        )
        
        extracted_vibe = response.choices[0].message["content"].strip()
        return extracted_vibe if extracted_vibe else vibe_summary.strip()[:200]
    except Exception as e:
        print(f"Error extracting vibe: {e}")
        # Fallback: return first 200 chars or full text if shorter
        return vibe_summary.strip()[:200]


def _find_restaurant_by_name(name: str, restaurants: list[Dict], search_results: Dict = None) -> Dict | None:
    """Find a restaurant by name from the restaurants list or search results.
    
    Args:
        name: Restaurant name to find
        restaurants: List of restaurants to search
        search_results: Optional search results dict that might contain more restaurants
    
    Returns:
        Restaurant dict if found, None otherwise
    """
    name_lower = name.lower().strip()
    
    # First, try exact match in restaurants list
    for r in restaurants:
        r_name = r.get('name', '').lower().strip()
        if r_name == name_lower:
            return r
    
    # Try substring matching in restaurants list
    for r in restaurants:
        r_name = r.get('name', '').lower().strip()
        if r_name in name_lower or name_lower in r_name:
            return r
    
    # If search_results provided, try to find in alternatives or other fields
    if search_results:
        # Check alternatives
        for alt in search_results.get('alternatives', []):
            alt_name = alt.get('name', '').lower().strip()
            if alt_name == name_lower or (alt_name in name_lower or name_lower in alt_name):
                return alt
        
        # Check top_match
        top_match = search_results.get('top_match', {})
        top_name = top_match.get('name', '').lower().strip()
        if top_name == name_lower or (top_name in name_lower or name_lower in top_name):
            return top_match
    
    return None


def inject_images_after_vibe(text: str, restaurants: list[Dict], search_results: Dict = None) -> str:
    """Inject restaurant images after 'Vibe at this restaurant:' sections in the LLM response.
    
    Args:
        text: The LLM-generated markdown text
        restaurants: List of restaurant dicts (top match first, then alternatives)
        search_results: Optional search results dict for looking up restaurants by name
    
    Returns:
        Text with images injected after vibe sections
    """
    import re
    
    if not restaurants:
        return text
    
    # Find all restaurant sections by header pattern
    # Pattern matches: ## Restaurant Name, ## 🏆 Restaurant Name, ## 1. Restaurant Name, ## Alternative Restaurant Name
    # Also matches: ## Restaurant Name: X format (LLM sometimes uses this)
    # More flexible pattern to catch all variations
    restaurant_pattern = r'(##\s*(?:🏆\s*)?(?:\d+\.\s*)?(?:Alternative\s+)?(?:[Rr]estaurant\s+[Nn]ame:\s*)?[^\n]+)'
    sections = re.split(restaurant_pattern, text)
    
    print(f"[DEBUG] Image injection: Found {len(sections)} sections, {len(restaurants)} restaurants")
    
    if len(sections) < 3:
        # No restaurant sections found, return as-is
        print(f"[DEBUG] Image injection: No restaurant sections found (only {len(sections)} sections)")
        return text
    
    result_parts = [sections[0]]  # Text before first restaurant
    restaurant_idx = 0
    
    # Build a lookup dict of all available restaurants by name (for faster matching)
    restaurant_lookup = {}
    for r in restaurants:
        r_name = r.get('name', '').lower().strip()
        if r_name:
            restaurant_lookup[r_name] = r
    
    # Also add restaurants from search_results if provided
    if search_results:
        for alt in search_results.get('alternatives', []):
            alt_name = alt.get('name', '').lower().strip()
            if alt_name and alt_name not in restaurant_lookup:
                restaurant_lookup[alt_name] = alt
        top_match = search_results.get('top_match', {})
        top_name = top_match.get('name', '').lower().strip()
        if top_name and top_name not in restaurant_lookup:
            restaurant_lookup[top_name] = top_match
    
    print(f"[DEBUG] Image injection: Built lookup with {len(restaurant_lookup)} restaurants")
    
    # Function to get restaurant data (from lookup or JSON file)
    def get_restaurant_data(name: str) -> Dict | None:
        """Get restaurant data from lookup first, then JSON file if not found."""
        name_lower = name.lower().strip()
        
        # Try lookup first
        restaurant = restaurant_lookup.get(name_lower)
        if restaurant:
            return restaurant
        
        # Try substring in lookup
        for lookup_name, lookup_rest in restaurant_lookup.items():
            if lookup_name in name_lower or name_lower in lookup_name:
                return lookup_rest
        
        # If not found in lookup, try JSON file
        restaurant = _find_restaurant_in_json(name)
        if restaurant:
            print(f"[DEBUG]   ✅ Found restaurant in JSON: {restaurant.get('name', 'Unknown')}")
            return restaurant
        
        return None
    
    # Process each restaurant section
    for i in range(1, len(sections), 2):
        if i + 1 < len(sections):
            header = sections[i]
            content = sections[i + 1]
            
            # Extract restaurant name from header for matching
            header_clean = header.replace('##', '').replace('🏆', '').strip()
            # Remove numbering if present
            header_clean = re.sub(r'^\d+\.\s*', '', header_clean)
            header_clean = re.sub(r'^Alternative\s+', '', header_clean, flags=re.IGNORECASE)
            # Handle "Restaurant Name: X" format
            if 'Restaurant Name:' in header_clean or 'restaurant name:' in header_clean.lower():
                header_clean = re.sub(r'^.*?[Rr]estaurant\s+[Nn]ame:\s*', '', header_clean)
            header_clean = header_clean.strip()
            
            print(f"[DEBUG] Image injection: Processing section {restaurant_idx}, header: '{header_clean[:50]}...'")
            
            result_parts.append(header)
            
            # CRITICAL: Capture restaurant_idx in a closure to ensure each section uses the correct index
            current_idx = restaurant_idx
            
            # Find "Vibe at this restaurant:" section and inject images after it
            # Pattern: **Vibe at this restaurant:** followed by text until next ** section or end
            vibe_pattern = r'(\*\*Vibe at this restaurant:\*\*\s*\n[^\*]+?)(?=\n\*\*|\n##|\Z)'
            
            def replace_vibe(match, idx=current_idx):
                vibe_text = match.group(1)
                
                # Try to match restaurant by index first, then by name if index doesn't work
                restaurant = None
                restaurant_name = "Unknown"
                
                if idx < len(restaurants):
                    # Use index-based matching (most reliable)
                    restaurant = restaurants[idx]
                    restaurant_name = restaurant.get('name', 'Unknown')
                else:
                    # Index out of range - try to match by name from header using lookup
                    print(f"[DEBUG]   ⚠️  Index {idx} out of range (max: {len(restaurants)-1}), trying name matching")
                    header_name_lower = header_clean.lower().strip()
                    print(f"[DEBUG]   Looking for restaurant matching header: '{header_clean[:50]}...'")
                    
                    # Try exact match in lookup first
                    restaurant = restaurant_lookup.get(header_name_lower)
                    if restaurant:
                        restaurant_name = restaurant.get('name', 'Unknown')
                        print(f"[DEBUG]   ✅ Exact lookup match: {restaurant_name}")
                    else:
                        # Try substring matching in lookup
                        for lookup_name, lookup_rest in restaurant_lookup.items():
                            if lookup_name in header_name_lower or header_name_lower in lookup_name:
                                restaurant = lookup_rest
                                restaurant_name = lookup_rest.get('name', 'Unknown')
                                print(f"[DEBUG]   ✅ Substring lookup match: {restaurant_name}")
                                break
                    
                    if not restaurant:
                        # Try to get restaurant data from JSON file if not in lookup
                        print(f"[DEBUG]   Trying get_restaurant_data for: '{header_clean}'")
                        restaurant = get_restaurant_data(header_clean)
                        if restaurant:
                            restaurant_name = restaurant.get('name', 'Unknown')
                            print(f"[DEBUG]   ✅ Found restaurant via get_restaurant_data: {restaurant_name}")
                        else:
                            # Try word-based matching in JSON directly (more lenient)
                            print(f"[DEBUG]   Trying _find_restaurant_in_json for: '{header_clean}'")
                            restaurant = _find_restaurant_in_json(header_clean)
                            if restaurant:
                                restaurant_name = restaurant.get('name', 'Unknown')
                                print(f"[DEBUG]   ✅ Found restaurant in JSON via word matching: {restaurant_name}")
                            else:
                                print(f"[DEBUG]   ⚠️  Could not find '{header_clean}' in JSON file")
                                # Try more aggressive matching in restaurants list
                                for r in restaurants:
                                    r_name = r.get('name', '').lower().strip()
                                    if r_name == header_name_lower:
                                        restaurant = r
                                        restaurant_name = r.get('name', 'Unknown')
                                        print(f"[DEBUG]   ✅ Exact name match: {restaurant_name}")
                                        break
                        
                        # If no exact match, try substring matching
                        if not restaurant:
                            for r in restaurants:
                                r_name = r.get('name', '').lower().strip()
                                # Check if restaurant name appears in header or vice versa
                                if r_name and (r_name in header_name_lower or header_name_lower in r_name):
                                    restaurant = r
                                    restaurant_name = r.get('name', 'Unknown')
                                    print(f"[DEBUG]   ✅ Substring name match: {restaurant_name}")
                                    break
                        
                        # If still no match, try word-based matching (more lenient)
                        if not restaurant:
                            header_words = set(word for word in header_name_lower.split() if len(word) >= 2)  # Lowered to 2 chars
                            for r in restaurants:
                                r_name = r.get('name', '').lower().strip()
                                r_words = set(word for word in r_name.split() if len(word) >= 2)
                                # Remove common words
                                common_words = {'the', 'restaurant', 'bar', 'grill', 'cafe', 'café', 'and', '&', 'name', 'restaurant'}
                                header_words_clean = header_words - common_words
                                r_words_clean = r_words - common_words
                                
                                if header_words_clean and r_words_clean:
                                    overlap = header_words_clean & r_words_clean
                                    # More lenient matching: if most words match or if it's a short name
                                    match_ratio = len(overlap) / len(header_words_clean) if header_words_clean else 0
                                    if len(overlap) >= 1 and (match_ratio >= 0.5 or len(header_words_clean) <= 2):
                                        restaurant = r
                                        restaurant_name = r.get('name', 'Unknown')
                                        print(f"[DEBUG]   ✅ Word-based match: {restaurant_name} (overlap: {overlap}, ratio: {match_ratio:.2f})")
                                        break
                        
                        # Last resort: try fuzzy matching with normalized names (remove punctuation, extra spaces)
                        if not restaurant:
                            header_normalized = re.sub(r'[^\w\s]', ' ', header_name_lower)
                            header_normalized = ' '.join(header_normalized.split())
                            for r in restaurants:
                                r_name = r.get('name', '').lower().strip()
                                r_normalized = re.sub(r'[^\w\s]', ' ', r_name)
                                r_normalized = ' '.join(r_normalized.split())
                                # Check if normalized names are similar
                                if header_normalized and r_normalized:
                                    # Check if one contains the other (after normalization)
                                    if header_normalized in r_normalized or r_normalized in header_normalized:
                                        restaurant = r
                                        restaurant_name = r.get('name', 'Unknown')
                                        print(f"[DEBUG]   ✅ Normalized match: {restaurant_name}")
                                        break
                
                if restaurant:
                    # Get images from restaurant data - prioritize restaurant_photos_urls
                    photos_urls = restaurant.get('restaurant_photos_urls', []) or restaurant.get('photos_urls', [])
                    
                    # Debug logging
                    print(f"[DEBUG] Image injection: Restaurant {idx} ({restaurant_name})")
                    print(f"[DEBUG]   Header matched: '{header_clean[:30]}...'")
                    print(f"[DEBUG]   Found {len(photos_urls)} photo URLs")
                    
                    if photos_urls:
                        # Filter valid URLs and limit to max 3 (or show 1 if that's all there is)
                        valid_photos = [url for url in photos_urls if url and isinstance(url, str) and url.strip()]
                        
                        # Limit to max 3 images, but show all if there's only 1
                        if len(valid_photos) > 3:
                            valid_photos = valid_photos[:3]
                        
                        if valid_photos:
                            print(f"[DEBUG]   ✅ Injecting {len(valid_photos)} images after vibe section for {restaurant_name}")
                            # Add images after vibe text
                            images_markdown = "\n\n"
                            for photo_url in valid_photos:
                                images_markdown += f"![Restaurant photo]({photo_url})\n"
                            return vibe_text + images_markdown
                        else:
                            print(f"[DEBUG]   ⚠️  No valid photos found (all filtered out)")
                    else:
                        print(f"[DEBUG]   ⚠️  No photo URLs in restaurant data for {restaurant_name}")
                else:
                    print(f"[DEBUG]   ⚠️  Could not match restaurant for header: '{header_clean[:30]}...'")
                
                return vibe_text
            
            # Replace vibe sections with images injected
            # Use lambda to properly capture current_idx
            content_with_images = re.sub(
                vibe_pattern, 
                lambda m: replace_vibe(m, current_idx), 
                content, 
                flags=re.DOTALL
            )
            result_parts.append(content_with_images)
            
            restaurant_idx += 1
        else:
            result_parts.append(sections[i])
    
    print(f"[DEBUG] Image injection: Processed {restaurant_idx} restaurant sections")
    return ''.join(result_parts)


def format_search_results(data: Dict, is_refined: bool = False, include_header: bool = True, single_restaurant: bool = False) -> str:
    """Format search results with the new structured format.
    
    Args:
        data: Search results dictionary (should include match_reasons and query)
        is_refined: Whether this is a refined search (affects header message)
        include_header: Whether to include the "Ana:" header message
        single_restaurant: If True, only show the top match (no alternatives)
    """
    if not data.get("success", False):
        result = f"{data.get('explanation', 'An error occurred.')}\n"
        if data.get("caveats"):
            for caveat in data.get("caveats", []):
                result += f"- {caveat}\n"
        return f"**Ana:** {result}" if include_header else result
    
    top_match = data.get("top_match", {})
    # Get all restaurants (top match + alternatives) for up to 6 total
    # Only filter alternatives if it's a single restaurant query
    alternatives_raw = data.get("alternatives", [])
    alternatives = [] if single_restaurant else alternatives_raw[:5]  # Up to 5 alternatives (6 total)
    match_reasons = data.get("match_reasons", [])
    query = data.get("query", "")
    
    # Debug: print what we're getting
    print(f"[DEBUG] Formatter: single_restaurant={single_restaurant}, alternatives_count={len(alternatives_raw)}, filtered={len(alternatives)}")
    
    # CRITICAL: Check if we have an LLM-generated explanation
    # The LLM explanation uses general knowledge and provides context beyond just data
    llm_explanation = data.get("explanation", "").strip()
    
    result = ""
    
    # Add header
    if include_header:
        if single_restaurant:
            result = "**Ana:** Here's what I found about this restaurant:\n\n"
        elif is_refined:
            result = "**Ana:** Perfect! Here's what I found based on your preferences:\n\n"
        else:
            result = "**Ana:** Here's what I found:\n\n"
    
    # PRIORITY: Use LLM-generated explanation if available
    # This explanation uses the LLM's general knowledge and provides context
    if llm_explanation:
        # Prepare restaurant list for image injection (top match + alternatives)
        all_restaurants = [top_match]
        if alternatives and not single_restaurant:
            all_restaurants.extend(alternatives)
        
        # Debug: Check image URLs in restaurant data
        print(f"[DEBUG] Formatting {len(all_restaurants)} restaurants for image injection")
        for idx, rest in enumerate(all_restaurants):
            rest_name = rest.get('name', 'Unknown')
            rest_photos = rest.get('restaurant_photos_urls', [])
            photos = rest.get('photos_urls', [])
            print(f"[DEBUG] Restaurant {idx} ({rest_name}): restaurant_photos_urls={len(rest_photos)}, photos_urls={len(photos)}")
            if rest_photos:
                print(f"[DEBUG]   Sample restaurant_photos_urls: {rest_photos[0]}")
            elif photos:
                print(f"[DEBUG]   Sample photos_urls: {photos[0]}")
        
        # Inject images after vibe sections in the LLM explanation
        # Pass search_results so we can look up restaurants by name if they're not in all_restaurants
        llm_with_images = inject_images_after_vibe(llm_explanation, all_restaurants, search_results=data)
        
        # Use the LLM-generated explanation with images injected
        result += llm_with_images + "\n\n"
    else:
        # Fallback: Use data-only formatting if LLM explanation is not available
        # This should rarely happen since ResponseGenerator.generate() always creates an explanation
        print("[WARNING] No LLM explanation found, falling back to data-only formatting")
        
        # Combine top match and alternatives into a list for consistent formatting
        all_restaurants = [top_match]
        if alternatives and not single_restaurant:
            all_restaurants.extend(alternatives)
        
        print(f"[DEBUG] Formatter: total_restaurants={len(all_restaurants)}")
        
        # Format all restaurants (up to 10) with structured format
        for i, restaurant in enumerate(all_restaurants):
            # Use trophy emoji only for top match, not for single restaurant queries
            if i == 0:
                if single_restaurant:
                    result += f"## {restaurant.get('name', 'N/A')}\n\n"
                else:
                    result += f"## 🏆 {restaurant.get('name', 'N/A')}\n\n"
                # Top match gets match_reasons
                result += format_restaurant_details(restaurant, query=query, match_reasons=match_reasons)
            else:
                # Alternatives get numbered
                result += f"## {i}. {restaurant.get('name', 'N/A')}\n\n"
                result += format_restaurant_details(restaurant, query=query, match_reasons=None)
            
            # Add separator between restaurants (except after the last one)
            if i < len(all_restaurants) - 1:
                result += "\n---\n\n"
    
    return result

