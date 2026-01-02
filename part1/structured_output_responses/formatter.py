"""Formatter for structured restaurant output."""
from typing import Dict, Optional
import os
import json
from pathlib import Path


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


def inject_images_after_vibe(text: str, restaurants: list[Dict]) -> str:
    """Inject restaurant images after 'Vibe at this restaurant:' sections in the LLM response.
    
    Args:
        text: The LLM-generated markdown text
        restaurants: List of restaurant dicts (top match first, then alternatives)
    
    Returns:
        Text with images injected after vibe sections
    """
    import re
    
    if not restaurants:
        return text
    
    # Find all restaurant sections by header pattern
    # Pattern matches: ## Restaurant Name or ## 🏆 Restaurant Name or ## 1. Restaurant Name
    restaurant_pattern = r'(##\s*(?:🏆\s*)?(?:\d+\.\s*)?[^\n]+)'
    sections = re.split(restaurant_pattern, text)
    
    if len(sections) < 3:
        # No restaurant sections found, return as-is
        return text
    
    result_parts = [sections[0]]  # Text before first restaurant
    restaurant_idx = 0
    
    # Process each restaurant section
    for i in range(1, len(sections), 2):
        if i + 1 < len(sections):
            header = sections[i]
            content = sections[i + 1]
            
            result_parts.append(header)
            
            # Find "Vibe at this restaurant:" section and inject images after it
            # Pattern: **Vibe at this restaurant:** followed by text until next ** section or end
            vibe_pattern = r'(\*\*Vibe at this restaurant:\*\*\s*\n[^\*]+?)(?=\n\*\*|\n##|\Z)'
            
            def replace_vibe(match):
                vibe_text = match.group(1)
                
                # Get images for current restaurant
                if restaurant_idx < len(restaurants):
                    restaurant = restaurants[restaurant_idx]
                    # Get images from restaurant data - prioritize restaurant_photos_urls
                    photos_urls = restaurant.get('restaurant_photos_urls', []) or restaurant.get('photos_urls', [])
                    
                    # Debug logging
                    print(f"[DEBUG] Injecting images for restaurant {restaurant_idx}: {restaurant.get('name', 'Unknown')}")
                    print(f"[DEBUG] Found {len(photos_urls)} photo URLs")
                    
                    if photos_urls:
                        # Filter valid URLs
                        valid_photos = [url for url in photos_urls[:3] if url and isinstance(url, str) and url.strip()]
                        
                        if valid_photos:
                            print(f"[DEBUG] Injecting {len(valid_photos)} images after vibe section")
                            # Add images after vibe text
                            images_markdown = "\n\n"
                            for photo_url in valid_photos:
                                images_markdown += f"![Restaurant photo]({photo_url})\n"
                            return vibe_text + images_markdown
                
                return vibe_text
            
            # Replace vibe sections with images injected
            content_with_images = re.sub(vibe_pattern, replace_vibe, content, flags=re.DOTALL)
            result_parts.append(content_with_images)
            
            restaurant_idx += 1
        else:
            result_parts.append(sections[i])
    
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
    # Get all restaurants (top match + alternatives) for up to 10 total
    # Only filter alternatives if it's a single restaurant query
    alternatives_raw = data.get("alternatives", [])
    alternatives = [] if single_restaurant else alternatives_raw[:9]  # Up to 9 alternatives (10 total)
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
        llm_with_images = inject_images_after_vibe(llm_explanation, all_restaurants)
        
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

