"""Gradio interface with streaming support and full ranking strategies."""
import gradio as gr
import httpx
import asyncio
import json
from typing import Dict, Optional, Tuple
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
else:
    load_dotenv()

API_URL = "http://localhost:8000"


def format_restaurant_info(restaurant: dict, is_top: bool = False) -> str:
    """Format complete restaurant information from metadata."""
    if not restaurant:
        return ""
    
    prefix = "🏆 **TOP MATCH**\n\n" if is_top else ""
    
    info = f"{prefix}## {restaurant.get('name', 'Unknown Restaurant')}\n\n"
    
    # Basic Info
    info += f"**Cuisine:** {restaurant.get('cuisine', 'N/A')}\n"
    info += f"**Price Level:** {restaurant.get('price_level_curated') or restaurant.get('price_level', 'N/A')}\n"
    
    # Location
    location_parts = []
    if restaurant.get('region'):
        location_parts.append(restaurant.get('region'))
    if restaurant.get('city'):
        location_parts.append(restaurant.get('city'))
    if restaurant.get('formatted_address'):
        location_parts.append(restaurant.get('formatted_address'))
    if location_parts:
        info += f"**Location:** {', '.join(location_parts)}\n"
    
    # Rating and Status
    info += f"**Rating:** {restaurant.get('rating', 'N/A')}/5.0 ⭐\n"
    if restaurant.get('business_status'):
        info += f"**Status:** {restaurant.get('business_status')}\n"
    if restaurant.get('is_open_now') is not None:
        open_status = "Yes" if restaurant.get('is_open_now') else "No"
        info += f"**Open Now:** {open_status}\n"
    
    # Scores
    if restaurant.get('final_score') is not None:
        info += f"**Match Score:** {restaurant.get('final_score', 0):.3f}\n"
        info += f"  - Vibe Score: {restaurant.get('vibe_score', 0):.3f}\n"
        info += f"  - Cuisine Score: {restaurant.get('cuisine_score', 0):.3f}\n"
        info += f"  - Price Score: {restaurant.get('price_score', 0):.3f}\n"
        info += f"  - Feature Score: {restaurant.get('feature_score', 0):.3f}\n"
    
    # Vibe Summary
    if restaurant.get('vibe_summary'):
        info += f"\n**Vibe:** {restaurant.get('vibe_summary')}\n"
    
    # Features
    features = restaurant.get('features', {})
    if features and isinstance(features, dict):
        active_features = [k.replace('_', ' ').title() for k, v in features.items() if v]
        if active_features:
            info += f"**Features:** {', '.join(active_features)}\n"
    
    # Contact Info
    if restaurant.get('national_phone') or restaurant.get('international_phone'):
        phones = []
        if restaurant.get('national_phone'):
            phones.append(restaurant.get('national_phone'))
        if restaurant.get('international_phone'):
            phones.append(restaurant.get('international_phone'))
        info += f"**Phone:** {', '.join(phones)}\n"
    
    if restaurant.get('website_uri'):
        info += f"**Website:** {restaurant.get('website_uri')}\n"
    
    if restaurant.get('google_maps_uri'):
        info += f"**Google Maps:** {restaurant.get('google_maps_uri')}\n"
    
    # Hours
    if restaurant.get('opening_hours_text'):
        info += f"**Hours:** {restaurant.get('opening_hours_text')}\n"
    
    if restaurant.get('serves_meal_times'):
        info += f"**Meals Served:** {', '.join(restaurant.get('serves_meal_times'))}\n"
    
    # Videos
    if restaurant.get('video_urls'):
        info += f"\n**Videos:**\n"
        for video_url in restaurant.get('video_urls', []):
            info += f"- {video_url}\n"
    
    # Photos
    if restaurant.get('restaurant_photos_urls'):
        info += f"\n**Photos:** {len(restaurant.get('restaurant_photos_urls', []))} photos available\n"
    
    info += "\n---\n\n"
    return info


def format_all_restaurants(metadata: dict) -> str:
    """Format all restaurant information from metadata."""
    if not metadata:
        return ""
    
    output = "\n\n---\n\n## 📋 Complete Restaurant Information\n\n"
    
    # Top match
    top_match = metadata.get("top_match", {})
    if top_match:
        output += format_restaurant_info(top_match, is_top=True)
    
    # Alternatives
    alternatives = metadata.get("alternatives", [])
    if alternatives and len(alternatives) > 0:
        output += f"### 🔄 Alternative Options ({len(alternatives)} found)\n\n"
        for i, alt in enumerate(alternatives, 1):
            output += f"### Option {i}\n\n"
            output += format_restaurant_info(alt, is_top=False)
    else:
        output += "### ℹ️ No Additional Alternatives\n\n"
        output += "Only one restaurant matched your query criteria. Try broadening your search to see more options.\n\n"
    
    confidence = metadata.get("confidence", "medium")
    output += f"\n**Search Confidence:** {confidence.upper()}\n"
    
    return output


def perform_search_streaming(query: str):
    """Perform streaming restaurant search via API with full ranking strategies.
    
    Uses /api/search/stream endpoint which:
    - Uses StreamingResponseGenerator for token-by-token streaming
    - Uses full ranking strategies (rank_with_award_priority, AdvancedScoreFusion)
    - Streams results in real-time for better UX
    - Shows ALL relevant restaurants (top match + up to 9 alternatives)
    """
    try:
        timeout = httpx.Timeout(300.0, connect=10.0)
        with httpx.stream(
            "POST",
            f"{API_URL}/api/search/stream",
            json={"query": query},
            timeout=timeout
        ) as response:
            response.raise_for_status()
            
            accumulated_text = ""
            metadata = None
            
            for line in response.iter_lines():
                if not line.startswith("data: "):
                    continue
                
                try:
                    data = json.loads(line[6:])  # Remove "data: " prefix
                    
                    if data.get("type") == "metadata":
                        # Store metadata for later use - contains all restaurant info
                        metadata = data
                        # Debug: log what we received
                        top_match_name = metadata.get("top_match", {}).get("name", "Unknown") if metadata else "None"
                        num_alternatives = len(metadata.get("alternatives", [])) if metadata else 0
                        print(f"[DEBUG] Metadata received: Top match={top_match_name}, Alternatives={num_alternatives}")
                        # Don't yield yet, wait for first token
                    
                    elif data.get("type") == "token":
                        # Accumulate tokens and yield incrementally
                        content = data.get("content", "")
                        accumulated_text += content
                        yield accumulated_text
                    
                    elif data.get("type") == "done":
                        # Streaming complete
                        break
                    
                    elif data.get("type") == "error":
                        yield f"**Ana:** {data.get('message', 'An error occurred.')}\n"
                        return
                        
                except json.JSONDecodeError:
                    continue
            
            # After streaming, ALWAYS append all restaurant information from metadata
            # This ensures we show ALL restaurants even if LLM only mentions one
            if metadata:
                # Format all restaurants from metadata (top match + all alternatives)
                restaurant_info = format_all_restaurants(metadata)
                if accumulated_text:
                    # Append restaurant info after LLM response
                    # Add a clear separator
                    final_output = accumulated_text + "\n\n---\n\n" + restaurant_info
                else:
                    # If no LLM text, just show restaurant info
                    final_output = restaurant_info
                yield final_output
            elif accumulated_text:
                # If we have text but no metadata, just show the text
                yield accumulated_text
            else:
                yield "**Ana:** No results found. Please try a different query."
                
    except httpx.ConnectError:
        yield "**Ana:** Cannot connect to API. Make sure the FastAPI server is running on port 8000."
    except httpx.TimeoutException:
        yield "**Ana:** Request timed out. The search is taking longer than expected. Please try again."
    except Exception as e:
        yield f"**Ana:** Error: {str(e)}"


def search_restaurant(query: str):
    """Search restaurant with streaming and full ranking strategies.
    
    Uses /api/search/stream endpoint which:
    - Uses StreamingResponseGenerator for real-time token streaming
    - Uses full ranking strategies (rank_with_award_priority, AdvancedScoreFusion)
    - Provides better perceived latency with streaming
    - Uses all scoring signals (vibe, cuisine, price, features) with proper weights
    """
    if not query or not query.strip():
        yield "**Ana:** Please enter a question about restaurants!"
        return
    
    query_stripped = query.strip()
    
    # Use streaming search which includes all ranking strategies
    # This generator yields tokens as they arrive from the API
    for chunk in perform_search_streaming(query_stripped):
        yield chunk


# Create Gradio interface
with gr.Blocks(title="Ana AI Restaurant Search", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🍽️ Ana AI Restaurant Search
        
        Your intelligent guide to Maui's dining scene. Ask me anything about restaurants, food culture, and dining experiences!
        """
    )
    
    with gr.Row():
        with gr.Column(scale=2):
            query_input = gr.Textbox(
                label="What are you looking for?",
                placeholder="e.g., 'what desserts are famous in Maui', 'best vegan restaurants with ocean view', 'romantic Italian restaurant in Lahaina', 'what is ULUPALAKUA RANCH STORE & GRILL famous for?'",
                lines=3
            )
            search_btn = gr.Button("Search", variant="primary", size="lg")
        
        with gr.Column(scale=1):
            gr.Markdown("### 💡 Try asking:")
            gr.Markdown(
                """
                - **Food culture:** "What desserts are famous in Maui?"
                - **Specific requests:** "Best vegan restaurants with ocean view"
                - **Restaurant info:** "What is MAMA'S FISH HOUSE known for?"
                - **Dining experiences:** "Romantic dinner spots in Wailea"
                - **Cuisine searches:** "Best Hawaiian spots in Lahaina"
                """
            )
    
    with gr.Row():
        with gr.Column():
            result_output = gr.Markdown(
                label="Search Results", 
                elem_classes=["search-results"],
                line_breaks=True  # Enable line breaks in markdown
            )
    
    # Connect the search function with streaming
    search_btn.click(
        fn=search_restaurant,
        inputs=query_input,
        outputs=result_output,
        show_progress=True
    )
    
    # Allow Enter key to submit
    query_input.submit(
        fn=search_restaurant,
        inputs=query_input,
        outputs=result_output,
        show_progress=True
    )


if __name__ == "__main__":
    demo.queue()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True, theme=gr.themes.Soft())

