"""Enhanced Gradio interface with structured output - uses LLM context for rich responses."""
import gradio as gr
import httpx
import asyncio
from typing import Dict, Optional, Tuple
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
else:
    load_dotenv()

# Import formatter and query parser for structured output
import sys
_current_dir = Path(__file__).parent
_parent_dir = _current_dir.parent
if str(_parent_dir) not in sys.path:
    sys.path.insert(0, str(_parent_dir))

try:
    from part1.structured_output_responses.formatter import format_search_results
    from part1.src.query_parser import QueryParser
except ImportError:
    try:
        from structured_output_responses.formatter import format_search_results
        from src.query_parser import QueryParser
    except ImportError:
        from .structured_output_responses.formatter import format_search_results
        from .src.query_parser import QueryParser

API_URL = "http://localhost:8000"

# Global query parser for restaurant name detection
_query_parser: Optional[QueryParser] = None

def get_query_parser() -> QueryParser:
    """Get or create query parser instance."""
    global _query_parser
    if _query_parser is None:
        _query_parser = QueryParser()
    return _query_parser

# Example queries for quick access
EXAMPLE_QUERIES = [
    "What desserts are famous in Maui?",
    "Best vegan restaurants with ocean view",
    "Romantic Italian restaurant in Lahaina",
    "What is MAMA'S FISH HOUSE known for?",
    "Best Hawaiian spots in Wailea",
    "Casual Thai food with outdoor seating",
    "Where can I find authentic poke?",
    "Upscale dining with live music",
]


def perform_search(query: str) -> tuple[Dict, Optional[str]]:
    """Perform restaurant search via API using structured endpoint with LLM capabilities.
    
    Returns:
        Tuple of (search_results_dict, error_message)
    """
    try:
        # Use structured endpoint which:
        # 1. Has LLM capabilities (uses ResponseGenerator.generate())
        # 2. Returns enriched restaurant data (EnrichedRestaurantMatch)
        # 3. Includes all fields needed for structured formatting
        timeout = httpx.Timeout(300.0, connect=10.0)
        response = httpx.post(
            f"{API_URL}/api/search/structured",
            json={"query": query},
            timeout=timeout
        )
        response.raise_for_status()
        return response.json(), None
    except httpx.ConnectError:
        return {
            "success": False,
            "explanation": "Cannot connect to API. Make sure the FastAPI server is running on port 8000."
        }, "Connection error"
    except httpx.TimeoutException:
        return {
            "success": False,
            "explanation": "Request timed out. The search is taking longer than expected. Please try again."
        }, "Timeout error"
    except Exception as e:
        return {
            "success": False,
            "explanation": f"Error: {str(e)}"
        }, str(e)


def extract_images_from_results(search_results: Dict) -> list:
    """Extract all restaurant images from search results.
    
    Returns a list of image URLs from top match and alternatives.
    Gradio Gallery accepts a list of URLs or file paths.
    """
    images = []
    
    # Get images from top match
    top_match = search_results.get("top_match", {})
    if top_match:
        top_images = top_match.get("restaurant_photos_urls", []) or top_match.get("photos_urls", [])
        if top_images:
            # Filter out None/empty values and add up to 3 images from top match
            valid_images = [img for img in top_images[:3] if img and isinstance(img, str) and img.strip()]
            images.extend(valid_images)
            print(f"[DEBUG] Top match images found: {len(valid_images)}")
    
    # Get images from alternatives
    alternatives = search_results.get("alternatives", [])
    for alt in alternatives[:5]:  # Limit to first 5 alternatives
        alt_images = alt.get("restaurant_photos_urls", []) or alt.get("photos_urls", [])
        if alt_images:
            # Filter out None/empty values and add up to 2 images per alternative
            valid_images = [img for img in alt_images[:2] if img and isinstance(img, str) and img.strip()]
            images.extend(valid_images)
    
    # Limit total images to prevent UI overload
    final_images = images[:15]  # Max 15 images total
    print(f"[DEBUG] Total images extracted: {len(final_images)}")
    if final_images:
        print(f"[DEBUG] First image URL: {final_images[0]}")
    
    return final_images


def search_restaurant(query: str) -> Tuple[str, list, str]:
    """Search restaurant and return formatted structured response with images.
    
    Uses /api/search/structured endpoint which:
    - Has full LLM capabilities (uses ResponseGenerator.generate())
    - Returns enriched restaurant data (EnrichedRestaurantMatch) with all fields
    - Includes LLM-generated natural language explanations
    - Provides structured format for consistent restaurant display
    
    Returns:
        Tuple of (formatted_text, list_of_image_urls)
    """
    if not query or not query.strip():
        return "**Ana:** Please enter a question about restaurants!", [], ""
    
    query_stripped = query.strip()
    
    # Perform search using structured endpoint
    # This gives us both LLM capabilities AND structured enriched data
    search_results, error = perform_search(query_stripped)
    
    if error or not search_results.get("success", False):
        # Error case - return explanation
        result = f"{search_results.get('explanation', 'An error occurred.')}\n"
        if search_results.get("caveats"):
            for caveat in search_results.get("caveats", []):
                result += f"- {caveat}\n"
        return f"**Ana:** {result}", [], ""
    
    # Check if query is about a specific restaurant for formatting
    # Only set single_restaurant=True if it's actually a specific restaurant query
    # For general queries (food culture, "best X", etc.), we want to show all alternatives
    parser = get_query_parser()
    detected_restaurant = parser._detect_restaurant_name(query_stripped)
    
    # Check if this is a food culture query (e.g., "best dessert", "famous dishes")
    # These should show alternatives even if a restaurant name is mentioned
    query_lower = query_stripped.lower()
    food_culture_indicators = [
        'best', 'famous', 'popular', 'favorite', 'must try', 'what', 'where to find',
        'dessert', 'desserts', 'dish', 'dishes', 'food', 'eat', 'try', 'specialty',
        'specialties', 'known for', 'famous for', 'recommend', 'suggest', 'top'
    ]
    is_food_culture_query = any(indicator in query_lower for indicator in food_culture_indicators)
    
    # Only treat as single restaurant if:
    # 1. A restaurant name is detected AND
    # 2. It's NOT a food culture query (which should show alternatives)
    single_restaurant = detected_restaurant is not None and not is_food_culture_query
    
    # Debug: check what we're getting from API
    alternatives_count = len(search_results.get("alternatives", []))
    print(f"[DEBUG] Gradio: single_restaurant={single_restaurant}, alternatives_from_api={alternatives_count}")
    
    # Format results using structured formatter
    # This will show:
    # 1. LLM-generated explanation (from ResponseGenerator.generate())
    # 2. Structured restaurant details (Good for, Vibe, Features, Videos, etc.)
    result_text = format_search_results(
        search_results, 
        is_refined=False, 
        include_header=True,  # Add header
        single_restaurant=single_restaurant
    )
    
    # Images are now injected inline in the markdown, so we don't need a separate gallery
    # Return empty list for images since they're embedded in the markdown
    images = []
    image_status_msg = ""
    
    return result_text, images, image_status_msg


def clear_results() -> Tuple[str, list, str]:
    """Clear the results area."""
    return "", [], ""


def make_example_handler(example: str):
    """Create a handler function for an example query."""
    def handler():
        return example
    return handler


# Create Gradio interface with enhanced UI
with gr.Blocks(title="Ana AI Restaurant Search") as demo:
    gr.Markdown(
        """
        # 🍽️ Ana AI Restaurant Search
        
        Your intelligent guide to Maui's dining scene. Ask me anything about restaurants, food culture, and dining experiences!
        
        *Powered by AI with deep knowledge of Hawaiian cuisine and dining culture.*
        """
    )
    
    with gr.Row():
        with gr.Column(scale=2):
            query_input = gr.Textbox(
                label="What are you looking for?",
                placeholder="e.g., 'what desserts are famous in Maui', 'best vegan restaurants with ocean view', 'romantic Italian restaurant in Lahaina'",
                lines=3,
                show_label=True
            )
            
            with gr.Row():
                search_btn = gr.Button("🔍 Search", variant="primary", size="lg", scale=3)
                clear_btn = gr.Button("🗑️ Clear", variant="secondary", size="lg", scale=1)
        
        with gr.Column(scale=1):
            gr.Markdown("### 💡 Example Queries")
            gr.Markdown("Click any example to try it:")
            
            # Create clickable example buttons
            example_buttons = []
            for example in EXAMPLE_QUERIES[:6]:  # Show first 6 examples
                btn = gr.Button(
                    example,
                    variant="secondary",
                    size="sm",
                    scale=1
                )
                # Connect button to set query input using factory function to avoid closure issues
                handler = make_example_handler(example)
                btn.click(
                    fn=handler,
                    inputs=None,
                    outputs=query_input,
                    show_progress=False
                )
                example_buttons.append(btn)
            
            gr.Markdown("---")
            
    with gr.Row():
        with gr.Column(scale=2):
            result_output = gr.Markdown(
                label="Search Results", 
                elem_classes=["search-results"],
                line_breaks=True,  # Enable line breaks in markdown
                show_label=False
            )
    
    # Images are now embedded inline in the markdown, so we don't need a separate gallery
    # Keep the components for compatibility but hide them
    with gr.Row(visible=False):  # Hide the separate image gallery
        with gr.Column():
            image_gallery = gr.Gallery(
                label="📸 Restaurant Photos",
                show_label=False,
                elem_id="restaurant-gallery",
                columns=4,
                rows=2,
                height=400,
                allow_preview=True,
                visible=False,
                value=[]  # Initialize with empty list
            )
            image_status = gr.Markdown(
                value="",
                visible=False,
                show_label=False
            )
    
    # Connect the search function
    search_btn.click(
        fn=search_restaurant,
        inputs=query_input,
        outputs=[result_output, image_gallery, image_status],
        show_progress=True
    )
    
    # Allow Enter key to submit
    query_input.submit(
        fn=search_restaurant,
        inputs=query_input,
        outputs=[result_output, image_gallery, image_status],
        show_progress=True
    )
    
    # Clear button functionality
    clear_btn.click(
        fn=clear_results,
        inputs=None,
        outputs=[result_output, image_gallery, image_status],
        show_progress=False
    )


if __name__ == "__main__":
    import socket
    
    def find_free_port(start_port=7861, max_attempts=10):
        """Find a free port starting from start_port."""
        for i in range(max_attempts):
            port = start_port + i
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.bind(('', port))
                    return port
                except OSError:
                    continue
        return start_port  # Fallback
    
    free_port = find_free_port(7861)
    print(f"Starting Gradio app on port {free_port}")
    
    demo.queue()
    demo.launch(server_name="0.0.0.0", server_port=free_port, share=True, theme=gr.themes.Soft())
