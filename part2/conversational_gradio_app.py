import gradio as gr
import httpx
import sys
from pathlib import Path
from dotenv import load_dotenv
from typing import Dict, Optional, Tuple

# Load environment variables
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
else:
    load_dotenv()

# Add current directory to path
_current_dir = Path(__file__).parent
if str(_current_dir) not in sys.path:
    sys.path.insert(0, str(_current_dir))

try:
    from structured_output_responses.formatter import format_search_results
except ImportError:
    sys.path.append(str(_current_dir.parent))
    from part2.structured_output_responses.formatter import format_search_results

API_URL = "http://localhost:8001" # Part 2 API port

EXAMPLE_QUERIES = [
    "What desserts are famous in Maui?",
    "Best vegan restaurants with ocean view",
    "Romantic Italian restaurant in Lahaina",
    "What is MAMA'S FISH HOUSE known for?",
    "Best Hawaiian spots in Wailea",
    "Casual Thai food with outdoor seating",
]

def perform_search(query: str) -> tuple[Dict, Optional[str]]:
    try:
        timeout = httpx.Timeout(300.0, connect=10.0)
        response = httpx.post(
            f"{API_URL}/api/search/structured",
            json={"query": query},
            timeout=timeout
        )
        response.raise_for_status()
        return response.json(), None
    except Exception as e:
        return {"success": False, "explanation": f"Error: {str(e)}"}, str(e)

def search_restaurant(query: str) -> str:
    if not query or not query.strip():
        return "**Ana:** Please enter a question about restaurants!"
    
    search_results, error = perform_search(query.strip())
    
    if error or not search_results.get("success", False):
        return f"**Ana:** {search_results.get('explanation', 'An error occurred.')}"
    
    # Format results using Part 2 structured formatter
    result_text = format_search_results(
        search_results, 
        is_refined=False, 
        include_header=True,
        single_restaurant=False # In agentic mode, Gemini decides if it's single
    )
    
    return result_text

with gr.Blocks(title="Ana AI Agentic Search (Part 2)", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🍽️ Ana AI Agentic Search (Part 2)")
    gr.Markdown("This version uses a multi-stage agentic workflow: Brainstorm -> Ground -> Synthesize.")
    
    with gr.Row():
        with gr.Column(scale=2):
            query_input = gr.Textbox(label="What are you looking for?", placeholder="e.g., 'best dessert in Maui'", lines=3)
            search_btn = gr.Button("🔍 Search", variant="primary")
        
        with gr.Column(scale=1):
            gr.Markdown("### 💡 Examples")
            for example in EXAMPLE_QUERIES:
                gr.Button(example).click(lambda x=example: x, None, query_input)

    result_output = gr.Markdown(label="Search Results")

    search_btn.click(fn=search_restaurant, inputs=query_input, outputs=[result_output])
    query_input.submit(fn=search_restaurant, inputs=query_input, outputs=[result_output])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7862, share=True)
