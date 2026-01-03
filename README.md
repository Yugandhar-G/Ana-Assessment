# Ana AI — AI Engineering Take-Home Assignment

> *Candidate: Yugandhar Gopu | Role: AI Engineer (Ana AI)*

Ana AI is a culinary assistant for Hawaii that combines RAG (for hard facts like menus, hours) with Persona/LLM (for "vibes" and cultural context). This repository contains the prototype implementations for the "Brain" of Ana AI.

---

## Assignment Overview

This project addresses two core challenges:

1. **Part 1: The Vibe Check** - Restaurant retrieval by mood and atmosphere
2. **Part 2: System Design** - Production design document (see `docs/system_design.md`)

---

## Quick Start

### Prerequisites

```bash
# 1. Clone the repository
git clone https://github.com/Yugandhar-G/Ana-Assessment.git
cd Ana-Assessment

# 2. Setup Python environment
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 3. Configure API Key
# Create a .env file in the root directory:
echo "GEMINI_API_KEY=your-api-key-here" > .env
echo "GEMINI_CHAT_MODEL=gemini-3-flash-preview" >> .env

# Get your API key from: https://makersuite.google.com/app/apikey
```

### Running the Application

#### Option 1: Conversational Gradio Interface (Recommended)

```bash
cd part1
python conversational_gradio_app.py
```

This launches a web interface at `http://localhost:7861` (or next available port) with:
- Interactive chat interface
- Clickable example queries
- Image display for restaurants
- Clear/reset functionality
- Real-time search results

#### Option 2: FastAPI Backend

```bash
cd part1
python -m uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

API endpoints:
- `POST /api/search/structured` - Structured search with enriched results
- `GET /api/health` - Health check
- API docs: `http://localhost:8000/docs`

#### Option 3: CLI Interface

```bash
# Run a vibe-based restaurant search
python -m part1.src.main --query "romantic spot with ocean views"

# Example queries:
python -m part1.src.main --query "night market feel, not fancy"
python -m part1.src.main --query "casual Thai food with outdoor seating"
```

See [`part1/README.md`](part1/README.md) for detailed documentation.

---

## Project Structure

```
.
├── part1/                              # Part 1: The Vibe Check
│   ├── src/                            # Source code
│   │   ├── pipeline.py                # Main search pipeline
│   │   ├── query_parser.py            # LLM-based query parsing
│   │   ├── query_enhancer.py          # Query enrichment with LLM context
│   │   ├── response_generator.py      # LLM response generation
│   │   ├── scorers/                   # Multi-signal scoring
│   │   └── fusion.py                  # Score fusion and ranking
│   ├── conversational_gradio_app.py   # Conversational web interface
│   ├── api.py                         # FastAPI backend
│   ├── prompts/                       # LLM prompt templates
│   ├── structured_output_responses/   # Structured response formatting
│   └── README.md                      # Part 1 detailed docs
│
├── data/                              # Shared data directory
│   ├── restaurants.json               # Restaurant data (360 restaurants)
│   └── video_metadata.json           # Video metadata
│
├── docs/                              # Project documentation
│   ├── system_design.md              # Part 2: Production system design
│   └── workflow_diagram.md           # Workflow documentation
│
├── scripts/                           # Utility scripts
│   ├── rebuild_vector_db.py          # Rebuild vector database
│   └── update_vibe_summaries.py      # Update vibe summaries
│
├── requirements.txt                   # Python dependencies
└── README.md                          # This file
```

---

## Part 1: The Vibe Check

**Problem:** Users search by mood ("night market feel") not just cuisine.

**Solution:** Hybrid retrieval system combining:
- **Vector search** for semantic vibe matching
- **Metadata filtering** for hard exclusions (e.g., "NOT fancy")
- **Multi-signal scoring** (vibe, cuisine, price, features)
- **LLM-powered** query understanding and response generation
- **Query enhancement** with LLM context for better RAG retrieval
- **LLM reasoning** for enhanced recommendations

### Key Features

- **Semantic Understanding**: Understands natural language queries like "romantic dinner with ocean view"
- **Vibe Matching**: Finds restaurants by atmosphere and mood, not just cuisine
- **Smart Filtering**: Handles negations ("not fancy", "no kids")
- **Context-Aware**: Uses LLM to enrich queries with cultural context before RAG
- **Intelligent Ranking**: Multi-signal score fusion for optimal restaurant ranking
- **Restaurant-Specific Queries**: Shows only the requested restaurant when user asks about a specific place
- **General Queries**: Shows top match + 5 alternatives for cuisine/vibe queries
- **Rich Responses**: LLM-generated explanations with restaurant photos and details

### Architecture

```
User Query 
  → Query Parser (LLM) 
  → Query Enhancer (LLM context) 
  → RAG Retrieval (Vector Search + Metadata)
  → Multi-Signal Scoring 
  → Score Fusion 
  → LLM Reasoning 
  → Response Generation (LLM)
```

### Key Components

1. **Query Parser** (`src/query_parser.py`): Uses Gemini 3 to understand natural language queries
2. **Query Enhancer** (`src/query_enhancer.py`): Adds cultural context and implicit requirements
3. **Hard Filters** (`src/filters.py`): Applies boolean filters based on must_not constraints
4. **Multi-Signal Scorers** (`src/scorers/`):
   - **Vibe Scorer**: Vector similarity search for atmosphere/vibe matching
   - **Cuisine Scorer**: Exact/approximate cuisine matching
   - **Price Scorer**: Price level preference matching
   - **Feature Scorer**: Boolean feature matching (outdoor seating, etc.)
5. **Score Fusion** (`src/fusion.py`): Advanced weighted combination of all signals
6. **Response Generator** (`src/response_generator.py`): LLM-based natural language response generation with reasoning

---

## Configuration

### Environment Variables

Create a `.env` file in the root directory:

```bash
# Required - Gemini API Key
GEMINI_API_KEY=your-api-key-here

# Optional - Model Selection (defaults shown)
GEMINI_CHAT_MODEL=gemini-3-flash-preview  # Latest Gemini 3 model
GEMINI_EMBEDDING_MODEL=models/text-embedding-004  # For vector embeddings

# Optional - Web Search (if enabled)
GOOGLE_SEARCH_API_KEY=your-search-api-key
GOOGLE_SEARCH_ENGINE_ID=your-search-engine-id
```

**Get your Gemini API key:** [Google AI Studio](https://makersuite.google.com/app/apikey)

**Note:** Enable billing in Google Cloud Console to use the $300 free credits and access Gemini 3 models.

---

## Tech Stack

| Component | Choice | Why |
|-----------|--------|-----|
| **LLM** | Google Gemini 3 Flash Preview | Latest model with enhanced reasoning and context understanding |
| **Embeddings** | Google Gemini (text-embedding-004) | High-quality embeddings for semantic search |
| **Vector Store** | ChromaDB | Persistent vector storage with metadata filtering |
| **API Framework** | FastAPI | Fast, async, auto-documented API |
| **Web Interface** | Gradio | Easy-to-use conversational interface |
| **Validation** | Pydantic | Type safety, clean schemas |

---

## Data

- **360 restaurants** from Maui, Hawaii
- Stored in `data/restaurants.json`
- Video metadata available in `data/video_metadata.json`
- Vector embeddings cached in `part1/chroma_db/` for fast retrieval

---

## Features

### Query Understanding
- Natural language parsing with Gemini 3
- Implicit requirement detection (e.g., "dessert" → adds dessert features)
- Cultural context enrichment (e.g., "best dessert in Maui" → adds Hawaiian dessert context)

### Smart Restaurant Selection
- **Restaurant-specific queries**: Shows only the requested restaurant
  - Example: "tell me about Mama's Fish House" → Only Mama's Fish House
- **General queries**: Shows top match + 5 alternatives
  - Example: "best Italian restaurants" → Top match + 5 alternatives
- Intelligent alternative selection based on relevance and ranking

### Response Generation
- LLM-generated explanations using Gemini 3
- Uses restaurant data + LLM general knowledge
- Includes restaurant photos (up to 3 per restaurant)
- Structured format with vibe summaries, features, and menu items

### User Interface
- Conversational Gradio interface
- Clickable example queries
- Image display with uniform sizing
- Clear/reset functionality
- Real-time search results

---

## Performance

- **First run**: ~30-60s to index 360 restaurants (embeddings via Gemini API)
- **Subsequent runs**: Fast (embeddings cached in ChromaDB)
- **Query latency**: <2s for most queries
- **API response**: <3s for structured search endpoint

---

## Example Queries

### General Queries (Shows alternatives)
- `"best Italian restaurants in Maui"`
- `"romantic dinner spot with ocean views"`
- `"best dessert in Maui"`
- `"family-friendly restaurants with kids menu"`
- `"casual Thai food, outdoor seating, not fancy"`

### Restaurant-Specific Queries (Shows only that restaurant)
- `"tell me about Mama's Fish House"`
- `"Mama's Fish House hours"`
- `"what is Merriman's Kapalua"`

---

## Part 2: System Design

See [`docs/system_design.md`](docs/system_design.md) for the production system design covering:
- Stack choices and rationale
- Latency vs. cost optimization strategies
- Data freshness and maintenance
- Scaling considerations

---

## Development

### Running Tests

```bash
# Test API key
python test_api_key.py

# Run CLI search
python -m part1.src.main --query "your query here"
```

### Rebuilding Vector Database

```bash
# Delete existing vector DB
rm -rf part1/chroma_db

# Restart API - it will rebuild automatically
python -m uvicorn part1.api:app --reload
```

---

## Author

**Yugandhar Gopu**  
AI Engineer Candidate - Ana AI

---

## License

This is a take-home assignment submission.
