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
# 1. Get Gemini API Key
# Visit https://makersuite.google.com/app/apikey and create an API key

# 2. Setup Python environment
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 3. Set environment variable
export GEMINI_API_KEY="your-api-key-here"
```

### Part 1: The Vibe Check

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
├── part1/                    # Part 1: The Vibe Check
│   ├── src/                  # Source code
│   ├── prompts/              # LLM prompt templates
│   ├── scripts/              # Data cleaning scripts
│   ├── docs/                 # Part 1 documentation
│   └── README.md             # Part 1 guide
│
├── data/                     # Shared data directory
│   ├── restaurants.json      # Restaurant data
│   └── video_metadata.json   # Video metadata
│
├── docs/                     # Project documentation
│   └── system_design.md      # Part 2: Production system design
│
├── requirements.txt          # Python dependencies
├── OLLAMA_SETUP.md          # Ollama setup guide (shared)
└── README.md                # This file
```

---

## Part 1: The Vibe Check

**Problem:** Users search by mood ("night market feel") not just cuisine.

**Solution:** Hybrid retrieval system combining:
- **Vector search** for semantic vibe matching
- **Metadata filtering** for hard exclusions (e.g., "NOT fancy")
- **Multi-signal scoring** (vibe, cuisine, price, features)
- **LLM-powered** query understanding and response generation

**Key Features:**
- Understands negations: "not fancy" → excludes upscale restaurants
- Semantic vibe matching: "night market feel" → finds similar atmospheres
- Weighted score fusion for optimal ranking

**Tech Stack:**
- Ollama (local LLM) with Metal GPU acceleration
- ChromaDB for vector storage
- Pydantic for type-safe schemas

---

## Part 2: System Design

See [`docs/system_design.md`](docs/system_design.md) for the production system design covering:
- Stack choices and rationale
- Latency vs. cost optimization strategies
- Data freshness and maintenance
- Scaling considerations

---

## Configuration

Set environment variables (required):

```bash
# Required - Gemini API Key
export GEMINI_API_KEY="your-api-key-here"

# Optional - Model Selection
export GEMINI_CHAT_MODEL="gemini-2.0-flash"  # or "gemini-2.5-pro", "gemini-2.5-flash"
export GEMINI_EMBEDDING_MODEL="models/embedding-001"  # Recommended for RAG
# Alternative: export GEMINI_EMBEDDING_MODEL="models/text-embedding-004"

# Optional - Web Search (if enabled)
export GOOGLE_SEARCH_API_KEY="your-search-api-key"
export GOOGLE_SEARCH_ENGINE_ID="your-search-engine-id"
```

Get your Gemini API key from [Google AI Studio](https://makersuite.google.com/app/apikey)

See [`GEMINI_SETUP.md`](GEMINI_SETUP.md) for detailed setup instructions.

---

## Tech Stack

| Component | Choice | Why |
|-----------|--------|-----|
| LLM | Google Gemini (gemini-1.5-flash) | Fast, cost-effective, excellent quality |
| Embeddings | Google Gemini (embedding-001 recommended, text-embedding-004 default) | Top-tier RAG embeddings for complex queries |
| Vector Store | ChromaDB | Simple, local, metadata filtering |
| Validation | Pydantic | Type safety, clean schemas |
| Web Search | Google Custom Search API | Optional web search integration |

---

## Data

### Data

- **360 restaurants** from Maui, Hawaii
- Stored in `data/restaurants.json`
- Video metadata available in `data/video_metadata.json`

---

## Performance

### Part 1
- **First run**: ~30-60s to index 95 restaurants (with Metal GPU)
- **Subsequent runs**: Fast (embeddings cached)
- **Query latency**: <2s for most queries
- **GPU**: Optimized for Metal GPU on macOS (6 concurrent requests)

---

## Development Notes

### Assumptions

1. **Menu Data**: Restaurant menus are available in structured format
2. **API Access**: System uses Gemini API for LLM and embeddings

### Code Organization

- Each part is self-contained in its own directory
- Shared utilities (like `ollama_client.py`) are in Part 1 (can be shared if needed)
- Data directory is at root level for shared access
- Documentation is organized per-part and at project level

---

## Author

**Yugandhar Gopu**  
AI Engineer Candidate - Ana AI

---

## License

This is a take-home assignment submission.
