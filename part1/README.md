# Part 1: The Vibe Check

> *"I want a place that feels like a night market in Bangkok—noisy, plastic chairs, super casual—not a fancy sit-down spot."*

A hybrid retrieval system that understands **how people actually search for food**: by feeling, not filters.

## Overview

This system solves the "vibe check" challenge by building a retrieval system that successfully identifies the best restaurant based on mood, atmosphere, and cultural context rather than just cuisine or price filters.

## Architecture

```
User Query → Query Understanding (LLM) → Hard Exclusions → Multi-Signal Scoring → Score Fusion → Response Generation
```

### Key Components

1. **Query Parser** (`src/query_parser.py`): Uses LLM to understand natural language queries and extract:
   - Semantic query for vector search
   - Hard exclusions (must_not filters)
   - Preferences (cuisine, price, features)
   - Dynamic signal weights

2. **Hard Filters** (`src/filters.py`): Applies boolean filters based on must_not constraints

3. **Multi-Signal Scorers** (`src/scorers/`):
   - **Vibe Scorer**: Vector similarity search for atmosphere/vibe matching
   - **Cuisine Scorer**: Exact/approximate cuisine matching
   - **Price Scorer**: Price level preference matching
   - **Feature Scorer**: Boolean feature matching (outdoor seating, etc.)

4. **Score Fusion** (`src/fusion.py`): Weighted combination of all signals

5. **Response Generator** (`src/response_generator.py`): LLM-based natural language response generation

## Data Schema

The `Restaurant` object includes:
- Basic info: `name`, `cuisine`, `price_level`, `region`, `rating`
- **Vibe attributes**: `formality`, `noise_level`, `vibe_summary` (text description)
- Features: `highlights` (list of features like "outdoor_seating", "live_music", etc.)

## Quick Start

```bash
# Install dependencies (from project root)
pip install -r requirements.txt

# Set Gemini API Key (required)
export GEMINI_API_KEY="your-api-key-here"
# Get your API key from https://makersuite.google.com/app/apikey

# Run a query
python -m part1.src.main --query "romantic spot with ocean views"

# Output JSON format
python -m part1.src.main --query "night market feel, not fancy" --json
```

## Example Queries

- `"I want a place that feels like a night market in Bangkok—noisy, plastic chairs, super casual"`
- `"romantic dinner spot with ocean views"`
- `"casual Thai food, outdoor seating, not fancy"`
- `"family-friendly restaurant with kids menu"`

## Configuration

Set environment variables:

```bash
# Required
export GEMINI_API_KEY="your-api-key-here"

# Optional - Model Selection
export GEMINI_CHAT_MODEL="gemini-2.0-flash"  # or "gemini-2.5-pro", "gemini-2.5-flash"
export GEMINI_EMBEDDING_MODEL="models/text-embedding-004"  # Default embedding model
```

## Data

Restaurant data is stored in `../data/restaurants.json` (shared at project root). The system automatically:
- Indexes restaurants into a vector store on first run
- Uses ChromaDB for persistent vector storage (in `chroma_db/` directory)
- Embeds vibe summaries using Gemini embedding models

## Files

- `src/`: Main source code
- `prompts/`: LLM prompt templates
- `docs/`: Additional documentation (scoring explanation, vibe calculation)

## Performance

- **First run**: ~30-60s to index restaurants (embeddings via Gemini API)
- **Subsequent runs**: Fast (embeddings cached in ChromaDB)
- **Query latency**: <2s for most queries

## See Also

- Main project README: `../README.md`
- System Design: `../docs/system_design.md`
- Gemini Setup: `../GEMINI_SETUP.md`

