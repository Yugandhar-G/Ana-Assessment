# Ana AI - The Vibe Check

A mood-based restaurant retrieval system for Hawaii that understands how people actually search for food — by feeling, not just filters.

> "I want a place that feels like a night market in Bangkok—noisy, plastic chairs, super casual—not a fancy sit-down spot."

Ana AI understands this query and finds the perfect match.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Data Schema](#data-schema)
- [Technical Decisions](#technical-decisions)
- [Assumptions](#assumptions)
- [Future Improvements](#future-improvements)

---

## Overview

### The Problem

Traditional restaurant search relies on explicit filters (cuisine, price, location). But users often search by **vibe**:
- "Somewhere romantic with ocean views"
- "A loud, fun spot for a birthday dinner"
- "Hole-in-the-wall with authentic food, not touristy"

These queries mix:
- **Semantic concepts** ("night market feel")
- **Explicit attributes** ("noisy", "casual")
- **Negations** ("NOT fancy")
- **Implicit expectations** (cheap, street food style)

No single retrieval method handles all of these well.

### The Solution

A **hybrid retrieval system** that combines:
1. **Hard filters** for explicit constraints and negations
2. **Vector search** for semantic vibe matching
3. **Metadata scoring** for cuisine, price, and feature preferences
4. **Weighted fusion** to combine all signals intelligently

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              USER QUERY                                     │
│         "night market in Bangkok, noisy, plastic chairs, not fancy"         │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         STAGE 1: QUERY UNDERSTANDING                        │
│                              (LLM Call #1)                                  │
│                                                                             │
│  Extracts:                                                                  │
│  • semantic_query: "Bangkok night market street food atmosphere..."         │
│  • must_not: {formality: ["upscale", "fine_dining"], price: ["$$$$"]}      │
│  • preferences: {cuisine: ["Thai", "Asian"], features: ["outdoor_seating"]} │
│  • weights: {vibe: 0.5, cuisine: 0.25, price: 0.15, features: 0.1}         │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      STAGE 2: HARD EXCLUSION FILTER                         │
│                           (Database Query)                                  │
│                                                                             │
│  Removes restaurants matching ANY exclusion criteria:                       │
│  • WHERE formality NOT IN ('upscale', 'fine_dining')                       │
│  • WHERE price_level != '$$$$'                                              │
│                                                                             │
│  500 restaurants → ~300 eligible candidates                                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    STAGE 3: PARALLEL SIGNAL RETRIEVAL                       │
│                                                                             │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐          │
│  │  VIBE SIGNAL     │  │  CUISINE SIGNAL  │  │  FEATURE SIGNAL  │          │
│  │  (Vector Search) │  │  (Metadata)      │  │  (Metadata)      │          │
│  │                  │  │                  │  │                  │          │
│  │  Embeds semantic │  │  Matches against │  │  Matches boolean │          │
│  │  query, finds    │  │  cuisine field   │  │  feature flags   │          │
│  │  similar vibes   │  │  with fuzzy      │  │  like outdoor    │          │
│  │                  │  │  matching        │  │  seating         │          │
│  │  Score: 0-1      │  │  Score: 0-1      │  │  Score: 0-1      │          │
│  └────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘          │
│           │                     │                     │                     │
│           └─────────────────────┼─────────────────────┘                     │
│                                 ▼                                           │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         STAGE 4: SCORE FUSION                               │
│                                                                             │
│  final_score = (0.50 × vibe_score)                                         │
│              + (0.25 × cuisine_score)                                       │
│              + (0.15 × price_score)                                         │
│              + (0.10 × feature_score)                                       │
│                                                                             │
│  Weights are dynamic — extracted from query based on user emphasis          │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      STAGE 5: RESPONSE GENERATION                           │
│                           (LLM Call #2)                                     │
│                                                                             │
│  Generates:                                                                 │
│  • Natural language explanation connecting query to restaurant              │
│  • Structured match_reasons with score breakdowns                           │
│  • Confidence level based on match quality                                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
                            STRUCTURED JSON OUTPUT
```

---

## Project Structure

```
ana-ai-vibe-check/
├── src/
│   ├── __init__.py
│   ├── main.py                 # Main entry point
│   ├── pipeline.py             # AnaVibeSearch orchestrator
│   ├── query_parser.py         # LLM-based query understanding
│   ├── filters.py              # Hard exclusion filtering
│   ├── scorers/
│   │   ├── __init__.py
│   │   ├── base.py             # SignalScorer base class
│   │   ├── vibe_scorer.py      # Vector similarity scoring
│   │   ├── cuisine_scorer.py   # Cuisine matching logic
│   │   ├── price_scorer.py     # Price preference scoring
│   │   └── feature_scorer.py   # Boolean feature scoring
│   ├── fusion.py               # Score fusion logic
│   ├── response_generator.py   # LLM-based response generation
│   └── schemas/
│       ├── __init__.py
│       ├── query.py            # ParsedQuery, filters, weights
│       ├── restaurant.py       # Restaurant, RestaurantVibe
│       └── response.py         # AnaResponse, MatchReason
├── data/
│   ├── restaurants.json        # Mock restaurant data (enriched)
│   └── sample_queries.json     # Test queries for evaluation
├── prompts/
│   ├── query_parser.txt        # System prompt for query parsing
│   └── response_generator.txt  # System prompt for response generation
├── notebooks/
│   └── demo.ipynb              # Interactive demonstration
├── tests/
│   ├── test_query_parser.py
│   ├── test_scorers.py
│   └── test_pipeline.py
├── docs/
│   └── system_design.md        # Part 3: System Design Document
├── requirements.txt
└── README.md
```

---

## Installation

### Prerequisites

- Python 3.10+
- OpenAI API key (for embeddings and LLM calls)

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd ana-ai-vibe-check

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export OPENAI_API_KEY="your-api-key-here"
```

### Dependencies

```
openai>=1.0.0
chromadb>=0.4.0
pydantic>=2.0.0
python-dotenv>=1.0.0
numpy>=1.24.0
```

---

## Usage

### Basic Usage

```python
from src.pipeline import AnaVibeSearch

# Initialize the search system
ana = AnaVibeSearch()

# Search by vibe
response = await ana.search(
    "I want a place that feels like a night market in Bangkok—noisy, plastic chairs, super casual—not a fancy sit-down spot."
)

print(response.top_match.name)
print(response.explanation)
```

### Example Output

```json
{
  "success": true,
  "top_match": {
    "name": "Noi's Thai Kitchen",
    "cuisine": "Thai",
    "price_level": "$",
    "vibe_summary": "No-frills Thai spot with plastic tables, loud music, and authentic Bangkok street food vibes",
    "final_score": 0.91
  },
  "explanation": "Noi's Thai Kitchen is exactly what you're looking for. It's got that authentic Bangkok night market energy—plastic chairs out front, loud and bustling, super casual. The food is legit Thai street food, not the watered-down tourist version.",
  "match_reasons": [
    {
      "signal": "vibe",
      "importance": "primary",
      "query_wanted": "night market feel, noisy, casual",
      "restaurant_has": "plastic outdoor seating, loud atmosphere, no-frills service",
      "score": 0.92
    },
    {
      "signal": "cuisine",
      "importance": "secondary",
      "query_wanted": "Thai, Asian, Street Food",
      "restaurant_has": "Thai",
      "score": 1.0
    }
  ],
  "confidence": "high"
}
```

### Running the Demo

```bash
# Interactive notebook
jupyter notebook notebooks/demo.ipynb

# Or run directly
python -m src.main --query "romantic dinner spot with ocean views"
```

---

## Data Schema

### Restaurant (Base + Enriched)

```python
class Restaurant:
    # Core Identity
    id: str
    name: str
    google_place_id: str

    # Structured Fields (from KaanaRestaurant schema)
    cuisine: str                    # "Thai", "Hawaiian", "Japanese"
    price_level: str                # "$", "$$", "$$$", "$$$$"
    region: str                     # "Lahaina", "Kihei", etc.
    rating: float

    # Boolean Features
    features: dict[str, bool]       # outdoor_seating, good_for_groups, etc.

    # Original Text Fields
    highlights: str
    details: str
    editorial_summary: str

    # ENRICHED: Vibe Profile (generated from text fields)
    vibe: RestaurantVibe

class RestaurantVibe:
    formality: str                  # "very_casual" | "casual" | "smart_casual" | "upscale" | "fine_dining"
    noise_level: str                # "quiet" | "moderate" | "lively" | "loud"
    atmosphere_tags: list[str]      # ["romantic", "family-friendly", "trendy", "hole-in-the-wall"]
    best_for: list[str]             # ["date night", "groups", "solo", "business"]
    vibe_summary: str               # Natural language description for embeddings
```

### Parsed Query

```python
class ParsedQuery:
    semantic_query: str             # Expanded query for vector search
    must_have: MustHaveFilters      # Hard requirements
    must_not: MustNotFilters        # Hard exclusions (handles "NOT X")
    preferences: Preferences        # Soft preferences (boost, don't exclude)
    weights: SignalWeights          # Dynamic weights based on query emphasis
```

### Response

```python
class AnaResponse:
    success: bool
    query_understood: ParsedQuery
    top_match: RestaurantMatch
    alternatives: list[RestaurantMatch]
    match_reasons: list[MatchReason]
    explanation: str                # Natural language from Ana
    confidence: str                 # "high" | "medium" | "low"
    caveats: list[str]              # Edge case notes
```

---

## Technical Decisions

| Component | Choice | Rationale |
|-----------|--------|-----------|
| **LLM** | GPT-4o-mini | Fast, cheap ($0.15/1M tokens), sufficient for parsing and generation |
| **Embeddings** | text-embedding-3-small | Good quality, cheap ($0.02/1M tokens), 1536 dimensions |
| **Vector Store** | ChromaDB | Simple local setup, good for prototyping, supports metadata filtering |
| **Framework** | Raw Python + Pydantic | Clean, readable, no abstraction overhead, easy to evaluate |
| **Retrieval** | Hybrid (filter + vector + metadata) | Handles negations explicitly, combines semantic + structured matching |

### Why Hybrid Over Pure Vector Search?

| Query Component | Pure Vector | Hybrid |
|-----------------|-------------|--------|
| "night market feel" | ✅ Good | ✅ Good |
| "NOT fancy" | ❌ May still return fancy places | ✅ Hard exclusion |
| "Thai food" | ⚠️ Approximate | ✅ Exact match boost |
| "outdoor seating" | ⚠️ Approximate | ✅ Boolean check |

---

## Assumptions

### Data Assumptions

1. **Vibe enrichment exists**: The pipeline assumes restaurants have enriched `vibe` profiles. In production, this would be generated via LLM from existing text fields (`highlights`, `details`, `reviews`).

2. **Mock data scope**: Demo uses 15-20 mock restaurants representing diverse vibes across Maui.

3. **Embedding pre-computation**: Restaurant vibe embeddings are pre-computed and stored in the vector database.

### Query Assumptions

1. **English language**: Queries are expected in English.

2. **Hawaii context**: The system is optimized for Hawaii restaurants. Queries like "best pizza in NYC" are out of scope.

3. **Single intent**: Each query has one primary dining intent (not "find me Thai food AND a coffee shop").

### System Assumptions

1. **API availability**: OpenAI API is available and responsive.

2. **Latency tolerance**: Target response time is <2 seconds for typical queries.

3. **No real-time data**: Menu items, hours, and availability are static (data freshness addressed in design doc).

---

## Edge Cases Handled

| Scenario | Handling |
|----------|----------|
| No results after filtering | Progressively relax filters, note in `caveats` |
| Low similarity scores (<0.5) | Return results but set `confidence: "low"` |
| Contradictory query ("fancy but cheap") | Flag conflict, return best compromise |
| Unknown cuisine type | Skip cuisine filter, search by vibe only |
| All negations, no positives | Default to broad search, ask for clarification |

---

## Future Improvements

### Short-term
- [ ] Add location-aware scoring (prefer nearby restaurants)
- [ ] Implement query caching for common vibe searches
- [ ] Add user feedback loop to improve rankings

### Long-term
- [ ] Real-time menu integration for availability checking
- [ ] Multi-turn conversation for refining preferences
- [ ] Personalization based on user history
- [ ] Support for Part 2: Video-to-Plate authenticity verification

---

## Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_pipeline.py -v
```

---

## License

This project is part of a take-home assessment for Kaana.

---

## Author

Yugandhar Gopu - AI Engineer Candidate
