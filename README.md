# Ana AI — Vibe-Based Restaurant Discovery

> *"Find me somewhere that feels like a Bangkok night market—loud, plastic chairs, no pretense."*

A hybrid retrieval system that understands **how people actually search for food**: by feeling, not filters.

---

## Architecture

```
                                    ┌──────────────┐
                                    │  User Query  │
                                    └──────┬───────┘
                                           │
                              ┌────────────▼────────────┐
                              │   Query Understanding   │
                              │        (LLM #1)         │
                              │                         │
                              │  → semantic_query       │
                              │  → must_not (negations) │
                              │  → preferences          │
                              │  → dynamic weights      │
                              └────────────┬────────────┘
                                           │
                              ┌────────────▼────────────┐
                              │    Hard Exclusions      │
                              │   (Database Filter)     │
                              │                         │
                              │  "NOT fancy" → exclude  │
                              │   upscale/fine_dining   │
                              └────────────┬────────────┘
                                           │
                    ┌──────────────────────┼──────────────────────┐
                    │                      │                      │
           ┌────────▼────────┐   ┌────────▼────────┐   ┌────────▼────────┐
           │   Vibe Signal   │   │ Cuisine Signal  │   │ Feature Signal  │
           │ (Vector Search) │   │   (Metadata)    │   │   (Boolean)     │
           └────────┬────────┘   └────────┬────────┘   └────────┬────────┘
                    │                      │                      │
                    └──────────────────────┼──────────────────────┘
                                           │
                              ┌────────────▼────────────┐
                              │     Score Fusion        │
                              │                         │
                              │  weighted combination   │
                              │  of all signals         │
                              └────────────┬────────────┘
                                           │
                              ┌────────────▼────────────┐
                              │  Response Generation    │
                              │        (LLM #2)         │
                              │                         │
                              │  natural language +     │
                              │  structured reasons     │
                              └────────────┬────────────┘
                                           │
                                    ┌──────▼───────┐
                                    │    Output    │
                                    └──────────────┘
```

---

## Why Hybrid Retrieval?

| Query Component | Pure Vector | This System |
|-----------------|-------------|-------------|
| "night market feel" | ✅ | ✅ |
| "NOT fancy" | ❌ still returns fancy | ✅ hard exclusion |
| "Thai food" | ~approximate | ✅ exact match |
| "outdoor seating" | ~approximate | ✅ boolean check |

---

## Quick Start

```bash
# Setup
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY="your-key"

# Run
python -m src.main --query "romantic spot with ocean views"
```

---

## Project Structure

```
src/
├── main.py              # Entry point
├── pipeline.py          # Orchestrator
├── query_parser.py      # LLM query understanding
├── filters.py           # Hard exclusions
├── scorers/             # Signal scoring modules
│   ├── vibe_scorer.py   # Vector similarity
│   ├── cuisine_scorer.py
│   ├── price_scorer.py
│   └── feature_scorer.py
├── fusion.py            # Weighted score combination
├── response_generator.py
└── schemas/             # Pydantic models

data/
├── restaurants.json     # Restaurant data
└── sample_queries.json  # Test queries

prompts/
├── query_parser.txt
└── response_generator.txt
```

---

## Tech Stack

| Component | Choice | Why |
|-----------|--------|-----|
| LLM | GPT-4o-mini | Fast, cheap, sufficient |
| Embeddings | text-embedding-3-small | Quality/cost balance |
| Vector Store | ChromaDB | Simple, local, metadata filtering |
| Validation | Pydantic | Type safety, clean schemas |

---

## Example Response

```json
{
  "top_match": {
    "name": "Noi's Thai Kitchen",
    "cuisine": "Thai",
    "price_level": "$",
    "final_score": 0.91
  },
  "explanation": "Exactly what you're looking for—plastic chairs, loud and bustling, authentic Bangkok street food vibes.",
  "confidence": "high"
}
```

---

## Author

Yugandhar Gopu
