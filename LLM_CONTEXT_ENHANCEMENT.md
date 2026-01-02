# LLM Context Enhancement for RAG

## Overview

Enhanced the system to use LLM's general knowledge to enrich queries BEFORE RAG retrieval, giving better context to the vector search system.

## Problem

Previously, when a user asked "what's the best dessert in Maui", the system would:
1. Parse query → "dessert" 
2. Search vector DB for "dessert"
3. Return results based on simple keyword matching

This missed cultural context, implicit requirements, and domain knowledge.

## Solution

Added **Query Enrichment** step that uses LLM to expand queries with context BEFORE RAG retrieval.

## Implementation

### 1. Query Enrichment (`query_parser.py`)

**New Method: `_enrich_query_with_context()`**
- Runs BEFORE vector search
- Uses LLM to understand query intent deeply
- Adds cultural context, implicit requirements, and domain knowledge
- Expands query with related terms restaurants would use

**Example:**
```
User Query: "best desserts in Maui"
Enriched: "best desserts in Maui, Hawaiian desserts, shave ice, malasadas, tropical fruit desserts, local specialties, famous sweets, popular treats"

User Query: "romantic dinner spot"
Enriched: "romantic dinner spot, intimate atmosphere, candlelit dining, quiet ambiance, cozy setting, special occasion restaurant"
```

### 2. Integration Flow

```
User Query
    ↓
Query Parser (parse)
    ↓
LLM Query Enrichment ← NEW: Adds context
    ↓
ParsedQuery.semantic_query (enriched)
    ↓
Pipeline (search)
    ↓
Cuisine Enhancement (if cuisine query) ← Existing
    ↓
Vector Search (uses enriched query)
    ↓
Results (better matches due to enriched context)
```

### 3. What Gets Added

**Cultural Context:**
- "desserts" → "Hawaiian desserts, shave ice, malasadas, tropical fruits"
- "breakfast" → "local favorites, Hawaiian breakfast, fresh fruit, acai bowls"

**Implicit Requirements:**
- "romantic" → "intimate atmosphere, candlelit, quiet, cozy"
- "casual" → "relaxed, family-friendly, laid-back"

**Domain Knowledge:**
- What's famous in Maui for this query type
- What restaurants typically describe themselves as
- Related terms that appear in restaurant descriptions

## Benefits

1. **Better RAG Retrieval**: Vector search finds more relevant restaurants because query has richer context
2. **Cultural Awareness**: Understands Maui/Hawaii context automatically
3. **Implicit Understanding**: Captures what user really wants, not just keywords
4. **Domain Knowledge**: Uses LLM's knowledge about what's typical/famous in Maui

## Example Workflow

**User Query:** "what desserts are famous in Maui?"

**Step 1 - Query Enrichment:**
- Original: "desserts famous Maui"
- Enriched: "desserts famous Maui, Hawaiian desserts, shave ice, malasadas, tropical fruit desserts, local specialties, famous sweets, popular treats, Maui dessert culture"

**Step 2 - Vector Search:**
- Searches with enriched query
- Finds restaurants that mention "shave ice", "malasadas", "Hawaiian desserts" in their descriptions
- Better matches than just searching for "dessert"

**Step 3 - Results:**
- More culturally relevant restaurants
- Better understanding of what user wants
- Results that match both explicit and implicit requirements

## Technical Details

- **Location**: `part1/src/query_parser.py::_enrich_query_with_context()`
- **When**: Called during `parse()` method, before semantic_query is finalized
- **Model**: Uses same Gemini model as query parsing
- **Temperature**: 0.3 (slight creativity for context expansion)
- **Max Tokens**: 200 (concise but comprehensive)

## Combined with Existing Features

This enhancement works alongside:
- ✅ RAG retrieval (vector search)
- ✅ Multi-signal scoring (vibe, cuisine, price, features)
- ✅ Score fusion and reranking
- ✅ LLM reasoning about results (post-retrieval)
- ✅ LLM response generation

The system now has:
1. **Pre-retrieval**: Query enrichment (NEW)
2. **Retrieval**: RAG with enriched queries
3. **Post-retrieval**: LLM reasoning about results
4. **Response**: LLM generation with knowledge + data

This creates a complete pipeline that leverages LLM knowledge at every stage while maintaining RAG precision.

