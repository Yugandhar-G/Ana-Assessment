# Structured Output Responses

This module provides structured restaurant search responses with enriched data for the conversational interface.

## Overview

The structured output system provides a standardized format for displaying restaurant information, making it easy to present consistent, well-formatted results in the conversational Gradio app.

## Structure

### Files

- **`schemas.py`**: Defines the data models:
  - `EnrichedRestaurantMatch`: Extended restaurant data with all fields needed for display
  - `StructuredSearchResponse`: Response wrapper for structured search results

- **`enrichment.py`**: Utilities to convert `ScoredRestaurant` objects to `EnrichedRestaurantMatch`:
  - `enrich_from_scored_restaurant()`: Extracts all needed fields from restaurant objects

- **`api.py`**: API endpoint function:
  - `search_structured()`: Performs search and returns enriched structured data

- **`formatter.py`**: Formatting functions for display:
  - `format_restaurant_details()`: Formats a single restaurant according to the specified structure
  - `format_search_results()`: Formats complete search results (top match + alternatives)

## Response Format

Each restaurant is formatted with the following structure:

1. **Relevance Explanation** (2-3 lines): Why this restaurant is relevant to the user's query
2. **Good for**: Top menu items (if available)
3. **Vibe at this restaurant**: Vibe description with images
4. **Features**: List of restaurant features
5. **Videos**: Video links (if available)

## API Endpoint

The structured search endpoint is available at:
- **POST** `/api/search/structured`
- **Request**: `{"query": "your search query"}`
- **Response**: `StructuredSearchResponse` with enriched restaurant data

## Usage

```python
from part1.structured_output_responses.api import search_structured
from part1.structured_output_responses.formatter import format_search_results

# In API endpoint
response = await search_structured(search_instance, query)
return response.model_dump()

# In Gradio app
results = perform_search(query)  # Calls /api/search/structured
formatted = format_search_results(results, is_refined=False)
```

## Notes

- This module doesn't modify the existing `/api/search` endpoint
- Uses `search_for_streaming()` to get scored restaurants with original objects
- Enriches data with all needed fields from restaurant schema
- Supports up to 10 restaurants (1 top match + 9 alternatives)

