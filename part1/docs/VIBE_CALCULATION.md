# How Vibe is Calculated

## Current Implementation

The vibe profile is currently generated using **rule-based inference** from available restaurant data. 

**Important:** We ARE using fields from your dataset:
- ✅ `highlights` - Rich descriptive text (used directly for vibe_summary)
- ✅ `details` - Additional description
- ✅ `editorial_summary` - Editorial content
- ✅ `theme` - Theme tags from dataset
- ✅ `top_menu_items` - Menu information
- ✅ All feature fields (live_music, outdoor_seating, etc.)

**However:** For structured fields (formality, noise_level, atmosphere_tags), we're doing rule-based inference instead of extracting from the rich text. Here's how it works:

### 1. Formality Level

Inferred from price level and theme:

```python
Price Level → Formality
$$$$ → "fine_dining"
$$$  → "upscale"
$$  → "smart_casual"
$    → "very_casual" (or "casual" if no other signals)

Additional signals:
- "hole-in-the-wall" in theme → "very_casual"
```

### 2. Noise Level

Inferred from features and highlights:

```python
Priority:
1. live_music feature → "loud"
2. Keywords in highlights:
   - "loud", "bustling", "energetic" → "loud"
   - "quiet", "peaceful", "tranquil" → "quiet"
   - "lively" → "lively"
3. Default → "moderate"
```

### 3. Atmosphere Tags

Built from multiple sources:

**From Theme:**
- "Local Favorite" → `local-favorite`
- "Hole-in-the-wall" → `hole-in-the-wall`
- "Vegan" → `vegan-friendly`

**From Price:**
- `$` → `budget-friendly`
- `$$$$` → `upscale`

**From Features:**
- `live_music` → `lively`
- `good_for_children` → `family-friendly`
- `good_for_groups` → `group-friendly`

**From Highlights (text analysis):**
- "romantic" or "intimate" → `romantic`
- "casual" or "relaxed" → `casual`
- "trendy" or "hip" → `trendy`

### 4. Best For

Inferred from features and atmosphere:

```python
- "romantic" in tags → "date night"
- good_for_groups → "groups"
- live_music → "nightlife"
- good_for_children → "families"
- serves_breakfast → "breakfast"
- serves_lunch → "lunch"
- serves_dinner → "dinner"
```

### 5. Vibe Summary

**Primary method:** Uses `highlights` field **directly from your dataset** if available (>50 chars)
- This is the rich descriptive text you provided
- Truncates to ~500 chars at sentence boundary
- Ensures proper punctuation

**Fallback method:** Builds from components (only if highlights missing):
```
[Formality] + [Theme tags] + [Features] + [Noise level] + [Cuisine]
```

**Example from your dataset:**
> "Relaxed local hangout with a friendly atmosphere, Creative island-inspired comfort food, Full bar offering unique cocktails, Unpretentious and welcoming neighborhood gem"

This comes directly from the `highlights` field in your CSV!

---

## Limitations of Current Approach

1. **Rule-based for structured fields** - Formality, noise_level, atmosphere_tags are inferred from rules rather than extracted from your rich text
2. **Keyword matching** - Misses context and subtlety in highlights/details
3. **Not using LLM** - Should extract vibe from your rich `highlights`/`details` text using LLM
4. **Wasted rich data** - Your dataset has excellent descriptive text that should be analyzed, not just keyword-matched

**The Good News:** 
- ✅ `vibe_summary` DOES use your `highlights` field directly (this is good!)
- ✅ We have all the rich text from your dataset available
- ⚠️ But we should use LLM to extract structured vibe (formality, noise_level, tags) from that rich text

---

## Recommended: LLM-Based Enrichment

According to the system design, vibe enrichment should ideally use **LLM** for better quality:

### Proposed LLM Enrichment

```python
def enrich_vibe_with_llm(restaurant_data):
    """Use LLM to generate rich vibe profile from restaurant data."""
    
    prompt = f"""
    Given this restaurant data, extract a comprehensive vibe profile:
    
    Name: {restaurant_data['name']}
    Cuisine: {restaurant_data['cuisine']}
    Price: {restaurant_data['price_level']}
    Highlights: {restaurant_data['highlights']}
    Details: {restaurant_data['details']}
    Features: {restaurant_data['features']}
    
    Generate:
    1. Formality level (very_casual, casual, smart_casual, upscale, fine_dining)
    2. Noise level (quiet, moderate, lively, loud)
    3. Atmosphere tags (romantic, trendy, hole-in-the-wall, etc.)
    4. Best for (date night, groups, families, etc.)
    5. Vibe summary (2-3 sentence natural language description)
    
    Output JSON format.
    """
    
    response = llm_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    
    return json.loads(response.choices[0].message.content)
```

### Benefits of LLM Enrichment

1. **Nuanced understanding** - Captures subtle vibes
2. **Context awareness** - Understands implicit signals
3. **Consistent quality** - All restaurants get rich profiles
4. **Better embeddings** - Richer vibe_summary = better vector search

---

## Current Data Quality

From the cleaned dataset:
- **95 restaurants** with vibe profiles
- **100% have vibe_summary** (81 use highlights, 14 built from components)
- **All have formality, noise_level, atmosphere_tags, best_for**

---

## Current Implementation Summary

**The system IS using your dataset correctly:**

1. ✅ **vibe_summary** = Uses `highlights` field directly from your dataset (when available)
2. ✅ **Vector embeddings** = Generated from `vibe_summary` (which comes from your `highlights`)
3. ✅ **Vector search** = Pre-computed embeddings, no LLM calls per query
4. ✅ **VibeScorer** = Uses `restaurant.vibe.vibe_summary` for semantic similarity

**The flow:**
```
Your Dataset (highlights) 
  → load_real_data.py extracts highlights
  → vibe_summary = highlights (truncated to 500 chars)
  → VectorStore embeds vibe_summary (batch, one-time)
  → Vector search uses pre-computed embeddings (fast!)
```

**What could be improved (optional, not necessary):**
- Use LLM to extract formality/noise_level from highlights (currently rule-based)
- But this is optional - the current system works well with your dataset's highlights field

