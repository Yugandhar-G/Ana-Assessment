# Vibe Summary Creation Bug Analysis

## How Vibe Summary is Created

The `vibe_summary` field is created in **`scripts/convert_csv_to_json.py`** during the CSV-to-JSON conversion process.

### The Function: `parse_vibe()`

Located at lines 56-95 in `scripts/convert_csv_to_json.py`:

```python
def parse_vibe(vibe_str: str, restaurant_name: str = "", cuisine: str = "", details: str = "", highlights: str = "") -> Dict[str, Any]:
    """Parse vibe string to dictionary. Generate vibe_summary if missing."""
    default_vibe = {
        "formality": "casual",
        "noise_level": "moderate",
        "atmosphere_tags": [],
        "best_for": [],
        "vibe_summary": ""
    }
    
    if not vibe_str or vibe_str.strip() == '':
        # Generate a basic vibe_summary from available data
        vibe_summary = highlights or details or f"{restaurant_name} serves {cuisine} cuisine."
        default_vibe["vibe_summary"] = vibe_summary[:500]
        return default_vibe
    
    # ... parsing logic ...
    
    vibe_summary = parsed.get("vibe_summary", "")
    # If vibe_summary is empty, generate one from available data
    if not vibe_summary or not vibe_summary.strip():
        vibe_summary = highlights or details or f"{restaurant_name} serves {cuisine} cuisine."
```

### The Bug: Missing Parameters

**Line 212** in `convert_row_to_restaurant()`:

```python
"vibe": parse_vibe(row.get("vibe", ""))
```

**Problem:** The function is called with ONLY the `vibe` field from the CSV, but the function signature expects:
- `restaurant_name`
- `cuisine`
- `details`
- `highlights`

These are **NOT being passed**, so they default to empty strings (`""`).

### What Happens

1. CSV row has empty or missing `vibe` field
2. `parse_vibe("")` is called (no other parameters)
3. Function checks: `if not vibe_str or vibe_str.strip() == ''` → **True**
4. Falls back to: `vibe_summary = highlights or details or f"{restaurant_name} serves {cuisine} cuisine."`
5. Since all parameters are empty strings:
   - `highlights = ""` (empty)
   - `details = ""` (empty)
   - `f"{restaurant_name} serves {cuisine} cuisine."` = `f" serves  cuisine."` = `" serves  cuisine."`

**Result:** 359 restaurants get the generic `" serves  cuisine."` vibe_summary!

### The Fix

**Change line 212 from:**
```python
"vibe": parse_vibe(row.get("vibe", ""))
```

**To:**
```python
"vibe": parse_vibe(
    row.get("vibe", ""),
    restaurant_name=name,
    cuisine=row.get("cuisine", "").strip() or "Unknown",
    details=row.get("details", "").strip() or "",
    highlights=row.get("highlights", "").strip() or ""
)
```

This ensures that when the `vibe` field is empty, the function can use `highlights`, `details`, or generate a proper summary from the restaurant name and cuisine.

---

## Alternative: Better Fallback Logic

The function could also be improved to handle empty strings better:

```python
def parse_vibe(vibe_str: str, restaurant_name: str = "", cuisine: str = "", details: str = "", highlights: str = "") -> Dict[str, Any]:
    # ... existing code ...
    
    if not vibe_str or vibe_str.strip() == '':
        # Try to use highlights or details first
        if highlights and highlights.strip():
            vibe_summary = highlights[:500]
        elif details and details.strip():
            vibe_summary = details[:500]
        elif restaurant_name and cuisine:
            # Only generate template if we have actual data
            vibe_summary = f"{restaurant_name} serves {cuisine} cuisine."
        else:
            # Last resort: generic fallback
            vibe_summary = "Restaurant serving various cuisine."
        
        default_vibe["vibe_summary"] = vibe_summary
        return default_vibe
```

---

## Impact

- **359 out of 360 restaurants** affected
- All have identical generic `vibe_summary`: `" serves  cuisine."`
- Causes vector search to fail (identical embeddings)
- Results in random/unpredictable restaurant recommendations
- Explains the "Morimoto hallucination" issue

---

## Solution Steps

1. **Fix the function call** to pass all required parameters
2. **Re-run the CSV conversion** to regenerate `restaurants.json` with proper vibe summaries
3. **Re-index the vector store** to create new embeddings from corrected vibe summaries
4. **Test** to ensure vector search now works correctly

---

## Related Files

- `scripts/convert_csv_to_json.py` - Contains the bug
- `data/restaurants.json` - Contains the corrupted data (359 restaurants with generic summaries)
- `part1/src/vector_store.py` - Uses vibe_summary for embeddings
- `part1/src/scorers/vibe_scorer.py` - Uses vibe_summary for scoring

