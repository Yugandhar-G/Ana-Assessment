# Codebase Issues and Fixes

This document identifies key issues found in the Ana-Assessment codebase that need to be fixed.

## 🔴 CRITICAL ISSUES

### 1. **Hard Filter Using Preferences (BUG)** ✅ FIXED
**Location:** `part1/src/filters.py` lines 33-51

**Problem:** The `_passes_filters` method was treating `parsed_query.preferences.cuisine` as a HARD filter (excluding restaurants), but preferences should be SOFT filters that only affect scoring, not filtering.

**Impact:** Restaurants with non-matching cuisines were completely excluded even if they might be good matches on other dimensions. This contradicted the design where preferences should boost scores but not exclude restaurants.

**Fix Applied:** Removed the cuisine preference filtering block entirely. Added comment explaining that preferences are soft filters that only affect scoring via `CuisineScorer`. Only `must_not.cuisine` is now used for hard exclusions.

---

### 2. **Location Treated as Hard Filter**
**Location:** `part1/src/filters.py` lines 62-68

**Problem:** Location is treated as a hard filter, but it should likely be a soft preference to give users flexibility.

**Impact:** Restaurants outside the specified location are completely excluded, even if they're perfect matches on other criteria.

**Consideration:** This might be intentional (strict location requirement), but consider making it a soft preference with location-based scoring.

---

## 🟡 MEDIUM PRIORITY ISSUES

### 3. **Weight Normalization Missing** ✅ FIXED
**Location:** `part1/src/query_parser.py` lines 142-154

**Problem:** SignalWeights from LLM don't necessarily sum to 1.0, which can cause incorrect score fusion.

**Fix Applied:** Added weight normalization after parsing. Weights are now normalized to sum to 1.0. If total is 0, falls back to default SignalWeights.
```python
weights_data = parsed_json.get("weights", {})
weights = SignalWeights(**weights_data)

# Normalize weights to sum to 1.0 for correct score fusion
total_weight = weights.vibe + weights.cuisine + weights.price + weights.features
if total_weight > 0:
    weights.vibe /= total_weight
    weights.cuisine /= total_weight
    weights.price /= total_weight
    weights.features /= total_weight
else:
    # Fallback to defaults if all weights are 0
    weights = SignalWeights()
```

---

### 4. **Type Safety Issue in Fusion** ✅ FIXED
**Location:** `part1/src/fusion.py` line 66

**Problem:** `AdvancedScoreFusion.fuse()` required `parsed_query` parameter (non-optional), but `ScoreFusion.fuse()` had it as optional. This could cause type issues.

**Fix Applied:** Made `parsed_query` parameter optional in `AdvancedScoreFusion.fuse()` (with type `ParsedQuery | None = None`) and added runtime check that raises ValueError if None is passed, ensuring type safety while maintaining compatibility with the base class interface.

---

### 5. **Duplicate Loop in Feature Scorer** ✅ FIXED
**Location:** `part1/src/scorers/feature_scorer.py` lines 61-75

**Problem:** There was a loop over `desired_atmosphere` that ran twice - once checking `atmosphere_tags` and once checking `best_for`, but the second loop didn't increment `total`. This could cause double-counting and was confusing.

**Fix Applied:** Refactored to combine both checks into a single loop. `atmosphere_tags` is checked first (primary source), and `best_for` is only checked as a fallback/bonus if not already matched in `atmosphere_tags`. This prevents double-counting and makes the logic clearer.

---

## 🟢 MINOR ISSUES / IMPROVEMENTS

### 6. **Missing Error Handling**
**Location:** Various locations

**Issues:**
- `vector_store.py`: No error handling for JSON serialization failures
- `query_parser.py`: JSON parsing errors are silently caught but could provide better feedback
- Embedding generation failures could be handled more gracefully

---

### 7. **Potential AttributeError** ✅ REVIEWED - SAFE
**Location:** `part1/src/filters.py` line 50

**Problem:** Uses `getattr(restaurant, "city", None)` but then accesses `.lower()` which could fail if `city` is None.

**Current Code:**
```python
city = getattr(restaurant, "city", None)
city_match = query_loc in city.lower() if city else False
```

**Status:** ✅ Actually safe due to the ternary check (`if city else False`). The `.lower()` is only called when `city` is not None. No fix needed.

---

### 8. **Hard-coded Values**
**Location:** Multiple files

- `part1/src/pipeline.py` line 111: Hard-coded `n_results=25`
- `part1/src/ollama_client.py` line 248: Hard-coded semaphore value `6`
- Various magic numbers throughout fusion.py

**Recommendation:** Move to constants or configuration.

---

### 9. **Code Duplication**
**Location:** Multiple files

- Similar error handling patterns repeated in `ollama_client.py`
- Restaurant name detection logic duplicated in `query_parser.py` and `pipeline.py`

---

## 📝 SUMMARY OF REQUIRED FIXES

### Must Fix (Critical):
1. ✅ **FIXED** - Remove cuisine preference hard filtering from `filters.py`
2. ⚠️ **PENDING DECISION** - Review location filtering (decide if it should be soft preference)

### Should Fix (Medium):
3. ✅ **FIXED** - Add weight normalization in query parser
4. ✅ **FIXED** - Fix type safety issue in fusion
5. ✅ **FIXED** - Review/refactor duplicate atmosphere loop in feature scorer

### Nice to Have (Minor):
6. ⚠️ **NOT FIXED** - Improve error handling (vector_store.py, query_parser.py, embedding failures)
7. ⚠️ **NOT FIXED** - Extract magic numbers to constants (n_results=25, semaphore=6, fusion magic numbers)
8. ⚠️ **NOT FIXED** - Reduce code duplication (error handling patterns, restaurant name detection)

---

## 🧪 TESTING RECOMMENDATIONS

After fixes, test:
1. Query with cuisine preference that doesn't match - should still return results (just lower scored)
2. Query with location - verify behavior matches expectations
3. Queries with various weight configurations
4. Edge cases: empty preferences, invalid weights, missing data

