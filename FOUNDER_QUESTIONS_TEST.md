# Founder Questions Test Results

This document shows how the Ana AI system handles the founder's test questions.

## Test Questions & System Responses

### 1. "Most popular menu items?"
**Status**: ✅ Handled
- System acknowledges that specific menu items aren't available in the database
- If `top_menu_items` data exists in the CSV, it will be included
- System mentions cuisine type and any relevant details from restaurant description
- **Example Response**: "While I don't have specific menu details for [Restaurant], it's known for its [Cuisine] cuisine, especially [details from description]..."

### 2. "Ingredients?"
**Status**: ✅ Handled
- System explains that ingredient-level details aren't available
- Can mention if restaurant serves vegetarian/vegan options (from features)
- **Example Response**: "I don't have specific ingredient information, but [Restaurant] does serve vegetarian options..."

### 3. "Vegan options?"
**Status**: ✅ **Works Well**
- System searches for restaurants with `serves_vegetarian` feature
- Can identify vegan restaurants specifically
- **Example Response**: "[Restaurant] is a fantastic choice for vegan options! They have an extensive 100% vegan menu..."

### 4. "Gluten free options?"
**Status**: ✅ Handled
- System mentions vegetarian options if available (from features)
- Acknowledges that specific gluten-free menu information isn't available
- **Example Response**: "[Restaurant] offers [Cuisine] cuisine, which could provide some gluten-free options. I don't have specific details about gluten-free items on their menu..."

### 5. "Organically sourced items?"
**Status**: ✅ Handled
- System acknowledges this information isn't available in the database
- **Example Response**: "I don't have information about organic sourcing for [Restaurant]. You may want to check their website or call them directly..."

### 6. "Value: price x quality?"
**Status**: ✅ Handled
- System uses price level and rating information
- Can provide value assessment based on price vs. rating
- **Example Response**: "[Restaurant] offers [Price Level] pricing with a [Rating]/5.0 rating, providing good value..."

### 7. "Specials?"
**Status**: ✅ Handled
- System explains that current specials aren't available
- Suggests checking website or calling
- **Example Response**: "I don't have information about current specials. I recommend checking their website at [website] or calling them at [phone]..."

### 8. "Best time/day to eat there?"
**Status**: ✅ **Works Well**
- System uses `opening_hours_text` to provide hours
- Uses `serves_meal_times` to suggest breakfast/lunch/dinner
- Uses atmosphere/vibe information to suggest optimal times
- **Example Response**: "[Restaurant] is open [hours]. With its [atmosphere] atmosphere, it's great for [meal time]..."

### 9. "Parking?"
**Status**: ✅ **Works Well**
- System parses `parking_options` field
- Provides specific parking details (free parking lot, valet, street parking, etc.)
- **Example Response**: "[Restaurant] offers free parking lot, free street parking..."

### 10. "Atmosphere?"
**Status**: ✅ **Works Well**
- System uses `vibe` object with:
  - `formality` (casual, upscale, etc.)
  - `noise_level` (quiet, moderate, lively, loud)
  - `atmosphere_tags` (family-friendly, romantic, etc.)
- **Example Response**: "[Restaurant] has a [formality] atmosphere with [noise_level] noise level. It's known for being [atmosphere_tags]..."

## Data Availability Summary

| Question Type | Data Available | System Handling |
|--------------|----------------|-----------------|
| Menu Items | Partial (`top_menu_items` if in CSV) | Acknowledges limitations, provides what's available |
| Ingredients | ❌ Not available | Explains limitation, mentions dietary options |
| Vegan Options | ✅ Yes (`serves_vegetarian` feature) | **Works well** |
| Gluten-Free | Partial (via vegetarian feature) | Acknowledges limitation |
| Organic Sourcing | ❌ Not available | Explains limitation, suggests contacting restaurant |
| Value (Price/Quality) | ✅ Yes (price_level + rating) | **Works well** |
| Specials | ❌ Not available | Explains limitation, suggests website/phone |
| Best Time/Day | ✅ Yes (hours + meal times + atmosphere) | **Works well** |
| Parking | ✅ Yes (`parking_options` field) | **Works well** |
| Atmosphere | ✅ Yes (`vibe` object) | **Works well** |

## Recommendations for Testing

1. **Test with specific restaurant names** for better results:
   - "Most popular menu items at MAMA'S FISH HOUSE"
   - "Parking at LE BAZAAR"
   - "Atmosphere at MAMA'S FISH HOUSE"

2. **Test general queries** to see restaurant discovery:
   - "Vegan options"
   - "Restaurants with parking"
   - "Best atmosphere for date night"

3. **System strengths**:
   - Excellent at finding restaurants by dietary needs (vegan, vegetarian)
   - Good at providing parking information
   - Good at describing atmosphere
   - Good at suggesting best times based on hours and vibe

4. **System limitations** (honestly communicated):
   - Menu item details are limited
   - Ingredient-level information not available
   - Current specials/promotions not tracked
   - Organic sourcing information not available

## How to Test

Run queries via:
```bash
python -m part1.src.main --query "YOUR QUESTION HERE"
```

Or use the Gradio interface:
```bash
cd part1 && python gradio_app.py
```

