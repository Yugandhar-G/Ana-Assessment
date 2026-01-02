#!/usr/bin/env python3
"""Audit restaurant vibe summaries for generic or incomplete descriptions."""
import json
from pathlib import Path
from typing import List, Dict

# Generic patterns that indicate poor quality vibe summaries
GENERIC_PATTERNS = [
    "serves cuisine",
    "serves food",
    "restaurant",
    "dining",
    "food",
    "cuisine",
    "place",
    "spot",
    "establishment",
]


def is_generic_vibe_summary(text: str) -> bool:
    """Check if vibe summary is too generic."""
    if not text or len(text.strip()) < 20:
        return True
    
    text_lower = text.lower().strip()
    
    # Check for generic patterns
    for pattern in GENERIC_PATTERNS:
        if text_lower == pattern or text_lower == f"{pattern}.":
            return True
    
    # Check if it's too short or just repeats the cuisine
    if len(text.split()) < 5:
        return True
    
    return False


def audit_restaurants(json_path: str) -> List[Dict]:
    """Audit all restaurants for generic vibe summaries."""
    with open(json_path) as f:
        restaurants = json.load(f)
    
    issues = []
    
    for restaurant in restaurants:
        name = restaurant.get("name", "Unknown")
        vibe = restaurant.get("vibe", {})
        vibe_summary = vibe.get("vibe_summary", "")
        
        if is_generic_vibe_summary(vibe_summary):
            issues.append({
                "name": name,
                "id": restaurant.get("id", "unknown"),
                "vibe_summary": vibe_summary,
                "has_highlights": bool(restaurant.get("highlights")),
                "has_details": bool(restaurant.get("details")),
                "has_editorial": bool(restaurant.get("editorial_summary")),
                "cuisine": restaurant.get("cuisine", "unknown"),
            })
    
    return issues


def main():
    """Main function."""
    data_path = Path(__file__).parent.parent / "data" / "restaurants.json"
    
    if not data_path.exists():
        print(f"❌ File not found: {data_path}")
        return
    
    print(f"🔍 Auditing restaurant vibe summaries...")
    print(f"📁 File: {data_path}\n")
    
    issues = audit_restaurants(str(data_path))
    
    if not issues:
        print("✅ All restaurants have good vibe summaries!")
        return
    
    print(f"⚠️  Found {len(issues)} restaurants with generic/incomplete vibe summaries:\n")
    
    for i, issue in enumerate(issues, 1):
        print(f"{i}. {issue['name']} (ID: {issue['id']})")
        print(f"   Cuisine: {issue['cuisine']}")
        print(f"   Current vibe_summary: \"{issue['vibe_summary']}\"")
        print(f"   Has highlights: {issue['has_highlights']}")
        print(f"   Has details: {issue['has_details']}")
        print(f"   Has editorial: {issue['has_editorial']}")
        print()
    
    print(f"\n💡 Recommendation: Update these restaurants' vibe_summary fields")
    print(f"   Use highlights, details, or editorial_summary as fallback sources")


if __name__ == "__main__":
    main()

