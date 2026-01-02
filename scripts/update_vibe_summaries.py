#!/usr/bin/env python3
"""
Script to update vibe_summary in restaurants.json using LLM.

This script reads restaurants.json, generates proper vibe_summary for each restaurant
by combining highlights, details, editorial_summary, formality, noise_level, 
atmosphere_tags, and best_for fields, then updates the JSON file.

Usage:
    python scripts/update_vibe_summaries.py [--dry-run] [--batch-size N] [--limit N]
"""

import json
import os
import sys
import asyncio
import argparse
from pathlib import Path
from typing import Dict, List, Optional
from dotenv import load_dotenv

# Add parent directory to path
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

from part1.src.gemini_client import AsyncGeminiClient

# Load environment variables
env_path = project_root / ".env"
if env_path.exists():
    load_dotenv(env_path)
else:
    load_dotenv()

# System prompt for generating vibe summaries
VIBE_SUMMARY_PROMPT = """You are a restaurant atmosphere expert. Your task is to generate a concise, well-structured 3-4 sentence vibe summary for a restaurant based on the provided information.

The vibe summary should:
1. Describe the atmosphere and ambiance (relaxed, elegant, casual, trendy, etc.)
2. Mention the formality level and noise level naturally
3. Include relevant atmosphere tags (romantic, group-friendly, etc.)
4. Mention what it's best for (dinner, groups, date night, etc.)
5. Be natural, engaging, and specific - NOT generic

Guidelines:
- Write in natural, conversational language
- Don't repeat information unnecessarily
- Focus on the EXPERIENCE and ATMOSPHERE
- Avoid award mentions or generic phrases like "offers great food"
- Be specific and descriptive (e.g., "lively and energetic" vs just "good")
- Keep it to 3-4 sentences maximum
- Each sentence should add value

Example good output:
"Sansei offers a vibrant and trendy atmosphere with an upscale yet energetic vibe. The restaurant is known for its lively setting, making it perfect for groups and dinner celebrations. The noise level is typically loud, reflecting the energetic ambiance, and the space is group-friendly with a modern, trendy aesthetic. It's an ideal spot for those seeking an exciting dining experience with creative Japanese-Pacific Rim cuisine."

Example bad output (generic, award-focused):
"'Aipono Honorable Mention for Best Sushi. Award-winning restaurant serving fresh sushi."

Now generate a vibe summary based on the restaurant information provided."""


async def generate_vibe_summary(
    client: AsyncGeminiClient,
    restaurant: Dict,
    restaurant_name: str
) -> Optional[str]:
    """Generate a vibe summary for a restaurant using LLM."""
    
    # Extract all relevant fields
    highlights = restaurant.get('highlights', '').strip()
    details = restaurant.get('details', '').strip()
    editorial_summary = restaurant.get('editorial_summary', '').strip()
    
    vibe = restaurant.get('vibe', {})
    if not isinstance(vibe, dict):
        vibe = {}
    
    formality = vibe.get('formality', '')
    noise_level = vibe.get('noise_level', '')
    atmosphere_tags = vibe.get('atmosphere_tags', [])
    best_for = vibe.get('best_for', [])
    cuisine = restaurant.get('cuisine', '')
    price_level = restaurant.get('price_level', '')
    
    # Build context for LLM
    context_parts = []
    
    if highlights:
        context_parts.append(f"Highlights: {highlights}")
    if details:
        context_parts.append(f"Details: {details}")
    if editorial_summary:
        context_parts.append(f"Editorial Summary: {editorial_summary}")
    
    vibe_info = []
    if formality:
        vibe_info.append(f"Formality: {formality}")
    if noise_level:
        vibe_info.append(f"Noise Level: {noise_level}")
    if atmosphere_tags:
        vibe_info.append(f"Atmosphere Tags: {', '.join(atmosphere_tags)}")
    if best_for:
        vibe_info.append(f"Best For: {', '.join(best_for)}")
    
    if vibe_info:
        context_parts.append(f"Vibe Information: {'; '.join(vibe_info)}")
    
    if cuisine:
        context_parts.append(f"Cuisine: {cuisine}")
    if price_level:
        context_parts.append(f"Price Level: {price_level}")
    
    context = "\n".join(context_parts)
    
    if not context.strip():
        print(f"  ⚠️  No context available for {restaurant_name}, skipping")
        return None
    
    user_prompt = f"""Restaurant Name: {restaurant_name}

{context}

Generate a 3-4 sentence vibe summary for this restaurant."""
    
    try:
        response = await client.chat.completions.create(
            model=client.default_chat_model,
            messages=[
                {"role": "system", "content": VIBE_SUMMARY_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=200,
        )
        
        vibe_summary = response.choices[0].message["content"].strip()
        
        # Clean up the response (remove quotes if LLM added them)
        if vibe_summary.startswith('"') and vibe_summary.endswith('"'):
            vibe_summary = vibe_summary[1:-1]
        if vibe_summary.startswith("'") and vibe_summary.endswith("'"):
            vibe_summary = vibe_summary[1:-1]
        
        return vibe_summary
        
    except Exception as e:
        print(f"  ❌ Error generating vibe summary for {restaurant_name}: {e}")
        return None


async def process_restaurants(
    restaurants: List[Dict],
    client: AsyncGeminiClient,
    dry_run: bool = False,
    batch_size: int = 10,
    limit: Optional[int] = None
) -> Dict[str, str]:
    """Process restaurants and generate vibe summaries."""
    
    updated_summaries = {}
    
    # Limit if specified
    restaurants_to_process = restaurants[:limit] if limit else restaurants
    
    total = len(restaurants_to_process)
    print(f"Processing {total} restaurants...")
    print(f"Dry run: {dry_run}")
    print(f"Batch size: {batch_size}\n")
    
    for i, restaurant in enumerate(restaurants_to_process, 1):
        restaurant_name = restaurant.get('name', f'Restaurant #{i}')
        
        print(f"[{i}/{total}] Processing: {restaurant_name}")
        
        # Check if vibe exists
        if 'vibe' not in restaurant:
            print(f"  ⚠️  No vibe field, skipping")
            continue
        
        if not isinstance(restaurant['vibe'], dict):
            print(f"  ⚠️  Vibe is not a dict, skipping")
            continue
        
        # Generate new vibe summary
        new_summary = await generate_vibe_summary(client, restaurant, restaurant_name)
        
        if new_summary:
            old_summary = restaurant['vibe'].get('vibe_summary', '')
            print(f"  ✅ Generated new summary ({len(new_summary)} chars)")
            if old_summary:
                print(f"     Old: {old_summary[:80]}...")
            print(f"     New: {new_summary[:80]}...")
            
            updated_summaries[restaurant_name] = new_summary
            
            if not dry_run:
                restaurant['vibe']['vibe_summary'] = new_summary
        
        # Rate limiting - wait between batches
        if i % batch_size == 0 and i < total:
            print(f"\n  ⏸️  Pausing after batch of {batch_size}...")
            await asyncio.sleep(2)  # 2 second pause between batches
    
    return updated_summaries


async def main():
    parser = argparse.ArgumentParser(description='Update vibe_summary in restaurants.json')
    parser.add_argument('--dry-run', action='store_true', help='Generate summaries but do not save to file')
    parser.add_argument('--batch-size', type=int, default=10, help='Number of restaurants to process before pausing (default: 10)')
    parser.add_argument('--limit', type=int, help='Limit number of restaurants to process (for testing)')
    parser.add_argument('--backup', action='store_true', default=True, help='Create backup before updating (default: True)')
    
    args = parser.parse_args()
    
    # Load restaurants.json
    json_path = project_root / "data" / "restaurants.json"
    
    if not json_path.exists():
        print(f"Error: {json_path} not found")
        sys.exit(1)
    
    print(f"Loading restaurants from {json_path}...")
    with open(json_path, 'r', encoding='utf-8') as f:
        restaurants = json.load(f)
    
    print(f"Loaded {len(restaurants)} restaurants\n")
    
    # Create backup if not dry-run
    if not args.dry_run and args.backup:
        backup_path = project_root / "data" / "restaurants.json.backup_vibe_update"
        print(f"Creating backup: {backup_path}")
        with open(backup_path, 'w', encoding='utf-8') as f:
            json.dump(restaurants, f, indent=2, ensure_ascii=False)
        print("✅ Backup created\n")
    
    # Initialize Gemini client
    try:
        client = AsyncGeminiClient()
    except Exception as e:
        print(f"Error initializing Gemini client: {e}")
        print("Make sure GEMINI_API_KEY is set in your .env file")
        sys.exit(1)
    
    # Process restaurants
    try:
        updated_summaries = await process_restaurants(
            restaurants,
            client,
            dry_run=args.dry_run,
            batch_size=args.batch_size,
            limit=args.limit
        )
        
        print(f"\n{'='*60}")
        print(f"Summary:")
        print(f"  Total processed: {len(updated_summaries)}")
        print(f"  Successfully generated: {len(updated_summaries)}")
        
        if args.dry_run:
            print(f"\n⚠️  DRY RUN - No changes saved to file")
            print(f"   Run without --dry-run to save changes")
        else:
            # Save updated JSON
            print(f"\nSaving updated restaurants.json...")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(restaurants, f, indent=2, ensure_ascii=False)
            print(f"✅ Saved {len(updated_summaries)} updated vibe summaries")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        if not args.dry_run:
            print("   Partial updates may have been saved")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())

