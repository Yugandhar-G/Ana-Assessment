import asyncio
import argparse
import json
from .pipeline import AnaVibeSearch


async def main():
    parser = argparse.ArgumentParser(description="Ana AI - Vibe-Based Restaurant Search")
    parser.add_argument(
        "--query", "-q",
        type=str,
        required=True,
        help="Natural language query for restaurant search"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON response"
    )
    
    args = parser.parse_args()
    
    # Initialize search
    ana = AnaVibeSearch()
    
    # Execute search
    response = await ana.search(args.query)
    
    if args.json:
        print(response.model_dump_json(indent=2))
    else:
        # Pretty print
        if response.success:
            print(f"\n🍽️  Top Match: {response.top_match.name}")
            print(f"   Cuisine: {response.top_match.cuisine} | Price: {response.top_match.price_level}")
            print(f"   Score: {response.top_match.final_score:.2f} | Confidence: {response.confidence}")
            print(f"\n💬 {response.explanation}")
            
            if response.alternatives:
                print(f"\n📋 Also consider:")
                for alt in response.alternatives:
                    print(f"   • {alt.name} ({alt.cuisine}, {alt.price_level}) - {alt.final_score:.2f}")
        else:
            print(f"\n❌ {response.explanation}")
            if response.caveats:
                for caveat in response.caveats:
                    print(f"   ⚠️ {caveat}")


if __name__ == "__main__":
    asyncio.run(main())

