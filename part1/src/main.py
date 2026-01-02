import asyncio
import argparse
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
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Disable LLM explanation for lower latency (structured scoring only)",
    )
    
    args = parser.parse_args()
    
    # Initialize search
    # In CLI we persist vectors by default for better latency across runs.
    ana = AnaVibeSearch(persist_vectors=True, use_llm=not args.no_llm)
    
    # Execute search
    response = await ana.search(args.query)
    
    if args.json:
        print(response.model_dump_json(indent=2))
    else:
        if response.success:
            print(f"\n Top Match: {response.top_match.name}")
            print(f"   Cuisine: {response.top_match.cuisine} | Price: {response.top_match.price_level}")
            print(f"   Score: {response.top_match.final_score:.2f} | Confidence: {response.confidence}")
            print(f"\n {response.explanation}")
            
            # Display video URLs if available
            if response.top_match.video_urls:
                print(f"\n 📹 Videos available for this restaurant:")
                for video_url in response.top_match.video_urls:
                    print(f"   • {video_url}")
            
            if response.alternatives:
                print(f"\n Also consider:")
                for alt in response.alternatives:
                    print(f"   • {alt.name} ({alt.cuisine}, {alt.price_level}) - {alt.final_score:.2f}")
                    # Display video URLs for alternatives too
                    if alt.video_urls:
                        for video_url in alt.video_urls:
                            print(f"     📹 {video_url}")
        else:
            print(f"\n {response.explanation}")
            if response.caveats:
                for caveat in response.caveats:
                    print(f"{caveat}")


if __name__ == "__main__":
    asyncio.run(main())

