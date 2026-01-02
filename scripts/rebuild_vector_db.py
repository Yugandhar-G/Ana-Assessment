#!/usr/bin/env python3
"""
Script to rebuild the vector database with updated vibe summaries.

This script deletes the existing chroma_db directory and rebuilds the vector store
from restaurants.json, which will use the updated vibe_summary fields.

Usage:
    python scripts/rebuild_vector_db.py
"""

import sys
import asyncio
import shutil
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

from part1.src.gemini_client import AsyncGeminiClient
from part1.src.vector_store import VectorStore, initialize_vector_store

# Load environment variables
env_path = project_root / ".env"
if env_path.exists():
    load_dotenv(env_path)
else:
    load_dotenv()


async def main():
    """Rebuild the vector database."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Rebuild vector database')
    parser.add_argument('--yes', '-y', action='store_true', help='Skip confirmation prompt')
    args = parser.parse_args()
    
    # ChromaDB persist directory is typically in part1/chroma_db
    chroma_db_path = project_root / "part1" / "chroma_db"
    
    print("=" * 60)
    print("Vector Database Rebuild Script")
    print("=" * 60)
    print(f"\nChromaDB path: {chroma_db_path}")
    
    # Check if directory exists
    if chroma_db_path.exists():
        print(f"\n⚠️  Found existing vector database at {chroma_db_path}")
        
        if not args.yes:
            response = input("Delete and rebuild? (yes/no): ").strip().lower()
            if response not in ['yes', 'y']:
                print("❌ Cancelled. Exiting.")
                return
        else:
            print("  (Skipping confirmation with --yes flag)")
        
        print(f"\n🗑️  Deleting {chroma_db_path}...")
        try:
            shutil.rmtree(chroma_db_path)
            print("✅ Deleted successfully")
        except Exception as e:
            print(f"❌ Error deleting directory: {e}")
            return
    else:
        print(f"\n📁 Vector database directory doesn't exist yet (will be created)")
    
    # Check restaurants.json exists
    restaurants_json = project_root / "data" / "restaurants.json"
    if not restaurants_json.exists():
        print(f"\n❌ Error: {restaurants_json} not found!")
        return
    
    print(f"✅ Found restaurants.json at {restaurants_json}")
    
    # Initialize Gemini client
    try:
        print("\n🔌 Initializing Gemini client...")
        client = AsyncGeminiClient()
        print("✅ Gemini client initialized")
    except Exception as e:
        print(f"❌ Error initializing Gemini client: {e}")
        print("Make sure GEMINI_API_KEY is set in your .env file")
        return
    
    # Initialize vector store (this will rebuild it)
    try:
        print(f"\n🔨 Rebuilding vector database from {restaurants_json}...")
        print("   This may take a few minutes depending on the number of restaurants...")
        
        store = await initialize_vector_store(
            persist_dir=str(chroma_db_path),
            gemini_client=client
        )
        
        count = store.get_count()
        print(f"\n✅ Vector database rebuilt successfully!")
        print(f"   Total restaurants: {count}")
        print(f"   Database location: {chroma_db_path}")
        
    except Exception as e:
        print(f"\n❌ Error rebuilding vector database: {e}")
        import traceback
        traceback.print_exc()
        return
    finally:
        await client.close()
    
    print("\n" + "=" * 60)
    print("✅ Rebuild complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())

