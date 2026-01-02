import json
import os
import warnings
from pathlib import Path

# Set environment variable to suppress gRPC verbose output BEFORE importing chromadb
# This prevents "grpc_wait_for_shutdown_with_timeout" warnings (harmless but noisy)
if "GRPC_VERBOSITY" not in os.environ:
    os.environ["GRPC_VERBOSITY"] = "ERROR"

import chromadb
from .gemini_client import AsyncGeminiClient
from .schemas import Restaurant, RestaurantVibe

# Suppress ChromaDB gRPC shutdown warnings (harmless but noisy)
# These warnings come from the gRPC C++ library during ChromaDB shutdown
# They don't affect functionality and can be safely ignored
# Note: The "grpc_wait_for_shutdown_with_timeout" warning is a known ChromaDB issue
warnings.filterwarnings("ignore", category=UserWarning, message=".*grpc.*")
# Suppress ChromaDB telemetry and other verbose logging
import logging
logging.getLogger("chromadb").setLevel(logging.WARNING)


class VectorStore:
    """ChromaDB-backed vector store for restaurant vibe embeddings."""
    
    def __init__(self, persist_dir: str | None = None, gemini_client: AsyncGeminiClient | None = None, embedding_model: str | None = None):
        if persist_dir:
            self.client = chromadb.PersistentClient(path=persist_dir)
        else:
            self.client = chromadb.Client()
        
        self.collection = self.client.get_or_create_collection(
            name="restaurant_vibes",
            metadata={"hnsw:space": "cosine"}
        )
        
        # CRITICAL: Use the same embedding model as the client, or the one explicitly provided
        # This ensures query embeddings match document embeddings in the vector store
        if gemini_client:
            self.gemini_client = gemini_client
            # Use the client's embedding model to ensure consistency
            self.embedding_model = embedding_model or gemini_client.default_embedding_model
        else:
            # If no client provided, create one with the specified or default model
            default_model = embedding_model or os.getenv("GEMINI_EMBEDDING_MODEL", "models/text-embedding-004")
            self.gemini_client = AsyncGeminiClient(
                api_key=os.getenv("GEMINI_API_KEY"),
                default_embedding_model=default_model,
            )
            self.embedding_model = default_model
    
    async def _get_embedding(self, text: str) -> list[float]:
        """Generate embedding using Gemini."""
        import logging
        logger = logging.getLogger(__name__)
        
        logger.debug(f"Generating embedding with model: {self.embedding_model}, text length: {len(text)}")
        response = await self.gemini_client.embeddings.create(
            model=self.embedding_model,
            input=text,
        )
        embedding = response.data[0].embedding
        logger.debug(f"Embedding generated: length={len(embedding)}, first_5={embedding[:5] if len(embedding) >= 5 else embedding}")
        return embedding
    
    def _build_document_text(self, restaurant: Restaurant) -> str:
        """Build document text for embedding that includes cuisine, menu items, and other important fields.
        
        This ensures cuisine and menu item information is captured in embeddings, making searches work.
        CRITICAL: Cuisine and menu items are included prominently so embeddings can match queries.
        
        NOTE: The vector store must be rebuilt (delete chroma_db folder) for this to take effect
        on existing restaurants.
        """
        parts = []
        
        # Include cuisine prominently at the start (most important for cuisine searches)
        # Repeat cuisine multiple times to boost it in embeddings
        if restaurant.cuisine:
            # Normalize cuisine for better matching (handle comma-separated, slashes, etc.)
            cuisine_normalized = restaurant.cuisine.replace('/', ' ').replace('-', ' ').lower()
            # Repeat cuisine 3 times for emphasis in embeddings
            cuisine_repeated = f"{cuisine_normalized} " * 3
            parts.append(f"Cuisine: {restaurant.cuisine}. {cuisine_repeated}cuisine restaurant.")
        
        # Include menu items prominently (important for menu item searches)
        if restaurant.top_menu_items:
            menu_items_text = ", ".join(restaurant.top_menu_items)
            # Repeat menu items to boost them in embeddings
            menu_items_repeated = f"{menu_items_text}. " * 2
            parts.append(f"Menu items: {menu_items_text}. {menu_items_repeated}Popular dishes, signature items, specialty food.")
        
        # Include vibe summary (original content)
        if restaurant.vibe.vibe_summary:
            parts.append(restaurant.vibe.vibe_summary)
        
        # Include name (helps with exact matches)
        if restaurant.name:
            parts.append(f"Restaurant: {restaurant.name}.")
        
        return " ".join(parts)
    
    async def add_restaurant(self, restaurant: Restaurant) -> None:
        """Add a restaurant's vibe embedding to the store."""
        text = self._build_document_text(restaurant)
        embedding = await self._get_embedding(text)
        
        # Chroma only allows primitive types in metadata; store a compact subset
        # plus the full restaurant record as a JSON string.
        meta = {
            "id": restaurant.id,
            "name": restaurant.name,
            "cuisine": restaurant.cuisine,
            "price_level": restaurant.price_level,
            "region": restaurant.region,
            "city": restaurant.city,
            "rating": float(restaurant.rating),
            "formality": restaurant.vibe.formality,
            "noise_level": restaurant.vibe.noise_level,
            "latitude": float(restaurant.latitude) if restaurant.latitude is not None else None,
            "longitude": float(restaurant.longitude) if restaurant.longitude is not None else None,
            "restaurant_json": json.dumps(restaurant.model_dump(mode="json")),
        }
        
        self.collection.upsert(
            ids=[restaurant.id],
            embeddings=[embedding],
            metadatas=[meta],
            documents=[text],
        )
    
    async def add_restaurants(self, restaurants: list[Restaurant]) -> None:
        """Batch add restaurants to the store."""
        if not restaurants:
            return
        
        ids = [r.id for r in restaurants]
        documents = [self._build_document_text(r) for r in restaurants]
        response = await self.gemini_client.embeddings.create(
            model=self.embedding_model,
            input=documents,
        )
        embeddings = [e.embedding for e in response.data]
        
        # Store compact primitives + full record as JSON string.
        metadatas = []
        for r in restaurants:
            metadatas.append({
                "id": r.id,
                "name": r.name,
                "cuisine": r.cuisine,
                "price_level": r.price_level,
                "region": r.region,
                "city": r.city,
                "rating": float(r.rating),
                "formality": r.vibe.formality,
                "noise_level": r.vibe.noise_level,
                "latitude": float(r.latitude) if r.latitude is not None else None,
                "longitude": float(r.longitude) if r.longitude is not None else None,
                "restaurant_json": json.dumps(r.model_dump(mode="json")),
            })
        
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents,
        )
    
    async def search(
        self,
        query: str,
        n_results: int = 10,
        where: dict | None = None,
        where_document: dict | None = None,
    ) -> list[dict]:
        """Search for restaurants by vibe similarity."""
        import logging
        logger = logging.getLogger(__name__)
        
        # Log query and collection info for debugging
        logger.debug(f"VectorStore.search: query='{query[:100]}...', n_results={n_results}, collection='{self.collection.name}', count={self.collection.count()}")
        
        query_embedding = await self._get_embedding(query)
        logger.debug(f"Query embedding generated: length={len(query_embedding)}, first_5={query_embedding[:5] if len(query_embedding) >= 5 else query_embedding}")
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where,
            where_document=where_document,
            include=["documents", "metadatas", "distances"],
        )
        
        matches = []
        for i in range(len(results["ids"][0])):
            distance = results["distances"][0][i]
            similarity = 1 - distance
            matches.append({
                "id": results["ids"][0][i],
                "document": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": distance,
                "similarity": similarity,
            })
            # Log first few results for debugging
            if i < 3:
                logger.debug(f"  Result {i+1}: id={results['ids'][0][i]}, distance={distance:.4f}, similarity={similarity:.4f}")
        
        return matches
    
    def get_count(self) -> int:
        """Get number of restaurants in the store."""
        return self.collection.count()


async def initialize_vector_store(persist_dir: str | None = None, gemini_client: AsyncGeminiClient | None = None) -> VectorStore:
    """Initialize vector store and load restaurant data."""
    import logging
    logger = logging.getLogger(__name__)
    
    store = VectorStore(persist_dir, gemini_client=gemini_client)
    count = store.get_count()
    
    logger.info(f"Vector store initialized. Current count: {count}")
    
    if count > 0:
        logger.info(f"   NOTE: If menu item searches aren't working, delete chroma_db folder to rebuild with menu item-enhanced embeddings")
        logger.info(f"Vector store already has {count} restaurants - using existing embeddings")
        logger.info("   NOTE: If cuisine searches aren't working, delete chroma_db folder to rebuild with cuisine-enhanced embeddings")
    else:
        logger.info("Vector store is empty - rebuilding from restaurants.json...")
        # Data directory is at project root, go up 3 levels from part1/src/vector_store.py
        data_path = Path(__file__).parent.parent.parent / "data" / "restaurants.json"
        if data_path.exists():
            logger.info(f"Loading restaurants from: {data_path}")
            with open(data_path) as f:
                data = json.load(f)
            
            logger.info(f"Found {len(data)} restaurants in JSON file")
            restaurants = []
            for item in data:
                vibe_data = item.pop("vibe", {})
                item["vibe"] = RestaurantVibe(**vibe_data)
                restaurants.append(Restaurant(**item))
            
            logger.info(f"Generating embeddings for {len(restaurants)} restaurants...")
            await store.add_restaurants(restaurants)
            logger.info(f"✅ Vector store rebuilt with {store.get_count()} restaurants")
            
            # Log sample document texts to verify they include cuisine
            sample_count = min(3, len(restaurants))
            logger.info(f"\nSample document texts (first {sample_count}):")
            for r in restaurants[:sample_count]:
                doc_text = store._build_document_text(r)
                preview = doc_text[:120] + "..." if len(doc_text) > 120 else doc_text
                logger.info(f"  • {r.name}: {preview}")
        else:
            logger.warning(f"⚠️  restaurants.json not found at {data_path}")
    
    return store

