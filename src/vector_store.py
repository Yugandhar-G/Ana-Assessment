import json
from pathlib import Path
import chromadb
from chromadb.config import Settings
from openai import OpenAI
from .schemas import Restaurant, RestaurantVibe


class VectorStore:
    """ChromaDB-backed vector store for restaurant vibe embeddings."""
    
    def __init__(self, persist_dir: str | None = None):
        if persist_dir:
            self.client = chromadb.PersistentClient(path=persist_dir)
        else:
            self.client = chromadb.Client()
        
        self.collection = self.client.get_or_create_collection(
            name="restaurant_vibes",
            metadata={"hnsw:space": "cosine"}
        )
        self.openai_client = OpenAI()
    
    def _get_embedding(self, text: str) -> list[float]:
        """Generate embedding using OpenAI."""
        response = self.openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=text,
        )
        return response.data[0].embedding
    
    def add_restaurant(self, restaurant: Restaurant) -> None:
        """Add a restaurant's vibe embedding to the store."""
        embedding = self._get_embedding(restaurant.vibe.vibe_summary)
        
        self.collection.upsert(
            ids=[restaurant.id],
            embeddings=[embedding],
            metadatas=[{
                "name": restaurant.name,
                "cuisine": restaurant.cuisine,
                "price_level": restaurant.price_level,
                "region": restaurant.region,
                "formality": restaurant.vibe.formality,
                "noise_level": restaurant.vibe.noise_level,
            }],
            documents=[restaurant.vibe.vibe_summary],
        )
    
    def add_restaurants(self, restaurants: list[Restaurant]) -> None:
        """Batch add restaurants to the store."""
        if not restaurants:
            return
        
        ids = [r.id for r in restaurants]
        documents = [r.vibe.vibe_summary for r in restaurants]
        
        # Batch embed
        response = self.openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=documents,
        )
        embeddings = [e.embedding for e in response.data]
        
        metadatas = [{
            "name": r.name,
            "cuisine": r.cuisine,
            "price_level": r.price_level,
            "region": r.region,
            "formality": r.vibe.formality,
            "noise_level": r.vibe.noise_level,
        } for r in restaurants]
        
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents,
        )
    
    def search(
        self,
        query: str,
        n_results: int = 10,
        where: dict | None = None,
        where_document: dict | None = None,
    ) -> list[dict]:
        """Search for restaurants by vibe similarity."""
        query_embedding = self._get_embedding(query)
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where,
            where_document=where_document,
            include=["documents", "metadatas", "distances"],
        )
        
        # Convert to list of dicts
        matches = []
        for i in range(len(results["ids"][0])):
            matches.append({
                "id": results["ids"][0][i],
                "document": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
                "similarity": 1 - results["distances"][0][i],  # cosine distance to similarity
            })
        
        return matches
    
    def get_count(self) -> int:
        """Get number of restaurants in the store."""
        return self.collection.count()


def initialize_vector_store(persist_dir: str | None = None) -> VectorStore:
    """Initialize vector store and load restaurant data."""
    store = VectorStore(persist_dir)
    
    # Load restaurants if store is empty
    if store.get_count() == 0:
        data_path = Path(__file__).parent.parent / "data" / "restaurants.json"
        if data_path.exists():
            with open(data_path) as f:
                data = json.load(f)
            
            restaurants = []
            for item in data:
                vibe_data = item.pop("vibe", {})
                item["vibe"] = RestaurantVibe(**vibe_data)
                restaurants.append(Restaurant(**item))
            
            print(f"Indexing {len(restaurants)} restaurants...")
            store.add_restaurants(restaurants)
            print(f"Done! {store.get_count()} restaurants indexed.")
    
    return store

