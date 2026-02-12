"""Multi-source web search supporting Google, Reddit, and blogs."""
import os
import httpx
from typing import List, Optional, Dict
from dataclasses import dataclass
from enum import Enum

class SearchSource(Enum):
    GOOGLE = "google"
    REDDIT = "reddit"
    BLOGS = "blogs"
    ALL = "all"

@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    source: str
    score: Optional[float] = None

class MultiSourceWebSearch:
    """Multi-source web search supporting Google, Reddit, and blogs."""
    
    def __init__(
        self,
        google_api_key: Optional[str] = None,
        google_search_engine_id: Optional[str] = None,
        reddit_client_id: Optional[str] = None,
        reddit_client_secret: Optional[str] = None,
        reddit_user_agent: Optional[str] = None,
    ):
        self.google_api_key = google_api_key or os.getenv("GOOGLE_SEARCH_API_KEY")
        self.google_search_engine_id = google_search_engine_id or os.getenv("GOOGLE_SEARCH_ENGINE_ID")
        self.reddit_client_id = reddit_client_id or os.getenv("REDDIT_CLIENT_ID")
        self.reddit_client_secret = reddit_client_secret or os.getenv("REDDIT_CLIENT_SECRET")
        self.reddit_user_agent = reddit_user_agent or os.getenv("REDDIT_USER_AGENT", "AnaAI/1.0")
        self._client = httpx.AsyncClient(timeout=30.0) if (self.google_api_key or self.reddit_client_id) else None
    
    async def search_google(self, query: str, num_results: int = 5) -> List[SearchResult]:
        """Search Google Custom Search API."""
        if not self.google_api_key or not self.google_search_engine_id:
            return []
        
        try:
            response = await self._client.get(
                "https://www.googleapis.com/customsearch/v1",
                params={
                    "key": self.google_api_key,
                    "cx": self.google_search_engine_id,
                    "q": query,
                    "num": min(num_results, 10),
                }
            )
            response.raise_for_status()
            data = response.json()
            
            results = []
            for item in data.get("items", [])[:num_results]:
                results.append(SearchResult(
                    title=item.get("title", ""),
                    url=item.get("link", ""),
                    snippet=item.get("snippet", ""),
                    source="google"
                ))
            return results
        except Exception as e:
            print(f"Google search error: {e}")
            return []
    
    async def search_reddit_via_google(self, query: str, num_results: int = 5) -> List[SearchResult]:
        """Search Reddit via Google Custom Search (no API needed)."""
        if not self.google_api_key or not self.google_search_engine_id:
            return []
        
        try:
            google_query = f"site:reddit.com {query}"
            response = await self._client.get(
                "https://www.googleapis.com/customsearch/v1",
                params={
                    "key": self.google_api_key,
                    "cx": self.google_search_engine_id,
                    "q": google_query,
                    "num": min(num_results, 10),
                }
            )
            response.raise_for_status()
            data = response.json()
            
            results = []
            for item in data.get("items", [])[:num_results]:
                results.append(SearchResult(
                    title=item.get("title", ""),
                    url=item.get("link", ""),
                    snippet=item.get("snippet", ""),
                    source="reddit"
                ))
            return results
        except Exception as e:
            print(f"Reddit search error: {e}")
            return []
    
    async def search_blogs(self, query: str, num_results: int = 5) -> List[SearchResult]:
        """Search blogs using Google Custom Search with site filters."""
        if not self.google_api_key or not self.google_search_engine_id:
            return []
        
        blog_domains = [
            "site:yelp.com",
            "site:tripadvisor.com",
            "site:eater.com",
            "site:timeout.com",
        ]
        
        all_results = []
        for domain in blog_domains[:2]:  # Limit API calls
            try:
                blog_query = f"{domain} {query} Maui Hawaii"
                response = await self._client.get(
                    "https://www.googleapis.com/customsearch/v1",
                    params={
                        "key": self.google_api_key,
                        "cx": self.google_search_engine_id,
                        "q": blog_query,
                        "num": 2,
                    }
                )
                response.raise_for_status()
                data = response.json()
                
                for item in data.get("items", []):
                    all_results.append(SearchResult(
                        title=item.get("title", ""),
                        url=item.get("link", ""),
                        snippet=item.get("snippet", ""),
                        source="blog"
                    ))
            except Exception as e:
                print(f"Blog search error for {domain}: {e}")
                continue
        
        return all_results[:num_results]
    
    async def search(
        self,
        query: str,
        sources: List[SearchSource] = None,
        num_results_per_source: int = 3,
    ) -> List[SearchResult]:
        """Search across multiple sources."""
        if sources is None:
            sources = [SearchSource.GOOGLE]
        
        if SearchSource.ALL in sources:
            sources = [SearchSource.GOOGLE, SearchSource.REDDIT, SearchSource.BLOGS]
        
        all_results = []
        import asyncio
        
        tasks = []
        if SearchSource.GOOGLE in sources:
            tasks.append(self.search_google(query, num_results_per_source))
        if SearchSource.REDDIT in sources:
            tasks.append(self.search_reddit_via_google(query, num_results_per_source))
        if SearchSource.BLOGS in sources:
            tasks.append(self.search_blogs(query, num_results_per_source))
        
        results_list = await asyncio.gather(*tasks, return_exceptions=True)
        
        for results in results_list:
            if isinstance(results, list):
                all_results.extend(results)
            elif isinstance(results, Exception):
                print(f"Search error: {results}")
        
        return all_results
    
    async def close(self):
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
