"""Gemini client adapter with web search support."""
import json
import os
from typing import Optional, List, Dict, AsyncGenerator
from dataclasses import dataclass
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import httpx


@dataclass
class ChatCompletionChoice:
    index: int
    message: Dict[str, str]
    finish_reason: Optional[str] = None


@dataclass
class ChatCompletion:
    id: str
    choices: List[ChatCompletionChoice]
    model: str
    usage: Optional[Dict[str, int]] = None


@dataclass
class EmbeddingData:
    embedding: List[float]
    index: int


@dataclass
class EmbeddingResponse:
    data: List[EmbeddingData]
    model: str
    usage: Optional[Dict[str, int]] = None


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str


class WebSearch:
    """Web search functionality using Google Custom Search API."""
    
    def __init__(self, api_key: Optional[str] = None, search_engine_id: Optional[str] = None):
        self.api_key = api_key or os.getenv("GOOGLE_SEARCH_API_KEY")
        self.search_engine_id = search_engine_id or os.getenv("GOOGLE_SEARCH_ENGINE_ID")
        self._client = httpx.AsyncClient(timeout=30.0) if self.api_key and self.search_engine_id else None
    
    async def search(self, query: str, num_results: int = 5) -> List[SearchResult]:
        """Perform web search and return results."""
        if not self._client:
            return []  # Web search not configured
        
        try:
            response = await self._client.get(
                "https://www.googleapis.com/customsearch/v1",
                params={
                    "key": self.api_key,
                    "cx": self.search_engine_id,
                    "q": query,
                    "num": min(num_results, 10),  # Google API max is 10 per request
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
                ))
            return results
        except Exception as e:
            print(f"Web search error: {e}")
            return []
    
    async def close(self):
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()


class AsyncGeminiClient:
    """Async Gemini client for chat completions and embeddings."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        default_chat_model: str = "gemini-2.0-flash",
        default_embedding_model: str = "models/text-embedding-004",  # For RAG: use "models/embedding-001" or "models/gemini-embedding-001" (better performance)
        enable_web_search: bool = True,
        search_api_key: Optional[str] = None,
        search_engine_id: Optional[str] = None,
    ):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY environment variable is required. "
                "Get your API key from https://makersuite.google.com/app/apikey"
            )
        
        genai.configure(api_key=self.api_key)
        
        self.default_chat_model = default_chat_model
        self.default_embedding_model = default_embedding_model
        self.web_search = WebSearch(search_api_key, search_engine_id) if enable_web_search else None
        
        # Safety settings - allow more content for restaurant recommendations
        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        }
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.web_search:
            await self.web_search.close()
    
    async def close(self):
        """Close clients."""
        if self.web_search:
            await self.web_search.close()
    
    @property
    def chat(self):
        """Chat completions interface matching OpenAI's structure."""
        return ChatInterface(self)
    
    @property
    def embeddings(self):
        """Embeddings interface."""
        return Embeddings(self)


class ChatInterface:
    """Chat interface that provides .completions to match OpenAI API."""
    
    def __init__(self, client: AsyncGeminiClient):
        self.completions = ChatCompletions(client)


class ChatCompletions:
    """Chat completions interface matching OpenAI's API."""
    
    def __init__(self, client: AsyncGeminiClient):
        self.client = client
    
    def _convert_messages_to_gemini(self, messages: List[Dict[str, str]]) -> tuple[str, str]:
        """Convert OpenAI-format messages to Gemini format.
        
        Gemini uses system_instruction for system messages and a single content string
        that combines all user/assistant messages. For simple cases, we combine user messages.
        """
        system_instruction = ""
        user_content_parts = []
        
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            
            if role == "system":
                if system_instruction:
                    system_instruction += "\n\n" + content
                else:
                    system_instruction = content
            elif role == "user":
                user_content_parts.append(content)
            # Note: We skip assistant messages in history for simplicity
            # For multi-turn conversations, you'd want to use chat sessions
        
        # Combine all user messages
        user_content = "\n\n".join(user_content_parts) if user_content_parts else ""
        return system_instruction, user_content
    
    async def create(
        self,
        model: Optional[str] = None,
        messages: List[Dict[str, str]] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> ChatCompletion:
        """Create a chat completion using Gemini."""
        model_name = model or self.client.default_chat_model
        
        system_instruction, user_content = self._convert_messages_to_gemini(messages or [])
        
        if not user_content:
            raise ValueError("At least one user message is required")
        
        # Handle JSON response format
        if response_format and response_format.get("type") == "json_object":
            if system_instruction:
                system_instruction += "\n\nIMPORTANT: You must respond with valid JSON only. Do not include any text outside the JSON object."
            else:
                system_instruction = "You must respond with valid JSON only. Do not include any text outside the JSON object."
        
        # Create Gemini model
        gemini_model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=system_instruction if system_instruction else None,
            safety_settings=self.client.safety_settings,
        )
        
        # Generate content
        generation_config = genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        
        import asyncio
        
        def _generate_sync():
            """Synchronous generation call (Gemini SDK doesn't have native async yet)."""
            try:
                response = gemini_model.generate_content(
                    user_content,
                    generation_config=generation_config,
                )
                return response
            except Exception as e:
                raise ConnectionError(
                    f"Error calling Gemini API: {str(e)}\n"
                    f"Make sure GEMINI_API_KEY is set correctly."
                ) from e
        
        # Run in executor to make it non-blocking (with timeout)
        loop = asyncio.get_event_loop()
        try:
            response = await asyncio.wait_for(
                loop.run_in_executor(None, _generate_sync),
                timeout=30.0  # 30 second timeout for API calls (optimized for speed)
            )
        except asyncio.TimeoutError:
            raise ConnectionError(
                "Gemini API call timed out after 60 seconds. "
                "The API might be slow or overloaded. Please try again."
            )
        
        message_content = response.text
        
        # Clean up JSON if needed
        if response_format and response_format.get("type") == "json_object":
            if "```json" in message_content:
                start = message_content.find("```json") + 7
                end = message_content.find("```", start)
                message_content = message_content[start:end].strip()
            elif "```" in message_content:
                start = message_content.find("```") + 3
                end = message_content.find("```", start)
                message_content = message_content[start:end].strip()
        
        # Estimate token usage (Gemini doesn't provide exact counts in some cases)
        prompt_tokens = len(user_content.split()) * 1.3  # Rough estimate
        completion_tokens = len(message_content.split()) * 1.3
        
        return ChatCompletion(
            id=f"chatcmpl-{hash(str(messages))}",
            model=model_name,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message={"role": "assistant", "content": message_content},
                    finish_reason="stop",
                )
            ],
            usage={
                "prompt_tokens": int(prompt_tokens),
                "completion_tokens": int(completion_tokens),
                "total_tokens": int(prompt_tokens + completion_tokens),
            }
        )
    
    async def create_stream(
        self,
        model: Optional[str] = None,
        messages: List[Dict[str, str]] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AsyncGenerator[Dict[str, any], None]:
        """Create a streaming chat completion."""
        model_name = model or self.client.default_chat_model
        
        system_instruction, user_content = self._convert_messages_to_gemini(messages or [])
        
        if not user_content:
            raise ValueError("At least one user message is required")
        
        gemini_model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=system_instruction if system_instruction else None,
            safety_settings=self.client.safety_settings,
        )
        
        generation_config = genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        
        import asyncio
        
        def _generate_stream_sync():
            """Synchronous streaming generation."""
            try:
                response = gemini_model.generate_content(
                    user_content,
                    generation_config=generation_config,
                    stream=True,
                )
                return list(response)  # Convert to list since we can't async iterate
            except Exception as e:
                raise ConnectionError(
                    f"Error calling Gemini API: {str(e)}\n"
                    f"Make sure GEMINI_API_KEY is set correctly."
                ) from e
        
        # Run in executor and then yield chunks (with timeout)
        loop = asyncio.get_event_loop()
        try:
            chunks = await asyncio.wait_for(
                loop.run_in_executor(None, _generate_stream_sync),
                timeout=30.0  # 30 second timeout for streaming API calls (optimized for speed)
            )
        except asyncio.TimeoutError:
            raise ConnectionError(
                "Gemini API streaming call timed out after 60 seconds. "
                "The API might be slow or overloaded. Please try again."
            )
        
        for chunk in chunks:
            yield {
                "content": chunk.text if hasattr(chunk, 'text') else "",
                "done": False,
            }
        
        yield {
            "content": "",
            "done": True,
        }


class Embeddings:
    """Embeddings interface using Gemini embedding models."""
    
    def __init__(self, client: AsyncGeminiClient):
        self.client = client
        import asyncio
        self._semaphore = asyncio.Semaphore(10)  # Higher concurrency for API-based embeddings
    
    async def create(
        self,
        model: Optional[str] = None,
        input: str | List[str] = None,
        **kwargs
    ) -> EmbeddingResponse:
        """Create embeddings using Gemini embedding models."""
        model_name = model or self.client.default_embedding_model
        inputs = [input] if isinstance(input, str) else input
        
        if len(inputs) == 1:
            text = inputs[0]
            embedding = await self._embed_single(model_name, text)
            return EmbeddingResponse(
                data=[EmbeddingData(embedding=embedding, index=0)],
                model=model_name,
                usage={
                    "prompt_tokens": len(text.split()),
                    "total_tokens": len(text.split()),
                }
            )
        
        # Batch processing
        import asyncio
        
        async def _embed_with_semaphore(model: str, text: str, index: int) -> EmbeddingData:
            async with self._semaphore:
                embedding = await self._embed_single(model, text)
                return EmbeddingData(embedding=embedding, index=index)
        
        tasks = [
            _embed_with_semaphore(model_name, text, idx)
            for idx, text in enumerate(inputs)
        ]
        
        embeddings_data = await asyncio.gather(*tasks)
        embeddings_data.sort(key=lambda x: x.index)
        
        return EmbeddingResponse(
            data=embeddings_data,
            model=model_name,
            usage={
                "prompt_tokens": sum(len(text.split()) for text in inputs),
                "total_tokens": sum(len(text.split()) for text in inputs),
            }
        )
    
    async def _embed_single(self, model: str, text: str) -> List[float]:
        """Embed a single text using Gemini embedding model."""
        import asyncio
        
        # Ensure model name has 'models/' prefix
        if not model.startswith("models/") and not model.startswith("tunedModels/"):
            model = f"models/{model}"
        
        def _embed_sync(model_name: str, text_content: str) -> List[float]:
            """Synchronous embedding call (Gemini SDK doesn't have async yet)."""
            try:
                # Use retrieval_document for consistency with how documents were embedded
                # For queries, we could use retrieval_query, but documents were embedded with retrieval_document
                # So we'll use retrieval_document for both to ensure compatibility
                result = genai.embed_content(
                    model=model_name,
                    content=text_content,
                    task_type="retrieval_document",  # Match the task_type used for document embeddings
                )
                return result["embedding"]
            except Exception as e:
                raise ConnectionError(
                    f"Error calling Gemini embedding API: {str(e)}\n"
                    f"Make sure GEMINI_API_KEY is set correctly."
                ) from e
        
        # Run in executor to make it non-blocking (with timeout)
        loop = asyncio.get_event_loop()
        try:
            return await asyncio.wait_for(
                loop.run_in_executor(None, _embed_sync, model, text),
                timeout=15.0  # 15 second timeout for embedding calls (optimized for speed)
            )
        except asyncio.TimeoutError:
            raise ConnectionError(
                "Gemini embedding API call timed out after 15 seconds. "
                "Please try again."
            )



