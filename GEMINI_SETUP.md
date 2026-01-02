# Gemini Setup Guide

This guide explains how to set up Google Gemini API for Ana AI instead of Ollama.

## Prerequisites

1. Google Cloud Account (or Google AI Studio account)
2. Gemini API Key

## Getting API Keys

### 1. Gemini API Key

1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy your API key

### 2. Google Custom Search API (Optional - for Web Search)

If you want to enable web search functionality:

1. Visit [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing one
3. Enable "Custom Search API"
4. Create credentials (API Key)
5. Visit [Custom Search Engine](https://programmablesearchengine.google.com/controlpanel/create)
6. Create a search engine (can search entire web)
7. Get your Search Engine ID

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# The google-generativeai package is already included in requirements.txt
```

## Environment Variables

Create a `.env` file in the project root or set environment variables:

```bash
# Required
export GEMINI_API_KEY="your-gemini-api-key-here"

# Optional - Model Selection
export GEMINI_CHAT_MODEL="gemini-2.0-flash"  # or "gemini-2.5-pro", "gemini-2.5-flash", "gemini-pro"
export GEMINI_EMBEDDING_MODEL="models/embedding-001"  # Recommended for RAG - better performance for complex queries
# Alternative: export GEMINI_EMBEDDING_MODEL="models/text-embedding-004"

# Optional - Web Search (if you want to enable web search)
export GOOGLE_SEARCH_API_KEY="your-google-search-api-key"
export GOOGLE_SEARCH_ENGINE_ID="your-search-engine-id"
```

## Model Options

### Chat Models
- `gemini-2.0-flash` (default) - Fast and efficient
- `gemini-2.5-flash` - Latest fast model
- `gemini-2.5-pro` - Most capable model
- `gemini-2.0-flash-exp` - Experimental features

### Embedding Models
- `models/text-embedding-004` (default) - Good general-purpose embeddings
- `models/embedding-001` ⭐ **Recommended for RAG** - Top-tier for complex, domain-specific data, outperforms Voyage and OpenAI in benchmarks
- `models/embedding-001` - Better semantic matching and efficiency for restaurant search RAG

## Usage

The code automatically uses Gemini when `GEMINI_API_KEY` is set. No code changes needed - the `AsyncOllamaClient` interface now uses Gemini under the hood.

```bash
# Run Part 1
python -m part1.src.main --query "romantic spot with ocean views"

```

## Web Search Integration

Web search is automatically available if you set `GOOGLE_SEARCH_API_KEY` and `GOOGLE_SEARCH_ENGINE_ID`.

The web search functionality can be accessed via the client:

```python
from part1.src.gemini_client import AsyncGeminiClient

client = AsyncGeminiClient(enable_web_search=True)
results = await client.web_search.search("best restaurants in Maui")
```

## Migration from Ollama

If you were using Ollama before:

1. Install Google Generative AI SDK: `pip install google-generativeai`
2. Set `GEMINI_API_KEY` environment variable
3. Remove Ollama-related environment variables (`OLLAMA_BASE_URL`, `OLLAMA_CHAT_MODEL`, etc.)
4. That's it! The code automatically uses Gemini now.

## Cost Considerations

- Gemini API has a free tier with generous limits
- `gemini-1.5-flash` is very cost-effective
- Embeddings are free up to certain limits
- Check [Google AI Pricing](https://ai.google.dev/pricing) for details

## Troubleshooting

### "GEMINI_API_KEY environment variable is required"
- Make sure you've set the `GEMINI_API_KEY` environment variable
- Check that your API key is valid

### "Error calling Gemini API"
- Verify your API key is correct
- Check your internet connection
- Ensure you have API access enabled in Google Cloud Console

### Embeddings not working
- Make sure you're using a valid embedding model name
- Check that `text-embedding-004` is available in your region
- Verify API key has access to embedding models

