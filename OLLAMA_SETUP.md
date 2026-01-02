# Ollama Setup Guide

This project uses **Ollama** for local LLM inference instead of OpenAI API calls. This means:
- ✅ No API costs
- ✅ No internet required (after initial model download)
- ✅ Complete privacy (data never leaves your machine)
- ✅ Full control over models

## Installation

### 1. Install Ollama

Visit [https://ollama.ai](https://ollama.ai) and download Ollama for your platform:
- **macOS**: Download the app or use `brew install ollama`
- **Linux**: `curl -fsSL https://ollama.ai/install.sh | sh`
- **Windows**: Download the installer

### 2. Pull Required Models

You need two models:

```bash
# Chat model (for query parsing and response generation)
# Recommended options:
ollama pull mistral                  # Fast, efficient, great quality
# OR
ollama pull qwen2.5                  # High quality, multilingual support
# OR
ollama pull llama3                   # Good balance of quality and speed

# Embedding model (for semantic search)
ollama pull nomic-embed-text
```

**Chat model recommendations:**

- **`mistral`** ⭐ **Recommended** - Fast, efficient, excellent quality for structured tasks
- **`qwen2.5`** - High quality, multilingual, great for complex queries
- **`qwen2.5:7b`** - Smaller version of qwen2.5, faster
- **`llama3`** - Good balance, widely used
- **`llama3.2`** - Newer, faster version of llama3

**Embedding models:**
- `nomic-embed-text` - Good quality, 768 dimensions (recommended)
- `all-minilm` - Smaller, faster
- `mxbai-embed-large` - Higher quality, larger

### 3. Verify Installation

```bash
# Check Ollama is running
ollama list

# Test a model
ollama run llama3 "Hello, how are you?"
```

## Configuration

### Environment Variables

Create a `.env` file (or set environment variables):

```bash
# Ollama server URL (default: http://localhost:11434)
OLLAMA_BASE_URL=http://localhost:11434

# Chat model for query parsing and response generation
# Options: mistral, qwen2.5, llama3, llama3.2
OLLAMA_CHAT_MODEL=mistral

# Embedding model for semantic search
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
```

### Custom Ollama Server

If you're running Ollama on a different machine or port:

```bash
export OLLAMA_BASE_URL=http://your-server:11434
```

## Usage

Once Ollama is installed and models are pulled, the system will automatically use Ollama:

```bash
python -m src.main --query "romantic spot with ocean views"
```

The first run will:
1. Index all restaurants (creates embeddings using Ollama)
2. This may take a few minutes depending on your hardware
3. Subsequent runs will be faster (embeddings are cached)

## Troubleshooting

### "Connection refused" error

**Problem:** Can't connect to Ollama server

**Solution:**
```bash
# Make sure Ollama is running
ollama serve

# Or start it in the background
ollama serve &
```

### "Model not found" error

**Problem:** Model hasn't been pulled yet

**Solution:**
```bash
# Pull the required model
ollama pull llama3
ollama pull nomic-embed-text
```

### Slow performance

**Problem:** Models running slowly

**Solutions:**
1. Use smaller/faster models:
   ```bash
   export OLLAMA_CHAT_MODEL=mistral        # Already fast, or use qwen2.5:7b
   export OLLAMA_EMBEDDING_MODEL=all-minilm
   ```

2. Ensure you have enough RAM (models need 4-8GB+)

3. Use GPU acceleration if available (Ollama auto-detects CUDA/Metal)

### Embedding dimension mismatch

**Problem:** Different embedding models have different dimensions

**Solution:** If you switch embedding models, delete the vector store:
```bash
rm -rf chroma_db/
# The system will re-index on next run
```

## Model Recommendations

### For Development/Testing (Fast)
- Chat: `mistral` or `qwen2.5:7b` (fast, efficient)
- Embeddings: `all-minilm` (smaller, faster)

### For Production Quality ⭐ Recommended
- Chat: **`mistral`** - Fast, efficient, excellent for structured JSON parsing
- Chat: **`qwen2.5`** - High quality, great for complex queries, multilingual
- Embeddings: `nomic-embed-text` (good quality, 768 dimensions)

### For Maximum Quality
- Chat: `qwen2.5:32b` or `qwen2.5:14b` (requires more RAM, best quality)
- Embeddings: `mxbai-embed-large` (larger, better quality)

### Quick Comparison

| Model | Speed | Quality | Best For |
|-------|-------|---------|----------|
| `mistral` | ⚡⚡⚡ | ⭐⭐⭐⭐ | **Recommended** - Fast + quality |
| `qwen2.5` | ⚡⚡ | ⭐⭐⭐⭐⭐ | Complex queries, multilingual |
| `qwen2.5:7b` | ⚡⚡⚡ | ⭐⭐⭐⭐ | Faster version of qwen2.5 |
| `llama3` | ⚡⚡ | ⭐⭐⭐⭐ | Good balance |
| `llama3.2` | ⚡⚡⚡ | ⭐⭐⭐ | Fastest, slightly lower quality |

## Performance Tips

1. **First run is slow** - Initial indexing creates embeddings for all restaurants
2. **Subsequent runs are fast** - Embeddings are cached in ChromaDB
3. **GPU acceleration** - Ollama automatically uses GPU if available
4. **Model size** - Larger models = better quality but slower inference

## Switching Back to OpenAI (Optional)

If you want to use OpenAI instead, you can modify the code to use OpenAI's client. The architecture supports both.

