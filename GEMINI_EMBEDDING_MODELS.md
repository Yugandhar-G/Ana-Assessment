# Gemini Embedding Models for RAG

## Recommended Models

### 1. `models/embedding-001` ⭐ **Recommended for RAG** (also available as `models/gemini-embedding-001`)

**Best choice for restaurant search RAG applications.**

- **Top-tier performance**: State-of-the-art results for RAG
- **Domain-specific excellence**: Outperforms Voyage and OpenAI embeddings in benchmarks
- **Complex queries**: Strong semantic matching for nuanced restaurant descriptions
- **Efficiency**: Optimized for retrieval-augmented generation tasks
- **Use case**: Perfect for restaurant vibe matching, cuisine searches, and complex query understanding

**Usage:**
```bash
export GEMINI_EMBEDDING_MODEL="models/embedding-001"
```

### 2. `models/text-embedding-004`

**General-purpose embedding model (default).**

- Good for general semantic search
- Smaller, faster than embedding-001
- Suitable for simpler search tasks
- Current default in the codebase

**Usage:**
```bash
export GEMINI_EMBEDDING_MODEL="models/text-embedding-004"
```

### 3. `EmbeddingGemma` (Open-source, on-device)

**For privacy-conscious deployments.**

- Open-source alternative
- Smaller model size (good for on-device)
- High-quality embeddings for its size
- Perfect for private, local deployments
- Note: Not currently supported in this codebase (requires different setup)

## Recommendation for Ana AI Restaurant Search

For the Ana AI restaurant search system, we recommend **`models/embedding-001`** because:

1. **Complex queries**: Restaurant searches often involve nuanced descriptions ("romantic spot with ocean views", "authentic Italian with live music")
2. **Vibe matching**: Requires strong semantic understanding of atmosphere, ambiance, and dining experience
3. **Domain-specific**: Restaurant data has domain-specific terminology and concepts
4. **Quality over speed**: Better results justify slightly slower embedding generation

## How to Switch

Update your `.env` file or environment variable:

```bash
# Recommended for best RAG performance
export GEMINI_EMBEDDING_MODEL="models/embedding-001"
```

The code will automatically use the model you specify. No code changes needed!

## Performance Comparison

Based on benchmarks:
- **embedding-001**: Best for complex, domain-specific RAG (legal, technical, restaurant reviews)
- **text-embedding-004**: Good for general semantic search
- **Voyage/OpenAI**: Generally outperformed by embedding-001 for domain-specific tasks

