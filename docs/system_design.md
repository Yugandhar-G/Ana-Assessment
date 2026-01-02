# Ana AI - System Design Document

## Part 3: Production Architecture for Mood-Based Restaurant Retrieval

**Author:** Yugandhar Gopu  
**Date:** December 2024

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [System Architecture](#2-system-architecture)
3. [Stack Choices & Justification](#3-stack-choices--justification)
4. [Latency vs. Cost Optimization](#4-latency-vs-cost-optimization)
5. [Data Freshness Strategy](#5-data-freshness-strategy)
6. [Scaling Considerations](#6-scaling-considerations)
7. [Monitoring & Observability](#7-monitoring--observability)
8. [Security & Privacy](#8-security--privacy)
9. [Cost Analysis](#9-cost-analysis)
10. [Risk Mitigation](#10-risk-mitigation)

---

## 1. Executive Summary

Ana AI is a mood-based restaurant recommendation system for Hawaii. Unlike traditional search that relies on explicit filters, Ana understands natural language queries like *"somewhere that feels like a Bangkok night market—noisy, casual, not fancy"* and returns semantically relevant matches with explanations.

### Key Design Goals

| Goal | Target | Approach |
|------|--------|----------|
| **Latency** | < 2 seconds p95 | Tiered retrieval, caching, async processing |
| **Cost** | < $0.01 per query | Small models for simple queries, batched embeddings |
| **Accuracy** | > 85% user satisfaction | Hybrid retrieval, explainable results |
| **Freshness** | < 24 hour menu lag | Event-driven updates, scheduled crawls |
| **Scale** | 10K queries/day initial | Horizontal scaling, edge caching |

### Architecture at a Glance

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CLIENT LAYER                                   │
│                    (Mobile App / Web / Conversational UI)                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              API GATEWAY                                    │
│                 (Rate Limiting, Auth, Request Routing)                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                    ┌─────────────────┼─────────────────┐
                    ▼                 ▼                 ▼
            ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
            │  Query      │   │  Search     │   │  Response   │
            │  Service    │   │  Service    │   │  Service    │
            └─────────────┘   └─────────────┘   └─────────────┘
                    │                 │                 │
                    └─────────────────┼─────────────────┘
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              DATA LAYER                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │ PostgreSQL   │  │ Qdrant       │  │ Redis        │  │ S3           │   │
│  │ (Metadata)   │  │ (Vectors)    │  │ (Cache)      │  │ (Media)      │   │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. System Architecture

### 2.1 High-Level Flow

```
User Query: "romantic spot with ocean views, not too loud"
                                │
                                ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                        1. QUERY CLASSIFICATION                            │
│                                                                           │
│  Fast classifier determines query complexity:                             │
│  • SIMPLE: "Thai food near Lahaina" → Direct DB lookup                   │
│  • COMPLEX: "night market vibe" → Full hybrid pipeline                   │
│  • AMBIGUOUS: "good food" → Clarification needed                         │
│                                                                           │
│  Model: Fine-tuned classifier or GPT-4o-mini (< 100ms)                   │
└───────────────────────────────────────────────────────────────────────────┘
                                │
                    ┌───────────┴───────────┐
                    ▼                       ▼
            ┌───────────────┐       ┌───────────────┐
            │ SIMPLE PATH   │       │ COMPLEX PATH  │
            │ (Fast Track)  │       │ (Full Pipeline)│
            │               │       │               │
            │ • Cache check │       │ • Query Parse │
            │ • Direct SQL  │       │ • Hard Filter │
            │ • Template    │       │ • Vector Search│
            │   response    │       │ • Score Fusion│
            │               │       │ • LLM Response│
            │ ~200ms        │       │ ~1.5s         │
            └───────────────┘       └───────────────┘
                    │                       │
                    └───────────┬───────────┘
                                ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                        RESPONSE ASSEMBLY                                  │
│                                                                           │
│  • Structured JSON for frontend                                           │
│  • Natural language explanation                                           │
│  • Match reasons with confidence                                          │
│  • Alternative suggestions                                                │
└───────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Service Breakdown

#### Query Service
- Receives raw user input
- Classifies query complexity
- Routes to appropriate pipeline
- Handles clarification dialogs

#### Search Service
- Executes hybrid retrieval
- Manages vector store queries
- Applies metadata filters
- Performs score fusion

#### Response Service
- Generates natural language explanations
- Formats structured output
- Applies personalization (future)
- Logs for analytics

#### Data Ingestion Service (Background)
- Crawls restaurant data sources
- Enriches with vibe profiles
- Generates/updates embeddings
- Maintains data freshness

---

## 3. Stack Choices & Justification

### 3.1 LLM Selection

| Use Case | Model | Rationale |
|----------|-------|-----------|
| **Query Parsing** | GPT-4o-mini | Fast (< 500ms), cheap ($0.15/1M input), excellent at structured extraction |
| **Response Generation** | GPT-4o-mini | Sufficient quality for explanations, consistent output |
| **Vibe Enrichment** (batch) | GPT-4o | Higher quality for offline enrichment, cost amortized over time |
| **Query Classification** | Fine-tuned DistilBERT | < 50ms, runs locally, no API cost |

**Why not Claude?**
- Claude Sonnet excels at nuanced reasoning but adds ~$3/1M tokens
- For structured parsing and template-based responses, GPT-4o-mini matches quality at 5% of the cost
- Would consider Claude for complex multi-turn conversations in future

**Why not open-source (Llama, Mistral)?**
- Adds infrastructure complexity (GPU hosting)
- Quality gap for structured extraction tasks
- May revisit when Llama 3.2 improves function calling

### 3.2 Embedding Model

| Option | Dimensions | Quality | Cost | Choice |
|--------|------------|---------|------|--------|
| text-embedding-3-small | 1536 | Good | $0.02/1M | **Selected** |
| text-embedding-3-large | 3072 | Best | $0.13/1M | Overkill for vibes |
| Cohere embed-v3 | 1024 | Good | Free tier | Backup option |
| BGE-large (self-hosted) | 1024 | Good | Infra cost | Future consideration |

**Rationale:** text-embedding-3-small provides excellent semantic matching for vibe queries at minimal cost. The 1536 dimensions capture sufficient nuance for restaurant atmosphere descriptions.

### 3.3 Vector Database

| Option | Pros | Cons | Verdict |
|--------|------|------|---------|
| **Qdrant** | Fast, great filtering, generous free tier, Rust performance | Newer, smaller community | **Selected** |
| Pinecone | Managed, scalable, easy | Vendor lock-in, cost at scale | Good alternative |
| Weaviate | Native hybrid search, GraphQL | Heavier, more complex | Overengineered |
| pgvector | SQL + vectors unified | Slower for large-scale vector ops | Good for MVP |
| Chroma | Simple, local | Not production-ready at scale | Dev/test only |

**Why Qdrant:**
1. **Native filtering** — Can filter by metadata during vector search (not post-filter)
2. **Performance** — Written in Rust, handles 1M+ vectors efficiently
3. **Cost** — Free tier sufficient for initial scale, predictable pricing
4. **Hybrid support** — Sparse + dense vectors for keyword + semantic
5. **Self-hostable** — Can run on own infrastructure if needed

### 3.4 Primary Database

| Option | Use Case | Verdict |
|--------|----------|---------|
| **PostgreSQL** | Restaurant metadata, user data, analytics | **Selected** |
| MongoDB | Flexible schema | Unnecessary complexity |
| DynamoDB | High-scale key-value | Overkill, vendor lock-in |

**Why PostgreSQL:**
- Mature, reliable, well-understood
- JSONB for flexible vibe profiles
- PostGIS for future location-based features
- Excellent tooling and hosting options (Supabase, RDS, Neon)

### 3.5 Caching Layer

| Layer | Technology | Purpose |
|-------|------------|---------|
| **Query Cache** | Redis | Cache parsed queries and responses |
| **Embedding Cache** | Redis | Avoid re-embedding identical queries |
| **CDN** | Cloudflare | Static assets, API response caching |

### 3.6 Complete Stack Summary

```
┌─────────────────────────────────────────────────────────────────┐
│                        PRODUCTION STACK                         │
├─────────────────────────────────────────────────────────────────┤
│  API Layer        │  FastAPI (Python 3.11+)                     │
│  LLM Provider     │  OpenAI (GPT-4o-mini)                       │
│  Embeddings       │  OpenAI text-embedding-3-small              │
│  Vector Store     │  Qdrant Cloud (or self-hosted)              │
│  Primary DB       │  PostgreSQL 15+ (Supabase/Neon)             │
│  Cache            │  Redis (Upstash or ElastiCache)             │
│  Queue            │  Redis Streams or SQS                       │
│  Hosting          │  AWS ECS / Railway / Fly.io                 │
│  Monitoring       │  Datadog / Grafana + Prometheus             │
│  Logging          │  Structured JSON → CloudWatch/Loki          │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. Latency vs. Cost Optimization

### 4.1 Latency Budget Breakdown

**Target: < 2 seconds end-to-end (p95)**

```
┌─────────────────────────────────────────────────────────────────┐
│                    LATENCY BUDGET (COMPLEX QUERY)               │
├─────────────────────────────────────────────────────────────────┤
│  Network (client → API)              │  ~100ms                  │
│  Query Classification                │  ~50ms   (local model)   │
│  Cache Check                         │  ~10ms   (Redis)         │
│  Query Parsing (LLM)                 │  ~400ms  (GPT-4o-mini)   │
│  Hard Filtering (PostgreSQL)         │  ~50ms                   │
│  Vector Search (Qdrant)              │  ~100ms                  │
│  Score Fusion                        │  ~20ms   (in-memory)     │
│  Response Generation (LLM)           │  ~500ms  (GPT-4o-mini)   │
│  Response Assembly                   │  ~20ms                   │
│  Network (API → client)              │  ~100ms                  │
├─────────────────────────────────────────────────────────────────┤
│  TOTAL                               │  ~1,350ms                │
│  Buffer for variance                 │  ~650ms                  │
│  TARGET                              │  < 2,000ms               │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 Optimization Strategies

#### Strategy 1: Tiered Query Routing

```python
class QueryRouter:
    """Route queries to appropriate pipeline based on complexity."""

    async def route(self, query: str) -> str:
        complexity = await self.classify(query)

        if complexity == "SIMPLE":
            # Direct database lookup, skip LLM parsing
            # "Thai food in Lahaina" → SQL query
            return "fast_path"

        elif complexity == "COMPLEX":
            # Full hybrid pipeline
            # "night market vibe, not fancy" → LLM + vectors
            return "full_pipeline"

        elif complexity == "AMBIGUOUS":
            # Request clarification
            # "good food" → "What cuisine or vibe are you looking for?"
            return "clarification"
```

**Impact:** 40% of queries can use fast path (< 300ms)

#### Strategy 2: Aggressive Caching

```
┌─────────────────────────────────────────────────────────────────┐
│                      CACHING STRATEGY                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Layer 1: Query Embedding Cache (Redis)                         │
│  ├─ Key: hash(normalized_query)                                 │
│  ├─ Value: embedding vector                                     │
│  ├─ TTL: 24 hours                                               │
│  └─ Hit rate: ~30% (common vibe phrases)                        │
│                                                                 │
│  Layer 2: Parsed Query Cache (Redis)                            │
│  ├─ Key: hash(query)                                            │
│  ├─ Value: ParsedQuery JSON                                     │
│  ├─ TTL: 1 hour                                                 │
│  └─ Hit rate: ~20% (repeated queries)                           │
│                                                                 │
│  Layer 3: Full Response Cache (Redis)                           │
│  ├─ Key: hash(query + location + filters)                       │
│  ├─ Value: Complete AnaResponse                                 │
│  ├─ TTL: 15 minutes (freshness tradeoff)                        │
│  └─ Hit rate: ~15% (popular queries)                            │
│                                                                 │
│  Layer 4: Restaurant Data Cache (In-memory)                     │
│  ├─ Hot restaurant profiles in application memory               │
│  ├─ Refreshed every 5 minutes                                   │
│  └─ Eliminates DB round-trip for top 100 restaurants            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Impact:**
- Cache hits reduce latency by 70-90%
- Estimated 40% overall hit rate across layers

#### Strategy 3: Parallel Execution

```python
async def search(self, parsed_query: ParsedQuery) -> list[ScoredRestaurant]:
    """Execute retrieval signals in parallel."""

    # Run all scorers concurrently
    vibe_task = asyncio.create_task(
        self.vibe_scorer.score(candidates, parsed_query)
    )
    cuisine_task = asyncio.create_task(
        self.cuisine_scorer.score(candidates, parsed_query)
    )
    price_task = asyncio.create_task(
        self.price_scorer.score(candidates, parsed_query)
    )
    feature_task = asyncio.create_task(
        self.feature_scorer.score(candidates, parsed_query)
    )

    # Wait for all to complete
    vibe_scores, cuisine_scores, price_scores, feature_scores = await asyncio.gather(
        vibe_task, cuisine_task, price_task, feature_task
    )

    return self.fuse_scores(...)
```

**Impact:** Reduces scoring phase from ~400ms (sequential) to ~150ms (parallel)

#### Strategy 4: Streaming Responses

```python
async def stream_response(self, query: str):
    """Stream response to reduce perceived latency."""

    # Immediately acknowledge
    yield {"status": "searching", "message": "Finding your perfect spot..."}

    # Stream partial results as available
    parsed = await self.parse_query(query)
    yield {"status": "understood", "query_intent": parsed.dict()}

    # Stream top match as soon as found
    results = await self.search(parsed)
    yield {"status": "found", "top_match": results[0].dict()}

    # Stream explanation (can be slower)
    explanation = await self.generate_explanation(results[0], query)
    yield {"status": "complete", "explanation": explanation}
```

**Impact:** User sees first result in ~800ms, full response streams over 1.5s

#### Strategy 5: Pre-computation

| What | When | Storage |
|------|------|---------|
| Restaurant embeddings | On data ingestion | Qdrant |
| Vibe enrichment | Nightly batch | PostgreSQL |
| Popular query responses | Every 15 min | Redis |
| Cuisine similarity matrix | Weekly | In-memory |

**Impact:** Eliminates runtime embedding generation for restaurants

### 4.3 Cost Optimization

#### Cost per Query Breakdown

```
┌─────────────────────────────────────────────────────────────────┐
│              COST PER QUERY (COMPLEX PATH)                      │
├─────────────────────────────────────────────────────────────────┤
│  Query Parsing (GPT-4o-mini)                                    │
│  ├─ Input: ~200 tokens × $0.15/1M = $0.00003                   │
│  └─ Output: ~300 tokens × $0.60/1M = $0.00018                  │
│                                                                 │
│  Query Embedding                                                │
│  └─ ~50 tokens × $0.02/1M = $0.000001                          │
│                                                                 │
│  Vector Search (Qdrant Cloud)                                   │
│  └─ ~$0.00001 per query (free tier covers initial scale)       │
│                                                                 │
│  Response Generation (GPT-4o-mini)                              │
│  ├─ Input: ~500 tokens × $0.15/1M = $0.000075                  │
│  └─ Output: ~200 tokens × $0.60/1M = $0.00012                  │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│  TOTAL (no cache)                                │ ~$0.0004     │
│  With 40% cache hit rate                         │ ~$0.00024    │
│  Target                                          │ < $0.01      │
└─────────────────────────────────────────────────────────────────┘
```

**Result:** Well under $0.01 per query target

#### Cost Reduction Strategies

1. **Batch embedding generation** — Generate restaurant embeddings in batches during ingestion, not at query time

2. **Response caching** — Popular queries like "best poke" cached aggressively

3. **Model downgrades for simple queries** — Use cheaper/faster models for straightforward lookups

4. **Prompt optimization** — Minimize token usage in system prompts through iteration

5. **Self-hosted fallbacks** — Consider Llama 3.1 8B for classification tasks if volume grows

---

## 5. Data Freshness Strategy

### 5.1 The Freshness Challenge

Restaurants change frequently:
- **Hours:** Holiday schedules, seasonal changes
- **Menus:** Daily specials, seasonal items, price changes
- **Status:** Temporary closures, permanent closures, new openings
- **Reviews:** New reviews affect vibe perception

Stale data = bad recommendations = lost trust

### 5.2 Data Source Hierarchy

```
┌─────────────────────────────────────────────────────────────────┐
│                    DATA SOURCE PRIORITY                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Priority 1: Direct Restaurant Integrations                     │
│  ├─ POS system integrations (real-time menu)                   │
│  ├─ Restaurant dashboard submissions                            │
│  └─ Freshness: Real-time to minutes                            │
│                                                                 │
│  Priority 2: Aggregator APIs                                    │
│  ├─ Google Places API (hours, status, reviews)                 │
│  ├─ Yelp API (reviews, photos)                                 │
│  └─ Freshness: Hours to daily                                  │
│                                                                 │
│  Priority 3: Web Scraping                                       │
│  ├─ Restaurant websites (menus, specials)                      │
│  ├─ Social media (Instagram for vibe signals)                  │
│  └─ Freshness: Daily to weekly                                 │
│                                                                 │
│  Priority 4: Manual Curation                                    │
│  ├─ Local team verification                                    │
│  ├─ User-submitted corrections                                 │
│  └─ Freshness: Weekly to monthly                               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 5.3 Update Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    DATA INGESTION PIPELINE                      │
└─────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌───────────────┐     ┌───────────────┐     ┌───────────────┐
│  SCHEDULED    │     │  EVENT-DRIVEN │     │  MANUAL       │
│  CRAWLERS     │     │  WEBHOOKS     │     │  SUBMISSIONS  │
│               │     │               │     │               │
│ • Google API  │     │ • POS updates │     │ • Admin UI    │
│   (6 hours)   │     │ • Restaurant  │     │ • User reports│
│ • Yelp API    │     │   dashboard   │     │               │
│   (daily)     │     │   changes     │     │               │
│ • Website     │     │               │     │               │
│   scrapes     │     │               │     │               │
│   (daily)     │     │               │     │               │
└───────────────┘     └───────────────┘     └───────────────┘
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      CHANGE DETECTION                           │
│                                                                 │
│  Compare incoming data with current state:                      │
│  • Hash comparison for text fields                              │
│  • Timestamp comparison for structured data                     │
│  • Diff generation for audit trail                              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      UPDATE PROCESSOR                           │
│                                                                 │
│  If changes detected:                                           │
│  1. Update PostgreSQL record                                    │
│  2. Queue for vibe re-enrichment (if text changed)             │
│  3. Queue for re-embedding (if vibe changed)                   │
│  4. Invalidate relevant caches                                  │
│  5. Log change for analytics                                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    VIBE RE-ENRICHMENT                           │
│                    (Async Background Job)                       │
│                                                                 │
│  If menu/reviews/description changed significantly:             │
│  1. Re-run LLM enrichment for vibe profile                     │
│  2. Generate new embedding                                      │
│  3. Upsert to Qdrant                                           │
│  4. Mark record as fresh                                        │
└─────────────────────────────────────────────────────────────────┘
```

### 5.4 Freshness Indicators

```python
class Restaurant:
    # ... other fields ...

    # Freshness metadata
    last_verified_at: datetime        # When data was last confirmed accurate
    data_source: str                  # "google_api", "manual", "scrape"
    confidence_score: float           # 0-1, decays over time

    @property
    def freshness_status(self) -> str:
        age = datetime.now() - self.last_verified_at
        if age < timedelta(hours=24):
            return "fresh"
        elif age < timedelta(days=7):
            return "recent"
        elif age < timedelta(days=30):
            return "stale"
        else:
            return "outdated"
```

### 5.5 Graceful Degradation

When data might be stale, communicate uncertainty:

```json
{
  "recommendation": "Mama's Fish House",
  "caveats": [
    "Hours were last verified 3 days ago — we recommend calling ahead to confirm"
  ],
  "freshness": {
    "status": "recent",
    "last_verified": "2024-12-14T10:30:00Z",
    "source": "google_api"
  }
}
```

### 5.6 Update Frequency Summary

| Data Type | Update Frequency | Method |
|-----------|------------------|--------|
| Operating hours | Every 6 hours | Google Places API |
| Menu items | Daily | Website scrape + manual |
| Reviews/Ratings | Daily | Google + Yelp APIs |
| Vibe enrichment | On text change | LLM batch job |
| Embeddings | On vibe change | Background job |
| Restaurant status | Real-time | Webhooks + manual flags |

---

## 6. Scaling Considerations

### 6.1 Current Scale Assumptions

- **Restaurants:** ~500 in Maui, ~2,000 across Hawaii
- **Queries:** 1,000-10,000/day initial
- **Data size:** ~10MB metadata, ~50MB embeddings

This is comfortably handled by a single-server deployment.

### 6.2 Scaling Triggers

| Metric | Threshold | Action |
|--------|-----------|--------|
| Queries/second | > 10 sustained | Add API server replicas |
| Vector search latency | > 200ms p95 | Upgrade Qdrant tier / add replicas |
| Database connections | > 80% pool | Add read replicas |
| Cache hit rate | < 30% | Increase Redis memory |
| LLM latency | > 1s p95 | Add request queuing, consider batching |

### 6.3 Horizontal Scaling Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     SCALED ARCHITECTURE                         │
└─────────────────────────────────────────────────────────────────┘

                         ┌─────────────┐
                         │   Route 53  │
                         │   (DNS)     │
                         └──────┬──────┘
                                │
                         ┌──────▼──────┐
                         │ CloudFront  │
                         │   (CDN)     │
                         └──────┬──────┘
                                │
                         ┌──────▼──────┐
                         │    ALB      │
                         │ (Load Bal.) │
                         └──────┬──────┘
                                │
            ┌───────────────────┼───────────────────┐
            ▼                   ▼                   ▼
     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
     │  API Pod 1  │     │  API Pod 2  │     │  API Pod N  │
     │  (ECS/K8s)  │     │  (ECS/K8s)  │     │  (ECS/K8s)  │
     └─────────────┘     └─────────────┘     └─────────────┘
            │                   │                   │
            └───────────────────┼───────────────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        ▼                       ▼                       ▼
 ┌─────────────┐         ┌─────────────┐         ┌─────────────┐
 │  Qdrant     │         │ PostgreSQL  │         │   Redis     │
 │  Cluster    │         │   Primary   │         │  Cluster    │
 │             │         │  + Replicas │         │             │
 └─────────────┘         └─────────────┘         └─────────────┘
```

### 6.4 Geographic Expansion

For expanding beyond Hawaii:

1. **Data isolation:** Separate vector collections per region
2. **Query routing:** Route based on user location / query context
3. **Edge caching:** Regional cache nodes for local popular queries
4. **Model fine-tuning:** Region-specific vibe vocabularies

---

## 7. Monitoring & Observability

### 7.1 Key Metrics

#### Latency Metrics
```
ana_query_duration_seconds{path="simple|complex", quantile="0.5|0.95|0.99"}
ana_llm_call_duration_seconds{model="gpt-4o-mini", operation="parse|generate"}
ana_vector_search_duration_seconds{collection="restaurants"}
ana_cache_hit_ratio{layer="query|embedding|response"}
```

#### Quality Metrics
```
ana_search_result_count{query_type="vibe|cuisine|location"}
ana_confidence_score{bucket="high|medium|low"}
ana_empty_result_rate
ana_user_feedback_score{rating="helpful|not_helpful"}
```

#### Cost Metrics
```
ana_llm_tokens_total{model="gpt-4o-mini", direction="input|output"}
ana_embedding_tokens_total
ana_estimated_cost_per_query
```

### 7.2 Alerting Rules

| Alert | Condition | Severity |
|-------|-----------|----------|
| High Latency | p95 > 3s for 5 min | Warning |
| Very High Latency | p95 > 5s for 2 min | Critical |
| Empty Results Spike | > 20% empty results for 10 min | Warning |
| LLM Errors | > 5% error rate for 5 min | Critical |
| Cache Degradation | Hit rate < 20% for 15 min | Warning |
| Cost Spike | Daily cost > 150% of average | Warning |

### 7.3 Logging Strategy

```python
# Structured log format
{
    "timestamp": "2024-12-17T10:30:00Z",
    "request_id": "uuid",
    "event": "search_complete",
    "query": "night market vibe...",
    "query_type": "complex",
    "latency_ms": 1250,
    "cache_hits": ["embedding"],
    "result_count": 3,
    "top_match_score": 0.91,
    "confidence": "high",
    "llm_tokens": {"input": 450, "output": 280},
    "user_id": "anonymous"  # or hashed
}
```

### 7.4 Dashboards

1. **Operations Dashboard**
   - Request rate, latency percentiles, error rate
   - Cache hit rates by layer
   - LLM API health

2. **Quality Dashboard**
   - Result confidence distribution
   - Empty result rate by query type
   - User feedback trends

3. **Cost Dashboard**
   - Daily/weekly LLM spend
   - Cost per query trending
   - Projected monthly costs

---

## 8. Security & Privacy

### 8.1 Data Protection

| Data Type | Sensitivity | Protection |
|-----------|-------------|------------|
| User queries | Medium | No PII stored, anonymized logs |
| Restaurant data | Low | Public information |
| User preferences | Medium | Encrypted at rest, minimal retention |
| API keys | High | Secrets manager (AWS Secrets Manager) |

### 8.2 API Security

- **Authentication:** API key for B2B, OAuth for consumer apps
- **Rate limiting:** 100 req/min per user, 1000 req/min per API key
- **Input validation:** Query length limits, sanitization
- **HTTPS only:** TLS 1.3 enforced

### 8.3 LLM Security

- **Prompt injection prevention:** Input sanitization, output validation
- **PII filtering:** Strip phone numbers, emails from LLM inputs
- **Content filtering:** Reject inappropriate queries

---

## 9. Cost Analysis

### 9.1 Monthly Cost Projection (10K queries/day)

```
┌─────────────────────────────────────────────────────────────────┐
│              MONTHLY COST ESTIMATE (10K queries/day)            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  LLM Costs (OpenAI)                                            │
│  ├─ Query parsing: 300K queries × $0.0002 = $60                │
│  ├─ Response gen:  300K queries × $0.0003 = $90                │
│  └─ Vibe enrichment (batch): ~$20/month                        │
│  Subtotal: ~$170                                                │
│                                                                 │
│  Infrastructure                                                 │
│  ├─ API servers (2× small instances): ~$50                     │
│  ├─ PostgreSQL (managed, small): ~$25                          │
│  ├─ Qdrant Cloud (starter): ~$25                               │
│  ├─ Redis (Upstash): ~$10                                      │
│  └─ Misc (logging, monitoring): ~$20                           │
│  Subtotal: ~$130                                                │
│                                                                 │
│  External APIs                                                  │
│  ├─ Google Places API: ~$50                                    │
│  └─ Other data sources: ~$20                                   │
│  Subtotal: ~$70                                                 │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│  TOTAL MONTHLY                               │ ~$370            │
│  Cost per query                              │ ~$0.0012         │
└─────────────────────────────────────────────────────────────────┘
```

### 9.2 Cost Scaling

| Scale | Queries/day | Est. Monthly Cost | Cost/Query |
|-------|-------------|-------------------|------------|
| MVP | 1,000 | ~$150 | $0.005 |
| Growth | 10,000 | ~$370 | $0.0012 |
| Scale | 100,000 | ~$2,500 | $0.0008 |

Economies of scale come from:
- Higher cache hit rates
- Batched LLM calls
- Reserved infrastructure pricing

---

## 10. Risk Mitigation

### 10.1 Technical Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| OpenAI API outage | Medium | High | Fallback to Anthropic/open-source |
| Vector DB corruption | Low | High | Regular backups, replication |
| Cache stampede | Medium | Medium | Staggered TTLs, request coalescing |
| Embedding drift | Low | Medium | Version embeddings, periodic refresh |
| Prompt injection | Medium | Medium | Input sanitization, output validation |

### 10.2 Business Risks

| Risk | Mitigation |
|------|------------|
| Stale recommendations damage trust | Freshness indicators, confidence scores |
| Biased results (always same restaurants) | Diversity injection, rotation |
| Negative user feedback | Feedback loop, rapid iteration |
| Data source changes (API deprecation) | Multiple sources, abstraction layer |

### 10.3 Fallback Strategies

```python
async def search_with_fallbacks(self, query: str) -> AnaResponse:
    """Graceful degradation when components fail."""

    try:
        # Primary path: Full hybrid search
        return await self.full_search(query)

    except LLMTimeoutError:
        # Fallback 1: Skip LLM parsing, use keyword extraction
        logger.warning("LLM timeout, using keyword fallback")
        keywords = extract_keywords(query)
        return await self.keyword_search(keywords)

    except VectorDBError:
        # Fallback 2: Metadata-only search
        logger.warning("Vector DB error, using metadata search")
        return await self.metadata_only_search(query)

    except Exception as e:
        # Fallback 3: Return popular/default results
        logger.error(f"Search failed completely: {e}")
        return self.get_popular_restaurants()
```

---

## 11. Implementation Roadmap

### Phase 1: MVP (Weeks 1-2)
- [ ] Core hybrid pipeline
- [ ] Mock data with 20 restaurants
- [ ] Basic caching (Redis)
- [ ] Simple API endpoint

### Phase 2: Production Hardening (Weeks 3-4)
- [ ] Real data integration
- [ ] Monitoring & alerting
- [ ] Rate limiting & auth
- [ ] Error handling & fallbacks

### Phase 3: Optimization (Weeks 5-6)
- [ ] Query classification tier
- [ ] Streaming responses
- [ ] Advanced caching
- [ ] Cost monitoring

### Phase 4: Scale & Iterate (Ongoing)
- [ ] User feedback integration
- [ ] A/B testing framework
- [ ] Geographic expansion

---

## 12. Conclusion

This design balances the competing demands of:

1. **User Experience** — Fast, relevant, explainable results
2. **Cost Efficiency** — Well under $0.01/query through smart caching and model selection
3. **Data Quality** — Multi-source ingestion with freshness tracking
4. **Operational Excellence** — Observable, recoverable, scalable

The hybrid retrieval approach is the key architectural decision—it handles the nuanced "vibe" queries that make Ana special while maintaining the reliability of structured search for simpler requests.

Ana isn't just a search engine; she's a knowledgeable local friend. This architecture ensures she can be that friend at scale.

---

## Appendix A: API Contract

```yaml
# OpenAPI snippet
paths:
  /v1/search:
    post:
      summary: Search restaurants by vibe
      requestBody:
        content:
          application/json:
            schema:
              type: object
              properties:
                query:
                  type: string
                  example: "romantic dinner spot with ocean views"
                location:
                  type: string
                  example: "Lahaina"
                limit:
                  type: integer
                  default: 5
      responses:
        200:
          description: Search results
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/AnaResponse'
```

## Appendix B: Sample Prompts

### Query Parser System Prompt

```
You are a query parser for Ana AI, a restaurant recommendation system for Hawaii.

Given a user's natural language request, extract structured search intent.

Output JSON with:
- semantic_query: Expanded description for vector search
- must_not: Hard exclusions (things they explicitly don't want)
- preferences: Soft preferences (nice to have)
- weights: How much each signal matters (must sum to 1.0)

Handle negations carefully: "not fancy" → must_not.formality: ["upscale", "fine_dining"]

Be generous in expanding the semantic_query to capture the full vibe.
```

### Response Generator System Prompt

```
You are Ana, a friendly and knowledgeable local food guide for Hawaii.

Given a user's query and the top restaurant match, write a warm, helpful explanation of why this restaurant is perfect for them.

Guidelines:
- Be conversational, like a friend giving advice
- Connect specific aspects of the restaurant to what they asked for
- Mention 2-3 concrete details (seating, vibe, food style)
- Keep it under 100 words
- Don't use generic superlatives—be specific
```
