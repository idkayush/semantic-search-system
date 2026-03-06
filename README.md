# Semantic Search System with Fuzzy Clustering and Intelligent Caching

A production-ready semantic search system built for the 20 Newsgroups dataset, featuring:
- **Fuzzy clustering** with probability distributions (not hard assignments)
- **Semantic cache** built from scratch (no Redis/Memcached)
- **Cluster-aware caching** for O(k) lookup instead of O(n)
- **FastAPI service** with proper state management

## 🏗️ Architecture

```
┌─────────────┐     ┌──────────────┐     ┌────────────┐
│   Query     │────▶│   Embedder   │────▶│   Cache    │
└─────────────┘     └──────────────┘     └────────────┘
                                               │
                                               ├─ Hit? → Return
                                               │
                                               ▼
                                         ┌──────────────┐
                                         │  Vector DB   │
                                         └──────────────┘
                                               │
                                               ▼
                                         ┌──────────────┐
                                         │   Results    │
                                         └──────────────┘
```

## 📋 Design Decisions

### Part 1: Embedding & Vector Database

**Embedding Model:** `all-MiniLM-L6-v2`
- Fast inference (~80ms per query)
- 384-dimensional embeddings (good balance)
- Excellent semantic understanding for general text
- Normalized embeddings for efficient cosine similarity

**Vector Database:** ChromaDB
- Lightweight, no external dependencies
- Built-in HNSW indexing for fast retrieval
- Persistent storage
- Metadata filtering support

**Data Preprocessing:**
- Removed headers, footers, and quotes (noise reduction)
- Filtered documents shorter than 100 characters
- Cleaned excessive whitespace and special characters
- **Rationale:** Focus on semantic content, not formatting artifacts

### Part 2: Fuzzy Clustering

**Clustering Method:** Gaussian Mixture Models (GMM)
- Provides **probability distributions** over clusters
- Captures fuzzy topic boundaries naturally
- Documents can belong to multiple clusters

**Number of Clusters:** Determined by BIC (Bayesian Information Criterion)
- Tested range: 15-30 clusters
- BIC balances fit quality vs. model complexity
- **Finding:** Optimal ~22-24 clusters (data-dependent)
- **Insight:** True semantic structure differs from 20 labeled categories

**Cluster Analysis:**
- Cluster purity scores
- Boundary document identification (high uncertainty)
- Category distribution within clusters
- UMAP visualizations showing overlap

### Part 3: Semantic Cache

**Data Structure:** Cluster-aware dictionary
```python
{
  cluster_id: [CacheEntry, CacheEntry, ...]
}
```

**Lookup Strategy:**
1. Determine query's dominant cluster via GMM
2. Search only that cluster's entries (~k entries)
3. Fallback to global search if needed
4. **Complexity:** O(k) instead of O(n) where k << n

**Similarity Threshold: THE KEY PARAMETER**
- Default: **0.85** (cosine similarity)
- Lower (0.7-0.8): Stricter matching, higher precision, more misses
- Higher (0.9-0.95): Looser matching, more hits, lower precision
- **Exploration:** Tested range reveals precision/recall tradeoff

**Why No Redis?**
- Simple in-memory structure sufficient for dataset size
- Cluster structure provides natural partitioning
- Full control over similarity matching logic
- **Production Note:** Could add LRU eviction or persistence layer

### Part 4: FastAPI Service

**State Management:**
- Models loaded once at startup (lifespan context manager)
- In-memory cache persists across requests
- Thread-safe operations

**Endpoints:**
1. `POST /query` - Semantic search with caching
2. `GET /cache/stats` - Cache metrics
3. `DELETE /cache` - Clear cache

## 🚀 Setup & Installation

### Prerequisites
- Python 3.10 or 3.11
- 4GB+ RAM
- (Optional) Docker

### Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd semantic-search-system
```

2. **Create virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Run the pipeline:**
```bash
# Full pipeline (first time)
python pipeline.py

# With caching (subsequent runs)
python pipeline.py --skip-data --skip-embeddings
```

This will:
- Download and preprocess 20 Newsgroups dataset
- Generate embeddings (~20 min on CPU)
- Create vector database
- Perform fuzzy clustering with BIC optimization
- Generate analysis and visualizations

### Running the API

**Option 1: Direct**
```bash
uvicorn api:app --reload
```

**Option 2: Docker**
```bash
# Build and run
docker-compose up --build

# Or build manually
docker build -t semantic-search-api .
docker run -p 8000:8000 semantic-search-api
```

The API will be available at `http://localhost:8000`

## 📊 Usage Examples

### Query Endpoint

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the best graphics cards for gaming?"}'
```

**Response (Cache Miss):**
```json
{
  "query": "What are the best graphics cards for gaming?",
  "cache_hit": false,
  "matched_query": null,
  "similarity_score": null,
  "result": [
    {
      "document": "...",
      "category": "comp.graphics",
      "distance": 0.23
    }
  ],
  "dominant_cluster": 7
}
```

**Response (Cache Hit):**
```json
{
  "query": "Best GPU for video games?",
  "cache_hit": true,
  "matched_query": "What are the best graphics cards for gaming?",
  "similarity_score": 0.91,
  "result": [...],
  "dominant_cluster": 7
}
```

### Cache Stats

```bash
curl "http://localhost:8000/cache/stats"
```

**Response:**
```json
{
  "total_entries": 42,
  "hit_count": 17,
  "miss_count": 25,
  "hit_rate": 0.405
}
```

### Clear Cache

```bash
curl -X DELETE "http://localhost:8000/cache"
```

## 🔬 Analysis & Insights

### Cluster Quality

After running the pipeline, check:
- `models/cluster_analysis.json` - Detailed cluster statistics
- `models/bic_scores.png` - BIC optimization plot
- `models/cluster_visualization.png` - UMAP 2D projection

**Key Findings:**
1. **Cluster overlap:** Documents about "gun legislation" belong to both politics and firearms clusters (as expected)
2. **Boundary cases:** ~5-10% of documents have high uncertainty (max probability < 0.4)
3. **Purity vs. Count:** Some clusters are very pure but small, others are large but mixed

### Threshold Impact

The `semantic_cache.py` script includes threshold exploration:

```python
python -c "from semantic_cache import *; # run exploration"
```

**Observations:**
- **Threshold 0.7:** Hit rate ~60%, but some false positives
- **Threshold 0.85:** Hit rate ~40%, good precision
- **Threshold 0.95:** Hit rate ~15%, very strict matching

**Insight:** The threshold reveals the tradeoff between:
- **High threshold:** Users get exactly what they asked for (precision)
- **Low threshold:** System recognizes more paraphrases (recall)

## 📁 Project Structure

```
semantic-search-system/
├── api.py                  # FastAPI service
├── pipeline.py             # Main pipeline script
├── data_loader.py          # Dataset loading and preprocessing
├── embeddings.py           # Text-to-vector conversion
├── vector_db.py            # ChromaDB integration
├── clustering.py           # GMM fuzzy clustering
├── semantic_cache.py       # From-scratch semantic cache
├── requirements.txt        # Python dependencies
├── Dockerfile              # Container definition
├── docker-compose.yml      # Docker orchestration
├── README.md               # This file
│
├── data/                   # Dataset and preprocessed data
│   ├── preprocessed_data.pkl
│   ├── metadata.json
│   └── chroma_db/         # Vector database storage
│
├── models/                 # Trained models and embeddings
│   ├── corpus_embeddings.npy
│   ├── gmm_model.pkl
│   ├── cluster_analysis.json
│   ├── bic_scores.png
│   └── cluster_visualization.png
│
└── cache/                  # Cache persistence
    ├── semantic_cache.json
    └── threshold_analysis.json
```

## 🧪 Testing

### Manual Testing

```bash
# Test cache behavior
python semantic_cache.py

# Test individual components
python data_loader.py
python embeddings.py
python clustering.py
```

### Interactive Testing

```bash
# Start API
uvicorn api:app --reload

# Open browser
http://localhost:8000/docs  # Swagger UI
```

## 🎯 Key Features

✅ **Soft Clustering:** Probability distributions, not hard labels  
✅ **Cluster-Aware Cache:** O(k) lookup complexity  
✅ **No External Cache:** Built from scratch  
✅ **Tunable Threshold:** Explore precision/recall tradeoff  
✅ **Production Ready:** Docker, health checks, proper state management  
✅ **Well Documented:** Inline comments explain design decisions  

## 📝 Submission Notes

**Dataset Preprocessing:**
- Removed headers/footers to reduce noise
- Minimum document length of 100 chars filters junk
- ~18,000 documents retained from original 20,000

**Embedding Choice:**
- `all-MiniLM-L6-v2` chosen for speed/quality balance
- 384 dimensions sufficient for semantic understanding
- Normalized embeddings optimize cosine similarity

**Clustering Justification:**
- BIC-based selection found ~22-24 optimal clusters
- More than 20 original categories (semantic structure ≠ labels)
- GMM captures fuzzy boundaries naturally

**Cache Design:**
- Cluster structure critical for scalability
- Threshold of 0.85 balances precision/recall
- No eviction needed for dataset size (could add LRU)

## 🔗 Links

- API Docs: http://localhost:8000/docs
- Health Check: http://localhost:8000/health
- GitHub: [https://github.com/idkayush/semantic-search-system]

## 📧 Contact

For questions or issues, contact: [ayusharvind2k18@gmail.com]
[7033762420]
---

## Quick Start

```bash
pip install -r requirements.txt
python pipeline.py
uvicorn api:app --reload

**Hope you ❤️ my project for Trademarkia**