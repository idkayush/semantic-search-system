"""
FastAPI Service for Semantic Search with Caching

This service exposes the semantic search system with a semantic cache layer.

Endpoints:
----------
1. POST /query - Perform semantic search with cache lookup
2. GET /cache/stats - Get cache statistics
3. DELETE /cache - Clear cache and reset stats
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import numpy as np
from contextlib import asynccontextmanager

# Import our modules
from embeddings import EmbeddingGenerator
from clustering import FuzzyClusterer
from vector_db import VectorDatabase
from semantic_cache import SemanticCache


# Global state (loaded on startup)
class AppState:
    embedder: Optional[EmbeddingGenerator] = None
    clusterer: Optional[FuzzyClusterer] = None
    vector_db: Optional[VectorDatabase] = None
    cache: Optional[SemanticCache] = None


state = AppState()


# Request/Response models
class QueryRequest(BaseModel):
    query: str = Field(..., description="Natural language query")


class QueryResponse(BaseModel):
    query: str
    cache_hit: bool
    matched_query: Optional[str] = None
    similarity_score: Optional[float] = None
    result: List[Dict[str, Any]]
    dominant_cluster: int


class CacheStatsResponse(BaseModel):
    total_entries: int
    hit_count: int
    miss_count: int
    hit_rate: float


# Lifespan context manager for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load models and initialize components on startup.
    
    This ensures the service starts with all necessary components loaded.
    """
    print("\n=== Starting Semantic Search Service ===")
    
    # Load embedding model
    print("Loading embedding model...")
    state.embedder = EmbeddingGenerator()
    
    # Load clustering model
    print("Loading clustering model...")
    state.clusterer = FuzzyClusterer()
    state.clusterer.load_model()
    
    # Load vector database
    print("Loading vector database...")
    state.vector_db = VectorDatabase()
    state.vector_db.load_collection()
    
    # Initialize semantic cache
    print("Initializing semantic cache...")
    state.cache = SemanticCache(
        gmm_model=state.clusterer.gmm,
        similarity_threshold=0.85  # The tunable parameter
    )
    
    print("Service ready!\n")
    
    yield
    
    # Cleanup on shutdown
    print("\n=== Shutting down ===")
    state.cache.save_cache()


# Create FastAPI app
app = FastAPI(
    title="Semantic Search API",
    description="Semantic search system with fuzzy clustering and intelligent caching",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "Semantic Search API",
        "version": "1.0.0"
    }


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Perform semantic search with cache lookup.
    
    Flow:
    1. Embed the query
    2. Check semantic cache
    3. If hit: return cached result
    4. If miss: search vector DB, cache result, return
    """
    try:
        # Embed the query
        query_embedding = state.embedder.embed_query(request.query)
        
        # Check cache
        cache_result = state.cache.lookup(request.query, query_embedding)
        
        if cache_result:
            # Cache hit!
            matched_entry, similarity_score = cache_result
            
            return QueryResponse(
                query=request.query,
                cache_hit=True,
                matched_query=matched_entry.query,
                similarity_score=round(similarity_score, 3),
                result=matched_entry.result,
                dominant_cluster=matched_entry.dominant_cluster
            )
        
        else:
            # Cache miss - perform actual search
            search_results = state.vector_db.search(
                query_embedding=query_embedding,
                n_results=5
            )
            
            # Format results
            formatted_results = []
            for i in range(len(search_results['documents'])):
                formatted_results.append({
                    'document': search_results['documents'][i],
                    'category': search_results['metadatas'][i].get('category', 'unknown'),
                    'distance': float(search_results['distances'][i])
                })
            
            # Determine dominant cluster
            cluster_probs = state.clusterer.gmm.predict_proba(
                query_embedding.reshape(1, -1)
            )
            dominant_cluster = int(np.argmax(cluster_probs))
            
            # Store in cache
            state.cache.store(
                query=request.query,
                query_embedding=query_embedding,
                result=formatted_results
            )
            
            return QueryResponse(
                query=request.query,
                cache_hit=False,
                matched_query=None,
                similarity_score=None,
                result=formatted_results,
                dominant_cluster=dominant_cluster
            )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/cache/stats", response_model=CacheStatsResponse)
async def get_cache_stats():
    """
    Get current cache statistics.
    
    Returns:
        Cache statistics including hit rate and entry count
    """
    try:
        stats = state.cache.get_stats()
        
        return CacheStatsResponse(
            total_entries=stats['total_entries'],
            hit_count=stats['hit_count'],
            miss_count=stats['miss_count'],
            hit_rate=stats['hit_rate']
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/cache")
async def clear_cache():
    """
    Clear the semantic cache and reset all statistics.
    
    Returns:
        Confirmation message
    """
    try:
        state.cache.clear()
        
        return {
            "status": "success",
            "message": "Cache cleared and statistics reset"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """
    Detailed health check showing component status.
    """
    return {
        "status": "healthy",
        "components": {
            "embedder": state.embedder is not None,
            "clusterer": state.clusterer is not None,
            "vector_db": state.vector_db is not None,
            "cache": state.cache is not None
        },
        "cache_stats": state.cache.get_stats() if state.cache else None
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )
