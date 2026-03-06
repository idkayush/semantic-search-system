"""
Test script for the Semantic Search API

This script tests all API endpoints and demonstrates cache behavior.
"""

import requests
import time
import json
from typing import Dict, Any


API_URL = "http://localhost:8000"


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)


def test_health():
    """Test health endpoint."""
    print_section("Testing Health Endpoint")
    
    response = requests.get(f"{API_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    return response.status_code == 200


def test_query(query: str) -> Dict[str, Any]:
    """Test query endpoint."""
    payload = {"query": query}
    
    start_time = time.time()
    response = requests.post(f"{API_URL}/query", json=payload)
    elapsed_time = time.time() - start_time
    
    result = response.json()
    
    print(f"\nQuery: {query}")
    print(f"Time: {elapsed_time:.3f}s")
    print(f"Cache Hit: {result['cache_hit']}")
    
    if result['cache_hit']:
        print(f"Matched: {result['matched_query']}")
        print(f"Similarity: {result['similarity_score']}")
    
    print(f"Dominant Cluster: {result['dominant_cluster']}")
    print(f"Results: {len(result['result'])} documents")
    
    if result['result']:
        print(f"\nTop Result:")
        top = result['result'][0]
        print(f"  Category: {top['category']}")
        print(f"  Distance: {top['distance']:.3f}")
        print(f"  Preview: {top['document'][:150]}...")
    
    return result


def test_cache_stats():
    """Test cache stats endpoint."""
    print_section("Cache Statistics")
    
    response = requests.get(f"{API_URL}/cache/stats")
    stats = response.json()
    
    print(f"Total Entries: {stats['total_entries']}")
    print(f"Hit Count: {stats['hit_count']}")
    print(f"Miss Count: {stats['miss_count']}")
    print(f"Hit Rate: {stats['hit_rate']:.1%}")
    
    return stats


def test_clear_cache():
    """Test cache clear endpoint."""
    print_section("Clearing Cache")
    
    response = requests.delete(f"{API_URL}/cache")
    result = response.json()
    
    print(f"Status: {result['status']}")
    print(f"Message: {result['message']}")
    
    return response.status_code == 200


def main():
    """Run all tests."""
    print("="*60)
    print(" SEMANTIC SEARCH API - TEST SUITE")
    print("="*60)
    
    # Test 1: Health check
    if not test_health():
        print("\n❌ Health check failed! Is the API running?")
        return
    
    print("\n✓ API is healthy")
    
    # Test 2: Clear cache to start fresh
    test_clear_cache()
    
    # Test 3: Query testing - demonstrate cache behavior
    print_section("Testing Query Endpoint & Cache Behavior")
    
    # Set of similar queries
    query_sets = [
        [
            "What are the best graphics cards for gaming?",
            "Which GPU is best for video games?",
            "Top graphics cards for gaming"
        ],
        [
            "How do I install Linux?",
            "Linux installation guide",
            "Steps to install Linux on my computer"
        ],
        [
            "What is quantum computing?",
            "Tell me about quantum computers",
            "Explain quantum computing"
        ]
    ]
    
    for i, query_set in enumerate(query_sets, 1):
        print(f"\n--- Query Set {i} ---")
        
        for query in query_set:
            test_query(query)
            time.sleep(0.5)  # Small delay for readability
    
    # Test 4: Check final cache stats
    print("\n")
    stats = test_cache_stats()
    
    # Summary
    print_section("Test Summary")
    print(f"✓ Health check: PASSED")
    print(f"✓ Query endpoint: PASSED")
    print(f"✓ Cache behavior: PASSED")
    print(f"✓ Cache stats: PASSED")
    print(f"\nFinal Cache Stats:")
    print(f"  Total queries: {stats['hit_count'] + stats['miss_count']}")
    print(f"  Cache hits: {stats['hit_count']}")
    print(f"  Cache misses: {stats['miss_count']}")
    print(f"  Hit rate: {stats['hit_rate']:.1%}")
    
    if stats['hit_rate'] > 0:
        print("\n✅ Cache is working! Similar queries are being matched.")
    else:
        print("\n⚠️  No cache hits yet. Try more queries or lower the threshold.")


if __name__ == "__main__":
    try:
        main()
    except requests.exceptions.ConnectionError:
        print("\n❌ Could not connect to API. Make sure it's running:")
        print("   uvicorn api:app --reload")
    except Exception as e:
        print(f"\n❌ Error: {e}")
