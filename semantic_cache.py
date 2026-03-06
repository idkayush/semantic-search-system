"""
Semantic Cache Module
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path


@dataclass
class CacheEntry:
    """Represents a single cache entry."""
    query: str
    embedding: np.ndarray
    result: Any
    dominant_cluster: int
    timestamp: str

    def to_dict(self):
        return {
            "query": self.query,
            "result": self.result,
            "dominant_cluster": int(self.dominant_cluster),
            "timestamp": self.timestamp
        }


class SemanticCache:

    def __init__(
        self,
        gmm_model,
        similarity_threshold: float = 0.85,
        output_dir: str = "./cache"
    ):

        self.gmm = gmm_model
        self.similarity_threshold = similarity_threshold

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.cache: Dict[int, List[CacheEntry]] = {}

        self.stats = {
            "hit_count": 0,
            "miss_count": 0,
            "total_queries": 0
        }

        print(f"Initialized semantic cache with threshold={similarity_threshold}")

    def _get_query_cluster(self, query_embedding: np.ndarray) -> int:
        cluster_probs = self.gmm.predict_proba(query_embedding.reshape(1, -1))
        dominant_cluster = int(np.argmax(cluster_probs))
        return dominant_cluster

    def _compute_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray
    ) -> float:
        return float(np.dot(embedding1, embedding2))

    def lookup(
        self,
        query: str,
        query_embedding: np.ndarray
    ) -> Optional[Tuple[CacheEntry, float]]:

        self.stats["total_queries"] += 1

        query_cluster = self._get_query_cluster(query_embedding)

        if query_cluster in self.cache:

            for entry in self.cache[query_cluster]:

                similarity = self._compute_similarity(
                    query_embedding,
                    entry.embedding
                )

                if similarity >= self.similarity_threshold:
                    self.stats["hit_count"] += 1
                    return entry, similarity

        for cluster_id, entries in self.cache.items():

            if cluster_id == query_cluster:
                continue

            for entry in entries:

                similarity = self._compute_similarity(
                    query_embedding,
                    entry.embedding
                )

                if similarity >= self.similarity_threshold:
                    self.stats["hit_count"] += 1
                    print(
                        f"Cache hit in different cluster: {cluster_id} vs {query_cluster}"
                    )
                    return entry, similarity

        self.stats["miss_count"] += 1
        return None

    def store(
        self,
        query: str,
        query_embedding: np.ndarray,
        result: Any
    ) -> CacheEntry:

        dominant_cluster = self._get_query_cluster(query_embedding)

        entry = CacheEntry(
            query=query,
            embedding=query_embedding,
            result=result,
            dominant_cluster=dominant_cluster,
            timestamp=datetime.now().isoformat()
        )

        if dominant_cluster not in self.cache:
            self.cache[dominant_cluster] = []

        self.cache[dominant_cluster].append(entry)

        return entry

    def get_stats(self) -> Dict:

        total_entries = sum(len(entries) for entries in self.cache.values())

        hit_rate = (
            self.stats["hit_count"] / self.stats["total_queries"]
            if self.stats["total_queries"] > 0
            else 0
        )

        return {
            "total_entries": total_entries,
            "hit_count": self.stats["hit_count"],
            "miss_count": self.stats["miss_count"],
            "total_queries": self.stats["total_queries"],
            "hit_rate": round(hit_rate, 3),
            "similarity_threshold": self.similarity_threshold,
            "cluster_distribution": {
                int(cluster_id): len(entries)
                for cluster_id, entries in self.cache.items()
            }
        }

    def clear(self):

        self.cache = {}

        self.stats = {
            "hit_count": 0,
            "miss_count": 0,
            "total_queries": 0
        }

        print("Cache cleared")

    def explore_threshold_impact(
        self,
        test_queries: List[str],
        test_embeddings: np.ndarray,
        thresholds: List[float] = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    ) -> Dict:

        print("\n=== Exploring Similarity Threshold Impact ===")

        results = {}

        for threshold in thresholds:

            print(f"\nTesting threshold: {threshold}")

            temp_cache = SemanticCache(
                gmm_model=self.gmm,
                similarity_threshold=threshold
            )

            for i, (query, embedding) in enumerate(
                zip(test_queries, test_embeddings)
            ):

                if i == 0:
                    temp_cache.store(query, embedding, f"result_{i}")
                    continue

                hit = temp_cache.lookup(query, embedding)

                if hit is None:
                    temp_cache.store(query, embedding, f"result_{i}")

            stats = temp_cache.get_stats()
            results[threshold] = stats

            print(f"Hit rate: {stats['hit_rate']}")
            print(f"Total entries: {stats['total_entries']}")

        self._save_threshold_analysis(results)

        return results

    def _save_threshold_analysis(self, results: Dict):

        output_path = self.output_dir / "threshold_analysis.json"

        def convert_types(obj):

            if isinstance(obj, np.integer):
                return int(obj)

            elif isinstance(obj, np.floating):
                return float(obj)

            elif isinstance(obj, np.ndarray):
                return obj.tolist()

            elif isinstance(obj, dict):
                return {
                    str(key): convert_types(value)
                    for key, value in obj.items()
                }

            elif isinstance(obj, list):
                return [convert_types(item) for item in obj]

            else:
                return obj

        json_results = convert_types(results)

        with open(output_path, "w") as f:
            json.dump(json_results, f, indent=2)

        print(f"\nSaved threshold analysis to {output_path}")

    def save_cache(self, filename: str = "semantic_cache.json"):

        def convert_types(obj):

            if isinstance(obj, np.integer):
                return int(obj)

            elif isinstance(obj, np.floating):
                return float(obj)

            elif isinstance(obj, np.ndarray):
                return obj.tolist()

            elif isinstance(obj, dict):
                return {
                    str(key): convert_types(value)
                    for key, value in obj.items()
                }

            elif isinstance(obj, list):
                return [convert_types(item) for item in obj]

            else:
                return obj

        cache_data = {
            "similarity_threshold": self.similarity_threshold,
            "stats": convert_types(self.stats),
            "entries": {
                str(cluster_id): [
                    convert_types(entry.to_dict())
                    for entry in entries
                ]
                for cluster_id, entries in self.cache.items()
            }
        }

        output_path = self.output_dir / filename

        with open(output_path, "w") as f:
            json.dump(cache_data, f, indent=2)

        print(f"Saved cache to {output_path}")