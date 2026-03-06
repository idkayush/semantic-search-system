"""
Fuzzy Clustering Module

This module implements soft clustering using Gaussian Mixture Models (GMM).

Design Decisions:
-----------------
1. Using GMM instead of K-Means:
   - Provides probability distributions over clusters (not hard assignments)
   - Better captures the fuzzy nature of topic boundaries
   - Each document gets a membership score for each cluster

2. Number of clusters determination:
   - Using BIC (Bayesian Information Criterion) to find optimal K
   - Testing range from 15 to 30 clusters (around the original 20 categories)
   - Lower BIC indicates better model fit with appropriate complexity penalty

3. Dimensionality reduction for visualization:
   - Using UMAP for 2D visualization (preserves local structure better than PCA)
   - Helps visualize cluster boundaries and overlap
"""

import json
import pickle
from pathlib import Path
from typing import Tuple, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.mixture import GaussianMixture


class FuzzyClusterer:
    """Implements fuzzy clustering using Gaussian Mixture Models."""

    def __init__(self, output_dir: str = "./models"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.gmm = None
        self.n_clusters = None
        self.cluster_probabilities = None

    def find_optimal_clusters(
        self,
        embeddings: np.ndarray,
        min_clusters: int = 15,
        max_clusters: int = 30,
        n_init: int = 5
    ) -> Tuple[int, List[float]]:
        """
        Find optimal number of clusters using BIC.

        The BIC score balances model fit against complexity. We test a range
        around 20 (the original category count) to see if the natural semantic
        structure differs from the labeled categories.

        Args:
            embeddings: Document embeddings
            min_clusters: Minimum clusters to test
            max_clusters: Maximum clusters to test
            n_init: Number of initializations per model (for stability)

        Returns:
            Tuple of (optimal_k, bic_scores)
        """
        print(f"Finding optimal number of clusters (testing {min_clusters}-{max_clusters})...")

        bic_scores = []
        cluster_range = range(min_clusters, max_clusters + 1)

        for k in cluster_range:
            print(f"Testing k={k}...", end=" ")

            gmm = GaussianMixture(
                n_components=k,
                covariance_type="full",
                n_init=n_init,
                random_state=42
            )

            gmm.fit(embeddings)
            bic = gmm.bic(embeddings)
            bic_scores.append(bic)

            print(f"BIC={bic:.2f}")

        optimal_idx = int(np.argmin(bic_scores))
        optimal_k = min_clusters + optimal_idx

        print(f"\nOptimal number of clusters: {optimal_k}")
        print(f"BIC score: {bic_scores[optimal_idx]:.2f}")

        self._plot_bic_scores(list(cluster_range), bic_scores, optimal_k)

        return optimal_k, bic_scores

    def fit(
        self,
        embeddings: np.ndarray,
        n_clusters: int,
        n_init: int = 10
    ) -> np.ndarray:
        """
        Fit GMM to embeddings.

        Args:
            embeddings: Document embeddings
            n_clusters: Number of clusters
            n_init: Number of initializations (more = more stable)

        Returns:
            Cluster probability matrix (n_docs × n_clusters)
        """
        print(f"Fitting GMM with {n_clusters} clusters...")

        self.n_clusters = int(n_clusters)
        self.gmm = GaussianMixture(
            n_components=n_clusters,
            covariance_type="full",
            n_init=n_init,
            random_state=42,
            verbose=1
        )

        self.gmm.fit(embeddings)
        self.cluster_probabilities = self.gmm.predict_proba(embeddings)

        print(f"Clustering complete. Shape: {self.cluster_probabilities.shape}")
        return self.cluster_probabilities

    def get_dominant_clusters(self) -> np.ndarray:
        """
        Get the most probable cluster for each document.

        While we use soft clustering, it's useful to identify the dominant
        cluster for analysis and caching purposes.
        """
        return np.argmax(self.cluster_probabilities, axis=1)

    def analyze_clusters(
        self,
        embeddings: np.ndarray,
        documents: List[str],
        labels: List[int],
        category_names: List[str],
        top_n_words: int = 10
    ) -> Dict:
        """
        Analyze cluster quality and semantic coherence.

        This analysis shows:
        1. What content lives in each cluster
        2. What sits at cluster boundaries (uncertain documents)
        3. How clusters relate to original categories

        Args:
            embeddings: Document embeddings
            documents: Original documents
            labels: Original category labels
            category_names: Category name mappings
            top_n_words: Number of top words to extract per cluster

        Returns:
            Analysis dictionary with cluster statistics
        """
        print("\nAnalyzing clusters...")

        dominant_clusters = self.get_dominant_clusters()

        analysis = {
            "n_clusters": int(self.n_clusters),
            "cluster_sizes": {},
            "cluster_purity": {},
            "boundary_documents": [],
            "cluster_examples": {}
        }

        for cluster_id in range(self.n_clusters):
            cluster_mask = dominant_clusters == cluster_id
            cluster_docs = [documents[i] for i, m in enumerate(cluster_mask) if m]
            cluster_labels = [labels[i] for i, m in enumerate(cluster_mask) if m]

            analysis["cluster_sizes"][int(cluster_id)] = int(len(cluster_docs))

            if cluster_labels:
                label_counts = {}
                for label in cluster_labels:
                    label = int(label)
                    label_counts[label] = label_counts.get(label, 0) + 1

                most_common_label = max(label_counts, key=label_counts.get)
                purity = label_counts[most_common_label] / len(cluster_labels)

                analysis["cluster_purity"][int(cluster_id)] = {
                    "purity_score": float(purity),
                    "dominant_category": category_names[most_common_label],
                    "category_distribution": {
                        category_names[int(label)]: int(count)
                        for label, count in label_counts.items()
                    }
                }

            if cluster_docs:
                analysis["cluster_examples"][int(cluster_id)] = [
                    doc[:200] + "..." for doc in cluster_docs[:3]
                ]

        max_probs = np.max(self.cluster_probabilities, axis=1)
        uncertainty_threshold = 0.4

        uncertain_indices = np.where(max_probs < uncertainty_threshold)[0]

        print(f"\nFound {len(uncertain_indices)} highly uncertain documents")

        for idx in uncertain_indices[:10]:
            top_clusters = np.argsort(self.cluster_probabilities[idx])[-3:][::-1]
            top_probs = self.cluster_probabilities[idx][top_clusters]

            analysis["boundary_documents"].append({
                "document_preview": documents[int(idx)][:200] + "...",
                "original_category": category_names[int(labels[int(idx)])],
                "cluster_probabilities": {
                    f"cluster_{int(cluster_id)}": float(prob)
                    for cluster_id, prob in zip(top_clusters, top_probs)
                }
            })

        self._save_analysis(analysis)
        return analysis

    def visualize_clusters(
        self,
        embeddings: np.ndarray,
        labels: List[int],
        output_path: str = None
    ):
        """
        Create 2D visualization of clusters using UMAP.

        This helps visualize:
        - Cluster separation
        - Overlap between clusters
        - Relationship to original categories
        """
        try:
            from umap import UMAP

            print("Creating 2D visualization with UMAP...")

            reducer = UMAP(n_components=2, random_state=42)
            embedding_2d = reducer.fit_transform(embeddings)

            dominant_clusters = self.get_dominant_clusters()

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

            scatter1 = ax1.scatter(
                embedding_2d[:, 0],
                embedding_2d[:, 1],
                c=dominant_clusters,
                cmap="tab20",
                alpha=0.6,
                s=10
            )
            ax1.set_title(f"Discovered Clusters (k={self.n_clusters})")
            ax1.set_xlabel("UMAP 1")
            ax1.set_ylabel("UMAP 2")
            plt.colorbar(scatter1, ax=ax1, label="Cluster ID")

            scatter2 = ax2.scatter(
                embedding_2d[:, 0],
                embedding_2d[:, 1],
                c=labels,
                cmap="tab20",
                alpha=0.6,
                s=10
            )
            ax2.set_title("Original Categories (20 classes)")
            ax2.set_xlabel("UMAP 1")
            ax2.set_ylabel("UMAP 2")
            plt.colorbar(scatter2, ax=ax2, label="Category ID")

            plt.tight_layout()

            if output_path is None:
                output_path = self.output_dir / "cluster_visualization.png"

            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            print(f"Saved visualization to {output_path}")
            plt.close()

        except ImportError:
            print("UMAP not available. Skipping visualization.")

    def _plot_bic_scores(
        self,
        cluster_range: List[int],
        bic_scores: List[float],
        optimal_k: int
    ):
        """Plot BIC scores across different cluster counts."""
        plt.figure(figsize=(10, 6))
        plt.plot(cluster_range, bic_scores, "bo-", linewidth=2)
        plt.axvline(
            x=optimal_k,
            color="r",
            linestyle="--",
            label=f"Optimal k={optimal_k}"
        )
        plt.xlabel("Number of Clusters", fontsize=12)
        plt.ylabel("BIC Score", fontsize=12)
        plt.title("Bayesian Information Criterion vs Number of Clusters", fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)

        output_path = self.output_dir / "bic_scores.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved BIC plot to {output_path}")
        plt.close()

    def _save_analysis(self, analysis: Dict):
        """Save cluster analysis to JSON."""
        output_path = self.output_dir / "cluster_analysis.json"

        def convert_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {str(key): convert_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(item) for item in obj]
            else:
                return obj

        analysis_converted = convert_types(analysis)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(analysis_converted, f, indent=2)

        print(f"Saved cluster analysis to {output_path}")

    def save_model(self, filename: str = "gmm_model.pkl"):
        """Save the trained GMM model."""
        model_data = {
            "gmm": self.gmm,
            "n_clusters": self.n_clusters,
            "cluster_probabilities": self.cluster_probabilities
        }

        output_path = self.output_dir / filename
        with open(output_path, "wb") as f:
            pickle.dump(model_data, f)

        print(f"Saved GMM model to {output_path}")

    def load_model(self, filename: str = "gmm_model.pkl"):
        """Load a trained GMM model."""
        model_path = self.output_dir / filename

        with open(model_path, "rb") as f:
            model_data = pickle.load(f)

        self.gmm = model_data["gmm"]
        self.n_clusters = model_data["n_clusters"]
        self.cluster_probabilities = model_data["cluster_probabilities"]

        print(f"Loaded GMM model from {model_path}")


if __name__ == "__main__":
    from data_loader import NewsGroupsLoader
    from embeddings import EmbeddingGenerator

    loader = NewsGroupsLoader()
    data = loader.load_preprocessed()

    embedder = EmbeddingGenerator()
    embeddings = embedder.load_embeddings()

    clusterer = FuzzyClusterer()
    optimal_k, bic_scores = clusterer.find_optimal_clusters(embeddings)

    cluster_probs = clusterer.fit(embeddings, n_clusters=optimal_k)

    analysis = clusterer.analyze_clusters(
        embeddings,
        data["documents"],
        data["labels"],
        data["category_names"]
    )

    clusterer.visualize_clusters(embeddings, data["labels"])
    clusterer.save_model()

    print("\nCluster Analysis Summary:")
    print(f"Number of clusters: {analysis['n_clusters']}")
    print(f"Cluster sizes: {analysis['cluster_sizes']}")
    print("\nSample boundary document:")
    if analysis["boundary_documents"]:
        bd = analysis["boundary_documents"][0]
        print(f"Document: {bd['document_preview']}")
        print(f"Cluster probabilities: {bd['cluster_probabilities']}")