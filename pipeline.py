"""
Main Pipeline Script

Runs the complete workflow:
1. Load and preprocess data
2. Generate embeddings
3. Create vector database
4. Perform fuzzy clustering
5. Analyze clusters

Run this before starting the API service.
"""

import argparse
from pathlib import Path

from data_loader import NewsGroupsLoader
from embeddings import EmbeddingGenerator
from vector_db import VectorDatabase
from clustering import FuzzyClusterer


def main(
    skip_data_load: bool = False,
    skip_embeddings: bool = False,
    skip_clustering: bool = False,
    min_clusters: int = 15,
    max_clusters: int = 30
):
    """
    Run the complete pipeline.
    
    Args:
        skip_data_load: Skip data loading if already done
        skip_embeddings: Skip embedding generation if already done
        skip_clustering: Skip clustering if already done
        min_clusters: Minimum clusters to test
        max_clusters: Maximum clusters to test
    """
    print("="*60)
    print("SEMANTIC SEARCH SYSTEM - PIPELINE")
    print("="*60)
    
    # Step 1: Load and preprocess data
    print("\n[1/5] Loading and preprocessing data...")
    loader = NewsGroupsLoader()
    
    if skip_data_load and Path("./data/preprocessed_data.pkl").exists():
        print("Skipping data load (using cached data)")
        data = loader.load_preprocessed()
    else:
        documents, labels, categories = loader.load_and_preprocess(subset='all')
        data = loader.load_preprocessed()
    
    print(f"✓ Loaded {data['num_documents']} documents")
    
    # Step 2: Generate embeddings
    print("\n[2/5] Generating embeddings...")
    embedder = EmbeddingGenerator()
    
    if skip_embeddings and Path("./models/corpus_embeddings.npy").exists():
        print("Skipping embedding generation (using cached embeddings)")
        embeddings = embedder.load_embeddings()
    else:
        embeddings = embedder.embed_documents(
            data['documents'],
            batch_size=32,
            show_progress=True
        )
        embedder.save_embeddings(embeddings)
    
    print(f"✓ Generated embeddings: {embeddings.shape}")
    
    # Step 3: Create vector database
    print("\n[3/5] Creating vector database...")
    db = VectorDatabase()
    
    # Check if collection exists
    try:
        db.load_collection()
        print("✓ Loaded existing vector database")
    except:
        print("Creating new vector database...")
        db.create_collection(
            embeddings=embeddings,
            documents=data['documents'],
            labels=data['labels'],
            category_names=data['category_names'],
            reset=True
        )
        print("✓ Vector database created")
    
    # Step 4: Fuzzy clustering
    print("\n[4/5] Performing fuzzy clustering...")
    clusterer = FuzzyClusterer()
    
    if skip_clustering and Path("./models/gmm_model.pkl").exists():
        print("Skipping clustering (using cached model)")
        clusterer.load_model()
        optimal_k = clusterer.n_clusters
    else:
        # Find optimal number of clusters
        optimal_k, bic_scores = clusterer.find_optimal_clusters(
            embeddings,
            min_clusters=min_clusters,
            max_clusters=max_clusters
        )
        
        # Fit with optimal k
        cluster_probs = clusterer.fit(embeddings, n_clusters=optimal_k)
        
        # Save model
        clusterer.save_model()
    
    print(f"✓ Clustering complete with k={optimal_k}")
    
    # Step 5: Cluster analysis and visualization
    print("\n[5/5] Analyzing clusters...")
    analysis = clusterer.analyze_clusters(
        embeddings=embeddings,
        documents=data['documents'],
        labels=data['labels'],
        category_names=data['category_names']
    )
    
    # Create visualizations
    clusterer.visualize_clusters(
        embeddings=embeddings,
        labels=data['labels']
    )
    
    print("✓ Analysis complete")
    
    # Update vector database with cluster assignments
    print("\nUpdating vector database with cluster assignments...")
    dominant_clusters = clusterer.get_dominant_clusters()
    db.update_cluster_assignments(dominant_clusters)
    print("✓ Cluster assignments stored")
    
    # Print summary
    print("\n" + "="*60)
    print("PIPELINE COMPLETE!")
    print("="*60)
    print(f"Documents: {data['num_documents']}")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    print(f"Number of clusters: {optimal_k}")
    print(f"Cluster sizes: {analysis['cluster_sizes']}")
    print("\nNext steps:")
    print("1. Review cluster analysis in ./models/cluster_analysis.json")
    print("2. Check visualizations in ./models/")
    print("3. Start the API: uvicorn api:app --reload")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the semantic search pipeline")
    
    parser.add_argument(
        "--skip-data",
        action="store_true",
        help="Skip data loading (use cached data)"
    )
    
    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Skip embedding generation (use cached embeddings)"
    )
    
    parser.add_argument(
        "--skip-clustering",
        action="store_true",
        help="Skip clustering (use cached model)"
    )
    
    parser.add_argument(
        "--min-clusters",
        type=int,
        default=15,
        help="Minimum number of clusters to test (default: 15)"
    )
    
    parser.add_argument(
        "--max-clusters",
        type=int,
        default=30,
        help="Maximum number of clusters to test (default: 30)"
    )
    
    args = parser.parse_args()
    
    main(
        skip_data_load=args.skip_data,
        skip_embeddings=args.skip_embeddings,
        skip_clustering=args.skip_clustering,
        min_clusters=args.min_clusters,
        max_clusters=args.max_clusters
    )
