"""
Vector Database Module

This module handles persistent storage and retrieval of document embeddings.

Design Decisions:
-----------------
1. Using ChromaDB:
   - Lightweight and easy to set up
   - Built-in support for metadata filtering
   - Persistent storage without external dependencies
   - Efficient similarity search with HNSW indexing
   
2. Storing metadata (original labels, cluster assignments) alongside embeddings
3. Using cosine similarity for retrieval (matches our normalized embeddings)
"""

import chromadb
from chromadb.config import Settings
import numpy as np
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import json


class VectorDatabase:
    """Manages vector storage and retrieval using ChromaDB."""
    
    def __init__(
        self, 
        persist_directory: str = "./data/chroma_db",
        collection_name: str = "newsgroups"
    ):
        """
        Initialize vector database.
        
        Args:
            persist_directory: Directory for persistent storage
            collection_name: Name of the collection
        """
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB with persistence
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory)
        )
        
        self.collection_name = collection_name
        self.collection = None
        
        print(f"Initialized vector database at {persist_directory}")
    
    def create_collection(
        self,
        embeddings: np.ndarray,
        documents: List[str],
        labels: List[int],
        category_names: List[str],
        reset: bool = False
    ):
        """
        Create and populate the vector collection.
        
        Args:
            embeddings: Document embeddings
            documents: Original document texts
            labels: Original category labels
            category_names: Category name mappings
            reset: Whether to delete existing collection
        """
        if reset and self.collection_name in [c.name for c in self.client.list_collections()]:
            self.client.delete_collection(name=self.collection_name)
            print(f"Deleted existing collection: {self.collection_name}")
        
        # Create collection with cosine similarity
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        print(f"Adding {len(documents)} documents to collection...")
        
        # Prepare data for insertion
        ids = [f"doc_{i}" for i in range(len(documents))]
        
        # Metadata includes original labels and category names
        metadatas = [
            {
                "label": int(labels[i]),
                "category": category_names[labels[i]],
                "doc_length": len(documents[i])
            }
            for i in range(len(documents))
        ]
        
        # Add to collection in batches
        batch_size = 1000
        for i in range(0, len(documents), batch_size):
            end_idx = min(i + batch_size, len(documents))
            
            self.collection.add(
                ids=ids[i:end_idx],
                embeddings=embeddings[i:end_idx].tolist(),
                documents=documents[i:end_idx],
                metadatas=metadatas[i:end_idx]
            )
            
            print(f"Added batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1}")
        
        print(f"Collection created with {self.collection.count()} documents")
    
    def load_collection(self):
        """Load existing collection."""
        self.collection = self.client.get_collection(name=self.collection_name)
        print(f"Loaded collection: {self.collection_name} ({self.collection.count()} documents)")
    
    def search(
        self,
        query_embedding: np.ndarray,
        n_results: int = 10,
        filter_metadata: Optional[Dict] = None
    ) -> Dict:
        """
        Search for similar documents.
        
        Args:
            query_embedding: Query vector
            n_results: Number of results to return
            filter_metadata: Optional metadata filters
        
        Returns:
            Search results with documents, distances, and metadata
        """
        if self.collection is None:
            self.load_collection()
        
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results,
            where=filter_metadata
        )
        
        return {
            'documents': results['documents'][0],
            'distances': results['distances'][0],
            'metadatas': results['metadatas'][0],
            'ids': results['ids'][0]
        }
    
    def search_by_cluster(
        self,
        query_embedding: np.ndarray,
        cluster_id: int,
        n_results: int = 10
    ) -> Dict:
        """
        Search within a specific cluster.
        
        Args:
            query_embedding: Query vector
            cluster_id: Cluster to search within
            n_results: Number of results
        
        Returns:
            Search results from specified cluster
        """
        return self.search(
            query_embedding,
            n_results,
            filter_metadata={"cluster_id": cluster_id}
        )
    
    def update_cluster_assignments(
        self,
        cluster_assignments: np.ndarray
    ):
        """
        Update cluster assignments in metadata.
        
        Args:
            cluster_assignments: Array of cluster IDs for each document
        """
        if self.collection is None:
            self.load_collection()
        
        print("Updating cluster assignments in database...")
        
        # Get all document IDs
        all_docs = self.collection.get()
        ids = all_docs['ids']
        metadatas = all_docs['metadatas']
        
        # Update metadata with cluster assignments
        for i, (doc_id, metadata) in enumerate(zip(ids, metadatas)):
            metadata['cluster_id'] = int(cluster_assignments[i])
        
        # Update in batches
        batch_size = 1000
        for i in range(0, len(ids), batch_size):
            end_idx = min(i + batch_size, len(ids))
            
            self.collection.update(
                ids=ids[i:end_idx],
                metadatas=metadatas[i:end_idx]
            )
        
        print("Cluster assignments updated")
    
    def get_collection_stats(self) -> Dict:
        """Get statistics about the collection."""
        if self.collection is None:
            self.load_collection()
        
        all_docs = self.collection.get()
        
        stats = {
            'total_documents': self.collection.count(),
            'collection_name': self.collection_name,
            'metadata_keys': list(all_docs['metadatas'][0].keys()) if all_docs['metadatas'] else []
        }
        
        return stats


if __name__ == "__main__":
    # Test the vector database
    from data_loader import NewsGroupsLoader
    from embeddings import EmbeddingGenerator
    
    # Load data
    loader = NewsGroupsLoader()
    data = loader.load_preprocessed()
    
    # Load or generate embeddings
    embedder = EmbeddingGenerator()
    try:
        embeddings = embedder.load_embeddings()
    except FileNotFoundError:
        embeddings = embedder.embed_documents(data['documents'])
        embedder.save_embeddings(embeddings)
    
    # Create vector database
    db = VectorDatabase()
    db.create_collection(
        embeddings=embeddings,
        documents=data['documents'],
        labels=data['labels'],
        category_names=data['category_names'],
        reset=True
    )
    
    # Test search
    query = "What are the latest developments in computer graphics?"
    query_emb = embedder.embed_query(query)
    results = db.search(query_emb, n_results=5)
    
    print("\nSearch Results:")
    for i, (doc, dist, meta) in enumerate(zip(
        results['documents'], 
        results['distances'], 
        results['metadatas']
    )):
        print(f"\n{i+1}. Category: {meta['category']} (distance: {dist:.3f})")
        print(f"   {doc[:150]}...")
