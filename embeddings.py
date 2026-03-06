"""
Embedding Module for Semantic Search

This module handles text-to-vector conversion using sentence transformers.

Design Decisions:
-----------------
1. Using 'all-MiniLM-L6-v2' model:
   - Fast inference (important for API responsiveness)
   - 384-dimensional embeddings (balance between expressiveness and efficiency)
   - Good semantic understanding for general text
   - Lightweight (~80MB) compared to larger models
   
2. Batch processing for efficiency when embedding corpus
3. Normalization of embeddings for cosine similarity optimization
4. Caching embeddings to disk to avoid recomputation
"""

from sentence_transformers import SentenceTransformer
import numpy as np
from pathlib import Path
import pickle
from typing import List, Union
from tqdm import tqdm


class EmbeddingGenerator:
    """Generates and manages text embeddings using sentence transformers."""
    
    def __init__(
        self, 
        model_name: str = 'all-MiniLM-L6-v2',
        cache_dir: str = './models'
    ):
        """
        Initialize embedding generator.
        
        Args:
            model_name: Sentence transformer model to use
            cache_dir: Directory to cache model and embeddings
        """
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"Model loaded. Embedding dimension: {self.embedding_dim}")
    
    def embed_documents(
        self, 
        documents: List[str],
        batch_size: int = 32,
        normalize: bool = True,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Generate embeddings for a list of documents.
        
        Args:
            documents: List of text documents
            batch_size: Batch size for encoding
            normalize: Whether to L2-normalize embeddings (recommended for cosine similarity)
            show_progress: Show progress bar
        
        Returns:
            Array of embeddings with shape (num_docs, embedding_dim)
        """
        print(f"Generating embeddings for {len(documents)} documents...")
        
        embeddings = self.model.encode(
            documents,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=normalize
        )
        
        return embeddings
    
    def embed_query(
        self, 
        query: str,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Generate embedding for a single query.
        
        Args:
            query: Query text
            normalize: Whether to L2-normalize embedding
        
        Returns:
            Query embedding vector
        """
        embedding = self.model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=normalize
        )
        
        return embedding
    
    def save_embeddings(
        self, 
        embeddings: np.ndarray,
        filename: str = 'corpus_embeddings.npy'
    ):
        """Save embeddings to disk."""
        output_path = self.cache_dir / filename
        np.save(output_path, embeddings)
        print(f"Saved embeddings to {output_path}")
        
        # Save metadata
        metadata = {
            'model_name': self.model_name,
            'embedding_dim': self.embedding_dim,
            'num_embeddings': embeddings.shape[0]
        }
        
        metadata_path = self.cache_dir / f'{filename}.metadata.pkl'
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
    
    def load_embeddings(
        self, 
        filename: str = 'corpus_embeddings.npy'
    ) -> np.ndarray:
        """Load embeddings from disk."""
        embeddings_path = self.cache_dir / filename
        
        if not embeddings_path.exists():
            raise FileNotFoundError(f"Embeddings not found at {embeddings_path}")
        
        embeddings = np.load(embeddings_path)
        print(f"Loaded {embeddings.shape[0]} embeddings from {embeddings_path}")
        
        return embeddings
    
    @staticmethod
    def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Compute cosine similarity between two vectors.
        
        If vectors are already normalized, this simplifies to dot product.
        """
        # If already normalized (L2 norm = 1), just use dot product
        return float(np.dot(vec1, vec2))
    
    @staticmethod
    def compute_similarity_matrix(
        embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Compute pairwise cosine similarity matrix.
        
        For normalized embeddings, this is just the dot product matrix.
        """
        return np.matmul(embeddings, embeddings.T)


if __name__ == "__main__":
    # Test the embedding generator
    from data_loader import NewsGroupsLoader
    
    # Load data
    loader = NewsGroupsLoader()
    data = loader.load_preprocessed()
    documents = data['documents'][:100]  # Test with first 100
    
    # Generate embeddings
    embedder = EmbeddingGenerator()
    embeddings = embedder.embed_documents(documents)
    
    print(f"\nEmbeddings shape: {embeddings.shape}")
    print(f"Sample embedding (first 10 dims): {embeddings[0][:10]}")
    
    # Test query embedding
    query = "What are the best computer graphics cards?"
    query_emb = embedder.embed_query(query)
    print(f"\nQuery embedding shape: {query_emb.shape}")
    
    # Test similarity
    similarities = embeddings @ query_emb
    top_idx = np.argmax(similarities)
    print(f"\nMost similar document (similarity={similarities[top_idx]:.3f}):")
    print(documents[top_idx][:200] + "...")
