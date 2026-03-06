"""
Data Loader for 20 Newsgroups Dataset

This module handles downloading, preprocessing, and preparing the 20 Newsgroups corpus.

Design Decisions:
-----------------
1. Using sklearn's fetch_20newsgroups for reliability and ease
2. Removing headers, footers, and quotes to reduce noise and focus on actual content
3. Filtering out documents shorter than 100 characters (likely noise/formatting artifacts)
4. Preserving original category labels for evaluation purposes
5. Creating clean text by removing excessive whitespace and special characters
"""

from sklearn.datasets import fetch_20newsgroups
import re
import json
from pathlib import Path
from typing import List, Dict, Tuple
import pickle


class NewsGroupsLoader:
    """Handles loading and preprocessing of 20 Newsgroups dataset."""
    
    def __init__(self, data_dir: str = "./data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def _clean_text(self, text: str) -> str:
        """
        Clean individual document text.
        
        - Remove excessive whitespace
        - Remove special characters that don't add semantic meaning
        - Preserve sentence structure and basic punctuation
        """
        # Remove multiple newlines
        text = re.sub(r'\n+', '\n', text)
        
        # Remove multiple spaces
        text = re.sub(r' +', ' ', text)
        
        # Remove tabs
        text = text.replace('\t', ' ')
        
        # Remove excessive punctuation (e.g., "!!!" -> "!")
        text = re.sub(r'([!?.]){2,}', r'\1', text)
        
        return text.strip()
    
    def load_and_preprocess(
        self, 
        subset: str = 'all',
        min_length: int = 100
    ) -> Tuple[List[str], List[int], List[str]]:
        """
        Load and preprocess the 20 Newsgroups dataset.
        
        Args:
            subset: 'train', 'test', or 'all'
            min_length: Minimum character length for documents (filters noise)
        
        Returns:
            Tuple of (documents, labels, category_names)
        """
        print(f"Loading 20 Newsgroups dataset (subset={subset})...")
        
        # Remove headers, footers, and quotes to focus on content
        # This is a deliberate choice to reduce noise and focus on semantic content
        newsgroups = fetch_20newsgroups(
            subset=subset,
            remove=('headers', 'footers', 'quotes'),
            shuffle=True,
            random_state=42
        )
        
        documents = []
        labels = []
        
        print("Preprocessing documents...")
        for doc, label in zip(newsgroups.data, newsgroups.target):
            # Clean the text
            cleaned = self._clean_text(doc)
            
            # Filter out very short documents (likely noise or formatting artifacts)
            # Min length of 100 chars ensures we have substantive content
            if len(cleaned) >= min_length:
                documents.append(cleaned)
                labels.append(int(label))
        
        category_names = newsgroups.target_names
        
        print(f"Loaded {len(documents)} documents across {len(category_names)} categories")
        print(f"Filtered out {len(newsgroups.data) - len(documents)} short documents")
        
        # Save preprocessed data
        self._save_preprocessed(documents, labels, category_names)
        
        return documents, labels, category_names
    
    def _save_preprocessed(
        self, 
        documents: List[str], 
        labels: List[int], 
        category_names: List[str]
    ):
        """Save preprocessed data for future use."""
        data = {
            'documents': documents,
            'labels': labels,
            'category_names': category_names,
            'num_documents': len(documents),
            'num_categories': len(category_names)
        }
        
        output_path = self.data_dir / 'preprocessed_data.pkl'
        with open(output_path, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"Saved preprocessed data to {output_path}")
        
        # Also save metadata as JSON for easy inspection
        metadata = {
            'num_documents': len(documents),
            'num_categories': len(category_names),
            'category_names': category_names,
            'avg_doc_length': sum(len(d) for d in documents) / len(documents)
        }
        
        metadata_path = self.data_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def load_preprocessed(self) -> Dict:
        """Load previously preprocessed data."""
        preprocessed_path = self.data_dir / 'preprocessed_data.pkl'
        
        if not preprocessed_path.exists():
            raise FileNotFoundError(
                "Preprocessed data not found. Run load_and_preprocess() first."
            )
        
        with open(preprocessed_path, 'rb') as f:
            data = pickle.load(f)
        
        print(f"Loaded {data['num_documents']} preprocessed documents")
        return data


if __name__ == "__main__":
    # Load and preprocess the dataset
    loader = NewsGroupsLoader()
    documents, labels, categories = loader.load_and_preprocess(subset='all')
    
    print("\nDataset Statistics:")
    print(f"Total documents: {len(documents)}")
    print(f"Categories: {len(categories)}")
    print(f"\nCategories: {categories}")
    print(f"\nSample document (first 200 chars):")
    print(documents[0][:200] + "...")
