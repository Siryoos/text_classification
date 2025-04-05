import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
import spacy
import re
from typing import List, Dict, Union, Optional, Tuple, Any
from dataclasses import dataclass
import logging
import time
import joblib
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class TextClassifierConfig:
    """Configuration for text classification."""
    min_similarity: float = 0.7  # Minimum similarity threshold for clustering
    use_embeddings: bool = True  # Use word embeddings
    use_tfidf: bool = True  # Use TF-IDF vectorization
    embedding_weight: float = 0.5  # Weight for embedding vs TF-IDF similarity
    n_gram_range: Tuple[int, int] = (1, 2)  # n-gram range for TF-IDF
    stop_words: str = 'english'  # Stop words to ignore
    spacy_model: str = 'en_core_web_md'  # SpaCy model for embeddings
    batch_size: int = 1000  # Batch size for processing large datasets
    n_jobs: int = -1  # Number of parallel jobs
    min_cluster_size: int = 2  # Minimum samples for a cluster

class TextSimilarityClassifier:
    """
    A class to classify text data based on similarity and token overlap.
    
    This classifier uses a combination of TF-IDF vectors and word embeddings
    to compute similarity between texts, then groups them using clustering.
    """
    
    def __init__(self, config: Optional[TextClassifierConfig] = None):
        """Initialize the classifier with given configuration."""
        self.config = config or TextClassifierConfig()
        self.vectorizer = None
        self.nlp = None
        self.labels = None
        self.vectors = None
        self.embeddings = None
        self.original_texts = None
        self.clusterer = None
        self.similarity_matrix = None
        
        # Initialize NLP if embeddings are used
        if self.config.use_embeddings:
            logger.info(f"Loading spaCy model: {self.config.spacy_model}")
            try:
                self.nlp = spacy.load(self.config.spacy_model)
                # Disable components we don't need for faster processing
                self.nlp.select_pipes(enable=["tok2vec", "tagger"])
            except OSError:
                logger.warning(f"SpaCy model {self.config.spacy_model} not found. Downloading it now...")
                spacy.cli.download(self.config.spacy_model)
                self.nlp = spacy.load(self.config.spacy_model)
                self.nlp.select_pipes(enable=["tok2vec", "tagger"])
        
        # Initialize TF-IDF vectorizer if used
        if self.config.use_tfidf:
            self.vectorizer = TfidfVectorizer(
                ngram_range=self.config.n_gram_range,
                stop_words=self.config.stop_words,
                min_df=2,  # Ignore terms that appear in less than 2 documents
                max_df=0.9  # Ignore terms that appear in more than 90% of documents
            )
    
    @staticmethod
    def preprocess_text(text: str) -> str:
        """Preprocess text by removing special characters and standardizing format."""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep spaces
        text = re.sub(r'[^\w\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def compute_tfidf_similarity(self, texts: List[str]) -> np.ndarray:
        """Compute TF-IDF vectors and similarity matrix."""
        logger.info("Computing TF-IDF vectors for %d texts", len(texts))
        start_time = time.time()
        
        # Fit and transform texts to TF-IDF vectors
        self.vectors = self.vectorizer.fit_transform(texts)
        
        # Compute cosine similarity between all pairs
        similarity_matrix = cosine_similarity(self.vectors)
        
        logger.info("TF-IDF similarity computation completed in %.2f seconds", 
                   time.time() - start_time)
        return similarity_matrix
    
    def compute_embedding_similarity(self, texts: List[str]) -> np.ndarray:
        """Compute word embeddings and similarity matrix."""
        logger.info("Computing word embeddings for %d texts", len(texts))
        start_time = time.time()
        
        # Process texts in batches to avoid memory issues
        batch_size = self.config.batch_size
        all_vectors = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            logger.info(f"Processing batch {i//batch_size + 1} ({len(batch)} texts)")
            docs = list(self.nlp.pipe(batch))
            # Extract document vectors
            batch_vectors = np.array([doc.vector for doc in docs])
            all_vectors.append(batch_vectors)
        
        # Combine all batch vectors
        self.embeddings = np.vstack(all_vectors)
        
        # Normalize vectors for cosine similarity
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        normalized_vectors = self.embeddings / np.maximum(norms, 1e-10)
        
        # Compute cosine similarity
        similarity_matrix = np.dot(normalized_vectors, normalized_vectors.T)
        
        logger.info("Embedding similarity computation completed in %.2f seconds", 
                   time.time() - start_time)
        return similarity_matrix
    
    def combine_similarity_matrices(self, tfidf_sim: np.ndarray, emb_sim: np.ndarray) -> np.ndarray:
        """Combine TF-IDF and embedding similarity matrices with weighting."""
        w = self.config.embedding_weight
        return w * emb_sim + (1 - w) * tfidf_sim
    
    def cluster_texts(self, similarity_matrix: np.ndarray) -> np.ndarray:
        """Cluster texts based on similarity matrix using DBSCAN."""
        logger.info("Clustering texts using DBSCAN")
        start_time = time.time()
        
        # Convert similarity to distance (1 - similarity)
        distance_matrix = 1 - similarity_matrix
        
        # Use DBSCAN for clustering with precomputed distances
        eps = 1 - self.config.min_similarity  # Convert similarity threshold to distance
        self.clusterer = DBSCAN(
            eps=eps, 
            min_samples=self.config.min_cluster_size,
            metric='precomputed',
            n_jobs=self.config.n_jobs
        )
        
        labels = self.clusterer.fit_predict(distance_matrix)
        
        logger.info("Clustering completed in %.2f seconds", time.time() - start_time)
        unique_labels = set(labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        n_unclustered = np.sum(labels == -1)
        
        logger.info("Found %d clusters and %d unclustered texts", n_clusters, n_unclustered)
        
        return labels
    
    def fit(self, texts: List[str]) -> 'TextSimilarityClassifier':
        """Fit the classifier on the provided texts."""
        logger.info("Fitting classifier on %d texts", len(texts))
        start_time = time.time()
        
        # Store original texts
        self.original_texts = texts
        
        # Preprocess texts
        preprocessed_texts = [self.preprocess_text(text) for text in texts]
        
        # Compute similarities
        if self.config.use_tfidf and self.config.use_embeddings:
            tfidf_sim = self.compute_tfidf_similarity(preprocessed_texts)
            emb_sim = self.compute_embedding_similarity(preprocessed_texts)
            self.similarity_matrix = self.combine_similarity_matrices(tfidf_sim, emb_sim)
        elif self.config.use_tfidf:
            self.similarity_matrix = self.compute_tfidf_similarity(preprocessed_texts)
        elif self.config.use_embeddings:
            self.similarity_matrix = self.compute_embedding_similarity(preprocessed_texts)
        else:
            raise ValueError("At least one of use_tfidf or use_embeddings must be True")
        
        # Cluster texts
        self.labels = self.cluster_texts(self.similarity_matrix)
        
        logger.info("Classifier fitting completed in %.2f seconds", time.time() - start_time)
        return self
    
    def predict(self, new_texts: List[str], method: str = 'refit') -> np.ndarray:
        """
        Predict cluster labels for new texts.
        
        Args:
            new_texts: List of new text strings to classify
            method: Method to use for prediction:
                   'refit' - Re-fit the model with old and new texts
                   'similarity' - Assign based on similarity to cluster representatives
            
        Returns:
            Array of predicted cluster labels
        """
        if self.labels is None:
            raise ValueError("Classifier must be fit before predicting")
        
        logger.info("Predicting cluster labels for %d new texts", len(new_texts))
        
        if method == 'refit':
            # Combine with original texts for processing
            all_texts = self.original_texts + new_texts
            
            # Re-fit and predict
            original_count = len(self.original_texts)
            self.fit(all_texts)
            
            # Return only the labels for the new texts
            return self.labels[original_count:]
            
        elif method == 'similarity':
            # Preprocess new texts
            preprocessed_new = [self.preprocess_text(text) for text in new_texts]
            
            # Get representatives for each cluster
            representatives = self.get_cluster_representatives()
            
            # Assign each new text to the most similar cluster
            predictions = []
            
            for text in preprocessed_new:
                max_similarity = -1
                assigned_cluster = -1
                
                for cluster_id, rep_text in representatives.items():
                    if cluster_id == -1:
                        continue
                        
                    # Compute similarity between new text and representative
                    similarity = self._compute_text_similarity(text, rep_text)
                    
                    if similarity > max_similarity and similarity >= self.config.min_similarity:
                        max_similarity = similarity
                        assigned_cluster = cluster_id
                
                predictions.append(assigned_cluster)
            
            return np.array(predictions)
        
        else:
            raise ValueError(f"Unknown prediction method: {method}")
    
    def _compute_text_similarity(self, text1: str, text2: str) -> float:
        """Compute similarity between two texts."""
        similarities = []
        
        if self.config.use_tfidf and self.vectorizer is not None:
            # Transform texts using fitted vectorizer
            vec1 = self.vectorizer.transform([text1])
            vec2 = self.vectorizer.transform([text2])
            tfidf_sim = cosine_similarity(vec1, vec2)[0][0]
            similarities.append((1 - self.config.embedding_weight) * tfidf_sim)
        
        if self.config.use_embeddings and self.nlp is not None:
            # Compute document vectors
            doc1 = self.nlp(text1)
            doc2 = self.nlp(text2)
            
            # Compute cosine similarity
            vec1 = doc1.vector
            vec2 = doc2.vector
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 > 0 and norm2 > 0:
                emb_sim = np.dot(vec1, vec2) / (norm1 * norm2)
                similarities.append(self.config.embedding_weight * emb_sim)
            else:
                similarities.append(0)
        
        return sum(similarities)
    
    def get_clusters(self) -> Dict[int, List[str]]:
        """Get the clustered texts organized by cluster label."""
        if self.labels is None:
            raise ValueError("Classifier must be fit before getting clusters")
        
        clusters = {}
        for i, label in enumerate(self.labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(self.original_texts[i])
        
        return clusters
    
    def get_cluster_representatives(self) -> Dict[int, str]:
        """Get a representative text for each cluster."""
        clusters = self.get_clusters()
        representatives = {}
        
        for label, texts in clusters.items():
            if label == -1:  # Skip unclustered texts
                continue
                
            # Choose the text closest to the cluster centroid as representative
            if self.config.use_tfidf:
                # Get indices of texts in this cluster
                cluster_indices = [i for i, l in enumerate(self.labels) if l == label]
                
                # Get TF-IDF vectors for this cluster
                cluster_vectors = self.vectors[cluster_indices]
                
                # Compute centroid
                centroid = cluster_vectors.mean(axis=0)
                
                # Find closest text to centroid
                similarities = cosine_similarity(cluster_vectors, centroid)
                closest_idx = np.argmax(similarities)
                
                representatives[label] = texts[closest_idx]
            else:
                # Just use the first text as representative
                representatives[label] = texts[0]
        
        return representatives
    
    def get_cluster_keywords(self, n_keywords: int = 5) -> Dict[int, List[str]]:
        """Get top keywords for each cluster using TF-IDF weights."""
        if not self.config.use_tfidf:
            raise ValueError("TF-IDF must be enabled to extract keywords")
        
        if self.labels is None or self.vectors is None:
            raise ValueError("Classifier must be fit before extracting keywords")
        
        # Get feature names from vectorizer
        feature_names = np.array(self.vectorizer.get_feature_names_out())
        
        # Get clusters
        cluster_keywords = {}
        
        for label in set(self.labels):
            if label == -1:  # Skip unclustered texts
                continue
                
            # Get indices of texts in this cluster
            cluster_indices = [i for i, l in enumerate(self.labels) if l == label]
            
            # Get TF-IDF vectors for this cluster
            cluster_vectors = self.vectors[cluster_indices]
            
            # Sum TF-IDF scores for all documents in the cluster
            cluster_sum = cluster_vectors.sum(axis=0)
            
            # Convert to array and flatten
            cluster_sum = np.asarray(cluster_sum).flatten()
            
            # Get top keyword indices
            top_indices = cluster_sum.argsort()[-n_keywords:][::-1]
            
            # Get top keywords
            keywords = feature_names[top_indices].tolist()
            
            cluster_keywords[label] = keywords
        
        return cluster_keywords
    
    def save(self, filepath: str) -> None:
        """Save the trained classifier to a file."""
        # Create directory if it doesn't exist
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(self, filepath)
        logger.info(f"Classifier saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'TextSimilarityClassifier':
        """Load a trained classifier from a file."""
        classifier = joblib.load(filepath)
        logger.info(f"Classifier loaded from {filepath}")
        return classifier


def main():
    """Example usage of TextSimilarityClassifier."""
    # Sample data
    df = pd.DataFrame({
        'text': [
            "The quick brown fox jumps over the lazy dog",
            "A fast brown fox leaps over a sleepy dog",
            "The rapid brown fox bounds over the tired canine",
            "Python is a programming language",
            "Python language is used for programming",
            "SQL is a database query language",
            "SQL is used for database management",
            "Data science involves programming and statistics",
            "Machine learning is a subset of artificial intelligence",
            "Deep learning is a type of machine learning"
        ]
    })
    
    # Create configuration
    config = TextClassifierConfig(
        min_similarity=0.6,
        use_embeddings=True,
        use_tfidf=True,
        embedding_weight=0.6,
        n_gram_range=(1, 3),
        spacy_model='en_core_web_md'
    )
    
    # Initialize and fit classifier
    classifier = TextSimilarityClassifier(config)
    classifier.fit(df['text'].tolist())
    
    # Get and print clusters
    clusters = classifier.get_clusters()
    print("Text Clusters:")
    for label, texts in clusters.items():
        if label == -1:
            print(f"\nUnclustered texts ({len(texts)}):")
        else:
            print(f"\nCluster {label} ({len(texts)} texts):")
        for text in texts:
            print(f"  - {text}")
    
    # Get and print cluster keywords
    keywords = classifier.get_cluster_keywords(n_keywords=3)
    print("\nCluster Keywords:")
    for label, words in keywords.items():
        print(f"Cluster {label}: {', '.join(words)}")
    
    # Predict new texts
    new_texts = [
        "The speedy fox jumps across the dog",
        "SQL queries are used in databases",
        "Neural networks are part of deep learning"
    ]
    
    pred_labels = classifier.predict(new_texts)
    print("\nPredicted clusters for new texts:")
    for text, label in zip(new_texts, pred_labels):
        print(f"Text: {text}")
        print(f"Predicted Cluster: {label}")


if __name__ == "__main__":
    main()
