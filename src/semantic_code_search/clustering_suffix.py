import numpy as np
from tqdm import tqdm

def get_suffix_tree_clusters(dataset, distance_threshold):
    """
    Cluster functions using a simplified similarity approach
    
    Args:
    - dataset: Dictionary containing embeddings and functions
    - distance_threshold: Maximum distance for clustering
    
    Returns:
    List of clusters, where each cluster is a dictionary with functions
    """
    embeddings = dataset.get('embeddings')
    functions = dataset.get('functions')
    
    # Normalize embeddings
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    # Prepare text features
    def extract_text_features(text):
        """Extract simple text features"""
        if not text:
            return set()
        # Extract words and short sequences
        words = set(text.split())
        sequences = set()
        # Add short sequences (2-3 words)
        for i in range(len(words)-1):
            if i+1 < len(words):
                sequences.add(' '.join(list(words[i:i+2])))
            if i+2 < len(words):
                sequences.add(' '.join(list(words[i:i+3])))
        return words.union(sequences)
    
    # Cluster similar functions
    clustered_functions = {}
    used_indices = set()
    
    # Use tqdm for progress tracking
    for idx in tqdm(range(len(functions)), desc="Clustering Functions"):
        if idx in used_indices:
            continue
        
        # Current function details
        func = functions[idx]
        embedding = embeddings[idx]
        
        # Extract text features
        func_features = extract_text_features(func.get('text', ''))
        
        # Start a new cluster
        cluster = [func]
        used_indices.add(idx)
        
        # Compare with other functions
        for compare_idx in range(len(functions)):
            if compare_idx in used_indices or compare_idx == idx:
                continue
            
            # Get comparison function details
            compare_func = functions[compare_idx]
            compare_embedding = embeddings[compare_idx]
            
            # Compute embedding distance
            embedding_distance = np.linalg.norm(embedding - compare_embedding)
            
            # Extract comparison text features
            compare_features = extract_text_features(compare_func.get('text', ''))
            
            # Compute feature similarity
            feature_similarity = len(func_features.intersection(compare_features)) / len(func_features.union(compare_features))
            
            # Combine metrics
            combined_similarity = (embedding_distance + (1 - feature_similarity)) / 2
            
            # Check if functions are similar enough
            if combined_similarity <= distance_threshold:
                cluster.append(compare_func)
                used_indices.add(compare_idx)
        
        # Only keep clusters with more than one function
        if len(cluster) > 1:
            clustered_functions[len(clustered_functions)] = {
                'avg_distance': distance_threshold,
                'functions': cluster
            }
    
    return list(clustered_functions.values())

# Add to clustering functions dictionary
