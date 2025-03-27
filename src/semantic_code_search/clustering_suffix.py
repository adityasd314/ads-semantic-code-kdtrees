import numpy as np
import pygtrie
import unicodedata
from tqdm import tqdm

def get_suffix_tree_clusters(dataset, distance_threshold):
    """
    Cluster functions using suffix-like similarity with progress bar
    
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
    
    # Utility function to normalize text
    def normalize_text(text):
        """Normalize text to remove special characters and convert to lowercase"""
        if not text:
            return ''
        # Remove accents and convert to lowercase
        text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
        return text.lower()
    
    # Cluster similar functions
    clustered_functions = {}
    used_indices = set()
    
    # Wrap the main processing with tqdm for progress tracking
    for idx, (func, embedding) in tqdm(
        enumerate(zip(functions, embeddings)), 
        total=len(functions), 
        desc="Clustering Functions", 
        unit="function"
    ):
        if idx in used_indices:
            continue
        
        # Normalize function text
        func_text = normalize_text(func.get('text', ''))
        if not func_text:
            continue
        
        # Create trie of suffixes
        suffix_trie = pygtrie.CharTrie()
        for i in range(len(func_text)):
            suffix_trie[func_text[i:]] = True
        
        cluster = [func]
        used_indices.add(idx)
        
        # Compare with other functions
        for compare_idx, (compare_func, compare_embedding) in enumerate(
            zip(functions, embeddings)
        ):
            if (compare_idx in used_indices or 
                compare_idx == idx):
                continue
            
            # Normalize comparison function text
            compare_text = normalize_text(compare_func.get('text', ''))
            if not compare_text:
                continue
            
            # Compute multiple similarity metrics
            embedding_distance = np.linalg.norm(embedding - compare_embedding)
            
            # Find longest common substring using trie
            def longest_common_substring(text1, text2):
                max_length = 0
                for i in range(len(text1)):
                    for j in range(len(text1) - i + 1):
                        substring = text1[i:i+j]
                        if substring and substring in text2:
                            max_length = max(max_length, len(substring))
                return max_length
            
            common_length = longest_common_substring(func_text, compare_text)
            text_similarity = common_length / max(len(func_text), len(compare_text))
            
            # Combine metrics
            combined_similarity = (embedding_distance + (1 - text_similarity)) / 2
            
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
