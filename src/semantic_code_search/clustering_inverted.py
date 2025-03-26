from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer

def build_inverted_index(functions):
    index = defaultdict(set)
    for i, func in enumerate(functions):
        tokens = set(func.get('text', '').split())  # Tokenize based on space
        for token in tokens:
            index[token].add(i)
    return index

def get_similar_functions(index, functions, similarity_threshold=0.5):
    tfidf = TfidfVectorizer()
    corpus = [func.get('text', '') for func in functions]
    tfidf_matrix = tfidf.fit_transform(corpus).toarray()
    
    clusters = defaultdict(list)
    for i, func in enumerate(functions):
        tokens = set(func.get('text', '').split())
        candidates = set()
        for token in tokens:
            candidates.update(index.get(token, []))
        candidates.discard(i)  # Remove self-matching
        
        similarities = [(j, np.dot(tfidf_matrix[i], tfidf_matrix[j])) for j in candidates]
        similarities = [(j, sim) for j, sim in similarities if sim >= similarity_threshold]
        similarities.sort(key=lambda x: -x[1])  # Sort by highest similarity
        
        if similarities:
            best_match = similarities[0][0]
            clusters[best_match].append(i)
    
    return clusters

def inverted_index_clustering(dataset, similarity_threshold=0.5):
    functions = dataset.get('functions', [])
    index = build_inverted_index(functions)
    clusters = get_similar_functions(index, functions, similarity_threshold)
    
    clustered_functions = []
    for cluster_id, indices in clusters.items():
        cluster_embeddings = np.array([functions[i].get('embedding') for i in indices if 'embedding' in functions[i]])
        if len(cluster_embeddings) < 2:
            continue  # Ignore clusters with insufficient data
        
        similarity_matrix = cosine_similarity(cluster_embeddings)
        avg_similarity = np.mean(similarity_matrix)
        
        if avg_similarity < similarity_threshold:
            continue  # Filter out weak clusters
        
        clustered_functions.append({
            'avg_similarity': avg_similarity,
            'functions': [functions[i] for i in indices]
        })
    
    return clustered_functions
