import numpy as np
from semantic_code_search.kd_tree import KDTree
# from sklearn.neighbors import KDTree
from sklearn.cluster import AgglomerativeClustering

def clustering_kdtrees(dataset, distance_threshold, k_neighbors=5):
    # Normalize embeddings 
    embeddings = dataset.get('embeddings')
    normalized_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    # Construct K-D Tree for efficient nearest neighbor search
    # print(type(normalized_embeddings))
    kdtree = KDTree(normalized_embeddings)
    
    # Find k-nearest neighbors for each embedding
    # This step uses K-D Tree's efficient search capabilities
    distances, indices = kdtree.query(normalized_embeddings, k=k_neighbors)
    
    # Create a similarity matrix based on k-nearest neighbors
    n = len(normalized_embeddings)
    similarity_matrix = np.zeros((n, n))
    
    for i in range(n):
        neighbor_indices = indices[i]
        for idx in neighbor_indices:
            similarity_matrix[i, idx] = 1 - distances[i][list(indices[i]).index(idx)]
            similarity_matrix[idx, i] = similarity_matrix[i, idx]
    
    # Use the similarity matrix for clustering
    clustering_model = AgglomerativeClustering(
    n_clusters=None,
    distance_threshold=distance_threshold,
    metric='precomputed',
    linkage='average'
)

    
    # Fit clustering using the similarity matrix (converted to distance)
    cluster_labels = clustering_model.fit_predict(1 - similarity_matrix)
    
    # Cluster processing similar to original function
    clustered_functions = {}
    for idx, cluster_id in enumerate(cluster_labels):
        if cluster_id not in clustered_functions:
            clustered_functions[cluster_id] = []
        
        ds_entry = dataset.get('functions')[idx].copy()
        ds_entry['idx'] = idx
        clustered_functions[cluster_id].append(ds_entry)
    
    # Filter clusters with more than one function
    clusters = [
        {
            'avg_distance': np.mean([
                np.linalg.norm(normalized_embeddings[f1['idx']] - normalized_embeddings[f2['idx']])
                for i, f1 in enumerate(functions) 
                for f2 in functions[i+1:]
            ]),
            'functions': functions
        }
        for cluster_id, functions in clustered_functions.items() 
        if len(functions) > 1
    ]
    return clusters