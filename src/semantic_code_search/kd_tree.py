import numpy as np

class KDTree:
    def __init__(self, points):
        """
        Initialize the K-D Tree
        
        Parameters:
        points (numpy.ndarray): Input points of shape (n_points, n_dimensions)
        """
        # Ensure input is a numpy array
        self.points = np.asarray(points)
        
        # Get dimensions
        self.n_points, self.n_dimensions = points.shape
        
        # Build the tree
        self.root = self._build_tree(np.arange(self.n_points), 0)

    def _build_tree(self, point_indices, depth):
        """
        Recursively build the K-D Tree
        
        Parameters:
        point_indices (numpy.ndarray): Indices of points to build tree from
        depth (int): Current depth in the tree
        
        Returns:
        dict: Node representation of the K-D Tree
        """
        # Base case: no points
        if len(point_indices) == 0:
            return None
        
        # Select axis based on depth
        axis = depth % self.n_dimensions
        
        # Sort point indices based on the current axis
        sorted_indices = point_indices[np.argsort(self.points[point_indices, axis])]
        
        # Choose median point
        median_idx = len(sorted_indices) // 2
        median_point_index = sorted_indices[median_idx]
        
        # Recursive tree construction
        return {
            'index': median_point_index,
            'axis': axis,
            'left': self._build_tree(sorted_indices[:median_idx], depth + 1),
            'right': self._build_tree(sorted_indices[median_idx + 1:], depth + 1)
        }

    def query(self, query_points, k=1):
        """
        Find k-nearest neighbors for given points
        
        Parameters:
        query_points (numpy.ndarray): Query points
        k (int): Number of nearest neighbors to find
        
        Returns:
        tuple: (distances, indices) of k-nearest neighbors
        """
        # Ensure query points is a 2D array
        query_points = np.asarray(query_points)
        if query_points.ndim == 1:
            query_points = query_points.reshape(1, -1)
        
        # Prepare output arrays
        distances = np.zeros((len(query_points), k))
        indices = np.zeros((len(query_points), k), dtype=int)
        
        # Find nearest neighbors for each query point
        for i, query_point in enumerate(query_points):
            # Priority queue to store k-nearest neighbors
            neighbors = []
            
            def search(node, depth=0):
                if node is None:
                    return
                
                # Current point
                point = self.points[node['index']]
                
                # Calculate distance
                dist = np.linalg.norm(point - query_point)
                
                # Update neighbors
                if len(neighbors) < k:
                    neighbors.append((dist, node['index']))
                    neighbors.sort()
                elif dist < neighbors[-1][0]:
                    neighbors[-1] = (dist, node['index'])
                    neighbors.sort()
                
                # Determine which subtree to search first
                axis = node['axis']
                if query_point[axis] < point[axis]:
                    first, second = node['left'], node['right']
                else:
                    first, second = node['right'], node['left']
                
                # Recursively search first subtree
                search(first, depth + 1)
                
                # Check if we need to search second subtree
                if len(neighbors) < k or \
                   abs(query_point[axis] - point[axis]) < neighbors[-1][0]:
                    search(second, depth + 1)
            
            # Start search from root
            search(self.root)
            
            # Extract distances and indices
            distances[i] = [n[0] for n in neighbors]
            indices[i] = [n[1] for n in neighbors]
        
        return distances, indices

    def __repr__(self):
        """
        String representation of the K-D Tree
        """
        return f"KDTree with {self.n_points} points in {self.n_dimensions} dimensions"