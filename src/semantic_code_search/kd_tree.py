import numpy as np

class KDTree:
    def __init__(self, points, logging=False):
        """
        Initialize the K-D Tree
        
        Parameters:
        points (numpy.ndarray): Input points of shape (n_points, n_dimensions)
        logging (bool): Whether to print debug information (default: False)
        """
        self.logging = logging  # Store logging preference
        self.points = np.asarray(points)
        self.n_points, self.n_dimensions = points.shape
        
        if self.logging:
            print(f"Building KD-Tree with {self.n_points} points in {self.n_dimensions} dimensions")
        
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
        if len(point_indices) == 0:
            return None
        
        axis = depth % self.n_dimensions
        sorted_indices = point_indices[np.argsort(self.points[point_indices, axis])]
        median_idx = len(sorted_indices) // 2
        median_point_index = sorted_indices[median_idx]

        if self.logging:
            print(f"Depth {depth}: Splitting on axis {axis}, median index {median_point_index}")

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
        query_points = np.asarray(query_points)
        if query_points.ndim == 1:
            query_points = query_points.reshape(1, -1)
        
        distances = np.zeros((len(query_points), k))
        indices = np.zeros((len(query_points), k), dtype=int)
        
        for i, query_point in enumerate(query_points):
            neighbors = []

            def search(node, depth=0):
                if node is None:
                    return
                
                point = self.points[node['index']]
                dist = np.linalg.norm(point - query_point)

                if len(neighbors) < k:
                    neighbors.append((dist, node['index']))
                    neighbors.sort()
                elif dist < neighbors[-1][0]:
                    neighbors[-1] = (dist, node['index'])
                    neighbors.sort()

                axis = node['axis']
                if query_point[axis] < point[axis]:
                    first, second = node['left'], node['right']
                else:
                    first, second = node['right'], node['left']

                search(first, depth + 1)

                if len(neighbors) < k or abs(query_point[axis] - point[axis]) < neighbors[-1][0]:
                    search(second, depth + 1)

            if self.logging:
                print(f"Query {i}: Searching nearest {k} neighbors")

            search(self.root)

            distances[i] = [n[0] for n in neighbors]
            indices[i] = [n[1] for n in neighbors]

        return distances, indices

    def __repr__(self):
        return f"KDTree with {self.n_points} points in {self.n_dimensions} dimensions"
