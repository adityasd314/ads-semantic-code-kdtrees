class MinHeap:
    def __init__(self):
        self.heap = []

    def parent(self, i):
        return (i - 1) // 2

    def left_child(self, i):
        return 2 * i + 1

    def right_child(self, i):
        return 2 * i + 2

    def swap(self, i, j):
        self.heap[i], self.heap[j] = self.heap[j], self.heap[i]

    def insert(self, key):
        """
        Insert a new element into the min heap
        """
        self.heap.append(key)
        self._heapify_up(len(self.heap) - 1)

    def _heapify_up(self, i):
        """
        Maintain heap property after insertion
        """
        while i > 0 and self.heap[self.parent(i)] > self.heap[i]:
            parent_idx = self.parent(i)
            self.swap(i, parent_idx)
            i = parent_idx

    def extract_min(self):
        """
        Remove and return the minimum element
        """
        if not self.heap:
            return None
        
        if len(self.heap) == 1:
            return self.heap.pop()
        
        min_val = self.heap[0]
        self.heap[0] = self.heap.pop()
        self._heapify_down(0)
        
        return min_val

    def _heapify_down(self, i):
        """
        Maintain heap property after extraction
        """
        min_idx = i
        left = self.left_child(i)
        right = self.right_child(i)
        
        # Find the smallest among root, left child, and right child
        if left < len(self.heap) and self.heap[left] < self.heap[min_idx]:
            min_idx = left
        
        if right < len(self.heap) and self.heap[right] < self.heap[min_idx]:
            min_idx = right
        
        # If smallest is not the root, swap and continue heapifying
        if min_idx != i:
            self.swap(i, min_idx)
            self._heapify_down(min_idx)

    def get_top_k(self, k):
        """
        Return top k smallest elements while maintaining heap structure
        """
        top_k = []
        heap_copy = self.heap.copy()
        
        for _ in range(min(k, len(self.heap))):
            top_k.append(self.extract_min())
        
        # Restore the heap
        self.heap = heap_copy
        
        return top_k

    def __len__(self):
        return len(self.heap)