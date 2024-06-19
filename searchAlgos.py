#linear search
def linear_search(arr, target):
    """
    Performs an optimized search using a set for frequent membership checks.
    """
    # Convert list to set for O(1) average-time complexity for membership checks
    arr_set = set(arr)
    
    if target in arr_set:
        # If found in the set, find its index in the original array
        for index, element in enumerate(arr):
            if element == target:
                return index
    return -1

# Example usage
arr = [3, 5, 2, 4, 9, 7]
target = 4
result = linear_search(arr, target)
print(f"Element {target} is at index {result}")  # Output: Element 4 is at index 3

#binary search
def binary_search(arr, target):
    """
    Performs binary search to find the index of the target element in a sorted array.
    Returns the index of the target if found, else returns -1.
    """
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = left + (right - left) // 2  # Prevents overflow in other languages
        
        # Check if target is present at mid
        if arr[mid] == target:
            return mid
        
        # If target greater, ignore left half
        elif arr[mid] < target:
            left = mid + 1
        
        # If target is smaller, ignore right half
        else:
            right = mid - 1
    
    # Target is not present in the array
    return -1

# Example usage
arr = [1, 2, 3, 4, 5, 6, 7, 8, 9]
target = 4
result = binary_search(arr, target)
print(f"Element {target} is at index {result}")  # Output: Element 4 is at index 3


#DFS
def dfs_iterative(graph, start):
    """
    Performs an iterative depth-first search on a graph starting from the given node.
    """
    visited = set()
    stack = [start]

    while stack:
        node = stack.pop()
        
        if node not in visited:
            visited.add(node)
            print(node)  # Process the node (can be replaced with other processing logic)

            # Add neighbors to stack in reverse order to maintain correct traversal order
            for neighbor in reversed(graph[node]):
                if neighbor not in visited:
                    stack.append(neighbor)
    
    return visited

# Example usage
graph = {
    'A': ['B', 'C'],
    'B': ['A', 'D', 'E'],
    'C': ['A', 'F'],
    'D': ['B'],
    'E': ['B', 'F'],
    'F': ['C', 'E']
}

start_node = 'A'
dfs_iterative(graph, start_node)

#BFS

from collections import deque

def bfs(graph, start):
    """
    Performs a breadth-first search on a graph starting from the given node.
    """
    visited = set()
    queue = deque([start])
    
    visited.add(start)
    
    while queue:
        node = queue.popleft()
        print(node)  # Process the node (can be replaced with other processing logic)
        
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    
    return visited

# Example usage
graph = {
    'A': ['B', 'C'],
    'B': ['A', 'D', 'E'],
    'C': ['A', 'F'],
    'D': ['B'],
    'E': ['B', 'F'],
    'F': ['C', 'E']
}

start_node = 'A'
bfs(graph, start_node)


