#sort functions
sorted([5, 2, 3, 1, 4])
#output: [1, 2, 3, 4, 5]

a = [5, 2, 3, 1, 4]
a.sort()
print(a)
#output: [1, 2, 3, 4, 5]

# Bubble Sort O(n^2)
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        # Traverse through all array elements
        for j in range(0, n-i-1):
            # Swap if the element found is greater than the next element
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr

# Example usage:
arr = [64, 34, 25, 12, 22, 11, 90]
sorted_arr = bubble_sort(arr)
print("Sorted array is:", sorted_arr)

# Insertion Sort O(n^2)
def insertion_sort(arr):
    n = len(arr)
    for i in range(1, n):
        current_value = arr[i]
        j = i - 1
        # Move elements of arr[0..i-1], that are greater than current_value,
        # to one position ahead of their current position
        while j >= 0 and arr[j] > current_value:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = current_value
    return arr

# Example usage:
arr = [64, 34, 25, 12, 22, 11, 90]
sorted_arr = insertion_sort(arr)
print("Sorted array is:", sorted_arr)

# Selection Sort O(n^2)
def selection_sort(arr):
    n = len(arr)
    for i in range(n - 1):
        min_idx = i
        # Find the index of the minimum element in remaining unsorted array
        for j in range(i + 1, n):
            if arr[j] < arr[min_idx]:
                min_idx = j
        # Swap the found minimum element with the first element of the unsorted array
        if min_idx != i:
            arr[i], arr[min_idx] = arr[min_idx], arr[i]
    return arr

# Example usage:
arr = [64, 25, 12, 22, 11, 90]
sorted_arr = selection_sort(arr)
print("Sorted array is:", sorted_arr)

# Quick Sort O(n^2)
def quick_sort(arr):
    def partition(arr, low, high):
        pivot = arr[(low + high) // 2]  # Choosing the middle element as pivot
        i = low - 1
        j = high + 1
        while True:
            i += 1
            while arr[i] < pivot:
                i += 1
            j -= 1
            while arr[j] > pivot:
                j -= 1
            if i >= j:
                return j
            arr[i], arr[j] = arr[j], arr[i]

    def quick_sort_recursive(arr, low, high):
        if low < high:
            partition_index = partition(arr, low, high)
            quick_sort_recursive(arr, low, partition_index)
            quick_sort_recursive(arr, partition_index + 1, high)

    quick_sort_recursive(arr, 0, len(arr) - 1)
    return arr

# Example usage:
arr = [64, 25, 12, 22, 11, 90]
sorted_arr = quick_sort(arr)
print("Sorted array is:", sorted_arr)

# Merge Sort O(nlogn)
def merge_sort(arr):
    if len(arr) > 1:
        mid = len(arr) // 2  # Finding the mid of the array
        left_half = arr[:mid]  # Dividing the array elements into 2 halves
        right_half = arr[mid:]

        merge_sort(left_half)  # Sorting the first half
        merge_sort(right_half)  # Sorting the second half

        # Merging the sorted halves
        i = j = k = 0
        while i < len(left_half) and j < len(right_half):
            if left_half[i] < right_half[j]:
                arr[k] = left_half[i]
                i += 1
            else:
                arr[k] = right_half[j]
                j += 1
            k += 1

        # Checking if any element was left
        while i < len(left_half):
            arr[k] = left_half[i]
            i += 1
            k += 1

        while j < len(right_half):
            arr[k] = right_half[j]
            j += 1
            k += 1

    return arr

# Example usage:
arr = [64, 25, 12, 22, 11, 90]
sorted_arr = merge_sort(arr)
print("Sorted array is:", sorted_arr)

# Heap Sort O(nlogn)
def heap_sort(arr):
    n = len(arr)

    # Build max heap
    for i in range(n // 2 - 1, -1, -1):
        heapify(arr, n, i)

    # Extract elements one by one
    for i in range(n - 1, 0, -1):
        arr[i], arr[0] = arr[0], arr[i]  # Swap root (max element) with last element
        heapify(arr, i, 0)  # Heapify reduced heap

    return arr

def heapify(arr, n, i):
    largest = i  # Initialize largest as root
    left = 2 * i + 1  # Left child
    right = 2 * i + 2  # Right child

    # If left child exists and is greater than root
    if left < n and arr[left] > arr[largest]:
        largest = left

    # If right child exists and is greater than root
    if right < n and arr[right] > arr[largest]:
        largest = right

    # Change root, if needed
    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]
        heapify(arr, n, largest)

# Example usage:
arr = [64, 25, 12, 22, 11, 90]
sorted_arr = heap_sort(arr)
print("Sorted array is:", sorted_arr)

# Radix Sort O(nk)
def radix_sort(arr):
    # Find the maximum number to know number of digits
    max_number = max(arr)
    exp = 1  # Initialize exponent (1, 10, 100, ...)
    n = len(arr)

    # Do counting sort for every digit
    while max_number // exp > 0:
        counting_sort(arr, exp)
        exp *= 10

def counting_sort(arr, exp):
    n = len(arr)
    output = [0] * n  # Output array that will have sorted numbers
    count = [0] * 10  # Initialize count array for digits 0-9

    # Store count of occurrences in count[]
    for i in range(n):
        index = arr[i] // exp
        count[index % 10] += 1

    # Change count[i] so that count[i] now contains actual position of this digit in output[]
    for i in range(1, 10):
        count[i] += count[i - 1]

    # Build the output array
    i = n - 1
    while i >= 0:
        index = arr[i] // exp
        output[count[index % 10] - 1] = arr[i]
        count[index % 10] -= 1
        i -= 1

    # Copy the sorted elements into original array
    for i in range(n):
        arr[i] = output[i]

# Example usage:
arr = [170, 45, 75, 90, 802, 24, 2, 66]
radix_sort(arr)
print("Sorted array is:", arr)

# Bucket Sort O(n+k)
def bucket_sort(arr):
    # Create empty buckets
    num_buckets = 10
    buckets = [[] for _ in range(num_buckets)]

    # Insert elements into their respective buckets
    for num in arr:
        bucket_index = num // 10  # Simple mapping function for demonstration
        buckets[bucket_index].append(num)

    # Sort each bucket (using another sorting algorithm or recursively bucket sort)
    for bucket in buckets:
        insertion_sort(bucket)  # Using Insertion Sort for simplicity

    # Concatenate all buckets into the original array
    k = 0
    for i in range(num_buckets):
        for num in buckets[i]:
            arr[k] = num
            k += 1

def insertion_sort(arr):
    n = len(arr)
    for i in range(1, n):
        current_value = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > current_value:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = current_value

# Example usage:
arr = [64, 25, 12, 22, 11, 90, 5, 75, 36]
bucket_sort(arr)
print("Sorted array is:", arr)

# Topological Sort
from collections import defaultdict, deque

def topological_sort(graph):
    # Initialize variables
    topo_order = deque()
    visited = set()
    stack = []

    # Helper function to perform DFS
    def dfs(v):
        visited.add(v)
        for neighbor in graph[v]:
            if neighbor not in visited:
                dfs(neighbor)
        stack.append(v)

    # Perform DFS on all vertices
    for vertex in graph:
        if vertex not in visited:
            dfs(vertex)

    # Build topological order from stack
    while stack:
        topo_order.appendleft(stack.pop())

    return topo_order

# Example usage:
graph = {
    'A': ['C', 'D'],
    'B': ['D'],
    'C': ['E'],
    'D': ['E'],
    'E': []
}

sorted_vertices = topological_sort(graph)
print("Topologically sorted vertices:", list(sorted_vertices))


