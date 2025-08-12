def f(arr, target):
    # Initialize statistics
    iterations = 0
    comparisons = 0

    left, right = 0, len(arr) - 1
    while left <= right:
        iterations += 1
        mid = (left + right) // 2

        # First comparison: check equality
        comparisons += 1
        if arr[mid] == target:
            return mid, {'iterations': iterations, 'comparisons': comparisons}

        # Second comparison: decide direction
        comparisons += 1
        if arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    # Target not found
    return -1, {'iterations': iterations, 'comparisons': comparisons}