def f(arr):
    # Initialize statistics
    comparisons = 0
    swaps = 0
    # Original bubble sort logic with integrated tracking
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            comparisons += 1  # Count each element comparison
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swaps += 1      # Count each swap operation
    return arr, {'comparisons': comparisons, 'swaps': swaps}