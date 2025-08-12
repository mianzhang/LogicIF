def f(arr):
    n = len(arr)
    bucket_assignments = 0
    insertion_comparisons = 0
    insertion_swaps = 0
    if n <= 1:
        return arr, {'bucket_assignments': bucket_assignments, 'insertion_comparisons': insertion_comparisons, 'insertion_swaps': insertion_swaps}
    mn = arr[0]
    mx = arr[0]
    for x in arr:
        if x < mn:
            mn = x
        if x > mx:
            mx = x
    rng = mx - mn
    k = int(n**0.5) or 1
    buckets = [[] for _ in range(k)]
    for x in arr:
        idx = int((x - mn)/(rng+1)*k)
        buckets[idx].append(x)
        bucket_assignments += 1
    result = []
    for b in buckets:
        for i in range(1, len(b)):
            key = b[i]
            j = i
            while j > 0:
                insertion_comparisons += 1
                if b[j-1] > key:
                    b[j] = b[j-1]
                    insertion_swaps += 1
                    j -= 1
                else:
                    break
            b[j] = key
        result += b
    return result, {'bucket_assignments': bucket_assignments, 'insertion_comparisons': insertion_comparisons, 'insertion_swaps': insertion_swaps}