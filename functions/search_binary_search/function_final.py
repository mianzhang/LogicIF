def f(arr, target):
    interpolations = 0
    binary_steps = 0
    phases = 0
    n = len(arr)
    if n == 0:
        return -1, {'phases': phases, 'interpolations': interpolations, 'binary_steps': binary_steps}
    left, right = 0, n - 1
    phases = 1
    max_interp = n.bit_length()
    for _ in range(max_interp):
        interpolations += 1
        if left > right or arr[left] == arr[right]:
            break
        pos = left + (target - arr[left]) * (right - left) // (arr[right] - arr[left])
        if pos < left or pos > right:
            break
        if arr[pos] == target:
            return pos, {'phases': phases, 'interpolations': interpolations, 'binary_steps': binary_steps}
        if arr[pos] < target:
            left = pos + 1
        else:
            right = pos - 1
    phases = 2
    while left <= right:
        binary_steps += 1
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid, {'phases': phases, 'interpolations': interpolations, 'binary_steps': binary_steps}
        if arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1, {'phases': phases, 'interpolations': interpolations, 'binary_steps': binary_steps}