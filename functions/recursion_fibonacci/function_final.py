def f(n):
    dp_steps = 0
    matrix_mults = 0
    binary_steps = 0
    if n < 2:
        return n, {'dp_steps': 0, 'matrix_mults': 0, 'binary_steps': 0}
    threshold = int(n**0.5) or 1
    cache = [0] * (threshold + 1)
    cache[0], cache[1] = 0, 1
    for i in range(2, threshold + 1):
        cache[i] = cache[i-1] + cache[i-2]
        dp_steps += 1
    def mat_mult(a, b):
        nonlocal matrix_mults
        matrix_mults += 1
        return [[a[0][0]*b[0][0] + a[0][1]*b[1][0], a[0][0]*b[0][1] + a[0][1]*b[1][1]],
                [a[1][0]*b[0][0] + a[1][1]*b[1][0], a[1][0]*b[0][1] + a[1][1]*b[1][1]]]
    def mat_pow(exp):
        nonlocal binary_steps
        binary_steps += 1
        if exp == 1:
            return [[1,1],[1,0]]
        if exp % 2 == 0:
            half = mat_pow(exp // 2)
            return mat_mult(half, half)
        return mat_mult(mat_pow(exp - 1), [[1,1],[1,0]])
    if n <= threshold:
        result = cache[n]
    else:
        m = mat_pow(n - threshold)
        result = m[0][0] * cache[threshold] + m[0][1] * cache[threshold-1]
    return result, {'dp_steps': dp_steps, 'matrix_mults': matrix_mults, 'binary_steps': binary_steps}