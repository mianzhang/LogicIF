def f(n):
    # Initialize statistics
    recursive_calls = 0
    additions = 0

    def _f(k):
        nonlocal recursive_calls, additions
        # Count each invocation as a recursive call
        recursive_calls += 1
        if k <= 1:
            return k
        # Recurse and track each addition operation
        result = _f(k - 1) + _f(k - 2)
        additions += 1
        return result

    original_result = _f(n)
    return original_result, {'recursive_calls': recursive_calls, 'additions': additions}