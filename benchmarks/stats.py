def median(values):
    s = sorted(values)
    n = len(s)
    mid = n // 2
    if n % 2 == 0:
        return (s[mid - 1] + s[mid]) / 2
    return s[mid]


def percentile(values, p):
    # TODO: implement this yourself
    # p is a number like 95, meaning "95th percentile"
    # approach: sort the values, find the index at position (p/100) * (n-1),
    # and return the value there (rounding the index is fine for our purposes)
    s = sorted(values)
    n = len(s)
    idx = int((p / 100) * (n - 1))
    return s[idx]


