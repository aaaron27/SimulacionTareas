def simpson_aux(f, ini, i, h, n):
    if i == n:
        return 0
    return (2 + 2 * (i % 2)) * f(ini + i * h) + simpson_aux(f, ini, i + 1, h, n)

def simpson(f, a, b, n):
    h = (b - a) / n
    return (h / 3) * (f(a) + f(b) + simpson_aux(f, a, 1, h, n))