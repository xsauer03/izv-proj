def calculate_dx (a, b, n):
    return (b-a)/float(n)

def rect_rule (f, a, b, n):
    total = 0.0
    dx = calculate_dx(a, b, n)
    for k in range (0, n):
        total = total + f((a + (k*dx)))
    return dx*total

def f(x):
    return 10 * x + 2

print(rect_rule(f, 0, 1, 10000))