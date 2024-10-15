import numpy as np
from matplotlib import pyplot as plt


def f(x):
    return 8*x + 8*np.sin(x)

def deriv_f(x):
    return 8 + 8*np.cos(x)

def gradient_descent(start, gradient, step_length, limit, domain):
    x = start
    points = [x]
    steps = 0
    while steps < limit and domain.min() < x < domain.max():
        d = gradient(x)
        x = x - step_length * d
        points.append(x)
        steps += 1
    return np.array(points)

xs = np.linspace(-4 * np.pi, 4 * np.pi)
points = gradient_descent(0.4, deriv_f, 0.1, 10, xs)
plt.plot(xs, f(xs))
plt.plot(points, f(points), '-o')
plt.show()

# print(gradient_descent(0, deriv_f, 0.2, 40, xes))