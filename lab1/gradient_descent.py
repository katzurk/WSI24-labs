import numpy as np
from matplotlib import pyplot as plt


def f(x):
    return 8 * x + 8 * np.sin(x)


def deriv_f(x):
    return 8 + 8 * np.cos(x)


def g(point):
    x, y = point
    return 3 * x * y / (np.e ** (x**2 + y**2))


def deriv_g(point):
    x, y = point
    dx = (3 - 6 * x**2) * y * np.e ** (-(x**2) - (y**2))
    dy = (3 - 6 * y**2) * x * np.e ** (-(x**2) - (y**2))
    return (dx, dy)


def gradient_descent(start, gradient, step_length, limit, domain):
    x = np.array(start)
    points = [x]
    steps = 0
    while steps <= limit and (domain.min() < x).any() and (domain.max() > x).any():
        d = np.array(gradient(x))
        x = x - step_length * d
        points.append(x)
        steps += 1
    return np.array(points)


def plot_f(points):
    D = np.linspace(-4 * np.pi, 4 * np.pi)
    plt.plot(D, f(D))
    plt.plot(points, f(points), '-o')
    plt.show()


def plot_g(points):
    D = np.linspace(-2, 2)
    x, y = np.meshgrid(D, D)
    plt.contourf(x, y, g([x, y]))
    plt.plot([p[0] for p in points], [p[1] for p in points], "-o", color="red")
    plt.show()
