import numpy as np
from matplotlib import pyplot as plt
import pandas as pd


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


def in_bounds(point, domain):
    return (domain.min() < point).any() and (domain.max() > point).any()


def gradient_descent(start, step_length, gradient, domain, limit=1000000):
    x = np.array(start)
    points = [x]
    steps = 0
    while steps < limit and in_bounds(x, domain):
        d = np.array(gradient(x))
        if np.all(abs(d) < 1e-04):
            break
        x = x - step_length * d
        if not in_bounds(x, domain):
            break
        points.append(x)
        steps += 1
    return np.array(points), steps


def plot_f(points, title, filename):
    D = np.linspace(-4 * np.pi, 4 * np.pi)
    plt.plot(D, f(D))
    plt.plot(points, f(points), "-o")
    plt.title(title)
    plt.xlabel("oś X")
    plt.ylabel("oś Y")
    plt.savefig(filename)
    plt.close()


def plot_g(points, title, filename):
    D = np.linspace(-2, 2)
    x, y = np.meshgrid(D, D)
    plt.contour(x, y, g([x, y]), cmap="viridis")
    plt.plot([p[0] for p in points], [p[1] for p in points], "-o")
    plt.title(title)
    plt.xlabel("oś X")
    plt.ylabel("oś Y")
    plt.savefig(filename)
    plt.close()
