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


def gradient_descent(start, gradient, step_length, limit, domain):
    x = np.array(start)
    points = [x]
    steps = 0
    while steps <= limit and in_bounds(x, domain):
        d = np.array(gradient(x))
        x = x - step_length * d
        if not in_bounds(x, domain): break
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


def generate_dataframe(data_list, column_names, filename):
    data = pd.DataFrame(data_list, columns=column_names)
    data = data.round(2)
    data.to_csv(filename)
    return data



def run_tests_function_f(start, step_length, limit):
    D = np.linspace(-4 * np.pi, 4 * np.pi)

    # start point
    data_list = []
    start_points = np.random.uniform(-4 * np.pi, 4 * np.pi, size=200)
    for point in start_points:
        points = gradient_descent(point, deriv_f, step_length, limit, D)
        result = f(points[-1])
        data_list.append([point, result])
    generate_dataframe(data_list, ["starting_point", "found local minimum"], "test1.csv")

    # step size
    data_list = []
    step_sizes = np.arange(0, 2, 0.05)[1:]
    for step in step_sizes:
        points = gradient_descent(start, deriv_f, step, limit, D)
        result = f(points[-1])
        data_list.append([step, result])
    data = generate_dataframe(data_list, ["step size", "found local minimum"], "test2.csv")

    return data



print(run_tests_function_f(1, 0.1, 100))
