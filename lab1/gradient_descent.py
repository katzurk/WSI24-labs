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


def gradient_descent(start, step_length, limit, gradient, domain):
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


def generate_dataframe(array, parameter_index, start, step_length, limit, D):
    data = []
    for point in array:
        match parameter_index:
            case 0:
                points = gradient_descent(point, step_length, limit, deriv_f, D)
                column = "starting point"
            case 1:
                points = gradient_descent(start, point, limit, deriv_f, D)
                column = "step size"
            case 2:
                points = gradient_descent(start, step_length, point, deriv_f, D)
                column = "iterations limit"
            case _:
                raise ValueError("Invalid parameter index")
        result = f(points[-1])
        data.append([point, result])
    dataframe = pd.DataFrame(data, columns=[column, "found local minimum"]).round(2)
    return dataframe



def run_tests_function_f(start, step_length, limit):
    D = np.linspace(-4 * np.pi, 4 * np.pi)

    # start point
    start_points = np.random.uniform(-4 * np.pi, 4 * np.pi, size=200)
    data = generate_dataframe(start_points, 0, start, step_length, limit, D)
    data.to_csv("test1.csv")

    # step size
    step_sizes = np.arange(0, 2, 0.05)[1:]
    data = generate_dataframe(step_sizes, 1, start, step_length, limit, D)
    data.to_csv("test2.csv")

    # iterations limit
    limits = np.arange(5, 200, 5)
    data = generate_dataframe(limits, 2, start, step_length, limit, D)
    data.to_csv("test3.csv")

    return data



print(run_tests_function_f(7, 0.1, 100))
