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


def plot_f(points):
    D = np.linspace(-4 * np.pi, 4 * np.pi)
    plt.plot(D, f(D))
    plt.plot(points, f(points), "-o")
    plt.show()


def plot_g(points):
    D = np.linspace(-2, 2)
    x, y = np.meshgrid(D, D)
    plt.contourf(x, y, g([x, y]))
    plt.plot([p[0] for p in points], [p[1] for p in points], "-o", color="red")
    plt.show()


def generate_dataframe(array, function, parameter_index, start, step_length, D, limit):
    data = []
    deriv = deriv_f if function == f else deriv_g
    for point in array:
        match parameter_index:
            case 0:
                points, steps = gradient_descent(point, step_length, deriv, D, limit)
                start = point
            case 1:
                points, steps = gradient_descent(start, point, deriv, D, limit)
                step_length = point
            case 2:
                points, steps = gradient_descent(start, step_length, deriv, D, point)
                steps = point
            case _:
                raise ValueError("Invalid parameter index")
        result = function(points[-1])
        data.append([start, step_length, points[-1], result, steps])
    dataframe = pd.DataFrame(data, columns=["starting point", "step_length", "location of the found local minimum", "value of the found local minimum", "number of steps"]).round(2)
    return dataframe


def repeat_dataframe_for_points(points_array, parameter_array, parameter_index, step_length, D, folder_name, function, limit):
    iteration = 1
    for start_point in points_array:
        data = generate_dataframe(parameter_array, function, parameter_index, start_point, step_length, D, limit)
        data.to_csv(f"function_{function.__name__}/{folder_name}/{iteration}.csv")
        iteration += 1


def run_tests_function_f(start, step_length, limit=1000000):
    D = np.linspace(-4 * np.pi, 4 * np.pi)

    # start point
    start_points = np.random.uniform(-4 * np.pi, 4 * np.pi, size=200)
    data = generate_dataframe(start_points, f, 0, start, step_length, D, limit)
    data.to_csv(f"function_f/starting_point/1.csv")

    # generate points for later tests
    start_points = np.random.uniform(-4 * np.pi, 4 * np.pi, size=10)

    # step length
    step_sizes = np.arange(0, 2, 0.05)[1:]
    repeat_dataframe_for_points(start_points, step_sizes, 1, step_length, D, "step_length", f, limit)

    # iterations limit
    limits = np.arange(5, 200, 5)
    repeat_dataframe_for_points(start_points, limits, 2, step_length, D, "iterations_limit", f, limit)

    return data


def run_tests_function_g(start, step_length, limit=1000000):
    D = np.linspace(-2, 2)

    # start point
    start_points = np.random.uniform(-2, 2, size=(200, 2))
    data = generate_dataframe(start_points, g, 0, start, step_length, D, limit)
    data.to_csv(f"function_g/starting_point/1.csv")

    # generate points for later tests
    start_points = np.random.uniform(-2, 2, size=(10, 2))

    # step length
    step_sizes = np.arange(0, 2, 0.05)[1:]
    repeat_dataframe_for_points(start_points, step_sizes, 1, step_length, D, "step_length", g, limit)

    # iterations limit
    limits = np.arange(5, 200, 5)
    repeat_dataframe_for_points(start_points, limits, 2, step_length, D, "iterations_limit", g, limit)

    return data


print(run_tests_function_g([0.4, 0.4], 0.3, 10000))
# gradient_descent([-1.72765308, 1.85675437],0.1, deriv_g, np.linspace(-2, 2))
# points, steps = gradient_descent([0.4, 0.5], 1.5, deriv_g, np.linspace(-2, 2))
# plot_g(points)
