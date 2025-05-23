from gradient_descent import *
import pandas as pd
import numpy as np


def generate_dataframe(
    array, function, parameter_index, start, step_length, D, limit, iteration=1
):
    # generates the dataframe object over a range of values of a specific parameter
    data = []
    deriv = deriv_f if function == f else deriv_g
    plot = plot_f if function == f else plot_g
    for id_p, point in enumerate(array):
        match parameter_index:
            case 0:
                points, steps = gradient_descent(point, step_length, deriv, D, limit)
                start = point
                folder = "starting_point"
            case 1:
                points, steps = gradient_descent(start, point, deriv, D, limit)
                step_length = point
                folder = "step_length"
            case 2:
                points, steps = gradient_descent(start, step_length, deriv, D, point)
                steps = point
                folder = "iterations_limit"
            case _:
                raise ValueError("Invalid parameter index")

        if id_p == 0 or id_p == len(array) // 2 or id_p == len(array) - 1:
            title = f"step length: {step_length} steps: {steps}"
            plot(
                points,
                title,
                f"function_{function.__name__}/{folder}/{iteration}/{id_p}.jpg",
            )

        points = np.round(points, 5)
        result = function(points[-1])
        data.append([np.round(start, 2), step_length, points[-1], result, steps])
    columns_names = [
        "starting point",
        "step_length",
        "location of the found local minimum",
        "value of the found local minimum",
        "number of steps",
    ]
    scalar_columns = [
        "step_length",
        "value of the found local minimum",
        "number of steps",
    ]
    dataframe = pd.DataFrame(data, columns=columns_names).round(5)
    data_statisctics = (
        dataframe[scalar_columns]
        .agg(["mean", "std", "min", "max"])
        .transpose()
        .round(2)
    )
    return dataframe, data_statisctics


def repeat_dataframe_for_points(
    points_array,
    parameter_array,
    parameter_index,
    step_length,
    D,
    folder_name,
    function,
    limit,
):
    # repeats the data generation for every starting point in points_array
    iteration = 1
    for start_point in points_array:
        data, statistics = generate_dataframe(
            parameter_array,
            function,
            parameter_index,
            start_point,
            step_length,
            D,
            limit,
            iteration,
        )
        data.to_csv(f"function_{function.__name__}/{folder_name}/{iteration}.csv")
        statistics.to_csv(
            f"function_{function.__name__}/{folder_name}/{iteration}_stat.csv"
        )
        iteration += 1


def run_tests_gradient_descent(start, function_name, step_length, D, limit=1000000):
    # generate points for later tests
    function = f if function_name == "f" else g
    if function == f:
        start_points = np.random.uniform(D.min(), D.max(), size=25)
        test_points = np.random.uniform(D.min(), D.max(), size=10)
    else:
        start_points = np.random.uniform(D.min(), D.max(), size=(25, 2))
        test_points = np.random.uniform(D.min(), D.max(), size=(10, 2))

    # start point
    data, statistics = generate_dataframe(
        start_points, function, 0, start, step_length, D, limit
    )
    data.to_csv(f"function_{function.__name__}/starting_point/1.csv")
    statistics.to_csv(f"function_{function.__name__}/starting_point/1_stat.csv")

    # step length
    step_sizes = np.arange(0, 0.8, 0.05)[1:]
    repeat_dataframe_for_points(
        test_points, step_sizes, 1, step_length, D, "step_length", function, limit
    )

    # iterations limit
    limits = np.arange(1, 200, 10)
    repeat_dataframe_for_points(
        test_points, limits, 2, step_length, D, "iterations_limit", function, limit
    )

    return data
