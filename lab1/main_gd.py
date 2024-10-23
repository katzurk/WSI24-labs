from gradient_descent import *
from run_tests import *
import numpy as np
import argparse
import os


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("function", choices=["f", "g"])
    parser.add_argument("start_point", nargs="+", type=float)  # 1D or 2D point
    parser.add_argument("step_length", type=float)
    parser.add_argument("--step_limit", type=int, nargs="?", default=1000000)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--plot_filename", type=str, default="plot.jpg")
    parser.add_argument("--run_tests", action="store_true")

    args = parser.parse_args()
    domain = (
        np.linspace(-4 * np.pi, 4 * np.pi)
        if args.function == "f"
        else np.linspace(-2, 2)
    )

    if args.run_tests:
        main_dir = f"function_{args.function}"
        directories = [
            f"{main_dir}/starting_point",
            f"{main_dir}/step_length",
            f"{main_dir}/iterations_limit",
        ]

        for id_d, directory in enumerate(directories):
            os.makedirs(directory, exist_ok=True)

            for i in range(1, 11):
                sub_dir = os.path.join(directory, f"{i}")
                os.makedirs(sub_dir, exist_ok=True)
                if id_d == 0:
                    break

        run_tests_gradient_descent(
            args.start_point, args.function, args.step_length, domain, args.step_limit
        )

    if args.visualize:
        deriv_function = deriv_f if args.function == "f" else deriv_g
        points, steps = gradient_descent(
            args.start_point, args.step_length, deriv_function, domain, args.step_limit
        )
        title = f"step length: {args} steps: {steps}"
        if args.function == "f":
            plot_f(points, title, args.plot_filename)
        else:
            plot_g(points, title, args.plot_filename)

        print("location of found local minimum: ", np.round(points[-1], 2))
        print("number of steps: ", steps)


if __name__ == "__main__":
    parse_arguments()
