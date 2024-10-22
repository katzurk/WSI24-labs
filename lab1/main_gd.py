from gradient_descent import *
from run_tests import *
import numpy as np
import argparse
import os


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("function", choices=["f", "g"])
    parser.add_argument("start_point", nargs="+", type=float) # 1D or 2D point
    parser.add_argument("step_length", type=float)
    parser.add_argument("--step_limit", type=int, nargs='?', default=1000000)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--run_tests", action="store_true")

    args = parser.parse_args()
    domain = np.linspace(-4 * np.pi, 4 * np.pi) if args.function == "f" else np.linspace(-2, 2)

    if args.run_tests:
        main_dir = f"function_{args.function}"
        directories = [f"{main_dir}/starting_point", f"{main_dir}/step_length", f"{main_dir}/iterations_limit"]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
        run_tests_gradient_descent(args.start_point, args.function, args.step_length, domain, args.step_limit)
        print("a")

    if args.visualize:
        deriv_function = deriv_f if args.function == "f" else deriv_g
        points, steps = gradient_descent(
            args.start_point, args.step_length, deriv_function, domain, args.step_limit
        )
        if args.function == "f":
            plot_f(points)
        else:
            plot_g(points)

        print("location of found local minimum: ", np.round(points[-1], 2))
        print("number of steps: ", steps)



if __name__ == "__main__":
    parse_arguments()
