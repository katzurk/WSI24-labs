from gradient_descent import gradient_descent, deriv_f, deriv_g, plot_f, plot_g
import numpy as np
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("function", choices=["f", "g"])
    parser.add_argument("start_point", nargs="+", type=float) # 1D or 2D point
    parser.add_argument("step_length", type=float)
    parser.add_argument("--step_limit", type=int, nargs='?', default=1000000)
    parser.add_argument("--visualize", action="store_true")


    args = parser.parse_args()
    function = True if args.function == "f" else False
    deriv_function = deriv_f if function else deriv_g
    domain = np.linspace(-4 * np.pi, 4 * np.pi) if function else np.linspace(-2, 2)
    points, steps = gradient_descent(
        args.start_point, args.step_length, deriv_function, domain, args.step_limit
    )

    if args.visualize:
        if function:
            plot_f(points)
        else:
            plot_g(points)

    print(points[-1])
    print(steps)


if __name__ == "__main__":
    parse_arguments()
