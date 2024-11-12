import argparse
import pathlib
import json

import numpy as np
import pandas as pd
from solution_utils import generate_solution, evaluate_solution
from evolutionary_algorithm import Evolutionary_Algorithm
from generate_stats import *

MINI_CITIES_NUM = 5


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cities-path",
        type=pathlib.Path,
        required=True,
        help="Path to cities csv file",
    )
    parser.add_argument(
        "--problem-size",
        choices=["mini", "full"],
        default="mini",
        help="Run algorithm on full or simplified problem setup",
    )
    parser.add_argument("--start", type=str, default="Łomża")
    parser.add_argument("--finish", type=str, default="Częstochowa")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--plot_filename", type=str, default="plot")
    parser.add_argument("--run_tests", action="store_true")
    parser.add_argument("--find_solution", type=int, default=1)
    return parser.parse_args()


def load_data(args):
    data = pd.read_csv(args.cities_path)
    data_without_start_and_finish = data[
        ~((data.index == args.finish) | (data.index == args.start))
    ]
    if args.problem_size == "mini":
        city_names = (
            [args.start]
            + data_without_start_and_finish.sample(n=MINI_CITIES_NUM - 2).index.tolist()
            + [args.finish]
        )
    else:
        city_names = (
            [args.start] + data_without_start_and_finish.index.tolist() + [args.finish]
        )

    return data[city_names].loc[city_names]


def main():
    args = parse_args()
    if args.seed is not None:
        np.random.seed(args.seed)

    data = load_data(args)

    with open("config.json", "r") as file:
        config = json.load(file)

    size = config["population_size"]
    mp = config["mutation_prob"]
    cp = config["crossover_prob"]
    limit = config["limit"]

    if args.visualize:
        EA = Evolutionary_Algorithm(evaluate_solution, data, size, mp, cp, limit)
        EA.start_algorithm()
        generate_plot(EA, args.plot_filename)

    if args.run_tests:
        stats = avg_std_best_parameter(data)
        stats.to_csv("stats.csv")
        c_p = crossover_prob_tests(data)
        c_p.to_csv("crossover_p.csv")

    results = generate_results(evaluate_solution, data, size, mp, cp, limit, args.find_solution)
    solution = best_result(results)
    solution.loc[0, "solution"] = decode_solution(data, solution.loc[0, "solution"])
    print(solution)
    solution.to_csv("solution.csv")


if __name__ == "__main__":
    main()
