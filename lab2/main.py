import argparse
import pathlib

import numpy as np
import pandas as pd
from solution_utils import generate_solution, evaluate_solution
from evolutionary_algorithm import Evolutionary_Algorithm

MINI_CITIES_NUM = 5


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cities-path", type=pathlib.Path, required=True, help="Path to cities csv file")
    parser.add_argument(
        "--problem-size",
        choices=["mini", "full"],
        default="mini",
        help="Run algorithm on full or simplified problem setup",
    )
    parser.add_argument("--start", type=str, default="Łomża")
    parser.add_argument("--finish", type=str, default="Częstochowa")
    parser.add_argument("--seed", type=int)
    return parser.parse_args()


def load_data(args):
    data = pd.read_csv(args.cities_path)
    data_without_start_and_finish = data[~((data.index == args.finish) | (data.index == args.start))]
    if args.problem_size == "mini":
        city_names = (
            [args.start] + data_without_start_and_finish.sample(n=MINI_CITIES_NUM - 2).index.tolist() + [args.finish]
        )
    else:
        city_names = [args.start] + data_without_start_and_finish.index.tolist() + [args.finish]

    return data[city_names].loc[city_names]


def main():
    args = parse_args()
    if args.seed is not None:
        np.random.seed(args.seed)

    data = load_data(args)
    population = [generate_solution(data) for i in range(20)]
    EA = Evolutionary_Algorithm(evaluate_solution, data, population, 0.8, 0.6, 10)
    o_, x_ = EA.start_algorithm()
    print(data)
    print(o_, x_)


if __name__ == "__main__":
    main()
