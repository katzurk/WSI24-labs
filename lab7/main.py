import numpy as np
import pandas as pd
from bayesian_network import *
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("statement", nargs=8, type=str)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    data = get_data()
    model = create_bayesian_network(data)
    visualize_network(model)
    res = create_inference(model, args.statement)
    for key, val in res.items():
        print(f"{key}: {val}")
