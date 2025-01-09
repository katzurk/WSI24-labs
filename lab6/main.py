import gymnasium as gym
import numpy as np
import pickle
from visualization import plot_q_values_map, postprocess
from reinforcement_learning import ReinforcementLearning
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning-rate", type=float, default=0.9)
    parser.add_argument("--discount-factor", type=int, default=0.9)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--training", action="store_true")
    parser.add_argument("--render", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    env = gym.make("CliffWalking-v0", render_mode="human" if args.render else None)
    rl = ReinforcementLearning(env, args.episodes, args.learning_rate, args.discount_factor, args.training)
    rl.run()
    print(rl.Qtable)
