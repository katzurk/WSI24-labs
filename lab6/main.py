import gymnasium as gym
import numpy as np
import pickle
from visualization import plot_q_values_map, postprocess
from reinforcement_learning import ReinforcementLearning

if __name__ == "__main__":
    render = False
    env = gym.make("CliffWalking-v0", render_mode="human" if render else None)
    rl = ReinforcementLearning(env, 1000)
    rl.run()
    print(rl.Qtable)
