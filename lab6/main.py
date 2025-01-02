import gymnasium as gym
import numpy as np

def run():
    env = gym.make("CliffWalking-v0", render_mode="human")

    state = env.reset()[0]
    terminated = False
    truncated = False

    while not terminated and not truncated:
        action = env.action_space.sample()
        new_state, reward, terminated, truncated, _ = env.step(action)
        state = new_state

    env.close()

if __name__ == "__main__":
    run()