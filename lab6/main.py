import gymnasium as gym
import numpy as np
import pickle
from visualization import plot_q_values_map

def run(episodes, training=True, render=False):
    env = gym.make("CliffWalking-v0", render_mode="human" if render else None)
    if training:
        Qtable = np.zeros((env.observation_space.n, env.action_space.n))
    else:
        f = open("stuff.pkl", "rb")
        Qtable = pickle.load(f)
        f.close()
    learning_rate = 0.9
    discount_factor = 0.9
    epsilon = 1
    epsilon_decay = 0.0001
    rng = np.random.default_rng()

    env.action_space.seed(1)

    for i in range(episodes):
        state = env.reset()[0]
        done = False

        while not done:
            if training:
                action = env.action_space.sample()
            else:
                action = np.argmax(Qtable[state,:])
            new_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            if training:
                delta = reward + discount_factor * np.max(Qtable[new_state,:]) - Qtable[state, action]
                Qtable[state, action] = Qtable[state, action] + learning_rate * delta
            state = new_state

    env.close()

    if training:
        f = open("stuff.pkl", "wb")
        pickle.dump(Qtable, f)
        f.close()

    plot_q_values_map(Qtable)

    return Qtable

if __name__ == "__main__":
    q = run(1, training=False, render=True)
    print(q)
