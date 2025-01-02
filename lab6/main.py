import gymnasium as gym
import numpy as np

def run(episodes, t_max, render=False):
    env = gym.make("CliffWalking-v0", render_mode="human" if render else None)
    Qtable = np.zeros((env.observation_space.n, env.action_space.n))
    learning_rate = 0.9
    discount_factor = 0.9

    env.action_space.seed(1)

    for i in range(episodes):
        t = 0
        state = env.reset()[0]
        terminated = False
        truncated = False

        while (not terminated and not truncated) or t < t_max:
            action = env.action_space.sample()
            new_state, reward, terminated, truncated, _ = env.step(action)
            delta = reward + discount_factor * np.max(Qtable[new_state,:]) - Qtable[state, action]
            Qtable[state, action] = Qtable[state, action] + learning_rate * delta
            state = new_state
            t += 1

    env.close()
    return Qtable

if __name__ == "__main__":
    q = run(100, 10)
    print(q)
