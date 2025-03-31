import gymnasium as gym
import numpy as np
import pickle
from visualization import plot_q_values_map, postprocess, plot_states_actions_distribution, plot_steps_and_rewards

class ReinforcementLearning:
    def __init__(self, env, episodes, lr=0.9, df=0.9, training=True, epsilon_greedy=True):
        self.env = env
        self.episodes = episodes
        self.learning_rate = lr
        self.discount_factor = df
        self.training = training
        self.epsilon_greedy = epsilon_greedy
        self.Qtable = np.zeros((env.observation_space.n, env.action_space.n)) if training else self.load_Qtable()

        self.epsilon = 1
        self.epsilon_decay = 0.0001
        self.rng = np.random.default_rng()

    def load_Qtable(self):
        f = open("CliffWalking_model.pkl", "rb")
        Qtable = pickle.load(f)
        f.close()
        return Qtable

    def save_Qtable(self):
        f = open("CliffWalking_model.pkl", "wb")
        pickle.dump(self.Qtable, f)
        f.close()

    def run(self):
        steps = []
        rewards = []
        all_states = []
        all_actions = []

        self.env.action_space.seed(1)

        for i in range(self.episodes):
            state = self.env.reset()[0]
            all_states.append(state)
            step = 0
            total_reward = 0
            done = False

            while not done:
                if self.epsilon_greedy:
                    if self.training and self.rng.random() < self.epsilon:
                        action = self.env.action_space.sample()
                    else:
                        action = np.argmax(self.Qtable[state,:])
                elif self.training:
                    action = self.env.action_space.sample()
                new_state, reward, terminated, truncated, _ = self.env.step(action)

                done = terminated or truncated
                if self.training:
                    all_actions.append(action)
                    all_states.append(new_state)

                    delta = reward + self.discount_factor * np.max(self.Qtable[new_state,:]) - self.Qtable[state, action]
                    self.Qtable[state, action] = self.Qtable[state, action] + self.learning_rate * delta

                total_reward += reward
                step += 1
                state = new_state

            steps.append(step)
            rewards.append(total_reward)

            if self.epsilon_greedy:
                self.epsilon = max(self.epsilon - self.epsilon_decay, 0)
                self.learning_rate = 0.0001 if not self.epsilon else self.learning_rate

        self.env.close()

        if self.training:
            self.save_Qtable()
            res = postprocess(self.episodes, rewards, steps)
            print(res)
            res.to_csv("learning.csv")
            plot_states_actions_distribution(all_states, all_actions)
            plot_steps_and_rewards(res)

        plot_q_values_map(self.Qtable)