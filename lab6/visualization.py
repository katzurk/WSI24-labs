from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

def postprocess(episodes, rewards, steps):
    """Convert the results of the simulation in dataframes."""
    res = pd.DataFrame(
        data={
            "Episodes": np.arange(1, episodes+1),
            "Rewards": rewards,
            "Steps": steps,
        }
    )

    return res


def qtable_directions_map(qtable, map_size=(4, 12)):
    """Get the best learned action & map it to arrows."""
    qtable_best_action = np.argmax(qtable, axis=1)
    directions = {0: "↑", 1: "→", 2: "↓", 3: "←"}

    qtable_directions = np.full(map_size, "", dtype=object)
    qtable_val_max = np.zeros(map_size)

    for state in range(qtable.shape[0]):
        row, col = divmod(state, map_size[1])
        best_action = qtable_best_action[state]
        if qtable[state, best_action] != 0:
            qtable_directions[row, col] = directions[best_action]
        qtable_val_max[row, col] = qtable[state, best_action]

    return qtable_val_max, qtable_directions

def plot_q_values_map(qtable, map_size=(4, 12)):
    """Plot the last frame of the simulation and the policy learned in one column."""
    qtable_val_max, qtable_directions = qtable_directions_map(qtable, map_size)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 8))

    sns.heatmap(
        qtable_val_max,
        annot=qtable_directions,
        fmt="",
        ax=ax,
        cmap=sns.color_palette("Blues", as_cmap=True),
        linewidths=0.7,
        linecolor="black",
        xticklabels=[],
        yticklabels=[],
        annot_kws={"fontsize": "xx-large"},
    ).set(title="Learned Q-values\nArrows represent best action")

    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_linewidth(0.7)
        spine.set_color("black")

    img_title = "policy.jpg"
    fig.savefig(img_title, bbox_inches="tight")
    plt.show()

def plot_states_actions_distribution(states, actions, map_size=(4, 12)):
    """Plot the distributions of states and actions."""
    labels = {"UP": 0, "RIGHT": 1, "DOWN": 2, "LEFT": 3}

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    sns.histplot(data=states, ax=ax[0], kde=True)
    ax[0].set_title("States")
    sns.histplot(data=actions, ax=ax[1])
    ax[1].set_xticks(list(labels.values()), labels=labels.keys())
    ax[1].set_title("Actions")
    fig.tight_layout()
    img_title = f"states_action_distribution.jpg"
    fig.savefig(img_title, bbox_inches="tight")
    plt.show()

def plot_steps_and_rewards(res_df):
    """Plot the steps and rewards from the dataframe."""
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    sns.lineplot(
        data=res_df, x="Episodes", y="Rewards", ax=ax[0]
    )
    ax[0].set(ylabel="Total rewards")

    sns.lineplot(data=res_df, x="Episodes", y="Steps", ax=ax[1])
    ax[1].set(ylabel="Steps number")

    fig.tight_layout()
    img_title = "steps_and_rewards.jpg"
    fig.savefig(img_title, bbox_inches="tight")
    plt.show()