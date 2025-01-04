from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns


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
