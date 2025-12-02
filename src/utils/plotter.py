# Just a placeholder for plotting utils if needed separately
# Currently plotting is handled in train.py and evaluate.py (video saving)
import matplotlib.pyplot as plt

def plot_rewards(rewards, title="Rewards", save_path=None):
    plt.figure()
    plt.plot(rewards)
    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    if save_path:
        plt.savefig(save_path)
    # plt.show() # Cannot show in headless
