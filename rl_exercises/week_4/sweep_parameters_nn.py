"""
Train different variations of Neural Networks.

you need to edit the configs/agent/dqn.yaml file to set the parameters you want to vary
and you must use lists in order for this to work
"""

import os

import gymnasium as gym
import hydra
import matplotlib
import matplotlib.pyplot as plt  # type: ignore[import]
from omegaconf import DictConfig
from rl_exercises.week_4.networks import DQNAgent, set_seed

matplotlib.use("Qt5Agg")  # Or 'Qt5Agg'


@hydra.main(config_path="../configs/agent/", config_name="dqn", version_base="1.1")
def main(cfg: DictConfig):
    # build env
    env = gym.make(cfg.env.name)
    set_seed(env, cfg.seed)

    # define parameters to vary neural networtk architecture, replay buffer size and batch size
    nn_width = cfg.network.hidden_dim
    if not isinstance(nn_width, (list, tuple)):
        raise TypeError("cfg.network.hidden_dim must be a list or tuple of values.")
    nn_depth = cfg.network.num_hidden_layers
    if not isinstance(nn_depth, (list, tuple)):
        raise TypeError(
            "cfg.network.num_hidden_layers must be a list or tuple of values."
        )
    replay_buffer_sizes = cfg.agent.buffer_capacity
    if not isinstance(replay_buffer_sizes, (list, tuple)):
        raise TypeError("cfg.agent.buffer_capacity must be a list or tuple of values.")
    batch_sizes = cfg.agent.batch_size
    if not isinstance(batch_sizes, (list, tuple)):
        raise TypeError("cfg.agent.batch_size must be a list or tuple of values.")

    # vary the neural network architecture -> width
    results_width = []
    for width in nn_width:
        agent = DQNAgent(env, hidden_dim=width)
        log = agent.train(cfg.train.num_frames, cfg.train.eval_interval)
        results_width.append((width, log))

    # vary the neural network architecture -> depth
    results_depth = []
    for depth in nn_depth:
        agent = DQNAgent(env, num_hidden_layers=depth)
        log = agent.train(cfg.train.num_frames, cfg.train.eval_interval)
        results_depth.append((depth, log))

    # vary the replay buffer size
    results_buffer = []
    for buffer_size in replay_buffer_sizes:
        agent = DQNAgent(env, buffer_capacity=buffer_size)
        log = agent.train(cfg.train.num_frames, cfg.train.eval_interval)
        results_buffer.append((buffer_size, log))

    # vary the batch size
    results_batch = []
    for batch_size in batch_sizes:
        agent = DQNAgent(env, batch_size=batch_size)
        log = agent.train(cfg.train.num_frames, cfg.train.eval_interval)
        results_batch.append((batch_size, log))

    # Plot all results
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(18, 10))

    # plot results of nn width variation
    for width, log in results_width:
        if log:
            frames, avg_rewards = zip(*log)
            ax[0, 0].plot(frames, avg_rewards, label=f"width: {width}")
            ax[0, 0].set_title("Vary dimension of hidden layers of NN", fontsize=12)
            ax[0, 0].set_xlabel("Frame")
            ax[0, 0].set_ylabel(f"Average Reward ({cfg.train.eval_interval} episodes)")
            ax[0, 0].legend()
            ax[0, 0].grid(True)

    # plot results of nn depth variation
    for depth, log in results_depth:
        if log:
            frames, avg_rewards = zip(*log)
            ax[0, 1].plot(frames, avg_rewards, label=f"depth: {depth}")
            ax[0, 1].set_title("Vary number of layers of NN", fontsize=12)
            ax[0, 1].set_xlabel("Frame")
            ax[0, 1].set_ylabel(f"Average Reward ({cfg.train.eval_interval} episodes)")
            ax[0, 1].legend()
            ax[0, 1].grid(True)

    # plot results of buffer size variation
    for buffer_size, log in results_buffer:
        if log:
            frames, avg_rewards = zip(*log)
            ax[1, 0].plot(frames, avg_rewards, label=f"buffer size: {buffer_size}")
            ax[1, 0].set_title("Vary replay buffer size", fontsize=12)
            ax[1, 0].set_xlabel("Frame")
            ax[1, 0].set_ylabel(f"Average Reward ({cfg.train.eval_interval} episodes)")
            ax[1, 0].legend()
            ax[1, 0].grid(True)

    # plot results of batch size variation
    for batch_size, log in results_batch:
        if log:
            frames, avg_rewards = zip(*log)
            ax[1, 1].plot(frames, avg_rewards, label=f"batch size: {batch_size}")
            ax[1, 1].set_title("Vary batch size", fontsize=12)
            ax[1, 1].set_xlabel("Frame")
            ax[1, 1].set_ylabel(f"Average Reward ({cfg.train.eval_interval} episodes)")
            ax[1, 1].legend()
            ax[1, 1].grid(True)

    # set common labels and title
    fig.suptitle(
        "DQN Training Progress: Comparison of NN architectures, replay buffer size and batch size",
        fontsize=16,
    )
    fig.text(
        0.5,
        0.94,
        "default values: hidden dimension (width) = 64, hidden layers (depth) = 1, buffer cap = 10000, batch size = 32",
        ha="center",
        fontsize=12,
    )

    # save the figure
    file_path = os.path.dirname(os.path.abspath(__file__))
    fig.savefig(f"{file_path}/plots/NN_architectures.png", dpi=300)

    plt.show()


if __name__ == "__main__":
    main()
