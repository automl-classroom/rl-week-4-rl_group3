"""
Train different variations of Neural Networks.
"""

import os

import gymnasium as gym
import hydra
import matplotlib
import matplotlib.pyplot as plt  # type: ignore[import]
import torch
from omegaconf import DictConfig
from rl_exercises.week_4.dqn import DQNAgent, set_seed

matplotlib.use("Qt5Agg")  # Or 'Qt5Agg'


@hydra.main(config_path="../configs/agent/", config_name="dqn", version_base="1.1")
def main(cfg: DictConfig):
    env_name = cfg.env.name
    seed = cfg.seed
    num_frames = cfg.train.num_frames
    eval_interval = cfg.train.eval_interval
    buf_cap = cfg.agent.buffer_capacity
    batch_size = cfg.agent.batch_size
    width = cfg.network.hidden_dim
    depth = cfg.network.num_hidden_layers

    # build env
    env = gym.make(env_name)  # render_mode="human" to see agent training
    set_seed(env, seed)

    # build agent
    agent = DQNAgent(
        env,
        buffer_capacity=buf_cap,
        batch_size=batch_size,
        hidden_dim=width,
        num_hidden_layers=depth,
    )
    log = agent.train(num_frames, eval_interval)

    # save model of agent
    file_path = os.path.dirname(os.path.abspath(__file__))
    model_name = f"width_{width}_depth_{depth}_buffer_{buf_cap}_batch_{batch_size}_seed_{seed}_frames_{num_frames}_eval_{eval_interval}"
    os.makedirs(os.path.join(file_path, "models"), exist_ok=True)
    torch.save(
        {"parameters": agent.q.state_dict(), "optimizer": agent.optimizer.state_dict()},
        os.path.join(file_path, "models", f"{model_name}.pth"),
    )

    # plot all results
    if log and len(log) > 0:
        plt.figure(figsize=(10, 6))
        frames, avg_rewards = zip(*log)
        plt.plot(frames, avg_rewards, label="Average Reward")
        plt.xlabel("Frame")
        plt.ylabel(f"Average Reward ({cfg.train.eval_interval} episodes)")
        plt.suptitle("DQN Training Progress", fontsize=16, ha="center")
        plt.title(
            f"parameters: width = {width}, depth = {depth}, buffer capacity = {buf_cap}, batch size = {batch_size}, seed = {seed}, frames = {num_frames}, eval = {eval_interval}",
            fontsize=12,
            ha="center",
        )
        plt.grid(True)

        # save the figure
        file_path = os.path.dirname(os.path.abspath(__file__))
        plots_dir = os.path.join(file_path, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        file_name = f"width_{width}_depth_{depth}_buffer_{buf_cap}_batch_{batch_size}_seed_{seed}_frames_{num_frames}_eval_{eval_interval}"
        plt.savefig(os.path.join(plots_dir, f"{file_name}.png"), dpi=300)

        plt.show()


if __name__ == "__main__":
    main()
