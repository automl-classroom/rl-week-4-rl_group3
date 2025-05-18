"""
Deep Q-Learning implementation.
"""

from typing import Any, Dict, List, Tuple

import os

import gymnasium as gym
import hydra
import matplotlib
import matplotlib.pyplot as plt  # type: ignore[import]
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig
from rl_exercises.agent import AbstractAgent
from rl_exercises.week_4.buffers import ReplayBuffer
from rl_exercises.week_4.networks import QNetwork

matplotlib.use("Qt5Agg")  # Or 'Qt5Agg'


def set_seed(env: gym.Env, seed: int = 0) -> None:
    """
    Seed Python, NumPy, PyTorch and the Gym environment for reproducibility.

    Parameters
    ----------
    env : gym.Env
        The Gym environment to seed.
    seed : int
        Random seed.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    env.reset(seed=seed)
    # some spaces also support .seed()
    if hasattr(env.action_space, "seed"):
        env.action_space.seed(seed)
    if hasattr(env.observation_space, "seed"):
        env.observation_space.seed(seed)


class DQNAgent(AbstractAgent):
    """
    Deep Q-Learning agent with ε-greedy policy and target network.

    Derives from AbstractAgent by implementing:
      - predict_action
      - save / load
      - update_agent
    """

    def __init__(
        self,
        env: gym.Env,
        buffer_capacity: int = 10000,
        batch_size: int = 32,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_final: float = 0.01,
        epsilon_decay: int = 500,
        target_update_freq: int = 1000,
        hidden_dim: int = 64,
        num_hidden_layers: int = 1,
        seed: int = 0,
    ) -> None:
        """
        Initialize replay buffer, Q-networks, optimizer, and hyperparameters.

        Parameters
        ----------
        env : gym.Env
            The Gym environment.
        buffer_capacity : int
            Max experiences stored.
        batch_size : int
            Mini-batch size for updates.
        lr : float
            Learning rate.
        gamma : float
            Discount factor.
        epsilon_start : float
            Initial ε for exploration.
        epsilon_final : float
            Final ε.
        epsilon_decay : int
            Exponential decay parameter.
        target_update_freq : int
            How many updates between target-network syncs.
        seed : int
            RNG seed.
        """
        super().__init__(
            env,
            buffer_capacity,
            batch_size,
            lr,
            gamma,
            epsilon_start,
            epsilon_final,
            epsilon_decay,
            target_update_freq,
            seed,
        )
        self.env = env
        set_seed(env, seed)

        obs_dim = env.observation_space.shape[0]  # number of neurons on the input layer
        n_actions = (
            env.action_space.n
        )  # number of neurons on the output layer = number of different actions

        # main Q-network and frozen target
        self.q = QNetwork(obs_dim, n_actions, hidden_dim, num_hidden_layers)
        self.target_q = QNetwork(obs_dim, n_actions, hidden_dim, num_hidden_layers)
        self.target_q.load_state_dict(self.q.state_dict())

        self.optimizer = optim.Adam(self.q.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_capacity)

        # hyperparams
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_final = epsilon_final
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq

        self.total_steps = 0  # for ε decay and target sync

    def epsilon(self) -> float:
        """
        Compute current ε by exponential decay.

        Returns
        -------
        float
            Exploration rate.
        """
        # implement exponential‐decayin
        # ε = ε_final + (ε_start - ε_final) * exp(-total_steps / ε_decay)
        # Currently, it is constant and returns the starting value ε
        epsilon = self.epsilon_final + (
            self.epsilon_start - self.epsilon_final
        ) * np.exp(-self.total_steps / self.epsilon_decay)
        # maybe try np.interp()
        return epsilon

    def predict_action(
        self, state: np.ndarray, evaluate: bool = False
    ) -> Tuple[int, Dict]:
        """
        Choose action via ε-greedy (or purely greedy in eval mode).

        Parameters
        ----------
        state : np.ndarray
            Current observation.
        info : dict
            Gym info dict (unused here).
        evaluate : bool
            If True, always pick argmax(Q).

        Returns
        -------
        action : int
        info_out : dict
            Empty dict (compatible with interface).
        """
        state_t = torch.tensor(np.array(state), dtype=torch.float32).unsqueeze(0)

        if evaluate:
            # select purely greedy action from Q(s)
            with torch.no_grad():
                qvals = self.q(state_t)
                action = torch.argmax(qvals, dim=1).item()
        else:
            if np.random.rand() < self.epsilon():
                # sample random action
                action = self.env.action_space.sample()
            else:
                # select purely greedy action from Q(s)
                with torch.no_grad():
                    qvals = self.q(state_t)
                    action = torch.argmax(qvals, dim=1).item()

        return action

    def save(self, path: str) -> None:
        """
        Save model & optimizer state to disk.

        Parameters
        ----------
        path : str
            File path.
        """
        torch.save(
            {
                "parameters": self.q.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            path,
        )

    def load(self, path: str) -> None:
        """
        Load model & optimizer state from disk.

        Parameters
        ----------
        path : str
            File path.
        """
        checkpoint = torch.load(path)
        self.q.load_state_dict(checkpoint["parameters"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])

    def update_agent(
        self, training_batch: List[Tuple[Any, Any, float, Any, bool, Dict]]
    ) -> float:
        """
        Perform one gradient update on a batch of transitions.

        Parameters
        ----------
        training_batch : list of transitions
            Each is (state, action, reward, next_state, done, info).

        Returns
        -------
        loss_val : float
            MSE loss value.
        """
        # unpack
        states, actions, rewards, next_states, dones, _ = zip(*training_batch)  # noqa: F841
        s = torch.tensor(np.array(states), dtype=torch.float32)  # noqa: F841
        a = torch.tensor(np.array(actions), dtype=torch.int64).unsqueeze(1)  # noqa: F841
        r = torch.tensor(np.array(rewards), dtype=torch.float32).unsqueeze(1)  # noqa: F841
        s_next = torch.tensor(np.array(next_states), dtype=torch.float32)  # noqa: F841
        mask = torch.tensor(np.array(dones), dtype=torch.float32).unsqueeze(1)  # noqa: F841

        # pass batched states through self.q and gather Q(s,a)
        qvals = self.q(s)
        pred = torch.gather(input=qvals, dim=1, index=a)

        # compute TD target with frozen network
        with torch.no_grad():
            target_qvals = self.target_q(s_next)
            max_target_qvals = target_qvals.max(dim=1, keepdim=True)[0]
            target = r + self.gamma * (1 - mask) * max_target_qvals

        loss = nn.MSELoss()(pred, target)

        # gradient step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # occasionally sync target network
        if self.total_steps % self.target_update_freq == 0:
            self.target_q.load_state_dict(self.q.state_dict())

        self.total_steps += 1
        return float(loss.item())

    def train(
        self, num_frames: int, eval_interval: int = 1000
    ) -> List[Tuple[int, float]]:
        """
        Run a training loop for a fixed number of frames.

        Parameters
        ----------
        num_frames : int
            Total environment steps.
        eval_interval : int
            Every this many episodes, print average reward.
        """
        state, _ = self.env.reset()
        ep_reward = 0.0
        recent_rewards: List[float] = []
        log: List[Tuple[int, float]] = []

        for frame in range(1, num_frames + 1):
            action = self.predict_action(state)
            next_state, reward, done, truncated, _ = self.env.step(action)

            # store and step
            self.buffer.add(state, action, reward, next_state, done or truncated, {})
            state = next_state
            ep_reward += reward

            # update if ready
            if len(self.buffer) >= self.batch_size:
                # sample a batch from replay buffer
                batch = self.buffer.sample(self.batch_size)
                _ = self.update_agent(batch)

            if done or truncated:
                state, _ = self.env.reset()
                recent_rewards.append(ep_reward)
                ep_reward = 0.0
                # logging
                if len(recent_rewards) % 10 == 0:
                    # compute avg over last eval_interval episodes and print
                    avg = float(np.mean(recent_rewards[-eval_interval:]))
                    log.append((frame, avg))
                    print(
                        f"Frame {frame}, AvgReward({eval_interval}): {avg:.2f}, ε={self.epsilon():.3f}"
                    )

        if log[-1][0] <= num_frames - eval_interval:
            # compute avg over last eval_interval episodes and print
            avg = np.mean(recent_rewards[-eval_interval:])
            log.append((num_frames, avg))
            print(
                f"Frame {num_frames}, AvgReward({eval_interval}): {avg:.2f}, ε={self.epsilon():.3f}"
            )

        print("Training complete.")
        return log


@hydra.main(config_path="../configs/agent/", config_name="dqn", version_base="1.1")
def main(cfg: DictConfig):
    # build env
    env = gym.make(cfg.env.name)
    set_seed(env, cfg.seed)

    # define parameters to vary neural networtk architecture, replay buffer size and batch size
    nn_width = cfg.network.hidden_dim
    nn_depth = cfg.network.num_hidden_layers
    replay_buffer_sizes = cfg.agent.buffer_capacity
    batch_sizes = cfg.agent.batch_size

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
