"""
Visualize CarPole-v1.
"""

import os

import gymnasium as gym
import hydra
import torch
from omegaconf import DictConfig
from rl_exercises.week_4.dqn import DQNAgent, set_seed


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
    env = gym.make(env_name, render_mode="human")
    gym.wrappers.TimeLimit(env, max_episode_steps=1000)
    set_seed(env, seed)

    # build agent
    agent = DQNAgent(
        env,
        buffer_capacity=buf_cap,
        batch_size=batch_size,
        hidden_dim=width,
        num_hidden_layers=depth,
    )

    # load model of agent
    file_path = os.path.dirname(os.path.abspath(__file__))
    model_name = f"width_{width}_depth_{depth}_buffer_{buf_cap}_batch_{batch_size}_seed_{seed}_frames_{num_frames}_eval_{eval_interval}"
    model = torch.load(f"{file_path}/models/{model_name}.pth")
    agent.q.load_state_dict(model["parameters"])
    agent.optimizer.load_state_dict(model["optimizer"])

    # run agent
    state, _ = env.reset(seed=seed)
    done = False
    while not done:
        action = agent.predict_action(state, evaluate=True)
        state, reward, done, truncated, _ = env.step(action)
        env.render()
        if done or truncated:
            break

    env.close()


if __name__ == "__main__":
    main()
