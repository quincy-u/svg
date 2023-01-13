import gym
import torch
from gym import spaces
from dmanip.envs import WarpEnv


class IsaacGymWrapper:
    def __init__(self, env):
        assert env.num_envs == 1, "svg is currently incompatible with parallelized envs"
        self.env = env
        self._max_episode_steps = self.env.max_episode_length

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def observation_space(self):
        return self.env.observation_space

    def step(self, action):
        if not torch.is_tensor(action):
            action = torch.as_tensor(action, dtype=torch.float32).to(self.env.device)
        obs, rew, done, info = self.env.step(action)
        if isinstance(obs, dict):
            obs = obs['obs']
        obs = obs.cpu().numpy().squeeze()
        rew = rew.cpu().numpy().item()
        done = done.to(bool).cpu().numpy().item()
        info.update({k: v.cpu().numpy() for k, v in info.items() if torch.is_tensor(v)})
        return obs, rew, done, info

    def reset(self):
        self.env.reset_idx([0], [0])
        obs = self.env.reset()
        if isinstance(obs, dict):
            obs = obs['obs']
        obs = obs.cpu().numpy().squeeze()
        return obs


class WarpEnvWrapper:
    def __init__(self, env):
        assert isinstance(env, WarpEnv), "WarpEnvWrapper is only compatible with warp envs"
        assert env.requires_grad is False, "env.requires_grad should be `False`"
        assert env.num_envs == 1, "svg is currently incompatible with parallelized envs"
        self.env = env
        self._max_episode_steps = self.env.episode_length

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def observation_space(self):
        return self.env.observation_space

    def step(self, action):
        if not torch.is_tensor(action):
            action = torch.as_tensor(action, dtype=torch.float32).to(self.env.device)
        obs, rew, done, info = self.env.step(action)
        obs = obs.cpu().numpy().squeeze()
        rew = rew.cpu().numpy().item()  # assumes single env
        done = done.to(bool).cpu().numpy().item()
        info.update({k: v.cpu().numpy() for k, v in info.items() if torch.is_tensor(v)})
        return obs, rew, done, info

    def reset(self):
        obs = self.env.reset()
        obs = obs.cpu().numpy().squeeze()
        return obs
