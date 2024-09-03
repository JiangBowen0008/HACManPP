import os, sys
from copy import deepcopy

import numpy as np
import gym
import gym.spaces as spaces

from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvObs, VecEnvStepReturn, VecEnvWrapper
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv, _flatten_obs
from hacman.utils.plotly_utils import plot_pcd, plot_action, plot_pcd_with_score
from typing import Any, Callable, List, Optional, Sequence, Tuple, Type, Union, Dict
from hacman.algos.location_policy import RandomLocation


def select_index_from_dict(data: dict, i: int):
    out = dict()
    for k, v in data.items():
        if isinstance(v, dict):
            out[k] = select_index_from_dict(v, i)
        else:
            out[k] = v[i]
    return out

class VecEnvwithLocationPolicy(VecEnvWrapper):
    def __init__(self,
                 venv: VecEnv,
                 observation_space: Optional[gym.spaces.Space] = None,
                 action_space: Optional[gym.spaces.Space] = None,
                 location_model: Callable[[Dict], Dict] = None):
        
        self.location_model = location_model
        if observation_space is not None:
            observation_space = self.update_observation_space(deepcopy(observation_space))
        super().__init__(venv, observation_space, action_space)
    
    def update_observation_space(self, obs_space: gym.spaces.Dict):
        """
        Updates the observation space with additional keys for action scores.
        """
        space_dict = obs_space.spaces
        # Add the HACMan spaces
        obj_pcd_size = extra_spaces['object_pcd_points'].shape[0]
        bg_pcd_size = extra_spaces['bg_pcd_points'].shape[0]
        pcd_size = obj_pcd_size + bg_pcd_size
        extra_spaces = {
            ## add all primitives action wrapper
            "action_location_score_all": gym.spaces.Box(-np.inf, np.inf, pcd_size, (len(self.venv.get_attr('prims')))),
            "action_location_score": gym.spaces.Box(-np.inf, np.inf, (pcd_size,)),
            "action_params": gym.spaces.Box(-np.inf, np.inf, (pcd_size, 3,)),}
        space_dict.update(extra_spaces)

        return spaces.Dict(space_dict)
    
    def process_obs(self, obs: Dict[str, np.ndarray]) -> VecEnvObs:
        location_infos = self.location_model.get_action(obs)
        obs.update(location_infos)
        keys = list(obs.keys())
        for i in range(self.venv.num_envs):
            sub_obs = {key: obs[key][i] for key in keys}
            assert self.env_method('set_prev_obs', sub_obs, indices=i)
        return obs
    
    def step_wait(self) -> VecEnvStepReturn:
        obs, rews, dones, infos = self.venv.step_wait()
        for i, done in enumerate(dones):
            if done:
                # NOTE: ensure that it will not be inplace modified when reset
                terminal_obs = infos[i]["terminal_observation"]
                location_info = self.location_model.get_action(terminal_obs)
                terminal_obs.update(location_info)
                infos[i]["terminal_observation"] = terminal_obs
                # if "terminal_observation" not in infos[i].keys():
                #     infos[i]["terminal_observation"] = select_index_from_dict(obs, i)
        obs = self.process_obs(obs) # New
        return obs, rews, dones, infos
    
    def reset(self) -> VecEnvObs:
        obs = self.venv.reset()
        self.process_obs(obs) # New
        return obs
    
class VecEnvwithPrimitivePolicy(VecEnvwithLocationPolicy):
    def update_observation_space(self, obs_space: gym.spaces.Dict):
        """
        Skip the update of observation space.
        """
        space_dict = obs_space.spaces
        return spaces.Dict(space_dict)


def flatten_obs(obs: List[Dict]):
    keys = list(obs[0].keys())
    return {key: np.stack([o[key] for o in obs]) for key in keys}
