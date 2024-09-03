from copy import deepcopy
import gym
import gym.spaces as spaces
import numpy as np
import torch as th
import time
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt

from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv, _flatten_obs
from mani_skill2.vector.vec_env import VecEnvWrapper as MSVecEnvWrapper
from mani_skill2.vector.vec_env import stack_obs
import mani_skill2.envs

import hacman_ms.envs
from hacman_ms.vec_envs.vec_obs_processing import MSVecObsProcessing


class PCDMSVecEnvWrapper(MSVecObsProcessing, MSVecEnvWrapper):
    """
    A vectorized environment that processes the individual sub-environments in parallel using the ManiSkill2 VecEnv.

    This environment wrapper inherits from two classes: `MSVecObsProcessing` and `MSVecEnvWrapper`. The former provides
    observation processing functionality, while the latter provides vectorization functionality.

    Note that this environment uses the ManiSkill2 VecEnv, which is different from the Stable Baselines 3 `SubprocVecEnv`.
    The latter is used by the `PCDSubprocVecEnv` class, which directly inherits from `SubprocVecEnv`.

    Args:
        env (gym.Env): The environment to wrap.
        object_pcd_size (int, optional): The size of the point cloud data for objects. Defaults to 400.
        background_pcd_size (int, optional): The size of the point cloud data for the background. Defaults to 400.

    Methods:
        reset_wait(self, **kwargs): Resets the environment and returns the processed observation.
        step_wait(self): Runs one timestep of the environment and returns the processed observation, reward, done flag,
            and info dictionary.
    """
    def __init__(self, env,
                 object_pcd_size=400, 
                 background_pcd_size=400,
                 background_clip_radius=0.8,
                 voxel_downsample_size=0.01,
                 primitives=[],):
        MSVecEnvWrapper.__init__(self, env)
        self.sub_observation_space = deepcopy(self.observation_space)
        MSVecObsProcessing.__init__(self,
                                    self.sub_observation_space,
                                    object_pcd_size=object_pcd_size, 
                                    background_pcd_size=background_pcd_size,
                                    background_clip_radius=background_clip_radius,
                                    downsample=True,
                                    voxel_downsample_size=voxel_downsample_size,
                                    primitives=primitives)
    
    def reset_wait(self, **kwargs):
        raw_obs = self.venv.reset_wait(**kwargs)
        return self.observation(raw_obs)
    
    def step_wait(self):
        raw_obs, reward, done, info = self.venv.step_wait()
        obs = self.observation(raw_obs)
        info = self.info(info)
        return obs, reward, done, info

class PCDSubprocVecEnv(MSVecObsProcessing, SubprocVecEnv):
    """
    A vectorized environment that processes the individual sub-environments in parallel using the Stable Baselines 3
    SubprocVecEnv.

    This environment wrapper inherits from the `SubprocVecEnv` class and adds observation processing functionality.

    Note that this environment uses the Stable Baselines 3 `SubprocVecEnv`, which is different from the ManiSkill2 VecEnv
    used by the `PCDMSVecEnvWrapper` class.

    Args:
        env_fns (list): A list of functions that create the individual sub-environments.
        start_method (str, optional): The multiprocessing start method to use. Defaults to "fork".
        object_pcd_size (int, optional): The size of the point cloud data for objects. Defaults to 400.
        background_pcd_size (int, optional): The size of the point cloud data for the background. Defaults to 400.

    Methods:
        reset(self): Resets the environment and returns the processed observation.
        step_wait(self): Waits for the asynchronous step to complete and returns the processed observation, reward, done
            flag, and info dictionary. Note: SubprocVecEnv calls step_async before step_wait.
    """
    def __init__(self, env_fns,
                 start_method=None,
                 object_pcd_size=400, 
                 background_pcd_size=400,
                 background_clip_radius=0.8,
                 voxel_downsample_size=0.01,
                 primitives=[],):
        SubprocVecEnv.__init__(self, env_fns, start_method)
        self.sub_observation_space = deepcopy(self.observation_space)
        MSVecObsProcessing.__init__(self,
                                    self.sub_observation_space,
                                    object_pcd_size=object_pcd_size, 
                                    background_pcd_size=background_pcd_size,
                                    background_clip_radius=background_clip_radius,
                                    downsample=True,
                                    voxel_downsample_size=voxel_downsample_size,
                                    primitives=primitives)
    
    def step_wait(self):
        # Note: SubprocVecEnv calls step_async before step_wait.
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        raw_obs, rews, dones, infos = zip(*results)
        stacked_obs = stack_obs(raw_obs, self.sub_observation_space)
        obs = self.observation(stacked_obs)
        infos = self.info(infos)
        return obs, np.stack(rews), np.stack(dones), infos
    
    def reset(self):
        for remote in self.remotes:
            remote.send(("reset", None))
        raw_obs = [remote.recv() for remote in self.remotes]
        stacked_obs = stack_obs(raw_obs, self.sub_observation_space)
        obs = self.observation(stacked_obs)
        return obs
    