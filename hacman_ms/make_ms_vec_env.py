from typing import Any, Callable, List, Optional, Sequence, Tuple, Type, Union, Dict
from tqdm.notebook import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch as th
from functools import partial

import gym
import gym.spaces as spaces
from gym.wrappers.time_limit import TimeLimit

from stable_baselines3.common.vec_env import VecMonitor
import mani_skill2.envs
from mani_skill2.utils.common import flatten_dict_space_keys, flatten_state_dict
from mani_skill2.utils.wrappers.sb3 import ContinuousTaskWrapper
from mani_skill2.vector import VecEnv, make as make_vec_env
from mani_skill2.vector.wrappers.sb3 import SB3VecEnvWrapper

from hacman.utils.transformations import transform_point_cloud, to_pose_mat, sample_idx
from hacman.envs.sim_envs.base_env import RandomLocation
from hacman.envs.env_wrappers import HACManActionWrapper, FlatActionWrapper, RegressedActionWrapper, FlatActionLogitWrapper

import hacman_ms.envs
import hacman_ms.env_wrappers.primitives
from hacman_ms.vec_envs import PCDSubprocVecEnv, PCDMSVecEnvWrapper
from hacman_ms.env_wrappers import HACManMobileActionWrapper


def make_maniskill_venv(
        env_id: str, num_envs: int, 
        max_episode_steps: int = 10,
        reward_mode: str = "dense",
        action_mode: str = "per_point",
        seed: int = 0,
        record_video: bool = False,
        vectorization: str = "ms",
        env_kwargs: Dict = {},
        env_wrapper_kwargs: Dict = {},
        vecenv_kwargs: Dict = {},):
    """
    Creates a ManiSkill2 vectorized environment with the specified parameters.

    Args:
        env_id (str): The ID of the environment to create.
        num_envs (int): The number of environments to create.
        max_episode_steps (int, optional): The maximum number of steps per episode. Defaults to 10.
        reward_mode (str, optional): The reward mode to use. Defaults to "dense".
        seed (int, optional): The random seed to use. Defaults to 0.
        record_video (bool, optional): Whether to record videos of the environment. Defaults to False.
        vectorization (str, optional): The type of vectorization to use. Can be either "ms" (ManiSkill2 vectorization) or "sb3" (Stable Baselines 3 vectorization). Defaults to "ms".
        wrapper_kwargs (optional): Additional keyword arguments to pass to the environment wrapper.
        vecenv_kwargs (optional): Additional keyword arguments to pass to the vectorized environment.

    Returns:
        env (VectorEnvWrapper): The created vectorized environment.

    Example usage:
        env = make_maniskill_venv(
            env_id="HACMan-PickCube-v0",
            num_envs=4,
            max_episode_steps=2,
            vectorization="sb3",
            wrapper_kwargs={"normalize_obs": True}
        )
    """
    
    wrappers = []

    # Video recorder wrapper
    if record_video:
        assert vectorization=="sb3", "Record video only works with stable-baselines3 parallelization"
        from mani_skill2.utils.wrappers import RecordEpisode
        wrappers.append(partial(RecordEpisode, output_dir=f"videos", info_on_video=True, save_trajectory=False))
    
    if env_id == "HACMan-OpenCabinetDoor-v1":
        control_mode = "base_pd_joint_vel_arm_pd_ee_delta_pose"
        wrapper_class = HACManMobileActionWrapper
    else:
        wrapper_class = {
            "per_point": HACManActionWrapper,
            'per_point_logit': FlatActionLogitWrapper,
            "flat": FlatActionWrapper,
            "per_primitive": RegressedActionWrapper,
        }[action_mode]
        control_mode = "pd_ee_delta_pose"
    wrapper_class = partial(wrapper_class, **env_wrapper_kwargs)
    wrappers.append(wrapper_class)

    # stable-baselines3 parralelization
    if vectorization == "sb3":
        from stable_baselines3.common.env_util import make_vec_env as make_sb3_vec_env
        ms_kwargs = dict(
            obs_mode="pointcloud",
            reward_mode=reward_mode,
            control_mode=control_mode,
            enable_segmentation=True,
        )

        # Collapse the wrappers into a single function
        wrappers.append(partial(TimeLimit, max_episode_steps=max_episode_steps))
        def wrapper_class(env, **kwargs):
            for wrapper in wrappers:
                env = wrapper(env)
            return env
        
        venv: VecEnv = make_sb3_vec_env(
            env_id, num_envs,
            seed=seed,
            vec_env_cls=PCDSubprocVecEnv,
            env_kwargs=ms_kwargs,
            wrapper_class=wrapper_class,
            vec_env_kwargs=vecenv_kwargs,
        )

    # Maniskill native parralelization
    elif vectorization == "ms":
        # Continuous task wrapper
        # wrappers.append(partial(ContinuousTaskWrapper, max_episode_steps=max_episode_steps))
        wrappers.append(partial(TimeLimit, max_episode_steps=max_episode_steps))
        venv: VecEnv = make_vec_env(
            env_id, num_envs,
            obs_mode="pointcloud",
            reward_mode=reward_mode,
            control_mode=control_mode,
            enable_segmentation=True,
            wrappers=wrappers,
        )
        venv = PCDMSVecEnvWrapper(venv, **vecenv_kwargs)
        venv = SB3VecEnvWrapper(venv)

    venv = VecMonitor(venv)
    venv.seed(seed)
    return venv


# Test the wrapper
if __name__ == "__main__":
    import mani_skill2.envs
    
    from mani_skill2.vector import VecEnv, make as make_vec_env
    from mani_skill2.vector.wrappers.sb3 import SB3VecEnvWrapper

    from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
    from hacman.envs.vec_env_wrappers import VecEnvwithLocationPolicy, VecEnvwithPrimitivePolicy,  DeterministicVecEnvWrapper
    from hacman.algos.location_policy import RandomLocation

    
    num_envs = 1
    max_episode_steps = 1
    primitives = [
        'poke',
        'pick_n_lift_fixed',
        'place',
        'open_gripper',
        'move'
        ]
    env_wrapper_kwargs = dict(
        primitives=primitives,
        end_on_reached=True,
        # enable_rotation=True,
        use_oracle_rotation=False,# only true for the peg insertion env
        # release_gripper=True,
        # use_oracle_motion=False,
        # bg_mapping_mode="bbox"
    )
    vecenv_kwargs = dict(
        primitives=primitives,
    )
    env = make_maniskill_venv(
       
                            "HACMan-StackCube-v0", # "HACMan-PegInsertionSideEnv-v0",  "HACMan-LiftCube-v0",
                              num_envs, max_episode_steps=max_episode_steps,
                              vectorization="sb3",
                              record_video=True,
                              env_wrapper_kwargs=env_wrapper_kwargs,
                              vecenv_kwargs=vecenv_kwargs,)

    env = VecEnvwithLocationPolicy(env, location_model=RandomLocation())
    env = VecMonitor(env)
    env.seed(0)
    obs = env.reset()

    action_dim = env.action_space.shape[0]
    for i in tqdm(range(1)):
        obs = env.reset()
        frame = env.env_method("render", indices=0, mode="rgb_array")
        for _ in range(max_episode_steps):
            action = np.random.uniform(-1, 1, (num_envs, action_dim))
            obs, reward, done, info = env.step(action)
            
        print(i)
    print("Done")