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

from stable_baselines3.common.vec_env import VecMonitor, DummyVecEnv
from hacman.utils.transformations import transform_point_cloud, to_pose_mat, sample_idx
from hacman.envs.sim_envs.base_env import RandomLocation
from hacman.envs.env_wrappers import HACManActionWrapper, FlatActionWrapper, RegressedActionWrapper
import hacman_adroit.env_wrappers.primitives
from hacman.envs.vec_env_wrappers import PCDSubprocVecEnv, PCDDummyVecEnv, WandbPointCloudRecorder
import imageio
import os

from gym.envs.registration import register
# register(
#     id='relocate-v0',
#     entry_point='mj_envs.hand_manipulation_suite:RelocateEnvV0',
#     max_episode_steps=200,
# )

def make_adroit_venv(
        env_id: str, num_envs: int, 
        max_episode_steps: int = 10,
        reward_mode: str = "dense",
        action_mode: str = "per_point",
        seed: int = 0,
        vectorization: str = "sb3",
        env_kwargs: Dict = {},
        env_wrapper_kwargs: Dict = {},
        vecenv_kwargs: Dict = {},):
    wrappers = []
    # Video recorder wrapper
    # if record_video:
    #     # assert vectorization=="sb3", "Record video only works with stable-baselines3 parallelization"
    #     from mani_skill2.utils.wrappers import RecordEpisode
    #     wrappers.append(partial(RecordEpisode, output_dir=f"videos", info_on_video=True, save_trajectory=False))
    
    wrapper_class = {
        "per_point": HACManActionWrapper,
        "flat": FlatActionWrapper,
        "per_primitive": RegressedActionWrapper,
    }[action_mode]
    wrapper_class = partial(wrapper_class, **env_wrapper_kwargs)
    wrappers.append(wrapper_class)

    # stable-baselines3 parralelization
    if vectorization in {"sb3", "dummy"}:
        from stable_baselines3.common.env_util import make_vec_env as make_sb3_vec_env
        # Collapse the wrappers into a single function
        wrappers.append(partial(TimeLimit, max_episode_steps=max_episode_steps))
        def wrapper_class(env, **kwargs):
            for wrapper in wrappers:
                env = wrapper(env)
            return env
        
        vec_env_cls = PCDSubprocVecEnv if vectorization == "sb3" else PCDDummyVecEnv
        venv = make_sb3_vec_env(
            env_id, num_envs,
            seed=seed,
            vec_env_cls=vec_env_cls,
            env_kwargs=env_kwargs,
            wrapper_class=wrapper_class,
            vec_env_kwargs=vecenv_kwargs,
        )
    else:
        raise NotImplementedError("Currently only supports sb3/dummy vectorization.")
    venv = VecMonitor(venv)
    venv.seed(seed)
    return venv


# Test the wrapper
if __name__ == "__main__":

    import cv2, time
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
    from hacman.envs.vec_env_wrappers import VecEnvwithLocationPolicy, VecEnvwithPrimitivePolicy,  DeterministicVecEnvWrapper
    from hacman.algos.location_policy import RandomLocation
    from hacman_adroit.envs.adroit_relocate_env import HACManAdroitRelocate 

    # env = gym.make("LiftCube-v0", obs_mode="pointcloud", control_mode="pd_ee_delta_pose")

    primitives = [
        'adroit-pick',
        'adroit-move-delta',
        # 'adroit-move',
        # 'adroit-open_gripper',
        # 'adroit-poke',
        # 'pick_n_drop',
        # 'pick_n_lift',
        # 'dummy',
        # 'place_w_gripper_action',
        # 'place_n_insert',
        # 'open_gripper',
        # "poke",
        # 'move'
        ]
    
    num_envs = 2
    max_episode_steps = 10
    record_video = True
    repeats = 2

    env_wrapper_kwargs = dict(
        primitives=primitives,
        end_on_reached=True,
        record_video=record_video,
        # enable_rotation=True,
        # use_oracle_rotation=True,
        # release_gripper=True,
        # use_oracle_motion=False,
        # bg_mapping_mode="bbox"
    )
    env_kwargs = dict(record_video=record_video)
    vecenv_kwargs = dict(
        primitives=primitives,
        background_pcd_size = 1000,
    )
    # vecenv_kwargs = None
    env = make_adroit_venv(HACManAdroitRelocate, 
                              num_envs, max_episode_steps=max_episode_steps,
                              vectorization="sb3",
                            #   record_video=True,
                            #   action_mode="flat",
                              env_wrapper_kwargs=env_wrapper_kwargs,
                              vecenv_kwargs=vecenv_kwargs,)

    env = VecEnvwithLocationPolicy(env, location_model=RandomLocation())
    # env = VecEnvwithPrimitivePolicy(env, location_model=RandomLocation())
    # env = DeterministicVecEnvWrapper(env)
    env = VecMonitor(env)
    env.seed(0)
    obs = env.reset()

    action_dim = env.action_space.shape[0]
    cam_frames = [list() for _ in range(num_envs)]
    start_time = time.time()
    for i in tqdm(range(20)):
        obs = env.reset()
        for _ in tqdm(range(max_episode_steps)):
            # action = np.zeros(6)
            action = np.random.uniform(-1, 1, (num_envs, action_dim))
            # print(action)
            # action = np.zeros((num_envs, 3))
            # action[:, i] = 0.1
            # print(action)
            # action[5] = -0.5
            obs, reward, done, info = env.step(action)
            # frame = env.env_method("render", indices=0, mode="offscreen")
            # print(frame)
            # cv2.imshow('show_adroit', frame[0])
            # cv2.waitKey(1)
            # plt.imshow(frame[0])
            # print(obs)

            for j in range(num_envs):
                cam_frames[j].extend(info[j]["cam_frames"])
            # frame = env.env_method("render", indices=0, mode="rgb_array")
        print(i)

        # Save the frames
        if record_video:
            os.makedirs("videos", exist_ok=True)
            for j in range(num_envs):
                frames = cam_frames[j]
                with imageio.get_writer(f'videos/video_{i}_{j}.mp4', mode='I', fps=30) as writer:
                    for frame in frames:
                        writer.append_data(frame)
                cam_frames[j] = []
    end_time = time.time()
    print("Time taken: ", end_time - start_time)
    print("fps: ", max_episode_steps * num_envs * repeats / (end_time - start_time))
    # print(obs)