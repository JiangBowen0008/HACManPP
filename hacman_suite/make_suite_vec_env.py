from typing import Any, Callable, List, Optional, Sequence, Tuple, Type, Union, Dict
import numpy as np
from functools import partial
import gym
import gym.spaces as spaces
from gym.wrappers.time_limit import TimeLimit

from stable_baselines3.common.vec_env import VecMonitor

import robosuite as suite

from hacman.utils.transformations import transform_point_cloud, to_pose_mat, sample_idx
from hacman.envs.sim_envs.base_env import RandomLocation
from hacman.envs.env_wrappers import HACManActionWrapper, FlatActionWrapper, RegressedActionWrapper
from hacman.envs.vec_env_wrappers import PCDSubprocVecEnv, PCDDummyVecEnv

import hacman_suite.env_wrappers.primitives

def make_suite_venv(
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

if __name__ == "__main__":
    # from hacman.envs.sim_envs.maniskill_env import SpatialPickCubeEnv
    from tqdm import tqdm
    import imageio, os, time
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
    from hacman.envs.vec_env_wrappers.location_policy_vec_wrappers import VecEnvwithLocationPolicy
    from hacman.algos.location_policy import RandomLocation

    from hacman_suite.envs import *
    # env_id = HACManSuiteStack
    # env_id = HACManSuiteNutAssembly
    # env_id = HACManSuitePickPlace
    env_id = HACManSuiteDoor
    # primitives = [
    #     'suite-poke',
    #     'suite-pick_n_lift_fixed',
    #     # 'bin-place_from_top_n_open_gripper',
    #     'suite-place',
    #     'suite-open_gripper',
    #     # 'place_from_top'
    # ]
    primitives = [
        # 'suite-poke',
        'suite-pick',
        # 'bin-place_from_top_n_open_gripper',
        'suite-move',
        'suite-open_gripper',
        # 'place_from_top'
    ]

    # env = gym.make("LiftCube-v0", obs_mode="pointcloud", control_mode="pd_ee_delta_pose")
    num_envs = 5
    max_episode_steps = 10
    record_video = True
    repeats = 2

    
    env_wrapper_kwargs = dict(
        primitives=primitives,
        end_on_reached=True,
        record_video=record_video,
        # enable_rotation=True,
        # use_oracle_motion=False,
    )
    env_kwargs = dict(record_video=record_video)
    vecenv_kwargs = dict(
        primitives=primitives,
        background_pcd_size = 1000,
    )

    
    env = make_suite_venv(env_id,
                            num_envs, max_episode_steps=max_episode_steps,
                            vectorization="sb3",
                            env_kwargs = env_kwargs,
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
    for i in tqdm(range(repeats)):
        obs = env.reset()
        for _ in range(max_episode_steps):
            # action = np.zeros(6)
            action = np.random.uniform(-1, 1, (num_envs, action_dim))
            # action = np.zeros((num_envs, 3))
            # action[:, i] = 0.1
            # print(action)
            # action[5] = -0.5
            obs, reward, done, info = env.step(action)
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