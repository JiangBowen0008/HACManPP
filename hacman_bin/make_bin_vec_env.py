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
from hacman.utils.transformations import transform_point_cloud, to_pose_mat, sample_idx
from hacman.envs.sim_envs.base_env import RandomLocation
from hacman.envs.env_wrappers import HACManActionWrapper, FlatActionWrapper, RegressedActionWrapper,FlatActionLogitWrapper
from hacman.envs.vec_env_wrappers import PCDSubprocVecEnv, PCDDummyVecEnv, WandbPointCloudRecorder
from hacman_bin.hacman_bin_env import HACManBinEnv
import imageio
import os
def make_bin_venv(
        env_id: str, num_envs: int, 
        max_episode_steps: int = 10,
        seed: int = 0,
        action_mode: str = "per_point",
        vectorization: str = "sb3", # sb3 or dummy
        env_kwargs: Dict = {},
        env_wrapper_kwargs: Dict = {},
        vecenv_kwargs: Dict = {},):
    """
    Creates a bin vectorized environment with the specified parameters.

    Args:
        env_id (str): The ID of the environment to create.
        num_envs (int): The number of environments to create.
        max_episode_steps (int, optional): The maximum number of steps per episode. Defaults to 10.
        reward_mode (str, optional): The reward mode to use. Defaults to "dense".
        seed (int, optional): The random seed to use. Defaults to 0.
        record_video (bool, optional): Whether to record videos of the environment. Defaults to False.
        vectorization (str, optional): "sb3" (Stable Baselines 3 vectorization). Currently only supports "ms".
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

   
    
    wrapper_class = {
        "per_point": HACManActionWrapper,
        "flat": FlatActionWrapper,
        "per_primitive": RegressedActionWrapper,
        'per_point_logit': FlatActionLogitWrapper,
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
            HACManBinEnv, num_envs,
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
    # from hacman.envs.sim_envs.maniskill_env import SpatialPickCubeEnv

    from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
    from hacman.envs.vec_env_wrappers.location_policy_vec_wrappers import VecEnvwithLocationPolicy
    from hacman.algos.location_policy import RandomLocation

    num_envs = 1
    max_episode_steps = 10

    primitives = [
        'bin-poke',
         'bin-pick_n_lift_fixed',
        'bin-open_gripper',
        'bin-place_from_top'
    ]
    env_wrapper_kwargs = dict(
        primitives=primitives,
        end_on_reached=True,
        rotation_type=1,

    )
    ## potential objects to test
    train_thick= [
     "bottle_021_bleach_cleanser_M",
    "bottle_frl_apartment_kitchen_utensil_09_M",
    "bowl_kitchen_set_kitchen_set_bowl_M",
    "camera_frl_apartment_camera_02_M",
    "can_002_heavy_master_chef_can_M",
    "can_005_tomato_soup_can_M",
    "canister_Nescafe_Tasters_Choice_Instant_Coffee_Decaf_House_Blend_Light_7_oz_M",
    "cup_065-a_cups_M",
    "cup_065-f_cups_M",
    "cup_BIA_Cordon_Bleu_White_Porcelain_Utensil_Holder_900028_M",
    "cup_Ecoforms_Cup_B4_SAN_M",
    "cup_frl_apartment_kitchen_utensil_06_M",
    "flashlight_HeavyDuty_Flashlight_M",
    "hand_bell_Cole_Hardware_School_Bell_Solid_Brass_38_M",
    "headphone_Razer_Kraken_Pro_headset_Full_size_Black_M",
    "lego_block_073-b_lego_duplo_M",
    "mug_025_mug_M",
    "mug_Room_Essentials_Mug_White_Yellow_M",
    "mug_Threshold_Porcelain_Coffee_Mug_All_Over_Bead_White_M",
    "mug_kitchen_set_kitchen_set_mug_2_M",
    "pencil_case_Olive_Kids_Robots_Pencil_Case_M",
    "pencil_case_Wishbone_Pencil_Case_M",
    "pill_bottle_Beta_Glucan_M",
    "pill_bottle_Germanium_GE132_M",
    "pitcher_Threshold_Porcelain_Pitcher_White_M",
    "plant_container_Central_Garden_Flower_Pot_Goo_425_M",
    "plant_container_Ecoforms_Plant_Container_Quadra_Turquoise_QP12_M",
    "rubiks_cube_077_rubiks_cube_M",
    "salt_shaker_Wilton_Pearlized_Sugar_Sprinkles_525_oz_Gold_M",
    "salt_shaker_frl_apartment_kitchen_utensil_03_M",
    "teapot_Threshold_Porcelain_Teapot_White_M",
    "teapot_kitchen_set_kitchen_set_kettle_M"]

    env_kwargs = dict(record_video=True, 
                        gripper_types = 'default', ## 'panda_festo', 'panda_festo_narrow', 'panda_narrow'
                        arena_type="double_bin", ## "bin"
                        friction_config='low_0.95',
                        gripper_friction= 0.4,
                        renderer='offscreen',  # 'onscreen'
                        object_init_pose = "any", ## 'fixed', 'fixed_off_center', 'any', 'any-ori'
                        goal_mode = 'any_var_size', ##'fixed_off_center', 'fixed', 'any'
                        goal_on_oppo_side = True, ## False
                        object_dataset = 'housekeep_all',
                        background_pcd_size = 1000,
                        object_name = 'rubiks_cube_077_rubiks_cube_M',
                        convex_decomposed=True,
                        object_scale_range = [0.6,1.0],
                        )
    vecenv_kwargs = dict(
        primitives=primitives,
    )
    
   
    camera_frames = []
    for iter in range(1):
        for i in tqdm(range(2)):
            env = make_bin_venv("HACManBinEnv-v0", 
                                num_envs, max_episode_steps=max_episode_steps,
                                vectorization="sb3",
                                env_kwargs = env_kwargs,
                                env_wrapper_kwargs=env_wrapper_kwargs,
                                vecenv_kwargs=vecenv_kwargs,)

            env = VecEnvwithLocationPolicy(env, location_model=RandomLocation())
            env.seed(0)


            action_dim = env.action_space.shape[0]
            obs = env.reset()
            for step in range(max_episode_steps):
                action = np.random.uniform(-1, 1, (num_envs, action_dim))
                obs, reward, done, info = env.step(action)
                print('step: ',  step)
                for l in range(num_envs):
                    frames = info[l]['cam_frames']
                    camera_frames.extend(frames)
    filename = 'test_rubiks_cube'+ '.mp4'
    vid_dir = './videos'
    vid_path = os.path.join(vid_dir, filename)
    os.makedirs(vid_dir, exist_ok=True)

    # Save video locally
    with imageio.get_writer(vid_path, mode='I', fps=10) as writer:
        for im in camera_frames:
            writer.append_data(im)
   
