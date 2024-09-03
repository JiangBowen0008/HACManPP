import numpy as np
import functools

from gym.wrappers.time_limit import TimeLimit

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

import mani_skill2.envs
import hacman_ms.envs
import hacman_suite.envs
import hacman_adroit.envs

from hacman_bin import make_bin_venv

from hacman.envs.vec_env_wrappers import VecEnvwithLocationPolicy, VecEnvwithPrimitivePolicy, DeterministicVecEnvWrapper
from hacman.envs.setup_location_policy import setup_location_policy, add_location_policy_config


def add_env_config(parser):
    parser.add_argument("--env", default='simple_env', type=str)
    parser.add_argument("--train_n_envs", default=20, type=int, help="Number of training envs in parallel")
    parser.add_argument("--eval_n_envs", default=None, type=int, help="Number of eval envs in parallel")
    parser.add_argument("--record_from_cam", default=None, type=str, help="Record video from camera.")
    parser.add_argument("--parallel_env", action="store_true", help="Use parallel env")

    # Location policy
    add_location_policy_config(parser)

    '''
    Shared
    '''
    parser.add_argument("--voxel_downsample_size", default=0.01, type=float, help="Voxel downsample size")
    parser.add_argument("--unrestrict_primitive", action="store_true", help="Unconstrain primitive")
    parser.add_argument("--free_move", action="store_true", help="Free move")
    
    '''
    HACMan Bin Env Specific Configs
    '''
    # Object mesh
    parser.add_argument("--object_dataset", default='housekeep_all', type=str, choices={'cube', 'housekeep', 'housekeep_all'})
    parser.add_argument("--object_types", default=None, nargs='*', type=str, help="Load from specific categories")
    parser.add_argument("--object_name", default=None, type=str, help="Load from a specific object name")
    parser.add_argument("--object_split_train", default='train_thick', type=str, help="Load from a specific split")
    parser.add_argument("--object_split_eval", default='train_thick', type=str, help="Load from a specific split")
    parser.add_argument("--object_scale_range", default=[0.8, 1.2], nargs=2, type=float, help="Scale range for object size randomization")
    parser.add_argument("--object_size_limit", default=[0.05, 0.05], nargs=2, type=float, help="Max/min object size")

    parser.add_argument("--object_voxel_down_sample", default=0.005, type=float, help="Downsample object point cloud voxel size")
    parser.add_argument("--convex_decomposed", action="store_true", help="Use convex decomposed mesh")
    parser.add_argument("--pcd_mode", default='reduced1', type=str, help="Point cloud sampling mode")
    parser.add_argument("--pcd_noise", default=None, nargs=4, type=float, help="Pcd noise")

    # Task
    parser.add_argument("--object_init_pose", default="any", type=str, help="Init pose sampling mode")
    parser.add_argument("--goal_mode", default='any_var_size', type=str, help="Goal sampling mode")
    parser.add_argument("--goal_mode_eval", default=None, type=str, help="Goal sampling mode")
    parser.add_argument("--goal_on_oppo_side", action="store_true", help="Goal on opposite side")
    parser.add_argument("--success_threshold", default=0.03, type=float, help="Success threshold")
    
    # Bin
    parser.add_argument("--table_base_size", default=None, nargs=3, type=float, help="Table size base (before randomization)")
    parser.add_argument("--bin_base_size", default=None, nargs=3, type=float, help="Bin size base (before randomization)")
    parser.add_argument("--arena_rand_size", default=None, nargs=3, type=float, help="Bin/table size randomization. Default 0.")
    parser.add_argument("--gripper_types", default='default', type=str, help="Gripper type")
    parser.add_argument("--gripper_friction", default=3., type=float, help="Gripper friction")
    parser.add_argument("--friction_config", default='default', type=str, help="friction settings")
    parser.add_argument("--obstacle_full_size", default=None, nargs=3, type=float, help="Obstacle size")    # (0.45, 0.02, 0.07)
    parser.add_argument("--exclude_wall_pcd", action="store_true", help="Exclude wall pcd")
    parser.add_argument("--exclude_gripper_pcd", action="store_true", help="Exclude gripper pcd")
    parser.add_argument("--move_gripper_to_object_pcd", action="store_true", help="Move gripper to object pcd")

    # Actions
    parser.add_argument("--action_repeat", default=3, type=int, help="Number of action repeats")
    parser.add_argument("--location_noise", default=0., type=float, help="Location noise")
    parser.add_argument("--reward_scale", default=1., type=float, help="Reward scale")
    parser.add_argument("--fixed_ep_len", action="store_true", help="Fixed episode length")
    parser.add_argument("--reward_gripper_distance", default=None, type=float, help="Whether to use distance in the reward calculation")

    '''
    Maniskill Env Specific Configs
    '''
    parser.add_argument("--background_pcd_size", default=400, type=int, help="Background pcd size")
    parser.add_argument("--object_pcd_size", default=400, type=int, help="Object pcd size")
    parser.add_argument("--use_oracle_motion", action="store_true", help="Use oracle motion")
    parser.add_argument("--use_oracle_rotation", action="store_true", help="Use oracle rotation")
    parser.add_argument("--bg_mapping_mode", default="bbox", type=str, help="Background mapping mode")
    parser.add_argument("--use_flow_reward", action="store_true", help="Use flow reward")
    parser.add_argument("--rotation_type", default=0, type=int, help='what rotation we use for primitives(0 for arctan(x,y), 1 for [-90, 90], 2 for [0, 180], 3 for [-180, 180], 4 for [0, 360]) ')
    parser.add_argument("--reward_aggregation", default="final", help="Cumulative reward mode")
    parser.add_argument("--primitives", default=["pick_n_drop"], nargs='*', type=str, help="Primitives to use")
    parser.add_argument("--background_clip_radius", default=0.5, type=float, help="Background clip radius")
    parser.add_argument("--max_episode_steps", default=2, type=int, help="Max episode steps")

    parser.add_argument("--end_on_collision", action="store_true", help="Stop on collision")
    parser.add_argument("--end_on_reached", action="store_true", help="Stop on reached")
    parser.add_argument("--pad_rewards", action="store_true", help="Pad rewards")
    parser.add_argument("--use_location_noise", action="store_true", help="add random noise to selected location")

    parser.add_argument("--pos_tolerance", default=0.005, type=float, help="Position tolerance")
    parser.add_argument("--rot_tolerance", default=0.1, type=float, help="Rotation tolerance")

    return

def setup_envs(config, eval_env_only=False): 
    train_env, eval_env = None, None
    config['eval_n_envs'] = config['train_n_envs'] if config['eval_n_envs'] is None else config['eval_n_envs']
    n_envs = config['train_n_envs'] if not eval_env_only else config['eval_n_envs']
    # ------------- Train/eval wrapper setup ------------- 
    if config["action_mode"] in {"per_point", "per_point_logit", "per_point_action", "per_primitive"}:
        location_model_train, location_model_eval = setup_location_policy(config)
    
    # ------------- Env Setup ------------- 
    ms_env_names = {
        "pick_cube": "HACMan-PickCube-v0",
        "lift_cube": "HACMan-LiftCube-v0",
        "stack_cube": "HACMan-StackCube-v0",
        "open_door": "HACMan-OpenCabinetDoor-v1",
        "insert_peg": "HACMan-PegInsertionSideEnv-v0",
        "plug_charger": "HACMan-PlugCharger-v0",
    }
    hacman_env_names = {"hacman_bin", "hacman_obstacle_bin", "hacman_double_bin"}
    suite_env_names = {
        "suite_stack_cube": hacman_suite.envs.HACManSuiteStack,
        "suite_nut_assembly": hacman_suite.envs.HACManSuiteNutAssembly,
        "suite_pick_place": hacman_suite.envs.HACManSuitePickPlace,
        "suite_door": hacman_suite.envs.HACManSuiteDoor,
    }
    adroit_env_names = {
        "adroit_relocate": hacman_adroit.envs.HACManAdroitRelocate,
    }
    env_wrapper_kwargs = dict(
        reward_scale=config['reward_scale'],
        use_oracle_motion=config['use_oracle_motion'],
        use_oracle_rotation=config['use_oracle_rotation'],
        use_location_noise = config['use_location_noise'],
        bg_mapping_mode=config['bg_mapping_mode'],
        use_flow_reward=config['use_flow_reward'],
        rotation_type=config['rotation_type'],
        reward_aggregation=config['reward_aggregation'],
        primitives=config['primitives'],
        restrict_primitive=(not config['unrestrict_primitive']),
        free_move=config['free_move'],
        end_on_collision=config['end_on_collision'],
        end_on_reached=config['end_on_reached'],
        pad_rewards=config['pad_rewards'],
        background_pcd_size=config['background_pcd_size'],
        object_pcd_size=config['object_pcd_size'],
        pos_tolerance=config['pos_tolerance'],
        rot_tolerance=config['rot_tolerance'],
        sample_action_logits=config['sample_action_logits'],
    )
    vecenv_kwargs = dict(
        primitives=config['primitives'],
        background_clip_radius=config['background_clip_radius'],
        background_pcd_size=config['background_pcd_size'],
        object_pcd_size=config['object_pcd_size'],
        voxel_downsample_size=config['voxel_downsample_size'],
    )
    if config['env'] in ms_env_names.keys():
        
        vectorization = 'ms'
        if config['record_video']:
            assert eval_env_only, "Record video only works with eval only mode (which sets vectorization to sb3)"
            env_wrapper_kwargs["record_video"] = True
            vectorization = "sb3"
        
        env_name = ms_env_names[config['env']]
        from hacman_ms import make_maniskill_venv
        train_env = make_maniskill_venv(env_name,
                                        n_envs,
                                        max_episode_steps=config['max_episode_steps'],
                                        vectorization=vectorization,
                                        action_mode=config['action_mode'],
                                        env_wrapper_kwargs=env_wrapper_kwargs,
                                        vecenv_kwargs=vecenv_kwargs)
        
        eval_env = train_env

    elif config['env'] in hacman_env_names:
        arena_type = {
            "hacman_bin": "bin",
            "hacman_obstacle_bin": "obstacle_bin",
            "hacman_double_bin": "double_bin"
        }[config['env']]
        env_kwargs = dict(
            arena_type=arena_type,
            reward_scale=config['reward_scale'],
            reward_gripper_distance=config['reward_gripper_distance'],
            object_pcd_size=config['object_pcd_size'],
            background_pcd_size=config['background_pcd_size'],
            goal_mode=config['goal_mode'],
            goal_on_oppo_side=config['goal_on_oppo_side'],
            pcd_mode=config['pcd_mode'],
            action_mode=config['action_mode'],
            action_repeat=config['action_repeat'],
            object_dataset=config['object_dataset'],
            object_types=config['object_types'],
            object_name=config['object_name'],
            convex_decomposed=config['convex_decomposed'],
            object_scale_range=config['object_scale_range'],
            object_init_pose=config['object_init_pose'],
            object_split=config['object_split_train'],
            object_size_limit=config['object_size_limit'],
            fixed_ep_len=config['fixed_ep_len'],
            table_base_size=config['table_base_size'],
            bin_base_size=config['bin_base_size'],
            arena_rand_size=config['arena_rand_size'],
            obstacle_full_size=config['obstacle_full_size'],
            exclude_wall_pcd=config['exclude_wall_pcd'],
            exclude_gripper_pcd=config['exclude_gripper_pcd'],
            move_gripper_to_object_pcd = config['move_gripper_to_object_pcd'],
            location_noise=config['location_noise'],
            pcd_noise=config['pcd_noise'],
            voxel_downsample_size=config['voxel_downsample_size'],
            friction_config=config['friction_config'],
            success_threshold=config['success_threshold'],
            gripper_types=config['gripper_types'],
            gripper_friction=config['gripper_friction'],
            record_video=config['record_video'],
            record_from_cam=config['record_from_cam'],
        )

        env_name = "HACManBinEnv-v0"
        vectorization = 'sb3'
        train_env = make_bin_venv(env_name,
                                n_envs,
                                max_episode_steps=config['max_episode_steps'],
                                vectorization=vectorization,
                                env_kwargs=env_kwargs,
                                env_wrapper_kwargs=env_wrapper_kwargs,
                                action_mode=config['action_mode'],
                                vecenv_kwargs=vecenv_kwargs)
    elif config['env'] in suite_env_names.keys():
        vectorization = 'sb3'
        if config['record_video']:
            assert eval_env_only, "Record video only works with eval only mode (which sets vectorization to sb3)"
            env_wrapper_kwargs["record_video"] = True
        
        env_name = suite_env_names[config['env']]
        env_kwargs = dict(
            voxel_downsample_size=config['voxel_downsample_size'],
            background_pcd_size=config['background_pcd_size'],
            object_pcd_size=config['object_pcd_size'],
            record_video=config['record_video'],
            record_from_cam=config['record_from_cam'],
        )
        from hacman_suite import make_suite_venv
        train_env = make_suite_venv(env_name,
                                    n_envs,
                                    max_episode_steps=config['max_episode_steps'],
                                    vectorization=vectorization,
                                    action_mode=config['action_mode'],
                                    env_kwargs=env_kwargs,
                                    env_wrapper_kwargs=env_wrapper_kwargs,
                                    vecenv_kwargs=vecenv_kwargs)
    elif config['env'] in adroit_env_names.keys():
        vectorization = 'sb3'
        if config['record_video']:
            assert eval_env_only, "Record video only works with eval only mode (which sets vectorization to sb3)"
            env_wrapper_kwargs["record_video"] = True
        
        env_name = adroit_env_names[config['env']]
        env_kwargs = dict(
            voxel_downsample_size=config['voxel_downsample_size'],
            background_pcd_size=config['background_pcd_size'],
            object_pcd_size=config['object_pcd_size'],
            record_video=config['record_video'],
            record_from_cam=config['record_from_cam'],
        )
        from hacman_adroit import make_adroit_venv
        train_env = make_adroit_venv(env_name,
                                    n_envs,
                                    max_episode_steps=config['max_episode_steps'],
                                    vectorization=vectorization,
                                    action_mode=config['action_mode'],
                                    env_kwargs=env_kwargs,
                                    env_wrapper_kwargs=env_wrapper_kwargs,
                                    vecenv_kwargs=vecenv_kwargs)

        
    eval_env = DeterministicVecEnvWrapper(train_env)
    if config['action_mode'] in {"per_point", "per_point_action", "per_point_logit"}:
        train_env = VecEnvwithLocationPolicy(train_env, location_model=location_model_train)
        eval_env = VecEnvwithLocationPolicy(eval_env, location_model=location_model_eval)
    elif config['action_mode'] == "per_primitive":
        train_env = VecEnvwithPrimitivePolicy(train_env, location_model=location_model_train)
        eval_env = VecEnvwithPrimitivePolicy(eval_env, location_model=location_model_eval)

    if eval_env_only:
        return eval_env
    
    else:
        return train_env, eval_env
