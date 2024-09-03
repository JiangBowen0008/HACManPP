import os
from hacman.utils.launch_utils import *
# use_freer_gpu()
# print(f"Using CUDA Device {os.environ['CUDA_VISIBLE_DEVICES']}")

from stable_baselines3.common.callbacks import EvalCallback, CallbackList

import wandb
from hacman.envs.vec_env_wrappers.wandb_wrappers import WandbPointCloudRecorder
from hacman.sb3_utils.custom_callbacks import ProgressBarCallback, CustomizedCheckpointCallback, EvalCallbackWPrimitiveInfo
from hacman.sb3_utils.evaluation import evaluate_policy
from hacman.algos.setup_model import setup_model, add_model_config
from hacman.envs.setup_envs import setup_envs, add_env_config

import argparse
import torch_geometric

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Exp and logging config
    parser.add_argument("--ExpID", default=9999, type=int, help="Exp ID")
    parser.add_argument("--name", default="default", type=str, help="Exp name")
    parser.add_argument("--ExpGroup", default=None, type=str, help="Exp group (for wandb grouping only)")
    parser.add_argument("--ExpVariant", default=None, type=str, help="Variant name (for wandb grouping only)")
    parser.add_argument("--seed", default=0, type=int, help="Random seed")
    parser.add_argument("--eval", default=None, type=int, help="Eval only. Eval epsiodes to run")
    parser.add_argument("--dirname", default=None, type=str, help="Path to save models")
    parser.add_argument("--debug", action="store_true", help="Use debug config")
    parser.add_argument("--train_steps", default=200000, type=int, help="Number of training steps")
    parser.add_argument("--obj", action="store_true", help='Record per object success rate and info')

    parser.add_argument("--action_mode", default="per_point", choices={"per_point", "flat", "per_primitive", "per_point_logit"}, type=str, help="Action mode")
    parser.add_argument("--n_eval_episodes", default=20, type=int, help="Number of eval episodes")
    parser.add_argument("--eval_freq", default=100, type=int, help="Eval per n env steps")
    parser.add_argument("--save_freq_latest", default=100, type=int, help="Save per n env steps")
    parser.add_argument("--save_freq_checkpoint", default=1000, type=int, help="Save per n env steps")
    parser.add_argument("--save_replay_buffer", action="store_true", help="Save buffer")
    parser.add_argument("--record_video", action="store_true", help="Record video.")
    parser.add_argument("--upload_video", action="store_true", help="Upload video to wandb.")
    parser.add_argument("--delete_video", action="store_true", help="Delete video after uploading.")

    parser.add_argument("--load_exp", default=None, type=str, help="Load all the config from a previous exp.")
    parser.add_argument("--load_dirname", default=None, type=str, help="Path to search for prev exps.")
    parser.add_argument('--log_object', default=0, type = int, help = 'wheteher to log per object success rate' )
    parser.add_argument("--override_args", default=[], nargs="*", help="Override the config with the given args. e.g. --override_args object_name table_size")

    add_model_config(parser)
    add_env_config(parser)
    
    args = parser.parse_args()
    eval_per_obj = args.log_object
    config = vars(args)

    # ------------- Config Setup -------------
    if config['load_exp'] is not None:
        config = load_exp_config(parser, config)

    if args.dirname is None or args.debug:
        dirname = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    else:
        dirname = args.dirname
    config['dirname'] = dirname
    config["fullname"] = config["name"]
    # For eval, replace the Exp with Eval in the name
    if (config['eval'] is not None) and (config['name'].startswith('Exp')):
        config['fullname'] = config['name'].replace('Exp', 'Eval', 1)
    config['log_object'] = eval_per_obj
    from pprint import pprint
    pprint(config)

    torch_geometric.seed_everything(config["seed"])
    np.random.seed(config["seed"])

    # ------------- Logging ------------- 
    result_folder = os.path.join(config["dirname"], config["fullname"])
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    resume = hasattr(config, "run_id")
    run_id = config['run_id'] if resume else wandb.util.generate_id()   
    run = wandb.init(
        name=config["fullname"], config=config, id=run_id, 
        dir=result_folder, sync_tensorboard=True, resume=resume)
    print(f'wandb run-id:{run.id}')
    print(f'Result folder: {result_folder}')
    
    # ------------- Task Setup -------------
    if config['eval'] is not None:
        eval_env = setup_envs(config, eval_env_only=True)
        model = setup_model(config, eval_env, result_folder, normalize_param=None)
        
        if hasattr(eval_env, "location_model"):
            eval_env.location_model.load_model(model)

        model.policy.eval()

    else:
        env, eval_env = setup_envs(config)
        model = setup_model(config, env, result_folder, normalize_param=None)
        
        if hasattr(env, "location_model") and hasattr(eval_env, "location_model"):
            env.location_model.load_model(model)
            eval_env.location_model.load_model(model)
    
    # ------------- Additional Setup for Logging Purposes ------------- 
    def global_step_counter():
        return model.num_timesteps
    eval_env = WandbPointCloudRecorder(eval_env, 
        global_step_counter=global_step_counter, 
        save_plotly=True, foldername=result_folder,
        record_video=config['record_video'],
        upload_video=config['upload_video'],
        log_plotly_once=(config['eval'] is None))
    if config['record_video']:
        eval_env.enable_video_recording()
    
    eval_callback = EvalCallbackWPrimitiveInfo( #EvalCallback(
        eval_env, 
        n_eval_episodes=config['n_eval_episodes'], 
        eval_freq=config['eval_freq'],
        best_model_save_path=os.path.join(result_folder, "model"))
    ckpt_callback = CustomizedCheckpointCallback(
        save_freq_latest=config['save_freq_latest'],
        save_freq_checkpoint=config['save_freq_checkpoint'],
        save_path=os.path.join(result_folder, f"model-{run.id}"),
        save_replay_buffer=config['save_replay_buffer'],
        save_vecnormalize=True)  # TODO: confirm?
    progress_bar_callback = ProgressBarCallback()
    callback_list = CallbackList([ckpt_callback, eval_callback, progress_bar_callback])
    
    # ------------- Start Training -------------
    if config['eval'] is None:
        remaining_timesteps = config['train_steps'] - model.num_timesteps
        model.learn(
            total_timesteps=remaining_timesteps,
            log_interval=10,
            callback=callback_list,
            reset_num_timesteps=False)

    # ------------- Eval -------------
    else:
        deterministic = False
        if config['load_ckpt'] is not None:
            model.policy.eval()
            deterministic = True

        from tqdm import tqdm
        import numpy as np
        import pandas as pd

        pbar = tqdm(total=config['eval'])

        per_object_succ = config['log_object'] > 0
        mean_reward, std_reward, succ, verbose_buffer, prim_perc = evaluate_policy(
            model, eval_env, n_eval_episodes=config['eval'], deterministic=deterministic,
            save_path=os.path.join(result_folder, f'obs_list_{0}.pkl'), pbar=pbar, per_object_succ=per_object_succ,
            return_success_rate=True, verbose=True, return_episode_primitiv_perc=True)

        
        uncertainty = 1.96 * np.sqrt(succ * (1 - succ) / config['eval'])
        print(f"succ={succ:.3f} +/- {uncertainty:.3f}. mean_reward={mean_reward:.2f} +/- {std_reward}")
        print(f"primitive usage:")
        for p, v in prim_perc.items():
            print(f"{p}: {v:.1f}%")

        # Upload videos
        if config['upload_video']:
            vid_dir = os.path.join(result_folder, 'video')
            upload_videos(vid_dir, config['fullname'], delete=config['delete_video'])

        # Export the verbose buffer
        eval_dir = os.path.join(result_folder, 'evals')
        if not os.path.exists(eval_dir):
            os.makedirs(eval_dir)
        import pickle
        with open(os.path.join(eval_dir, f'verbose_eval.pkl'), 'wb') as f:
            pickle.dump(verbose_buffer, f)