import os
import functools
import numpy as np

from stable_baselines3.td3.policies import TD3Policy
from stable_baselines3 import TD3
from stable_baselines3.sac.policies import SACPolicy
from stable_baselines3.sac import SAC
from stable_baselines3.common.noise import NormalActionNoise, VectorizedActionNoise

from hacman.algos import (
    HACManTD3,
    MixTD3Policy,
    MultiTD3,
    MultiTD3Policy,
    LogitTD3Policy,
)
from hacman.algos.feature_extractors.feature_extractors import PointCloudExtractor, PointCloudGlobalExtractor, StatesExtractor


def add_model_config(parser):
    parser.add_argument("--load_ckpt", default=None, type=str, help="Ckpt path. Set to \"latest\" to use the latest checkpoint.")
    parser.add_argument("--load_best", action="store_true", help="Load the best model instead of the latest model.")
    parser.add_argument("--algo", default='TD3', type=str, help="RL algorithm")
    parser.add_argument("--gradient_steps", default=40, type=int, help="Gradient step per env step")
    parser.add_argument("--batch_size", default=64, type=int, help="Batch Size")
    parser.add_argument("--ent_coef", default='auto', type=str, help="Entropy Coefficient mode")
    parser.add_argument("--clip_grad_norm", default=None, type=float, help="Clip gradient norm for critic")
    parser.add_argument("--clamp_critic_max", default=None, type=float, help="Clamp critic value")
    parser.add_argument("--clamp_critic_min", default=None, type=float, help="Clamp critic value")
    parser.add_argument("--gamma", default=0.99, type=float, help="Discount factor")
    parser.add_argument("--actor_update_interval", default=4, type=int, help="Actor update interval")
    parser.add_argument("--target_update_interval", default=4, type=int, help="Target update interval")
    parser.add_argument("--action_noise", default=None, type=float, help="Action noise sigma")
    parser.add_argument("--initial_timesteps", default=10000, type=int, help="Initial env steps before training starts")
    parser.add_argument("--mean_q", action="store_true", help="Use mean Q instead of min Q")
    parser.add_argument("--learning_rate", default=1e-4, type=float, help="Learning rate")
    parser.add_argument("--tau", default = 0.005, type = float, help = 'tau for soft_update')
    parser.add_argument("--share_features_extractor", action="store_true", help="share features extractor")
    parser.add_argument("--pos_in_feature", action="store_true", help="use pos in feature")
    parser.add_argument("--preprocessing_fn", default="flow", type=str, help="Input processing function")
    parser.add_argument('--net_arch', default=[128, 128, 128], nargs="+", type=int)
    parser.add_argument('--sample_action_logits', action="store_true", help="Sample action logits instead of argmax at action output")

    # Baseline options
    parser.add_argument("--feature_mode", default='points', type=str,
                        choices={'points', 'states'})
    return


def setup_model(config, env, result_folder, normalize_param=None):
    # ----------- Search for ckpt  ------------
    if config['load_ckpt'] is not None:
        if not config['load_ckpt'].endswith('.zip'):
            config['load_ckpt'] = search_for_the_latest_ckpt(
                config['load_dirname'], config['load_ckpt'], load_best=config['load_best'])

    # ----------- Feature extractors ------------ 
    if config["feature_mode"] == "states":
        if config["action_mode"] in {"flat", "per_primitive"}:
            features_extractor_class = StatesExtractor
        else:
            raise ValueError
        
    if config["feature_mode"] == "points":
        if config["action_mode"] in {"per_point_action", "per_point",'per_point_logit'}:
            features_extractor_class = functools.partial(PointCloudExtractor, preprocessing_fn=config['preprocessing_fn'])
        elif config["action_mode"] in {"flat", "per_primitive"}:
            features_extractor_class = functools.partial(PointCloudGlobalExtractor, preprocessing_fn=config['preprocessing_fn'],
                                                         include_gripper=True)
        else:
            raise ValueError
        
        features_extractor_class = functools.partial(features_extractor_class, pos_in_feature=config['pos_in_feature'])

        if normalize_param is not None:
            features_extractor_class = functools.partial(features_extractor_class, normalize_pos_param=normalize_param)

        repeat_features = (config["action_mode"] in {"per_point_action", "per_point", "per_point_logit","per_primitive"}) and (config["algo"] in {"TD3", "SAC"})
        features_extractor_class = functools.partial(features_extractor_class, repeat_features=repeat_features)
    
    # --------- Action Noise -----------
    vec_noise = None
    if config['action_noise'] is not None:
        action_dim = env.action_space.shape[0]
        base_noise = NormalActionNoise(mean=np.zeros(action_dim), sigma=config['action_noise'])
        vec_noise = VectorizedActionNoise(base_noise=base_noise, n_envs=env.num_envs)

    # features_extractor_class = functools.partial(features_extractor_class, primitives=config['primitives'])
    # ----------- Model ------------
    model_kwargs = dict(batch_size=config['batch_size'], gamma=config['gamma'],
                verbose=1, learning_starts=config['initial_timesteps'], learning_rate=config['learning_rate'],
                tau = config['tau'], 
                actor_update_interval=config['actor_update_interval'], 
                target_update_interval=config['target_update_interval'], train_freq=1,
                action_noise=vec_noise,
                tensorboard_log=result_folder+'/tb', gradient_steps=config['gradient_steps'], 
                seed=config["seed"])
    # TD3-based model specific kwargs
    if "TD3" in config['algo']:
        td3_kwargs = model_kwargs.copy()
        td3_kwargs.update(dict(
            clip_critic_grad_norm=config['clip_grad_norm'], 
            clamp_critic_min=config['clamp_critic_min'],
            clamp_critic_max=config['clamp_critic_max'],
            mean_q=config['mean_q'],
        ))
    
    if config['algo'] == 'TD3':
        if config["action_mode"] == "flat" and config["sample_action_logits"]:
            policy_class = functools.partial(LogitTD3Policy, num_primitives=len(config['primitives']))
        else:
            policy_class = TD3Policy
        
        policy = functools.partial(policy_class,
                        features_extractor_class=features_extractor_class,
                        share_features_extractor=config["share_features_extractor"],
                        net_arch=config['net_arch'])
        if config['load_ckpt'] is None:
            model = TD3(policy, env, **td3_kwargs)
        else:
            model = TD3.load(path=config['load_ckpt'], env=env, **td3_kwargs) # Overwrite hyperparameters of the loaded model
            print(f"Loaded policy: {config['load_ckpt']}")

    elif config['algo'] == 'HybridTD3':
        policy = functools.partial(TD3Policy,
                        features_extractor_class=features_extractor_class,
                        share_features_extractor=False,
                        net_arch=config['net_arch'])
        hybrid_td3_kwargs = td3_kwargs.copy()
        hybrid_td3_kwargs.update({"temperature": config['location_model_temperature']})
        if config['load_ckpt'] is None:
            model = HACManTD3(policy, env, **hybrid_td3_kwargs)
        else:
            model = HACManTD3.load(path=config['load_ckpt'], env=env, **hybrid_td3_kwargs) # Overwrite hyperparameters of the loaded model
            print(f"Loaded policy: {config['load_ckpt']}")
    
    elif config['algo'] == 'MultiTD3':
        if config["action_mode"] == "per_point_logit":
            num_prim = 1 ## flat all primitives parameters as action output so we use 1 fake primitive for the model
        else:
            num_prim = len(config['primitives'])
        policy = functools.partial(MultiTD3Policy,
                        features_extractor_class=features_extractor_class,
                        share_features_extractor=config["share_features_extractor"],
                        net_arch=config['net_arch'],
                        num_primitives=len(config['primitives']))
        if config['load_ckpt'] is None:
            model = MultiTD3(policy, env,**td3_kwargs)
        else:
            model = MultiTD3.load(path=config['load_ckpt'], env=env, **td3_kwargs) # Overwrite hyperparameters of the loaded model
            print(f"Loaded policy: {config['load_ckpt']}")
    
    elif config['algo'] == "SAC":
        policy_class = functools.partial(SACPolicy, 
                        features_extractor_class=features_extractor_class,
                        share_features_extractor=config["share_features_extractor"],
                        net_arch=config['net_arch'])
        sac_kwargs = model_kwargs.copy()
        sac_kwargs.update({"ent_coef": config['ent_coef']})
        if config['load_ckpt'] is None:
            model = SAC(policy_class, env, **sac_kwargs)
        else:
            model = SAC.load(path=config['load_ckpt'], env=env, **td3_kwargs)
            print(f"Loaded policy: {config['load_ckpt']}")

    # Load the replay buffer
    if config['load_ckpt'] is not None:
        ckpt_dir = os.path.dirname(config['load_ckpt'])
        buffer_path = os.path.join(ckpt_dir, "rl_model_replay_buffer_latest.pkl")
        if os.path.exists(buffer_path):
            model.load_replay_buffer(buffer_path)
            print(f"Loaded replay buffer: {buffer_path}")
        else:
            Warning(f"Replay buffer not found: {buffer_path}")

    return model

def search_for_the_latest_ckpt(dir, keyword, load_best=False):
    ckpt_list = []
    
    # Recursively walk through all directories and subdirectories
    for root, dirs, files in os.walk(dir):
        
        # If a keyword is provided, skip directories that don't contain the keyword
        if keyword and keyword not in root:
            continue
            
        for file in files:
            if load_best:
                if file.endswith(".zip") and "best_model" in file:
                    full_path = os.path.join(root, file)
                    ckpt_list.append(full_path)
            else:
                if file.endswith(".zip"):
                    full_path = os.path.join(root, file)
                    ckpt_list.append(full_path)
                
    # Sort by modification time
    ckpt_list = sorted(ckpt_list, key=os.path.getmtime)
    ckpt_path = ckpt_list[-1]
    print(f"Found ckpt: {ckpt_path}")
    
    return ckpt_path

if __name__ == "__main__":
    ckpt_name = search_for_the_latest_ckpt("logs/hacman", "Exp2035-1")
    print(ckpt_name)