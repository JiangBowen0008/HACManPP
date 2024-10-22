import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import pickle
import gym
import numpy as np
from collections import defaultdict

from stable_baselines3.common import base_class
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecMonitor, is_vecenv_wrapped


def evaluate_policy(
    model: "base_class.BaseAlgorithm",
    env: Union[gym.Env, VecEnv],
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    render: bool = False,
    callback: Optional[Callable[[Dict[str, Any], Dict[str, Any]], None]] = None,
    reward_threshold: Optional[float] = None,
    return_success_rate: bool = False,
    return_episode_rewards: bool = False,
    return_episode_primitiv_perc: bool = False,
    warn: bool = True,
    save_path: str = None,
    pbar = None,
    per_object_succ = False,
    verbose = False,
) -> Union[Tuple[float, float], Tuple[List[float], List[int]]]:
    """
    Runs policy for ``n_eval_episodes`` episodes and returns average reward.
    If a vector env is passed in, this divides the episodes to evaluate onto the
    different elements of the vector env. This static division of work is done to
    remove bias. See https://github.com/DLR-RM/stable-baselines3/issues/402 for more
    details and discussion.

    .. note::
        If environment has not been wrapped with ``Monitor`` wrapper, reward and
        episode lengths are counted as it appears with ``env.step`` calls. If
        the environment contains wrappers that modify rewards or episode lengths
        (e.g. reward scaling, early episode reset), these will affect the evaluation
        results as well. You can avoid this by wrapping environment with ``Monitor``
        wrapper before anything else.

    :param model: The RL agent you want to evaluate.
    :param env: The gym environment or ``VecEnv`` environment.
    :param n_eval_episodes: Number of episode to evaluate the agent
    :param deterministic: Whether to use deterministic or stochastic actions
    :param render: Whether to render the environment or not
    :param callback: callback function to do additional checks,
        called after each step. Gets locals() and globals() passed as parameters.
    :param reward_threshold: Minimum expected reward per episode,
        this will raise an error if the performance is not met
    :param return_episode_rewards: If True, a list of rewards and episode lengths
        per episode will be returned instead of the mean.
    :param warn: If True (default), warns user about lack of a Monitor wrapper in the
        evaluation environment.
    :return: Mean reward per episode, std of reward per episode.
        Returns ([float], [int]) when ``return_episode_rewards`` is True, first
        list containing per-episode rewards and second containing per-episode lengths
        (in number of steps).
    """
    is_monitor_wrapped = False
    # Avoid circular import
    from stable_baselines3.common.monitor import Monitor

    if not isinstance(env, VecEnv):
        env = DummyVecEnv([lambda: env])

    is_monitor_wrapped = is_vecenv_wrapped(env, VecMonitor) or env.env_is_wrapped(Monitor)[0]

    if not is_monitor_wrapped and warn:
        warnings.warn(
            "Evaluation environment is not wrapped with a ``Monitor`` wrapper. "
            "This may result in reporting modified episode lengths and rewards, if other wrappers happen to modify these. "
            "Consider wrapping environment first with ``Monitor`` wrapper.",
            UserWarning,
        )

    n_envs = env.num_envs
    episode_rewards = []
    episode_lengths = []
    episode_prim_percs = []
    episode_lift_succs = []
    traj_list = []

    episode_counts = np.zeros(n_envs, dtype="int")
    # Divides episodes among different sub environments in the vector as evenly as possible
    episode_count_targets = np.array([(n_eval_episodes + i) // n_envs for i in range(n_envs)], dtype="int")

    current_rewards = np.zeros(n_envs)
    current_lengths = np.zeros(n_envs, dtype="int")
    success_buffer = []
    verbose_buffer = []
    if per_object_succ:
        per_object_results = defaultdict(dict)
    observations = env.reset()
    
    traj_list.append(observations)
    states = None
    episode_starts = np.ones((env.num_envs,), dtype=bool)
    
    while (episode_counts < episode_count_targets).any():
        actions, states = model.predict(observations, state=states, episode_start=episode_starts, deterministic=deterministic)
        observations, rewards, dones, infos = env.step(actions)
        traj_list.append((actions, observations, rewards, dones, infos))
        
        current_rewards += rewards
        current_lengths += 1
        for i in range(n_envs):
            if episode_counts[i] < episode_count_targets[i]:

                # unpack values so that the callback can access the local variables
                reward = rewards[i]
                done = dones[i]
                info = infos[i]
                episode_starts[i] = done

                if callback is not None:
                    callback(locals(), globals())

                # if dones[i]:
                if dones[i]:
                    # if save_path is not None:
                    #     pickle.dump(traj_list, open(save_path, 'wb'))

        
                    if is_monitor_wrapped:
                        # Atari wrapper can send a "done" signal when
                        # the agent loses a life, but it does not correspond
                        # to the true end of episode
                        if "episode" in info.keys():
                            # Do not trust "done" with episode endings.
                            # Monitor wrapper includes "episode" key in info if environment
                            # has been wrapped with it. Use those rewards instead.
                            episode_rewards.append(info["episode"]["r"])
                            episode_lengths.append(info["episode"]["l"])
                            # Only increment at the real end of an episode
                            episode_counts[i] += 1
                    else:
                        episode_rewards.append(current_rewards[i])
                        episode_lengths.append(current_lengths[i])
                        episode_counts[i] += 1
                   

                    # Compute the percentage of primitive used
                    if return_episode_primitiv_perc:
                        prim_perc = compute_primitive_perc(info)
                        episode_prim_percs.append(prim_perc)
                    
                    # Compute the percentage of lift success
                    if "lift_success" in info.keys():
                        episode_lift_succs.append(info["lift_success"])

                    maybe_is_success = info.get("is_success")
                    if maybe_is_success is not None:
                        if per_object_succ:
                            # per_object_results[info['object_name']].append(maybe_is_success)
                            if 'success' not in per_object_results[info['object_name']]:
                                per_object_results[info['object_name']] = dict(success = [maybe_is_success], episode_length = [current_lengths[i]])
                            else:
                                per_object_results[info['object_name']]['success'].append(maybe_is_success)
                                per_object_results[info['object_name']]['episode_length'].append(current_lengths[i])
                               
                        success_buffer.append(maybe_is_success)
                        verbose_buffer.append([
                            maybe_is_success,
                            i,
                            current_lengths[i],
                            reward
                        ])
                    current_rewards[i] = 0
                    current_lengths[i] = 0
                    
                    if pbar is not None:
                        pbar.update(1)

        if render:
            env.render()

    if pbar is not None:
        pbar.update(np.sum(dones))
        
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    if reward_threshold is not None:
        assert mean_reward > reward_threshold, "Mean reward below threshold: " f"{mean_reward:.2f} < {reward_threshold:.2f}"
        
    if return_episode_rewards:
        return_list = [episode_rewards, episode_lengths]
    else:
        return_list = [mean_reward, std_reward]
    
    if return_success_rate:
        assert len(success_buffer) > 0
        success_rate = np.average(success_buffer)
        return_list.append(success_rate)
    print('per obj result', per_object_succ)

    if per_object_succ:
        return_list.append(per_object_results)
    elif verbose:
        return_list.append(verbose_buffer)
    
    if return_episode_primitiv_perc:
        avg_prim_percs = average_primitive_perc(episode_prim_percs)
        return_list.append(avg_prim_percs)
    
    return return_list

def compute_primitive_perc(info):
    assert "executed_prims" in info.keys(), f"Primitive info not found in info dict. Available keys: {info.keys()}"
    assert "available_prims" in info.keys(), f"Primitive info not found in info dict. Available keys: {info.keys()}"
    executed_prims = info["executed_prims"]
    available_prims = info["available_prims"]
    
    # count the number of times each primitive was executed
    prim_counts = defaultdict(int)
    for prim in executed_prims:
        prim_counts[prim] += 1

    # compute the percentage of times each primitive was executed
    prim_perc = {p: 100*prim_counts[p] / len(executed_prims) for p in available_prims} 
    return prim_perc

def average_primitive_perc(episode_prim_percs):
    # average primitive percentages across episodes
    prim_percs = defaultdict(list)
    for prim_perc in episode_prim_percs:
        for prim in prim_perc.keys():
            prim_percs[prim].append(prim_perc[prim])
    avg_prim_percs = {prim: np.mean(prim_percs[prim]) for prim in prim_percs.keys()}
    return avg_prim_percs