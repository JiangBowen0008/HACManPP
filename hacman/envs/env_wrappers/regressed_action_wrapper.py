from typing import Any, Callable, List, Optional, Sequence, Tuple, Type, Union, Dict
import gym
import gym.spaces as spaces
from copy import deepcopy
import numpy as np

# import mani_skill2.envs
from mani_skill2.utils.visualization.misc import put_text_on_image, append_text_to_image
from sapien.core import Pose
import traceback
import os
from hacman.utils.primitive_utils import GroundingTypes, Primitive, get_primitive_class, MAX_PRIMITIVES
# import hacman_ms.env_wrappers.primitives

from .base_obs_wrapper import BaseObsWrapper
from .action_wrapper import HACManActionWrapper

class RegressedActionWrapper(HACManActionWrapper):
    """
    Wrapper for handling a variety of primitives and their groundings in a robotics environment.       
    """
    def __init__(self, 
                 env,
                 bg_mapping_mode="bbox",
                 **kwargs):
        super().__init__(env, **kwargs)
        for i in range(self.num_prims):
            self.prim_param_dims[i] += 3    # Predict the location
        max_param_dim = max(self.prim_param_dims)
        self.action_space = gym.spaces.Box(-1, 1, (max_param_dim,))
            
        self.observation_space = self.update_observation_space(self.observation_space)
        self.bg_mapping_mode = bg_mapping_mode
    
    def update_observation_space(self, obs_space: spaces.Dict):
        """
        Updates the observation space with additional keys for using multi-primitive with HACMan.
        """
        space_dict = obs_space.spaces
        # Add the HACMan spaces
        extra_spaces = {
            "prim_idx": gym.spaces.Box(-np.inf, np.inf, (1,)),
            "available_prims": gym.spaces.Box(-np.inf, np.inf, (1,)),
            "prim_groundings": gym.spaces.Box(-np.inf, np.inf, (MAX_PRIMITIVES,)),
            "prim_param_dims": gym.spaces.Box(-np.inf, np.inf, (self.num_prims,))}
        space_dict.update(extra_spaces)

        return spaces.Dict(space_dict)
    
    def observation(self, obs):
        """
        Appends primitive related observation.
        """
        obs = super().observation(obs)
        prim_states = self.unwrapped.get_primitive_states()
        # Append the primitive groundings to the observation
        prim_grounding = np.copy(self.prim_groundings)
        if self.restrict_primitive:
            for i, prim in enumerate(self.prims):
                prim_grounding[i] = prim_grounding[i] if prim.is_valid(prim_states) else GroundingTypes.NONE.value
        
        obs['prim_idx'] = np.array([-np.inf])
        obs['available_prims'] = np.array([self.num_prims])
        obs['prim_groundings'] = np.array(prim_grounding)
        obs['prim_param_dims'] = np.array(self.prim_param_dims)
        return obs
    
    def log_error(self, exception, prim_name, prim_param):
        print(f'Exception at primitive {prim_name} execution step: {exception}, {exception.__traceback__}')
        ## create a pickle file and dump all the debug information into the pickle file
        ## full error info
        error_str = traceback.format_exception(etype = type(exception), value = exception, tb = exception.__traceback__)
        print(error_str)
        all_rewards = [[-1.]]
        raw_obs, reward, done, info = self.unwrapped.compute_step_return()
        return raw_obs, all_rewards, done, info

    def step(self, action):
        """
        Takes a step in the environment using a HACMan action.
        *   Notices one HACMan action correpsonds to a number of sim_step() calls,
            which directly calls the underlying environment's step function.
        """
        # action = np.random.uniform(-1, 1, self.action_space.shape)
        # Find out which primitive to use
        prim_idx = int(self.prev_obs['prim_idx'][0])
        prim = self.prims[prim_idx]
        prim_name = deepcopy(type(prim).__name__)
        self.executed_prims.append(prim_name)
        
        # Index the corresponding primitive param 
        action_params = np.copy(action)
        prim_loc = action_params[:3]
        prim_param = action_params[3:]
        
        prim_loc = self.map_location(prim_loc, prim.grounding_type)
        try: 
            raw_obs, all_rewards, done, info = prim.execute(prim_loc, prim_param, normal=None)
        except Exception as e:
            ## if the mujoco error occurs, log the related object, action, image info to debug folder
            raw_obs, all_rewards, done, info  = self.log_error(exception=e, prim_name=prim_name, prim_param = prim_param)
     

        # Compute the step outcome
        obs = self.observation(raw_obs)
        reward = self.aggregate_rewards(all_rewards, mode=self.reward_aggregation)

        # Log the step info
        info.update({
            "action_location": prim_loc,
            "action_param": prim.visualize(prim_param),
            "executed_prims": self.executed_prims,
            "prim_name": prim_name,
            "available_prims": [type(p).__name__ for p in self.prims],
            "is_success": info["success"],})
        if 'cam_frames' not in info.keys():
            info["cam_frames"] = deepcopy(self.cam_frames)
            self.cam_frames.clear()
        info['cam_frames'] = self.process_frames(info['cam_frames'])
        return obs, reward, done, info
    
    def map_location(self, loc, grounding_type):
        if grounding_type == GroundingTypes.OBJECT_ONLY:
            obj_pos = self.get_object_pose().p
            obj_dim = self.get_object_dim()
            loc = obj_pos + loc * obj_dim / 2
            return loc
        else:
            return self.unwrapped.map_location(loc, mode=self.bg_mapping_mode)
    
# Test the wrapper
if __name__ == "__main__":
    from mani_skill2.utils.wrappers import RecordEpisode
    from mani_skill2.vector import VecEnv, make as make_vec_env
    wrappers = [HACManActionWrapper, ]
    num_envs = 5
    env: VecEnv = make_vec_env(
        "PickCube-v0",
        num_envs,
        obs_mode="pointcloud",
        reward_mode="dense",
        control_mode="pd_ee_delta_pose",
        enable_segmentation=True,
        wrappers = wrappers,

    )
    # env.seed()
    # env = RecordEpisode(env, f"videos", info_on_video=True)
    # env = HACManObservationWrapper(HACManActionWrapper(ContactVisWrapper(MoreCamWrapper(env))))
    for i in range(3):
        for _ in range(3):
            obs = env.reset()
            # action = np.zeros(6)
            # action = np.random.uniform(-1, 1, 3)
            action = np.zeros((num_envs, 3))
            action[:, i] = 0.1
            print(action)
            # action[5] = -0.5
            obs, reward, done, info = env.step(action)
            print(obs.keys())
    env.close()
    # print(obs)