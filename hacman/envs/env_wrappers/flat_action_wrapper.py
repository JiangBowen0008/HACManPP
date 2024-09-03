from typing import Any, Callable, List, Optional, Sequence, Tuple, Type, Union, Dict
import gym
import gym.spaces as spaces
from copy import deepcopy
import numpy as np

# import mani_skill2.envs
from mani_skill2.utils.visualization.misc import put_text_on_image, append_text_to_image
from sapien.core import Pose

from hacman.utils.primitive_utils import GroundingTypes, Primitive, get_primitive_class, MAX_PRIMITIVES
# import hacman_ms.env_wrappers.primitives

from .base_obs_wrapper import BaseObsWrapper
from .action_wrapper import HACManActionWrapper

class FlatActionWrapper(HACManActionWrapper):
    """
    Wrapper for handling a variety of primitives and their groundings in a robotics environment.       
    """
    def __init__(self, 
                 env,
                 bg_mapping_mode="bbox",
                 sample_action_logits=False,
                 **kwargs):
        super().__init__(env, **kwargs)
        for i in range(self.num_prims):
            self.prim_param_dims[i] += 3    # Predict the location
        action_dim = np.sum(self.prim_param_dims) + self.num_prims
        self.action_param_idx = np.cumsum(self.prim_param_dims)
            
        self.action_space = gym.spaces.Box(-1, 1, (action_dim,))
        self.observation_space = self.update_observation_space(self.observation_space)
        
        self.sample_action_logits = sample_action_logits
        self.bg_mapping_mode = bg_mapping_mode
    
    def update_observation_space(self, obs_space: spaces.Dict):
        """
        Updates the observation space with additional keys for using multi-primitive with HACMan.
        """
        space_dict = obs_space.spaces
        # Add the HACMan spaces
        extra_spaces = {
            "available_prims": gym.spaces.Box(-np.inf, np.inf, (1,)),
            "prim_groundings": gym.spaces.Box(-np.inf, np.inf, (MAX_PRIMITIVES,))}
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
        
        obs['available_prims'] = np.array([self.num_prims])
        obs['prim_groundings'] = np.array(prim_grounding)
        return obs

    def step(self, action):
        """
        Takes a step in the environment using a HACMan action.
        *   Notices one HACMan action correpsonds to a number of sim_step() calls,
            which directly calls the underlying environment's step function.
        """
        # Find out which primitive to use
        logits = action[-self.num_prims:]
        logits = self.mask_invalid_prims(logits)  # Mask out invalid primitives
        if self.deterministic or self.sample_action_logits: # if sample_action_logits, the logits are alreayd one-hot here
            prim_idx = np.argmax(logits)
        else:
            probs = np.exp(logits / 0.5)
            softmax = probs / np.sum(probs)
            prim_idx = np.random.choice(self.num_prims, p=softmax)

        prim = self.prims[prim_idx]
        prim_name = deepcopy(type(prim).__name__)
        self.executed_prims.append(prim_name)
        
        # Index the corresponding primitive param 
        action_params = action[self.num_prims:]
        end_idx = self.action_param_idx[prim_idx]
        start_idx = end_idx - self.prim_param_dims[prim_idx]
        prim_loc = action_params[start_idx:start_idx+3]
        prim_param = action_params[start_idx+3:end_idx]
        
        motion = np.copy(prim_param)
        prim_loc = self.map_location(prim_loc, prim.grounding_type)
        raw_obs, all_rewards, done, info = prim.execute(prim_loc, motion, normal=None)

        # Compute the step outcome
        obs = self.observation(raw_obs)
        reward = self.aggregate_rewards(all_rewards, mode=self.reward_aggregation)

        # Log the step info
        info.update({
            "action_location": prim_loc,
            "action_param": prim.visualize(motion),
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
    
    def mask_invalid_prims(self, logits):
        prim_states = self.unwrapped.get_primitive_states()
        logit_mask = np.zeros_like(logits)
        if self.restrict_primitive:
            for i, prim in enumerate(self.prims):
                logit_mask[i] = -np.inf if not prim.is_valid(prim_states) else 0.
        return logits + logit_mask

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