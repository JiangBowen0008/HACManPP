from typing import Any, Callable, List, Optional, Sequence, Tuple, Type, Union, Dict
import gym
import gym.spaces as spaces
from copy import deepcopy
import numpy as np

# import mani_skill2.envs
from mani_skill2.utils.visualization.misc import put_text_on_image, append_text_to_image
from sapien.core import Pose
import traceback
from hacman.utils.primitive_utils import GroundingTypes, Primitive, get_primitive_class, MAX_PRIMITIVES
# import hacman_ms.env_wrappers.primitives
import os
from .base_obs_wrapper import BaseObsWrapper
from .action_wrapper import HACManActionWrapper

class FlatActionLogitWrapper(HACManActionWrapper):
    """
    Wrapper for handling a variety of primitives and their groundings in a robotics environment.       
    """
    def __init__(self, 
                 env,
                 bg_mapping_mode="bbox",
                 sample_action_logits=False,
                 **kwargs):
        super().__init__(env, **kwargs)
        # for i in range(self.num_prims):
            # self.prim_param_dims[i] += 3    # Predict the location
        action_dim = np.sum(self.prim_param_dims) + self.num_prims ## 5*5 + 5 
        self.fake_prim_dims = np.array([action_dim])
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
            "prim_idx": gym.spaces.Box(-np.inf, np.inf, (1,)),
            "poke_idx": gym.spaces.Box(-np.inf, np.inf, (1,)),
            "available_prims": gym.spaces.Box(-np.inf, np.inf, (1,)),
            "prim_groundings": gym.spaces.Box(-np.inf, np.inf, (1,)),#MAX_PRIMITIVES
            "prim_param_dims": gym.spaces.Box(-np.inf, np.inf, (1,))}
        space_dict.update(extra_spaces)

        return spaces.Dict(space_dict)
    
    def observation(self, obs):
        """
        Appends primitive related observation.
        """
        obs = super().observation(obs)
        prim_states = self.unwrapped.get_primitive_states()
        # Append the primitive groundings to the observation
        # prim_grounding = np.copy(self.prim_groundings)
        # if self.restrict_primitive:
            # for i, prim in enumerate(self.prims):
                # prim_grounding[i] = prim_grounding[i] if prim.is_valid(prim_states) else GroundingTypes.NONE.value
        prim_grounding = np.array([GroundingTypes.OBJECT_AND_BACKGROUND])
        obs['prim_idx'] = np.array([-np.inf])
        obs['poke_idx'] = np.array([-np.inf])
        obs['available_prims'] = np.array([1])## have a fake available prims with flat actions
        obs['prim_groundings'] = np.array(prim_grounding)
        obs['prim_param_dims'] = np.array(self.fake_prim_dims)
        return obs

    def log_error(self, exception, prim_name, location, motion):
        print(f'Exception at primitive {prim_name} execution step: {exception}, {exception.__traceback__}')
        ## create a pickle file and dump all the debug information into the pickle file
        ## full error info
        error_str = traceback.format_exception(etype = type(exception), value = exception, tb = exception.__traceback__)
        # Add object name to the info
        if hasattr(self.cube, 'mesh_name'):
            obj_name = self.cube.mesh_name
        else:
            obj_name = "cube"
        ## add action related information, including primitive, location and motion
        primitive_name = prim_name
        location_info = location
        motion_info = motion
        ## add current env image for debugging purpose
        current_image, frame_info = self.render_offscreen()
        dump_info = dict(obj_name = obj_name, prim_name=primitive_name, location_info = location_info, motion_info=motion_info, frame_info=frame_info, image = current_image, error_info = error_str)
        # debug_folder = './debug'
        # if not os.path.exists(debug_folder):
            # os.makedirs(debug_folder)
        # num_file = len(os.listdir('./debug/'))
        # with open(os.path.join(debug_folder, 'mujoco_error_file_'+str(num_file+1)+'.pkl'),'wb') as file:
            # pickle.dump(dump_info, file)
        # raw_obs = {}
        # if 'pointcloud' not in raw_obs.keys():
        #     raw_obs['pointcloud'] = np.zeros((self.pcd_size, 10))
        all_rewards = [[-1.]]
        raw_obs, reward, done, info = self.unwrapped.compute_step_return()
        # done = False
        # info = dict(success = False, obj_name=obj_name)
        return raw_obs, all_rewards, done, info
    
    def step(self, action):
        """
        Takes a step in the environment using a HACMan action.
        *   Notices one HACMan action correpsonds to a number of sim_step() calls,
            which directly calls the underlying environment's step function.
        """
        # action = np.random.uniform(-1, 1, self.action_space.shape)
        # Find out which primitive to use
        obj_points, bg_points = self.prev_obs['object_pcd_points'], self.prev_obs['background_pcd_points']
        points = np.concatenate([obj_points, bg_points], axis=0)
        poke_idx = int(self.prev_obs['poke_idx'][0])
        location = points[poke_idx]
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
        # prim_loc = action_params[start_idx:start_idx+3]
        # prim_param = action_params[start_idx+3:end_idx]
        prim_param = action_params[start_idx:end_idx]
        
        # self.unwrapped.contact_site.set_pose(Pose(location))  # Add contact point visualization
        motion = np.copy(prim_param)
        normal = None
        if 'object_pcd_normals' in self.prev_obs.keys():
            obj_normals = self.prev_obs['object_pcd_normals']
            bg_normals = self.prev_obs['background_pcd_normals']
            normals = np.concatenate([obj_normals, bg_normals], axis=0)
            normal = np.array(normals[poke_idx])
        # normal = np.array([0, 0, 3])
        # prim_loc = self.map_location(prim_loc, prim.grounding_type)
        # raw_obs, all_rewards, done, info = prim.execute(prim_loc, motion, normal=None)
        try: 
            raw_obs, all_rewards, done, info = prim.execute(location, motion, normal=normal, rotation_type=self.rotation_type, squash_output=self.squash_output)
        except Exception as e:
            ## if the mujoco error occurs, log the related object, action, image info to debug folder
            raw_obs, all_rewards, done, info  = self.log_error(exception=e, prim_name=prim_name, location=location, motion=motion)

        # Compute the step outcome
        obs = self.observation(raw_obs)
        reward = self.aggregate_rewards(all_rewards, mode=self.reward_aggregation)

        # Log the step info
        info.update({
            "action_location": location,
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