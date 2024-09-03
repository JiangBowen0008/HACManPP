from typing import Any, Callable, List, Optional, Sequence, Tuple, Type, Union, Dict
import gym
import gym.spaces as spaces
from copy import deepcopy
import numpy as np
import os
import pickle
import open3d as o3d
# import mani_skill2.envs
# from mani_skill2.utils.visualization.misc import put_text_on_image, append_text_to_image
from hacman.utils.plotly_utils import put_text_on_image, append_text_to_image
from PIL import Image
from sapien.core import Pose
import traceback
from hacman.utils.primitive_utils import GroundingTypes, Primitive, get_primitive_class, MAX_PRIMITIVES
from hacman.utils.transformations import to_pose_mat, transform_point_cloud
# import hacman_ms.env_wrappers.primitives

from .base_obs_wrapper import BaseObsWrapper

class HACManActionWrapper(BaseObsWrapper):
    """
    Wrapper for handling a variety of primitives and their groundings in a robotics environment.

    Parameters
    ----------
    env : gym.Env
        The underlying environment to wrap.
    traj_steps : int, optional
        Number of trajectory steps, default is 1.
    use_oracle_motion : bool, optional
        Whether to use oracle-based motion prediction, default is False.
    use_oracle_rotation : bool, optional
        Whether to use oracle-based rotation prediction, default is False.
    use_flow_reward : bool, optional
        Whether to use flow-based reward computation, default is False.
    pad_rewards : bool, optional
        Whether to pad rewards, default is False.
    primitives : list, optional
        List of primitive classes to initialize, default is an empty list.
    restrict_primitive : bool, optional
        Whether to restrict the primitive, default is True.
    end_on_reached : bool, optional
        Whether to terminate the episode when the object has reached the target, default is True.
    end_on_collision : bool, optional
        Whether to terminate the episode when a collision occurs, default is False.
    **kwargs : dict
        Additional keyword arguments.        
    """
    def __init__(self, 
                 env,
                 traj_steps=1,
                 use_oracle_motion=False,
                 use_oracle_rotation=False,
                 use_location_noise=False,
                 rotation_type=0,
                 use_flow_reward=False,
                 pad_rewards=False,
                 primitives=[],
                 restrict_primitive=True,
                 free_move=False,
                 end_on_reached=True,
                 end_on_collision=False,
                 pos_tolerance=0.002,
                 rot_tolerance=0.05,
                 **kwargs):
        super().__init__(env, **kwargs)
        self.prims: List[Type[Primitive]] = []
        for primtive_name in primitives:
            prim_class = get_primitive_class(primtive_name)
            # print(prim_class)
            prim: Primitive = prim_class(self, 
                                    traj_steps=traj_steps,
                                    end_on_reached=end_on_reached,
                                    end_on_collision=end_on_collision,
                                    pad_rewards=pad_rewards,
                                    use_oracle_rotation=use_oracle_rotation,
                                    free_move=free_move,
                                    pos_tolerance=pos_tolerance,
                                    rot_tolerance=rot_tolerance,)
            self.prims.append(prim)

        self.num_prims = len(self.prims)
        self.prim_param_dims = [prim.param_dim for prim in self.prims]
        self.prim_groundings = [prim.grounding_type for prim in self.prims]
        self.prim_groundings = np.pad(
            self.prim_groundings, (0, MAX_PRIMITIVES - self.num_prims),
            constant_values=GroundingTypes.NONE)    # Pad the primitive groundings to length MAX_PRIMITIVES 
        self.executed_prims = []
        self.prim_name_mapping = {
            'Pick_N_Lift_Fixed': 'Grasp',
            'OpenGripper': 'Open Gripper',
            'Move': 'Move delta',
            'PlaceFromTop': 'Move to',
            'Poke': 'Poke',
            'Place': 'Move to',
            'Pick_N_Lift': 'Grasp'
        }
        max_param_dim = max(self.prim_param_dims)
        self.action_space = gym.spaces.Box(-1, 1, (max_param_dim,))
        self.observation_space = self.update_observation_space(self.observation_space)

        self.restrict_primitive = restrict_primitive
        self.elapsed_steps = 0
        self.rotation_type = rotation_type
        # Debug options
        self.use_location_noise = use_location_noise
        self.use_oracle_motion = use_oracle_motion
        self.use_flow_reward = use_flow_reward
        self.use_location_noise = use_location_noise
    
    def reset(self, reconfigure=True, **kwargs):
        # Reconfigure is manually set to True because SB3 does not set this
        self.executed_prims = []
        self.elapsed_steps = 0
        return super().reset(reconfigure=reconfigure, **kwargs)
    
    def update_observation_space(self, obs_space: spaces.Dict):
        """
        Updates the observation space with additional keys for using multi-primitive with HACMan.
        """
        space_dict = obs_space.spaces
        # Add the HACMan spaces
        extra_spaces = {
            "poke_idx": gym.spaces.Box(-np.inf, np.inf, (1,)),
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
        if hasattr(self.env, 'get_observation'):
            obs = self.env.get_observation()
        obs = super().observation(obs)
        prim_states = self.unwrapped.get_primitive_states()
        # Append the primitive groundings to the observation
        prim_grounding = np.copy(self.prim_groundings)
        if self.restrict_primitive:
            for i, prim in enumerate(self.prims):
                prim_grounding[i] = prim_grounding[i] if prim.is_valid(prim_states) else GroundingTypes.NONE.value
        
        obs['poke_idx'] = np.array([-np.inf])
        obs['prim_idx'] = np.array([-np.inf])
        obs['available_prims'] = np.array([self.num_prims])
        obs['prim_groundings'] = np.array(prim_grounding)
        obs['prim_param_dims'] = np.array(self.prim_param_dims)
        return obs
    
    def log_error(self, exception, prim_name, location, motion):
        print(f'Exception at primitive {prim_name} execution step: {exception}, {exception.__traceback__}')
        ## full error info
        error_str = traceback.format_exception(etype = type(exception), value = exception, tb = exception.__traceback__)
        print(error_str)
        all_rewards = [[-1.]]
        raw_obs, reward, done, info = self.unwrapped.compute_step_return()
        return raw_obs, all_rewards, done, info

    def step(self, action, debug=False):
        """
        Takes a step in the environment using a HACMan action.
        *   Notices one HACMan action correpsonds to a number of sim_step() calls,
            which directly calls the underlying environment's step function.
        """
        # Use the previous observation to get the location
        obj_points, bg_points = self.prev_obs['object_pcd_points'], self.prev_obs['background_pcd_points']
        points = np.concatenate([obj_points, bg_points], axis=0)
        poke_idx = int(self.prev_obs['poke_idx'][0])
        prim_idx = int(self.prev_obs['prim_idx'][0])
        location = points[poke_idx]
        if self.use_location_noise:
            location += np.clip(np.random.normal(0, 0.005, size=location.shape), -0.01, 0.01)
        normal = None
        if 'object_pcd_normals' in self.prev_obs.keys():
            obj_normals = self.prev_obs['object_pcd_normals']
            bg_normals = self.prev_obs['background_pcd_normals']
            normals = np.concatenate([obj_normals, bg_normals], axis=0)
            normal = np.array(normals[poke_idx])
        
        prim = self.prims[prim_idx]
        prim_name = deepcopy(type(prim).__name__)
        self.executed_prims.append(prim_name)

        # Sample some background points and actions for visualization
        if prim.grounding_type == GroundingTypes.OBJECT_ONLY.value:
            sampled_idx = np.random.choice(len(obj_points), size=50)
        elif prim.grounding_type == GroundingTypes.BACKGROUND_ONLY.value:
            sampled_idx = np.random.choice(len(bg_points), size=50) + len(obj_points)
        elif prim.grounding_type == GroundingTypes.OBJECT_AND_BACKGROUND.value:
            sampled_idx = np.random.choice(len(points), size=50)
        sampled_points = points[sampled_idx]
        sampled_action_params = self.prev_obs['action_params'][sampled_idx]
        
        # self.unwrapped.contact_site.set_pose(Pose(location))  # Add contact point visualization
        motion = np.copy(action)
        ## add try exception to catchup nan error
        try:
            assert not np.any(np.isnan(motion))
        except:
            print('nan action encountered')
            motion = np.zeros_like(action)
            motion[-2] = 1. ## prevent 0 division for angle prediction
        ## add try exeception here to catch the mujoco error
            
        if debug:
            self._visualize_action(location, action, prim)
        try: 
            raw_obs, all_rewards, done, info = prim.execute(location, motion, normal=normal, rotation_type=self.rotation_type)
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
            "sampled_points": sampled_points,
            "sampled_action_params": prim.visualize(sampled_action_params),
            "executed_prims": self.executed_prims,
            "elapsed_steps": self.elapsed_steps,
            "prim_name": prim_name,
            "available_prims": [type(p).__name__ for p in self.prims],
            "prim_grounding": prim.grounding_type,
            "is_success": info["success"],})
        if 'cam_frames' not in info.keys():
            info["cam_frames"] = deepcopy(self.cam_frames)
            self.cam_frames.clear()
        info['cam_frames'] = self.process_frames(
            info['cam_frames'])
        self.elapsed_steps += 1
        return obs, reward, done, info
    
    def get_object_pose(self) -> Pose:
        pose_vector = self.unwrapped.get_object_pose(format="vector")
        p, q = pose_vector[:3], pose_vector[3:]
        return Pose(p, q)
    
    def get_goal_pose(self) -> Pose:
        pose_vector = self.unwrapped.get_goal_pose(format="vector")
        p, q = pose_vector[:3], pose_vector[3:]
        return Pose(p, q)
    
    def get_object_dim(self) -> float:
        return self.unwrapped.get_object_dim()
    
    def get_gripper_pose(self) -> Pose:
        pose_vector = self.unwrapped.get_gripper_pose(format="vector")
        p, q = pose_vector[:3], pose_vector[3:]
        return Pose(p, q)
    
    def set_site_pos(self, pos):
        self.unwrapped.contact_site.set_pose(Pose(pos))

    def sim_step(self, action):
        return self.env.step(action)
    
    def process_frames(self, frames):
        '''
        frames is a list of (image, info) tuples
        '''
        processed_frames = []
        for img, info in frames:
            info['Primitive'] = self.map_primitive_name(self.executed_prims[-1])
            # info['elapsed_steps'] = self.elapsed_steps
            info['Time Step'] = self.elapsed_steps
            img = self._put_info_on_image(img, info)
            processed_frames.append(img)
        return processed_frames
    
    def map_primitive_name(self, name):
        for k, v in self.prim_name_mapping.items():
            if name in k or k in name:
                return v

    def record_cam_frame(self, extra_info={}):
        if self.record_video:
            img = self.unwrapped.render(mode="cameras")
            info = self.unwrapped.get_primitive_states()
            info.update(extra_info)
            self.cam_frames.append((img, info))
    
    def _put_info_on_image(self, image, info: Dict[str, float], extras=None, overlay=True):
        lines = [f"{k}: {v}" for k, v in info.items()]
        if extras is not None:
            lines.extend(extras)
        if overlay:
            return put_text_on_image(image, lines)
        else:
            return append_text_to_image(image, lines)
    
    def _visualize_action(self, location, motion, prim):
        prim_name = type(prim).__name__
        print(f'Primitive: {prim_name}, Location: {location}, Motion: {motion}')
        obj_o3d, bg_o3d = self.unwrapped.object_pcd_o3d, self.unwrapped.background_pcd_o3d
        obj_o3d.paint_uniform_color([1, 0.706, 0])
        bg_o3d.paint_uniform_color([0.5, 0.5, 0.5])

        object_pose, goal_pose = self.prev_obs['object_pose'], self.prev_obs['goal_pose']
        object_pcd_points = self.prev_obs['object_pcd_points']
        goal_pcd_points = transform_point_cloud(object_pose, goal_pose, object_pcd_points)
        goal_o3d = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(goal_pcd_points))
        goal_o3d.paint_uniform_color([0, 0.651, 0.929])

        location_o3d = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.array([location])))
        location_o3d.paint_uniform_color([0, 1, 0])
        action_vec = prim.visualize(motion)
        action_vec = np.linspace(location, location + action_vec, 100)
        action_o3d = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(action_vec))
        action_o3d.paint_uniform_color([1, 0, 0])

        origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
        o3d.visualization.draw_geometries([origin, obj_o3d, goal_o3d, bg_o3d, location_o3d, action_o3d])
    
 

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
            # print(obs.keys())
    env.close()
    # print(obs)