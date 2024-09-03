"""
Should different action modes be included here?
"""

import os, sys

import numpy as np
import imageio
import pickle
import copy
import gym
from functools import cached_property
# from gym.envs.registration import register

# from .base_env import BaseEnv, 
from hacman.utils.transformations import to_pose_mat, transform_point_cloud, decompose_pose_mat, sample_idx

from hacman.utils.plotly_utils import plot_pcd, plot_action, plot_pcd_with_score
from hacman_bin.util import angle_diff
from hacman_bin.bin_env import BinEnv
import hacman_bin.primitives
# import plotly.graph_objects as go
from typing import Any, Callable, List, Optional, Sequence, Tuple, Type, Union, Dict
from hacman.algos.location_policy import RandomLocation

# from mani_skill2.utils.registration import register_env
# @register_env(uid='HACManBinEnv-v0')
class HACManBinEnv(BinEnv, gym.Env):
    def __init__(self, 
                 reward_mode="flow",
                 real_robot=False,
                 action_repeat=10,
                 action_mode="per_point_action",
                 fixed_ep_len=False,
                 reward_gripper_distance=None,
                 success_threshold=0.03,
                 object_pcd_size=400,
                 background_pcd_size=400,
                 **kwargs):
        self.goal = None
        self.target_euler = None
        self.step_count = 0
        self.missing_object_count = 0
        self.motion_failure_count = 0
        self.real_robot = real_robot
        if self.real_robot:
            raise NotImplementedError
            from franka_env_polymetis import FrankaPokeEnv
            self.env = FrankaPokeEnv(**kwargs)
        # else:
            # if action_mode == 'regress_action_only':
            #     # Only use position limits when the action continues from previous position
            #     kwargs['ignore_position_limits'] = False
        BinEnv.__init__(self, **kwargs)
        self.object_pcd_size = object_pcd_size
        self.background_pcd_size = background_pcd_size
        self.pcd_size = object_pcd_size + background_pcd_size
        self.observation_space = gym.spaces.Dict(
            spaces={
                "object_pose": gym.spaces.Box(-np.inf, np.inf, (4,4)),
                "goal_pose": gym.spaces.Box(-np.inf, np.inf, (4,4)),
                # "gripper_pose": gym.spaces.Box(-np.inf, np.inf, (4,4)),
                "pointcloud": gym.spaces.Box(-np.inf, np.inf, (self.pcd_size, 10)),
            }
        )
        self.action_space = gym.spaces.Box(-1, 1, (3,))
        self.prev_reward = None
        self.action_repeat = action_repeat
        self.action_mode = action_mode
        self.fixed_ep_len = fixed_ep_len
        self.reward_mode = reward_mode
        self.reward_gripper_distance = reward_gripper_distance
        self.success_threshold = success_threshold
    
    def reset(self, **kwargs):
        if self.real_robot:
            for i in range(10):
                obs = BinEnv.reset(self, **kwargs)
                # obs = self.process_observation(raw_obs)
                success, reward = self._evaluate_obs(obs)
                if success:
                    print(f'Resample goal because t=0 is at the goal: Attempt {i}')
                else:
                    break
            if success:
                Warning(f'Cannot find a goal within 10 attempts.')
        else:
            for i in range(10):
                if self.action_mode == "regress_action_only":
                    obs = BinEnv.reset(
                        self, **kwargs, start_gripper_above_obj=True)
                else:
                    obs = BinEnv.reset(self, **kwargs)
                # obs = self.process_observation(raw_obs)
                if not self.get_primitive_states()["is_lifted"]:
                    break
                
        # self.prev_obs = obs
        return obs
    
    def compute_step_return(self, info={}):
        """
        Used in primitives, givent raw_obs and info generated from executing a primitive,
        Calculate the corresponding processed observation and update the info.
        """
        # Calculate step outcome
        self.step_count += 1
        obs = self._get_observations(force_update=True)

        if "poke_success" in info.keys() and not info['poke_success']:
            self.motion_failure_count += 1
            if self.motion_failure_count % 100 == 0:
                print(f"{self.motion_failure_count/self.step_count*100:.2f}% motion failure within "
                    f"{self.step_count} total steps")
        
        # Add object name to the info
        if hasattr(self.cube, 'mesh_name'):
            obj_name = self.cube.mesh_name
        else:
            obj_name = "cube"
        info.update({"object_name": obj_name})

        # Get the camframes
        info.update({"cam_frames": copy.copy(self.cam_frames)})
        self.cam_frames.clear()

        # Evaluate the observation
        box_in_the_view = (obs["pointcloud"] is not None)
        if not box_in_the_view:
            self.missing_object_count += 1
            print(f"{self.missing_object_count/self.step_count*100:.2f}% missing object within "
                f"{self.step_count} total steps")
            obs = self.reset(hard_reset=False)
            success, reward = False, -1
        else:
            success, reward = self._evaluate_obs(obs)
        info.update({"success": success,
                     "box_in_the_view": box_in_the_view,})
                
        done = False
        if not self.fixed_ep_len:
            done = success
        
        self.prev_reward = reward
        return obs, reward, done, info
    
    def get_segmentation_ids(self):       
        object_ids = np.array([1])  # TODO: confirm
        # background_ids = np.array([cubeB_id, background_id])
        background_ids = np.array([0])
        return {"object_ids": object_ids, "background_ids": background_ids}

    def get_goal_pose(self, format="mat"):
        goal = self.goal.copy()
        p, q = goal[:3], goal[3:]
        if format == "mat":
            return to_pose_mat(p, q, input_wxyz=True)
        elif format == "vector":
            return np.concatenate([p, q])
    
    def get_object_pose(self, format="mat"):
        # Notice: the cube pos and quat may not be updated in the env 
        # if _get_observation is not called
        pose = BinEnv.get_cube_pose(self).copy()
        p, q = pose[:3], pose[3:]
        if format == "mat":
            return to_pose_mat(p, q, input_wxyz=True)
        elif format == "vector":
            return np.concatenate([p, q])
    
    def get_gripper_pose(self, format="mat"):
        # Notice: the cube pos and quat may not be updated in the env 
        # if _get_observation is not called
        p = BinEnv.get_gripper_pos(self).copy()
        q = BinEnv.get_gripper_quat(self).copy()
        if format == "mat":
            return to_pose_mat(p, q, input_wxyz=True)
        elif format == "vector":
            return np.concatenate([p, q])
    
    def get_primitive_states(self):
        return {"is_lifted": self.is_cube_lifted, 'is_grasped': self.is_cube_grasped}

    def get_object_dim(self):
        return np.max(self.get_cube_size()) * 2
    
    def get_prev_reward(self):
        prev_reward = self.prev_reward
        if prev_reward is None:
            current_obs = self._get_observations(force_update=True, compute_normals=False)
            success, prev_reward = self._evaluate_obs(current_obs)
        return prev_reward
    
    def map_location(self, loc, mode="recentered_ws"):
        # mode is ignored in hacman_bin
        loc = self.location_center + self.location_scale * loc
        return loc
    
    @property
    def is_cube_lifted(self):
        # check if the object is lifted
        obj_pos = self.get_object_pose(format="vector")[:3]
        max_obj_size = np.max(self.get_cube_size())
        return (obj_pos[2] - self.table_offset[2]) > max_obj_size * 2.0#1.2
    
    @property
    def is_cube_grasped(self):
        return self._check_grasp(self.robots[0].gripper, object_geoms=self.cube)
    
    def _get_observations(self, force_update=False, compute_normals=False):
        # Subsample PCDs to fixed lengths
        raw_obs = BinEnv._get_observations(
            self, force_update=force_update, compute_normals=compute_normals)
        obs = {}
        if not 'pointcloud' in raw_obs.keys() or raw_obs['pointcloud'] is None:
            obs['pointcloud'] = None
            return obs

        pointcloud = raw_obs['pointcloud']
        segs = pointcloud[:,-1]
        obj_pcd = pointcloud[segs == 1]
        bg_pcd = pointcloud[segs == 0] 
        
        # Sample points to a fixed length
        obj_idx = sample_idx(len(obj_pcd), self.object_pcd_size)
        sampled_obj_pcd = obj_pcd[obj_idx, :]
        bg_idx = sample_idx(len(bg_pcd), self.background_pcd_size)
        sampled_bg_pcd = bg_pcd[bg_idx, :]
        sampled_pcd = np.vstack([sampled_obj_pcd, sampled_bg_pcd])
        obs['pointcloud'] = sampled_pcd
        
 

        if self.real_robot:
            # Save goal frames
            obs['raw_goal_frames'] = obs['goal_frames']
            
        return obs
    
    def _evaluate_obs(self, obs):  # goal
        if obs['pointcloud'] is None:
            reward = -1
            success = False
            return success, reward
    
        if self.reward_mode == "flow":
            if self.goal_mode == "lifted":
                obj_pos = self.get_object_pose(format="vector")[:3]
                max_obj_size = np.max(self.get_cube_size())
                lifted_dist = (obj_pos[2] - self.table_offset[2])
                max_lifted_dist = max_obj_size * 1.5
                reward = min(lifted_dist - max_lifted_dist, 0.0)
                success = self.is_cube_lifted
            else:
                seg_ids = self.get_segmentation_ids()
                obj_ids = seg_ids['object_ids']
                pointcloud = obs['pointcloud']
                current_pcd = pointcloud[np.isin(pointcloud[:,-1], obj_ids)][..., :3]
                
                current_pose = self.get_object_pose(format="mat")
                goal_pose = self.get_goal_pose(format="mat")

                goal_pcd = transform_point_cloud(current_pose, goal_pose, current_pcd)
                flow = np.linalg.norm(goal_pcd - current_pcd, axis=-1)
                mean_flow = np.mean(flow, axis=-1)
                reward = -mean_flow

                # Add a distance term
                if self.action_mode in ["regress_action_only", "regress_location_and_action"] and (self.reward_gripper_distance is not None):
                    gripper_pos, _ = decompose_pose_mat(obs['gripper_pose'])
                    dists = np.linalg.norm(current_pcd - gripper_pos, axis=1)
                    min_dist = np.min(dists)
                    reward -= max(min_dist - 0.05, 0.0) * self.reward_gripper_distance
            
                success = mean_flow < self.success_threshold
        
        elif self.reward_mode == "pose_diff":
            DeprecationWarning("Pose diff reward is deprecated.")
            object_pose = self.get_object_pose(format="vector")
            object_pos, object_ori = object_pose[:3], object_pose[3:]
            goal_pose = self.get_goal_pose(format="vector")
            goal_pos, goal_ori = goal_pose[:3], goal_pose[3:]
            pos_diff = np.linalg.norm(object_pos - goal_pos)
            ori_diff = angle_diff(object_ori, goal_ori) / np.pi * 180.0
            reward = - (50 * pos_diff + 0.2 * ori_diff)
            success = - reward < self.success_threshold
        
        else:
            raise NotImplementedError

        return success, reward

    def render_offscreen(self):
        (img, frame_info) = BinEnv.render_offscreen(self)
        frame_info.update(self.get_primitive_states())
    
        return (img, frame_info)


if __name__ == "__main__":
    env = HACManBinEnv()