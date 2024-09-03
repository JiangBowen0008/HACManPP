import numpy as np
from typing import Any, Callable, List, Optional, Sequence, Tuple, Type, Union, Dict
from scipy.spatial.transform import Rotation

import gym

from hacman.utils.primitive_utils import GroundingTypes, Primitive, register_primitive
# from .action_wrapper import HACManActionWrapper
# from .mobile_action_wrapper import HACManMobileActionWrapper

def get_primitive_class(name) -> Type["Primitive"]:
    return {
        "pick_n_drop": Pick_N_Drop,
        "pick_n_move": Pick_N_Move,
        "pick_n_lift": Pick_N_Lift,
        "move": Move,
        "place": Place,
        "place_on_top": PlaceOnTop,
        "place_from_top": PlaceFromTop,
    }[name]

class MSPrimitive(Primitive):    
    '''
    Low-level primitives
    '''
    def move_to(self, location, gripper_action=0, max_steps=50, speed=10.):
        rewards = []
        for _ in range(max_steps):
            raw_obs, reward, done, info = self.step_gripper_to(location, speed=speed, gripper_action=gripper_action)
            rewards.append(reward)
            if (self.gripper_reached_pos(location) and self.stop_on_reached) or done:
                break
        # pad the rewards
        if (len(rewards) < max_steps) and self.pad_rewards:
            max_reward = max(rewards)
            rewards += [max_reward] * (max_steps - len(rewards))
        return raw_obs, rewards, done, info

    def post_contact_motion(self, motion, max_steps=40, positive_z=True):
        rewards = []
        # Post contact motion
        traj_steps = motion.reshape(self.traj_steps, 3)
        for m_pos in traj_steps:
            # m_pos[2] = (m_pos[2] + 1.) / 2.     # Scale z to be within [0, 1]
            if positive_z:
                m_pos[2] = np.clip(m_pos[2], 0, 1)
            if self.env.use_oracle_motion:
                # Translation between object and the goal
                obj_pos = self.env.get_object_pose().p
                goal_pos = self.env.get_goal_pose().p
                oracle_motion = (goal_pos - obj_pos) * 0.65
                m_pos = oracle_motion
            m = np.concatenate([m_pos, np.array([0, 0, 0, -1])])  # Set the gripper to be closed
            for _ in range(max_steps):
                raw_obs, reward, done, info = self.sim_step(m)
                rewards.append(reward)
                if done:
                    break
        # pad the rewards
        if len(rewards) < max_steps and self.pad_rewards:
            last_reward = rewards[-1]
            rewards += [last_reward] * (max_steps - len(rewards))
        return raw_obs, rewards, done, info
            
    def grasp_from_top(self, location):
        all_rewards = []

        # Move to the top of the object
        precontact_location = location + np.array([0, 0, 0.05]) 
        raw_obs, rewards, done, info = self.move_to(precontact_location, gripper_action=1., speed=10., max_steps=30)
        all_rewards.append(rewards)

        # Move down and grasp
        grasp_location = location + np.array([0, 0, -0.02])
        raw_obs, rewards, done, info = self.move_to(grasp_location, speed=5., gripper_action=1., max_steps=15)
        all_rewards.append(rewards)
        raw_obs, rewards, done, info = self.move_to(grasp_location, speed=3., gripper_action=-1., max_steps=10) # Grasp
        all_rewards.append(rewards)    
        return raw_obs, rewards, done, info

    def open_gripper(self, max_steps=20):
        rewards = []
        for _ in range(max_steps):
            raw_obs, reward, done, info = self.sim_step(np.array([0, 0, 0.5, 0, 0, 0, 1]))
            rewards.append(reward)
            if done:
                break
        # pad the rewards
        if (len(rewards) < max_steps) and self.pad_rewards:
            last_reward = rewards[-1]
            rewards += [last_reward] * (max_steps - len(rewards))
        return raw_obs, rewards, done, info
    

    '''
    Utility motion functions
    '''
    def step_gripper_to(self, pos, speed=3., gripper_action=0.):
        delta_pos = pos - self.env.get_gripper_pose().p
        pos_control = delta_pos * speed
        control = np.concatenate([pos_control, np.zeros(3), [gripper_action]])
        raw_obs, reward, success, info = self.sim_step(control)
        return raw_obs, reward, success, info
    
    def gripper_reached_pos(self, pos):
        gripper_pos = self.env.get_gripper_pose().p
        return np.linalg.norm(gripper_pos - pos) < 0.01

    def sim_step(self, action):
        if self.env.control_mode == "pd_ee_delta_pose":
            action_pos = action[:3]
            action_euler = action[3:6]
            action_gripper = action[6]
        elif self.env.control_mode == "pd_ee_delta_pos":
            action_pos = action[:3]
            action_euler = np.zeros(3)
            action_gripper = action[6]
        else:
            raise NotImplementedError

        # Convert the action pos to the gripper frame
        tcp_pose = self.env.get_gripper_pose()
        to_tcp_mat = tcp_pose.inv().to_transformation_matrix()
        to_tcp_rot = Rotation.from_matrix(to_tcp_mat[:3, :3])
        pos_control = to_tcp_rot.apply(action_pos)

        control = [pos_control, action_euler, [action_gripper]]
        control = np.concatenate(control)
        raw_obs, reward, done, info = self.env.sim_step(control)

        # Replace the reward with flow reward
        if self.env.use_flow_reward:
            _, reward = self.env.evaluate_flow(raw_obs)
        else:
            reward = self.env.reward_scale * reward
        self.env.record_cam_frame()
        return raw_obs, reward, done, info

@register_primitive("pick_n_drop", GroundingTypes.OBJECT_ONLY, motion_dim=3)
class Pick_N_Drop(MSPrimitive):
    def execute(self, location, motion, **kwargs):
        all_rewards = []
        # Grasp primitive
        raw_obs, rewards, done, info = self.grasp_from_top(location)
        all_rewards.append(rewards)
        # Post contact motion
        raw_obs, rewards, done, info = self.post_contact_motion(motion)
        all_rewards.append(rewards)
        # Open the gripper
        if not done:
            raw_obs, rewards, done, info = self.open_gripper()
            all_rewards.append(rewards)
        
        return raw_obs, all_rewards, done, info
    
    def is_valid(self, states: Dict) -> bool:
        return not states['is_grasped']
    
    @property
    def motion_dim(self):
        return 3 * self.traj_steps
    
    @property
    def grounding_type(self):
        return GroundingTypes.OBJECT_ONLY.value
    

class Pick_N_Move(MSPrimitive):
    def execute(self, location, motion, **kwargs):
        all_rewards = []
        # Grasp primitive
        raw_obs, rewards, done, info = self.grasp_from_top(location)
        all_rewards.append(rewards)
        # Post contact motion
        motion = self.scale_motion_param(motion)
        raw_obs, rewards, done, info = self.post_contact_motion(motion, max_steps=20, positive_z=True)
        all_rewards.append(rewards)
        
        return raw_obs, all_rewards, done, info
    
    def is_valid(self, states: Dict) -> bool:
        return not states['is_grasped']
    
    def scale_motion_param(self, motion):
        return motion * 0.3
    
    @property
    def motion_dim(self):
        return 3 * self.traj_steps
    
    @property
    def grounding_type(self):
        return GroundingTypes.OBJECT_ONLY.value


class Pick_N_Lift(MSPrimitive):
    def execute(self, location, motion, **kwargs):
        all_rewards = []
        # Grasp primitive
        raw_obs, rewards, done, info = self.grasp_from_top(location)
        all_rewards.append(rewards)
        # Post contact motion
        motion = self.scale_motion_param(motion)
        raw_obs, rewards, done, info = self.post_contact_motion(motion, max_steps=20, positive_z=True)
        all_rewards.append(rewards)
        
        return raw_obs, all_rewards, done, info
    
    def is_valid(self, states: Dict) -> bool:
        return not states['is_grasped']
    
    def scale_motion_param(self, motion):
        motion_ = np.zeros_like(motion).T
        motion_[2] = 0.5
        motion_ = motion_.T
        return motion_
    
    @property
    def motion_dim(self):
        return 3 * self.traj_steps
    
    @property
    def grounding_type(self):
        return GroundingTypes.OBJECT_ONLY.value

class Move(MSPrimitive):
    def execute(self, location, motion, **kwargs):
        all_rewards = []
        raw_obs, rewards, done, info = self.post_contact_motion(motion)
        all_rewards.append(rewards)
        
        return raw_obs, all_rewards, done, info
    
    @property
    def motion_dim(self):
        return 3 * self.traj_steps
    
    @property
    def grounding_type(self):
        return GroundingTypes.OBJECT_ONLY.value

class Place(MSPrimitive):
    def execute(self, location, motion, **kwargs):
        all_rewards = []
        # Move to the target location
        target_pos = location + self.scale_motion_param(motion)
        raw_obs, rewards, done, info = self.move_to(target_pos, speed=3., gripper_action=-1., max_steps=50)
        all_rewards.append(rewards)

        # Interupt if the gripper collides with the object
        collided = not self.gripper_reached_pos(target_pos)
        if collided and self.stop_on_collision:
            return raw_obs, all_rewards, done, info
        
        # Open the gripper
        if not done:
            raw_obs, rewards, done, info = self.open_gripper()
            all_rewards.append(rewards)
        return raw_obs, all_rewards, done, info

    def scale_motion_param(self, motion):
        return motion * 0.05

    def is_valid(self, states: Dict) -> bool:
        return states['is_grasped']
    
    @property
    def motion_dim(self):
        return 3
    
    @property
    def grounding_type(self):
        return GroundingTypes.BACKGROUND_ONLY.value

class PlaceFromTop(MSPrimitive):
    def execute(self, location, motion, **kwargs):
        all_rewards = []
        # Move to the target location
        # motion = np.array([0, 0, 1])
        target_pos = location + self.scale_motion_param(motion)
        precontact_pos = np.copy(target_pos)
        precontact_pos[2] += 0.3

        raw_obs, rewards, done, info = self.move_to(precontact_pos, speed=7., gripper_action=-1., max_steps=20)
        all_rewards.append(rewards)

        if not done:
            raw_obs, rewards, done, info = self.move_to(target_pos, speed=3., gripper_action=-1., max_steps=30)
            all_rewards.append(rewards)

            # Interupt if the gripper collides with the object
            collided = not self.gripper_reached_pos(target_pos)
            if collided and self.stop_on_collision:
                return raw_obs, all_rewards, done, info
        
        # Open the gripper
        if not done:
            raw_obs, rewards, done, info = self.open_gripper()
            all_rewards.append(rewards)
        return raw_obs, all_rewards, done, info
    
    def scale_motion_param(self, motion):
        return motion * 0.05

    def is_valid(self, states: Dict) -> bool:
        return states['is_grasped']
    
    @property
    def motion_dim(self):
        return 3
    
    @property
    def grounding_type(self):
        return GroundingTypes.BACKGROUND_ONLY.value
    

class PlaceOnTop(MSPrimitive):
    def execute(self, location, motion, **kwargs):
        all_rewards = []
        # Move to the target location
        target_pos = location + self.scale_motion_param(motion)
        raw_obs, rewards, done, info = self.move_to(target_pos, speed=5., gripper_action=-1., max_steps=50)
        all_rewards.append(rewards)

        # Interupt if the gripper collides with the object
        collided = not self.gripper_reached_pos(target_pos)
        if collided and self.stop_on_collision:
            return raw_obs, all_rewards, done, info

        # Open the gripper
        if not done:
            raw_obs, rewards, done, info = self.open_gripper()
            all_rewards.append(rewards)
        return raw_obs, all_rewards, done, info

    def is_valid(self, states: Dict) -> bool:
        return states['is_grasped']
    
    def scale_motion_param(self, motion):
        motion_ = np.zeros_like(motion).T
        motion_[2] = 0.05
        motion_ = motion_.T
        return motion_
    
    @property
    def motion_dim(self):
        return 3
    
    @property
    def grounding_type(self):
        return GroundingTypes.BACKGROUND_ONLY.value