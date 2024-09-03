import numpy as np
import cv2
from typing import Any, Callable, List, Optional, Sequence, Tuple, Type, Union, Dict
from scipy.spatial.transform import Rotation as R
import quaternion

def xyzw2wxyz(quat : np.ndarray) -> np.ndarray:
    assert len(quat) == 4, f'quaternion size must be 4, got {len(quat)}'
    return np.asarray([quat[3], quat[0], quat[1], quat[2]])

def wxyz2xyzw(quat : np.ndarray) -> np.ndarray:
    assert len(quat) == 4, f'quaternion size must be 4, got {len(quat)}'
    return np.asarray([quat[1], quat[2], quat[3], quat[0]])


def get_pose_from_matrix(matrix, pose_size : int = 7) -> np.ndarray:

    mat = np.array(matrix)
    assert mat.shape == (4, 4), f'pose must contain 4 x 4 elements, but got {mat.shape}'
    
    pos = matrix[:3, 3]
    rot = None

    if pose_size == 6:
        rot = R.from_matrix(matrix[:3, :3]).as_rotvec()
    elif pose_size == 7:
        rot = R.from_matrix(matrix[:3, :3]).as_quat()
        rot = xyzw2wxyz(rot)
    elif pose_size == 9:
        rot = (matrix[:3, :2].T).reshape(-1)
            
    pose = list(pos) + list(rot)

    return np.array(pose)

def get_matrix_from_pose(pose) -> np.ndarray:
    assert len(pose) == 6 or len(pose) == 7 or len(pose) == 9, f'pose must contain 6 or 7 elements, but got {len(pose)}'
    pos_m = np.asarray(pose[:3])
    rot_m = np.identity(3)

    if len(pose) == 6:
        rot_m = R.from_rotvec(pose[3:]).as_matrix()
    elif len(pose) == 7:
        quat = wxyz2xyzw(pose[3:])
        rot_m = R.from_quat(quat).as_matrix()
    elif len(pose) == 9:
        rot_xy = pose[3:].reshape(2, 3)
        rot_m = np.vstack((rot_xy, np.cross(rot_xy[0], rot_xy[1]))).T
            
    ret_m = np.identity(4)
    ret_m[:3, :3] = rot_m
    ret_m[:3, 3] = pos_m

    return ret_m

import gym

from hacman.utils.primitive_utils import GroundingTypes, Primitive, register_primitive


class AdroitPrimitive(Primitive):    
    '''
    Utility primitives
    '''
    sim_step_count = 0

    def __init__(self,
                 env: gym.ObservationWrapper, # Union[HACManActionWrapper, HACManMobileActionWrapper]
                 grounding_type: GroundingTypes,
                 traj_steps: int = 1,
                 motion_dim: int = 30,
                 end_on_reached: bool = False,
                 end_on_collision: bool = False,
                 use_oracle_rotation: bool = False,
                 free_move=False,
                 pad_rewards: bool = False,
                 penalize_collision: bool = False,
                 pos_tolerance: float = 0.005,
                 rot_tolerance: float = 0.05):
        
        super().__init__(
            env=env,
            grounding_type=grounding_type,
            traj_steps=traj_steps,
            motion_dim=motion_dim,
            end_on_reached=end_on_reached,
            end_on_collision=end_on_collision,
            use_oracle_rotation=use_oracle_rotation,
            free_move=free_move,
            pad_rewards=pad_rewards,
            penalize_collision=penalize_collision,
            pos_tolerance=pos_tolerance,
            rot_tolerance=rot_tolerance
        )

        self.pos_low_limits = [-0.25, -0.5, 0.12]
        self.pos_high_limits = [ 0.25,  0.3, 0.3]
    
    def make_reach_joint_pos(self, wrist_pos, gripper_action=0):
        joint_pos = np.zeros(30)
        joint_pos[:3] = wrist_pos
        joint_pos[7]  = 0
        # forefinger
        joint_pos[8] = 0.1
        joint_pos[9:12] = gripper_action
        # middle finger
        joint_pos[12] = 0
        joint_pos[13:16] = gripper_action
        # ring finger
        joint_pos[16] = -0.1
        joint_pos[17:21] = gripper_action
        # little finger
        joint_pos[21] = -0.2
        joint_pos[22:25] = gripper_action
        # thumb
        joint_pos[25] = 0.4
        joint_pos[26] = gripper_action
        joint_pos[27] = 0.0
        joint_pos[28:30] = gripper_action

        return joint_pos

    def make_move_and_grasp_joint_pos(self, wrist_pos, gripper_action=1):
        joint_pos = np.zeros(30)
        joint_pos[:3] = wrist_pos
        joint_pos[7]  = 0.1
        # forefinger
        joint_pos[8] = 0.1
        joint_pos[9:12] = gripper_action
        # middle finger
        joint_pos[12] = 0
        joint_pos[13:16] = gripper_action
        # ring finger
        joint_pos[16] = -0.1
        joint_pos[17:21] = gripper_action
        # little finger
        joint_pos[21] = -0.2
        joint_pos[22:25] = gripper_action
        # thumb
        joint_pos[25] = 0.4
        joint_pos[26] = gripper_action
        joint_pos[27] = 1
        joint_pos[28:30] = gripper_action

        return joint_pos

    def make_release_joint_pos(self, wrist_pos):
        joint_pos = np.zeros(30)
        joint_pos[:3] = wrist_pos
        joint_pos[7]  = -0.2
        # forefinger
        joint_pos[8] = 0.1
        joint_pos[9:12] = 0
        # middle finger
        joint_pos[12] = 0
        joint_pos[13:16] = 0
        # ring finger
        joint_pos[16] = -0.1
        joint_pos[17:21] = 0
        # little finger
        joint_pos[21] = -0.2
        joint_pos[22:25] = 0
        # thumb
        joint_pos[25] = 0
        joint_pos[26] = 1
        joint_pos[27] = 0
        joint_pos[28:30] = -0.1

        return joint_pos

    def make_poke_joint_pos(self, wrist_pos):
        joint_pos = np.zeros(30)
        joint_pos[:3] = wrist_pos
        joint_pos[7]  = 0.4
        # forefinger
        joint_pos[8] = 0.1
        joint_pos[9:12] = 0
        # joint_pos[9:12] = 1
        # middle finger
        joint_pos[12] = 0
        joint_pos[13:16] = 0
        # joint_pos[13:16] = 1
        # ring finger
        joint_pos[16] = -0.1
        joint_pos[17:21] = 0
        # joint_pos[17:21] = 1
        # little finger
        joint_pos[21] = -0.2
        joint_pos[22:25] = 0
        # joint_pos[22:25] = 1
        # thumb
        joint_pos[25] = 0
        joint_pos[26] = 0
        # joint_pos[26] = 1
        joint_pos[27] = 0
        joint_pos[28:30] = 0
        # joint_pos[28:30] = 1

        return joint_pos

    # [Debug]
    def move_to(self, location, euler=None, gripper_action=0, max_steps=50, speed=10., action_type='none', use_wpt=True, end_on_reached=None, force_euler=True):
        rewards = []
        # self.env.set_site_pos(location)
        # if euler is None:
        #     tcp_pose = self.env.get_gripper_pose()
        #     rot = R.from_matrix(tcp_pose.to_transformation_matrix()[:3, :3])
        #     euler = rot.as_euler('XYZ')

        # [TODO]: fix step_gripper_to first
        gripper_position = self.env.get_gripper_pose().p
        diff = location - gripper_position
        for step in range(max_steps):

            if use_wpt:
                location_ = gripper_position + diff * ((step + 1) / max_steps) * speed
                location_ = location if np.linalg.norm((location_ - gripper_position)) > np.linalg.norm(diff) else location_
                raw_obs, reward, done, info = self.step_gripper_to(
                    location_, euler=euler, speed=speed, gripper_action=gripper_action, action_type=action_type)
            else:
                raw_obs, reward, done, info = self.step_gripper_to(
                    location, euler=euler, speed=speed, gripper_action=gripper_action, action_type=action_type)
                
            rewards.append(reward)

            reach = np.linalg.norm(location - self.env.get_gripper_pose().p) < self.pos_tolerance

            end_on_reached = self.end_on_reached if end_on_reached is None else end_on_reached
            if (end_on_reached and reach):
                # print('======================================')
                # print('=========== End on reached ===========')
                # print('======================================')
                break
            if done:
                break
        
        # pad the rewards
        if (len(rewards) < max_steps) and self.pad_rewards:
            max_reward = max(rewards)
            rewards += [max_reward] * (max_steps - len(rewards))
        return raw_obs, rewards, done, info

    # def repeat_motion(self, translation, euler=None, gripper=None, max_steps=40, positive_z=False):
    #     rewards = []
    #     # euler = np.zeros(3) if euler is None else euler
    #     gripper = 0. if gripper is None else gripper
    #     # Divide into traj steps
    #     traj_steps = translation.reshape(self.traj_steps, -1)
    #     for m_pos in traj_steps:
    #         # m_pos[2] = (m_pos[2] + 1.) / 2.     # Scale z to be within [0, 1]
    #         if positive_z:
    #             m_pos[2] = np.clip(m_pos[2], 0, 1)
    #         if self.env.use_oracle_motion:
    #             # Translation between object and the goal
    #             obj_pos = self.env.get_object_pose().p
    #             goal_pos = self.env.get_goal_pose().p
    #             oracle_motion = (goal_pos - obj_pos) * 0.65
    #             m_pos = oracle_motion

    #         # m = np.concatenate([m_pos, euler, [gripper]])  # Set the gripper to be closed
    #         for _ in range(max_steps):
    #             raw_obs, reward, done, info = self.sim_step(m)
    #             rewards.append(reward)
    #             if done:
    #                 break
    #     # pad the rewards
    #     if len(rewards) < max_steps and self.pad_rewards:
    #         last_reward = rewards[-1]
    #         rewards += [last_reward] * (max_steps - len(rewards))
    #     return raw_obs, rewards, done, info

    def lift_gripper(self, max_steps=20, open_gripper=True):
        rewards = []
        gripper_action = 0. if open_gripper else 1.
        for i in range(max_steps):
            # if open_gripper and i <=4:
            #     up_motion = 0.
            # else:
            #     up_motion = 0.4
            gripper_pos = self.env.get_gripper_pose().p
            motion = self.make_release_joint_pos(gripper_pos)
            raw_obs, reward, done, info = self.sim_step(motion)
            rewards.append(reward)
            if done:
                break
        # pad the rewards
        if (len(rewards) < max_steps) and self.pad_rewards:
            last_reward = rewards[-1]
            rewards += [last_reward] * (max_steps - len(rewards))
        return raw_obs, rewards, done, info
    
    '''
    Low-level motor controls
    '''
    # [Debug]
    def step_gripper_to(self, pos, euler=None, speed=0.4, gripper_action=0., action_type='none'):
        delta_pos = pos - self.env.get_gripper_pose().p
        reached_pos = np.linalg.norm(delta_pos) < self.pos_tolerance
        # print(np.linalg.norm(delta_pos), reached_pos)
        # pos_control = delta_pos * speed # relative
        pos_control = pos # absolute

        # # Calculate the delta euler required
        # if euler is None:
        #     # euler_control = np.zeros(3)
        #     reached_euler = True
        #     euler = self.env.get_gripper_pose().q
        # else:
        #     # Correct
        #     euler_rot = R.from_euler('XYZ', euler)
        #     tcp_pose = self.env.get_gripper_pose()
        #     to_tcp_mat = tcp_pose.inv().to_transformation_matrix()
        #     to_tcp_rot = R.from_matrix(to_tcp_mat[:3, :3])
        #     delta_rot = to_tcp_rot * euler_rot
        #     euler_control = delta_rot.as_euler('XYZ')
        #     reached_euler = np.linalg.norm(euler_control) < self.rot_tolerance # 0.1 = ~5.7 deg

        #     # euler_control *= speed * 3.
        #     # if np.max(abs(euler_control)) > 1:
        #     #     euler_control /= np.max(abs(euler_control))
            
        #     # [Debug]
        #     euler_control = euler
        
        # # Cauculate the joint pos
        # if gripper_action > 0.5:
        #     joint_qp = np.ones(24)
        # elif gripper_action < -0.5:
        #     joint_qp = -np.ones(24)
        # else:
        #     qp = self.env.get_joint_pose()
        #     joint_qp = qp[6:]

        # control = np.concatenate([pos_control, euler_control, joint_qp])
        
        control = np.zeros(self.motion_dim)
        if action_type == 'grasp': # grasp
            control = self.make_move_and_grasp_joint_pos(pos_control, gripper_action)
        elif action_type == 'move':
            control = self.make_reach_joint_pos(pos_control, gripper_action)
        elif action_type == 'poke':
            control = self.make_poke_joint_pos(pos_control)
        else :
            control = self.make_reach_joint_pos(pos_control, gripper_action)
        
        raw_obs, reward, success, info = self.sim_step(control)
        # raw_obs, reward, success, info = self.sim_step(self.env.get_joint_pose())
        info.update({"reached_pos": reached_pos, "reached_euler": True})
        return raw_obs, reward, success, info

    # [Debug]
    def sim_step(self, action):
        # Assuming OSC Pose control
        # action_pos = action[:3]
        # action_euler = action[3:6]
        # action_wrist = action[6:8]
        # action_fingers = action[8:]

        # Convert the action pos to the gripper frame
        # tcp_pose = self.env.get_gripper_pose()
        # to_tcp_mat = tcp_pose.inv().to_transformation_matrix()
        # to_tcp_rot = R.from_matrix(to_tcp_mat[:3, :3])
        # pos_control = to_tcp_rot.apply(action_pos)
        # pos_control = action_pos

        # control = [action_pos, action_euler, action_wrist, action_fingers]
        # control = np.concatenate(control)

        # raw_obs, reward, done, info = self.env.sim_step(control)
        raw_obs, reward, done, info = self.env.sim_step(action)

        # img = self.env.render()
        # cv2.imshow('show_adroit', img)
        # cv2.waitKey(1)

        # # [TODO]
        # # Replace the reward with flow reward
        # if self.env.use_flow_reward:
        #     _, reward = self.env.evaluate_flow(raw_obs)
        # else:
        #     reward = self.env.reward_scale * reward
        
        if hasattr(self.env, 'record_cam_frame'):
            self.env.record_cam_frame(extra_info={"reward": reward})
        self.env.render_viewer()

        # Add success info
        info['success'] = done
        return raw_obs, reward, done, info

# Debug
@register_primitive("adroit-dummy", GroundingTypes.OBJECT_AND_BACKGROUND, motion_dim=3)
class Dummy(AdroitPrimitive):
    def execute(self, location, motion, **kwargs):
        all_rewards = []
        # translation, euler, gripper = motion[:3], motion[3:6], motion[6]

        current_location =  self.env.get_gripper_pose().p
        # motion[:3] = 0.2 * motion[:3] / np.linalg.norm(motion[:3]) if \
        #             np.linalg.norm(motion) > 0.2 else \
        #             motion[:3]
        # motion[:3] = 0.2 * motion[:3] / np.linalg.norm(motion[:3]) if \
        #             np.linalg.norm(motion) > 0.2 else \
        #             motion[:3]
        # translation = current_location + motion[:3] # Max 20 cm
        translation = current_location + motion[:3] * 0.2 # Max 20 cm
        
        # raw_obs, rewards, done, info = self.repeat_motion(
        #     translation, euler=euler, gripper=gripper, max_steps=10)
        raw_obs, rewards, done, info = self.move_to(
            translation, euler=None, speed=1., gripper_action=0, max_steps=40,
            action_type='reach', force_euler=True, end_on_reached=True)
        all_rewards.append(rewards)
        return raw_obs, all_rewards, done, info
    
    def is_valid(self, states: Dict) -> bool:
        return True
    
    def visualize(self, motion):
        return motion[..., :3]

# Debug
@register_primitive("adroit-poke", GroundingTypes.OBJECT_ONLY, motion_dim=3)
class Poke(AdroitPrimitive):
    def execute(self, location, motion, **kwargs):
        all_rewards = []

        # z_rot = motion[-1] * np.pi
        # if self.use_oracle_rotation:
        #     z_rot = self.env.get_default_z_rot()    # TODO: remove this
        # euler = np.array([np.pi, 0, z_rot])

        # First move to the precontact location
        location = location + np.array([0.035, -0.15, -0.05])
        precontact = location + np.array([0, -0.05, 0])
        precontact = np.clip(precontact, self.pos_low_limits, self.pos_high_limits) 
        raw_obs, rewards, done, info = self.move_to(
            precontact, euler=None, speed=1., gripper_action=0, max_steps=20,
            action_type='poke', use_wpt=False, force_euler=True, end_on_reached=True)
        all_rewards.append(rewards)
        
        # # Then poke
        if not done:
            direction = motion[:3] / np.linalg.norm(motion[:3] + 1e-7)
            action = location + direction * 0.15
            raw_obs, rewards, done, info = self.move_to(
                action, euler=None, speed=1., gripper_action=0, max_steps=20,
                action_type='poke', force_euler=True, end_on_reached=True)
            all_rewards.append(rewards)
        
        # if not done:
        #     end_contact = location + motion[:3] * 0.15   # max 15 cms
        #     end_contact = np.clip(end_contact, self.pos_low_limits, self.pos_high_limits) 

        #     raw_obs, rewards, done, info = self.move_to(
        #         end_contact, euler=None, speed=1.5, gripper_action=0, max_steps=20,
        #         action_type='poke', force_euler=True, end_on_reached=True)
        #     all_rewards.append(rewards)
        
        # # Then lifting the gripper to not block the view
        # if not done:
        #     raw_obs, rewards, done, info = self.lift_gripper(open_gripper=False)
        
        return raw_obs, all_rewards, done, info
    
    def is_valid(self, states: Dict) -> bool:
        return not states['is_grasped']
    
    def visualize(self, motion):
        return motion[..., :3] * 0.15
    
# [Debug]
@register_primitive("adroit-pick_n_lift_fixed", GroundingTypes.OBJECT_ONLY, motion_dim=3)
class Pick(AdroitPrimitive):
    def execute(self, location, motion, **kwargs):
        # Grasp primitive
        all_rewards = []

        precontact_location = location + np.array([0, -0.04, 0.04])
        precontact_location = np.clip(precontact_location, self.pos_low_limits, self.pos_high_limits) 
        raw_obs, rewards, done, info = self.move_to(
            precontact_location, euler=None, gripper_action=0, speed=2,
            action_type='move', use_wpt=True, max_steps=90, force_euler=True, end_on_reached=True)

        if not done:
            grasp_location = location + np.array([0, -0.04, -0.05]) 
            grasp_location = np.clip(grasp_location, self.pos_low_limits, self.pos_high_limits) 
            raw_obs, rewards, done, info = self.move_to(
                grasp_location, euler=None, gripper_action=0, speed=2,
                action_type='move', use_wpt=False, max_steps=50, force_euler=True, end_on_reached=False)
            all_rewards.append(rewards)

        # Move down and grasp
        if not done:
            raw_obs, rewards, done, info = self.move_to(
                grasp_location, euler=None, gripper_action=1, speed=2,
                action_type='grasp', use_wpt=False, max_steps=20, force_euler=False)
            all_rewards.append(rewards)

        # if not done:
        #     motion = np.array([0, 0, 0.1])
        #     raw_obs, rewards, done, info = self.repeat_motion(motion, max_steps=2, positive_z=True)
        
        return raw_obs, all_rewards, done, info
    
    def is_valid(self, states: Dict) -> bool:
        return not states['is_grasped']
    
    def visualize(self, motion):
        motion_ = np.zeros_like(motion)[..., :3]
        motion_[..., 2] = 0.005
        return motion_

# [Debug]
@register_primitive("adroit-move", GroundingTypes.OBJECT_ONLY, motion_dim=3)
class Move(AdroitPrimitive):
    def execute(self, location, motion, **kwargs):
        all_rewards = []

        # [Question]: Why using gripper pose rather than object pose? 
        current_location =  self.env.get_gripper_pose().p
        # motion[:3] = 0.2 * motion[:3] / np.linalg.norm(motion[:3]) if \
        #             np.linalg.norm(motion[:3]) > 0.2 else \
        #             motion[:3]
        # target_location = current_location + motion[:3] # Max 20 cm
        target_location = current_location + motion[:3] * 0.2 # Max 20 cm
        target_location = np.clip(target_location, self.pos_low_limits, self.pos_high_limits) 

        raw_obs, rewards, done, info = self.move_to(
            target_location, euler=None, gripper_action=1, speed=1.5,
            action_type='grasp', use_wpt=True, max_steps=100, force_euler=False, end_on_reached=True)

        all_rewards.append(rewards)
        return raw_obs, all_rewards, done, info

    def is_valid(self, states: Dict) -> bool:
        return states['is_grasped']
    
    def visualize(self, motion):
        return motion[..., :3] * 0.3
    
# [Debug]
@register_primitive("adroit-place", GroundingTypes.OBJECT_ONLY, motion_dim=3)
class Move(AdroitPrimitive):
    def execute(self, location, motion, **kwargs):
        all_rewards = []

        # z_rot = motion[-1] * np.pi
        # if self.use_oracle_rotation:
        #     z_rot = self.env.get_default_z_rot()
        # euler = np.array([np.pi, 0, z_rot])

        target_location = location + motion[:3] * self.env.get_object_dim() 
        target_location = np.clip(location, self.pos_low_limits, self.pos_high_limits) 

        raw_obs, rewards, done, info = self.move_to(
            target_location, euler=None, gripper_action=1, speed=0.5,
            action_type='grasp', use_wpt=True, max_steps=100, force_euler=False, end_on_reached=True)
        # raw_obs, rewards, done, info = self.move_to(
        #     target_location, euler=euler, speed=5., gripper_action=1, max_steps=30)
        all_rewards.append(rewards)
        return raw_obs, all_rewards, done, info

    def is_valid(self, states: Dict) -> bool:
        return states['is_grasped']
    
    def visualize(self, motion):
        return motion[..., :3] * 0.3
    
# Debug
@register_primitive("adroit-open_gripper", GroundingTypes.BACKGROUND_ONLY, motion_dim=1)
class OpenGripper(AdroitPrimitive):
    def execute(self, location, motion, **kwargs):
        all_rewards = []
        raw_obs, rewards, done, info = self.lift_gripper(max_steps=20,open_gripper=True)
        all_rewards.append(rewards)
        return raw_obs, all_rewards, done, info
    
    def is_valid(self, states: Dict) -> bool:
        return states['is_grasped']
    
    def visualize(self, motion):
        return np.zeros_like(motion)[..., :3]   # Since the motion dim will always be larger than 3 due to the existence of other primitives

# # [Debug]
# @register_primitive("adroit-place", GroundingTypes.BACKGROUND_ONLY, motion_dim=5)
# class Place(AdroitPrimitive):
#     def execute(self, location, motion, **kwargs):
#         all_rewards = []
#         # Move to the target location
#         offset = motion[:3] * self.env.get_object_dim()
#         target_pos = location + offset
#         # z_rot = motion[-1] * np.pi
#         # if self.use_oracle_rotation:
#         #     z_rot = self.env.get_default_z_rot()    # TODO: remove this
#         # euler = np.array([np.pi, 0, z_rot])

#         # raw_obs, rewards, done, info = self.move_to(
#         #     target_pos, euler=euler, speed=3., gripper_action=1, 
#         #     max_steps=50, force_euler=True)

#         raw_obs, rewards, done, info = self.move_to(
#             target_pos, euler=None, gripper_action=1, speed=1.5,
#             action_type='reach', max_steps=100, force_euler=False, end_on_reached=True)
#         all_rewards.append(rewards)

#         # Interupt if the gripper collides with the object
#         collided = not info['reached_pos']
#         if collided and self.end_on_collision:
#             return raw_obs, all_rewards, done, info
        
#         return raw_obs, all_rewards, done, info
    
#     def is_valid(self, states: Dict) -> bool:
#         return states['is_grasped']

#     def visualize(self, motion):
#         return motion[..., :3] * self.env.get_object_dim()
    
# @register_primitive("adroit-pick_n_move", GroundingTypes.OBJECT_ONLY, motion_dim=5)
# class Pick_N_Move(Pick):
#     def execute(self, location, motion, **kwargs):
#         # [Move: 3, (Pick: 2)]
#         pick_motion = motion[3:]
#         raw_obs, all_rewards, done, info = super().execute(location, pick_motion, **kwargs)
        
#         # Post contact motion
#         if not done:
#             z_rot = motion[-1] * np.pi
#             if self.use_oracle_rotation:
#                 z_rot = self.env.get_default_z_rot()
#             euler = np.array([np.pi, 0, z_rot])

#             current_location =  self.env.get_gripper_pose().p
#             target_location = current_location + motion[:3] * 0.25   # max 25 cms
#             raw_obs, rewards, done, info = self.move_to(
#                 target_location, euler=euler, speed=5., gripper_action=1, max_steps=30)

#             all_rewards.append(rewards)
#         return raw_obs, all_rewards, done, info
    
    # def visualize(self, motion):
    #     return motion[..., :3] * 0.25

# @register_primitive("adroit-pick_n_drop", GroundingTypes.OBJECT_ONLY, motion_dim=6)
# class Pick_N_Drop(Pick_N_Move):
#     def execute(self, location, motion, **kwargs):
#         # [Move: 3, (Pick: 2)]
#         raw_obs, all_rewards, done, info = super().execute(location, motion, **kwargs)
#         # Open the gripper
#         release_gripper = motion[3] > 0
#         if release_gripper and not done:
#             raw_obs, rewards, done, info = self.lift_gripper()
#             all_rewards.append(rewards)
#         return raw_obs, all_rewards, done, info

# @register_primitive("adroit-pick_n_lift", GroundingTypes.OBJECT_ONLY, motion_dim=5)
# class Pick_N_Lift(Pick_N_Move):
#     def execute(self, location, motion, **kwargs):
#         # [Move: 3, (Pick: 2)]
#         # Change the motion to be in the z direction 
#         motion[:3] = motion[:3] * np.array([0, 0, 1])
#         raw_obs, all_rewards, done, info = super().execute(location, motion, **kwargs)
#         return raw_obs, all_rewards, done, info
    
#     def visualize(self, motion):
#         motion = motion[..., :3] * np.array([0, 0, 1])
#         return super().visualize(motion)

# @register_primitive("adroit-pick_n_lift_fixed", GroundingTypes.OBJECT_ONLY, motion_dim=5)
# class Pick_N_Lift_Fixed(Pick_N_Lift):
#     def execute(self, location, motion, **kwargs):
#         # [Move: 3, (Pick: 2)]
#         # Change the motion to be in the z direction 
#         motion[:3] = np.array([0, 0, 0.5])
#         raw_obs, all_rewards, done, info = super().execute(location, motion, **kwargs)
#         return raw_obs, all_rewards, done, info
    
#     def visualize(self, motion):
#         motion[..., :2] = 0
#         motion[..., 2] = 0.5
#         return super().visualize(motion)

# #
# @register_primitive("adroit-multi_step_move", GroundingTypes.OBJECT_ONLY, motion_dim=10)
# class Move(AdroitPrimitive):
#     def execute(self, location, motion, **kwargs):
#         all_rewards = []
#         num_steps = 2
#         current_location =  self.env.get_gripper_pose().p

#         # Calculate the waypoints
#         target_eulers, target_locations = [], []
#         for i in range(num_steps):
#             current_motion = motion[5*i:5*(i+1)]
#             z_rot = np.arctan2(current_motion[-1], current_motion[-2])
#             if self.use_oracle_rotation:
#                 z_rot = self.env.get_default_z_rot()
#             target_euler = np.array([np.pi, 0, z_rot])
#             target_eulers.append(target_euler)
#             target_location = current_location + current_motion[:3] * 0.2   # Max 20 cm
#             target_locations.append(target_location)

#         # Execute the motion
#         for i in range(num_steps):
#             raw_obs, rewards, done, info = self.move_to(
#                 target_locations[i], euler=target_eulers[i], speed=5., gripper_action=1, max_steps=30)
#             all_rewards.append(rewards)

#         return raw_obs, all_rewards, done, info

#     def is_valid(self, states: Dict) -> bool:
#         return states['is_grasped']
    
#     def visualize(self, motion):
#         return motion[..., -5:-2] * 0.2

# @register_primitive("adroit-place_w_gripper_action", GroundingTypes.BACKGROUND_ONLY, motion_dim=6)
# class PlaceWGripperAction(Place):
#     def execute(self, location, motion, **kwargs):
#         # Peform the normal place
#         raw_obs, all_rewards, done, info = super().execute(location, motion[1:], **kwargs)

#         # Perform the gripper motion
#         if not done:
#             if motion[0] > 0:
#                 raw_obs, rewards, done, info = self.lift_gripper(open_gripper=True)
#                 all_rewards.append(rewards)
        
#         return raw_obs, all_rewards, done, info

# @register_primitive("adroit-place_n_open_gripper", GroundingTypes.BACKGROUND_ONLY, motion_dim=5)
# class PlaceNOpenGripper(Place):
#     def execute(self, location, motion, **kwargs):
#         # Peform the normal place
#         raw_obs, all_rewards, done, info = super().execute(location, motion, **kwargs)

#         # Perform the gripper motion
#         if not done:
#             raw_obs, rewards, done, info = self.lift_gripper(open_gripper=True)
#             all_rewards.append(rewards)
        
#         return raw_obs, all_rewards, done, info

# @register_primitive("adroit-place_from_top", GroundingTypes.BACKGROUND_ONLY, motion_dim=5)
# class PlaceFromTop(Place):
#     def execute(self, location, motion, **kwargs):
#         all_rewards = []
#         # Move to the target location
#         # motion = np.array([0, 0, 1])
#         offset = motion[:3] * self.env.get_object_dim()
#         target_pos = location + offset
#         z_rot = motion[-1] * np.pi
#         euler = np.array([np.pi, 0, z_rot])

#         # Move to precontact position
#         precontact_pos = np.copy(target_pos)
#         precontact_pos[2] += 0.05
#         raw_obs, rewards, done, info = self.move_to(
#             precontact_pos, speed=7., gripper_action=1,
#             max_steps=20, euler=euler, force_euler=False)
#         all_rewards.append(rewards)

#         if not done:
#             raw_obs, all_rewards2, done, info = super().execute(location, motion, **kwargs)
#             all_rewards.extend(all_rewards2)

#         return raw_obs, all_rewards, done, info

# @register_primitive("adroit-place_n_insert", GroundingTypes.BACKGROUND_ONLY, motion_dim=3)
# class PlaceNInsert(Place):
#     def execute(self, location, motion, **kwargs):
#         # Place the object
#         raw_obs, all_rewards, done, info = super().execute(location, motion, **kwargs)
#         # Insert the object
#         euler = np.array([np.pi, 0, self.env.get_default_z_rot()])
#         location = self.env.get_object_pose().p + np.array([0.00, 0.0, 0.0])
#         if not done:
#             try:
#                 raw_obs, rewards, done, info = self.move_to(
#                     location, speed=3., gripper_action=1,
#                     max_steps=20, euler=euler, force_euler=True)
#                 # raw_obs, rewards, done, info = self.post_contact_motion(motion, max_steps=40)
#                 all_rewards.append(rewards)
#             except:
#                 pass
#         return raw_obs, all_rewards, done, info

# @register_primitive("adroit-place_n_insert_oracle", GroundingTypes.BACKGROUND_ONLY, motion_dim=3)
# class PlaceNInsertOracle(PlaceNInsert):
#     def execute(self, location, motion, **kwargs):
#         # location = self.env.get_goal_pose().p + np.array([0.00, -0.3, -0.005])
#         location = self.env.get_goal_pose().p + np.array([-0.06, 0.0, -0.005])
#         motion = 0.0 * motion
#         # Place the object
#         raw_obs, all_rewards, done, info = super().execute(location, motion, **kwargs)
#         return raw_obs, all_rewards, done, info