import numpy as np
from typing import Any, Callable, List, Optional, Sequence, Tuple, Type, Union, Dict
from scipy.spatial.transform import Rotation

import gym

from hacman.utils.primitive_utils import GroundingTypes, Primitive, register_primitive


class MSPrimitive(Primitive):    
    '''
    Utility primitives
    '''
    def move_to(self, location, euler=None, gripper_action=0, max_steps=50, speed=10., end_on_reached=None, force_euler=True):
        rewards = []
        self.env.set_site_pos(location)
        if euler is None:
            tcp_pose = self.env.get_gripper_pose()
            rot = Rotation.from_matrix(tcp_pose.to_transformation_matrix()[:3, :3])
            euler = rot.as_euler('XYZ')

        for _ in range(max_steps):
            raw_obs, reward, done, info = self.step_gripper_to(
                location, euler=euler, speed=speed, gripper_action=gripper_action)
            rewards.append(reward)
            end_on_reached = self.end_on_reached if end_on_reached is None else end_on_reached
            if (info['reached_pos'] and (info['reached_euler'] or not force_euler) and end_on_reached):
                break
            if done:
                break
        # pad the rewards
        if (len(rewards) < max_steps) and self.pad_rewards:
            max_reward = max(rewards)
            rewards += [max_reward] * (max_steps - len(rewards))
        return raw_obs, rewards, done, info

    def repeat_motion(self, translation, euler=None, gripper=None, max_steps=40, positive_z=False):
        rewards = []
        euler = np.zeros(3) if euler is None else euler
        gripper = 0. if gripper is None else gripper
        # Divide into traj steps
        traj_steps = translation.reshape(self.traj_steps, -1)
        for m_pos in traj_steps:
            if positive_z:
                m_pos[2] = np.clip(m_pos[2], 0, 1)
            if self.env.use_oracle_motion:
                # Translation between object and the goal
                obj_pos = self.env.get_object_pose().p
                goal_pos = self.env.get_goal_pose().p
                oracle_motion = (goal_pos - obj_pos) * 0.65
                m_pos = oracle_motion

            m = np.concatenate([m_pos, euler, [gripper]])  # Set the gripper to be closed
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

    def lift_gripper(self, max_steps=10, open_gripper=True):
        rewards = []
        gripper_action = 1. if open_gripper else -1.
        for i in range(max_steps):
            if open_gripper and i <=4:
                up_motion = 0.
            else:
                up_motion = 0.4
            raw_obs, reward, done, info = self.sim_step(np.array([0, 0, up_motion, 0, 0, 0, gripper_action]))
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
    def step_gripper_to(self, pos, euler=None, speed=3., gripper_action=0.):
        delta_pos = pos - self.env.get_gripper_pose().p
        reached_pos = np.linalg.norm(delta_pos) < self.pos_tolerance
        pos_control = delta_pos * speed

        # Calculate the delta euler required
        if euler is None:
            euler_control = np.zeros(3)
            reached_euler = True
        else:
            # Correct
            euler_rot = Rotation.from_euler('XYZ', euler)
            tcp_pose = self.env.get_gripper_pose()
            to_tcp_mat = tcp_pose.inv().to_transformation_matrix()
            to_tcp_rot = Rotation.from_matrix(to_tcp_mat[:3, :3])
            delta_rot = to_tcp_rot * euler_rot
            euler_control = delta_rot.as_euler('XYZ')
            reached_euler = np.linalg.norm(euler_control) < self.rot_tolerance # 0.1 = ~5.7 deg

            euler_control *= speed * 3.
            if np.max(abs(euler_control)) > 1:
                euler_control /= np.max(abs(euler_control))

        control = np.concatenate([pos_control, euler_control, [gripper_action]])
        raw_obs, reward, success, info = self.sim_step(control)
        info.update({"reached_pos": reached_pos, "reached_euler": reached_euler})
        return raw_obs, reward, success, info

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
        self.env.record_cam_frame(extra_info={"reward": reward})
        return raw_obs, reward, done, info

@register_primitive("dummy", GroundingTypes.OBJECT_AND_BACKGROUND, motion_dim=7)
class Dummy(MSPrimitive):
    def execute(self, location, motion, **kwargs):
        all_rewards = []
        translation, euler, gripper = motion[:3], motion[3:6], motion[6]
        raw_obs, rewards, done, info = self.repeat_motion(
            translation, euler=euler, gripper=gripper, max_steps=10)
        all_rewards.append(rewards)
        return raw_obs, all_rewards, done, info
    
    def is_valid(self, states: Dict) -> bool:
        return True
    
    def visualize(self, motion):
        return motion[..., :3]

@register_primitive("poke", GroundingTypes.OBJECT_ONLY, motion_dim=5)
class Poke(MSPrimitive):
    def execute(self, location, motion, **kwargs):
        all_rewards = []

        z_rot = np.arctan2(motion[-1], motion[-2])
        if self.use_oracle_rotation:
            z_rot = self.env.get_default_z_rot()    # TODO: remove this
        euler = np.array([np.pi, 0, z_rot])

        # First move to the precontact location
        direction = motion[:3] / np.linalg.norm(motion[:3])
        precontact = location - direction * 0.03
        raw_obs, rewards, done, info = self.move_to(
            precontact, euler=euler, speed=7., gripper_action=-1., max_steps=30)
        all_rewards.append(rewards)
        
        # Then poke
        if not done:
            end_contact = location + motion[:3] * 0.15   # max 15 cms
            raw_obs, rewards, done, info = self.move_to(
                end_contact, euler=euler, speed=3., gripper_action=-1., max_steps=20)
            all_rewards.append(rewards)
        
        # # Then lifting the gripper to not block the view
        if not done:
            raw_obs, rewards, done, info = self.lift_gripper(open_gripper=False)
        
        return raw_obs, all_rewards, done, info
    
    def is_valid(self, states: Dict) -> bool:
        return not states['is_grasped']
    
    def visualize(self, motion):
        return motion[..., :3] * 0.15
    
@register_primitive("pick", GroundingTypes.OBJECT_ONLY, motion_dim=5)
class Pick(MSPrimitive):
    def execute(self, location, motion, **kwargs):
        # Grasp primitive
        all_rewards = []
        
        # Move to the top of the object
        z_rot = np.arctan2(motion[-1], motion[-2])
        if self.use_oracle_rotation:
            z_rot = self.env.get_default_z_rot()    # TODO: remove this
        euler = np.array([np.pi, 0, z_rot])
        precontact_location = location + np.array([0, 0, 0.05]) 
        raw_obs, rewards, done, info = self.move_to(
            precontact_location, euler=euler, gripper_action=1., speed=10.,
            max_steps=30, force_euler=True)
        all_rewards.append(rewards)

        # Move down and grasp
        if not done:
            grasp_location = location + np.array([0, 0, -0.02])
            raw_obs, rewards, done, info = self.move_to(
                grasp_location, euler=None, speed=5., gripper_action=1.,
                max_steps=15, force_euler=False, end_on_reached=True)
            all_rewards.append(rewards)

        if not done:
            raw_obs, rewards, done, info = self.move_to(
                grasp_location, euler=None, speed=3., gripper_action=-1.,
                max_steps=25, force_euler=False, end_on_reached=False) # Grasp
            all_rewards.append(rewards)

        
        return raw_obs, all_rewards, done, info
    
    def is_valid(self, states: Dict) -> bool:
        return not states['is_grasped']
    
    def visualize(self, motion):
        motion_ = np.zeros_like(motion)[..., :3]
        motion_[..., 2] = 0.005
        return motion_
    
@register_primitive("pick_n_move", GroundingTypes.OBJECT_ONLY, motion_dim=5)
class Pick_N_Move(Pick):
    def execute(self, location, motion, **kwargs):
        # [Move: 3, (Pick: 2)]
        pick_motion = motion[3:]
        raw_obs, all_rewards, done, info = super().execute(location, pick_motion, **kwargs)
        
        # Post contact motion
        if not done:
            z_rot = np.arctan2(motion[-1], motion[-2])
            if self.use_oracle_rotation:
                z_rot = self.env.get_default_z_rot()
            euler = np.array([np.pi, 0, z_rot])

            current_location =  self.env.get_gripper_pose().p
            target_location = current_location + motion[:3] * 0.25   # max 25 cms
            raw_obs, rewards, done, info = self.move_to(
                target_location, euler=euler, speed=5., gripper_action=-1., max_steps=30)

            all_rewards.append(rewards)
        return raw_obs, all_rewards, done, info
    
    def visualize(self, motion):
        return motion[..., :3] * 0.25

@register_primitive("pick_n_drop", GroundingTypes.OBJECT_ONLY, motion_dim=6)
class Pick_N_Drop(Pick_N_Move):
    def execute(self, location, motion, **kwargs):
        # [Move: 3, (Pick: 2)]
        raw_obs, all_rewards, done, info = super().execute(location, motion, **kwargs)
        # Open the gripper
        release_gripper = motion[3] > 0
        if release_gripper and not done:
            raw_obs, rewards, done, info = self.lift_gripper()
            all_rewards.append(rewards)
        return raw_obs, all_rewards, done, info

@register_primitive("pick_n_lift", GroundingTypes.OBJECT_ONLY, motion_dim=5)
class Pick_N_Lift(Pick_N_Move):
    def execute(self, location, motion, **kwargs):
        # [Move: 3, (Pick: 2)]
        # Change the motion to be in the z direction 
        motion[:3] = motion[:3] * np.array([0, 0, 1])
        raw_obs, all_rewards, done, info = super().execute(location, motion, **kwargs)
        return raw_obs, all_rewards, done, info
    
    def visualize(self, motion):
        motion = motion[..., :3] * np.array([0, 0, 1])
        return super().visualize(motion)

@register_primitive("pick_n_lift_fixed", GroundingTypes.OBJECT_ONLY, motion_dim=5)
class Pick_N_Lift_Fixed(Pick_N_Lift):
    def execute(self, location, motion, **kwargs):
        # [Move: 3, (Pick: 2)]
        # Change the motion to be in the z direction 
        motion[:3] = np.array([0, 0, 0.5])
        raw_obs, all_rewards, done, info = super().execute(location, motion, **kwargs)
        return raw_obs, all_rewards, done, info
    
    def visualize(self, motion):
        motion[..., :2] = 0
        motion[..., 2] = 0.5
        return super().visualize(motion)

@register_primitive("move", GroundingTypes.OBJECT_ONLY, motion_dim=5)
class Move(MSPrimitive):
    def execute(self, location, motion, **kwargs):
        all_rewards = []

        z_rot = np.arctan2(motion[-1], motion[-2])
        if self.use_oracle_rotation:
            z_rot = self.env.get_default_z_rot()
        euler = np.array([np.pi, 0, z_rot])

        current_location =  self.env.get_gripper_pose().p
        target_location = current_location + motion[:3] * 0.2   # Max 20 cm

        raw_obs, rewards, done, info = self.move_to(
            target_location, euler=euler, speed=5., gripper_action=-1., max_steps=30)
        all_rewards.append(rewards)
        return raw_obs, all_rewards, done, info

    def is_valid(self, states: Dict) -> bool:
        return states['is_grasped']
    
    def visualize(self, motion):
        return motion[..., :3] * 0.3

@register_primitive("multi_step_move", GroundingTypes.OBJECT_ONLY, motion_dim=10)
class Move(MSPrimitive):
    def execute(self, location, motion, **kwargs):
        all_rewards = []
        num_steps = 2
        current_location =  self.env.get_gripper_pose().p

        # Calculate the waypoints
        target_eulers, target_locations = [], []
        for i in range(num_steps):
            current_motion = motion[5*i:5*(i+1)]
            z_rot = np.arctan2(current_motion[-1], current_motion[-2])
            if self.use_oracle_rotation:
                z_rot = self.env.get_default_z_rot()
            target_euler = np.array([np.pi, 0, z_rot])
            target_eulers.append(target_euler)
            target_location = current_location + current_motion[:3] * 0.2   # Max 20 cm
            target_locations.append(target_location)

        # Execute the motion
        for i in range(num_steps):
            raw_obs, rewards, done, info = self.move_to(
                target_locations[i], euler=target_eulers[i], speed=5., gripper_action=-1., max_steps=30)
            all_rewards.append(rewards)

        return raw_obs, all_rewards, done, info

    def is_valid(self, states: Dict) -> bool:
        return states['is_grasped']
    
    def visualize(self, motion):
        return motion[..., -5:-2] * 0.2

@register_primitive("place", GroundingTypes.BACKGROUND_ONLY, motion_dim=5)
class Place(MSPrimitive):
    def execute(self, location, motion, **kwargs):
        all_rewards = []
        # Move to the target location
        offset = motion[:3] * self.env.get_object_dim()
        target_pos = location + offset
        z_rot = np.arctan2(motion[-1], motion[-2])
        if self.use_oracle_rotation:
            z_rot = self.env.get_default_z_rot()    # TODO: remove this
        euler = np.array([np.pi, 0, z_rot])

        raw_obs, rewards, done, info = self.move_to(
            target_pos, euler=euler, speed=3., gripper_action=-1., 
            max_steps=50, force_euler=True)
        all_rewards.append(rewards)

        # Interupt if the gripper collides with the object
        collided = not info['reached_pos']
        if collided and self.end_on_collision:
            return raw_obs, all_rewards, done, info
        
        return raw_obs, all_rewards, done, info
    
    def is_valid(self, states: Dict) -> bool:
        return states['is_grasped']

    def visualize(self, motion):
        return motion[..., :3] * self.env.get_object_dim()

@register_primitive("open_gripper", GroundingTypes.BACKGROUND_ONLY, motion_dim=5)
class OpenGripper(MSPrimitive):
    def execute(self, location, motion, **kwargs):
        all_rewards = []
        raw_obs, rewards, done, info = self.lift_gripper(open_gripper=True)
        all_rewards.append(rewards)
        return raw_obs, all_rewards, done, info
    
    def is_valid(self, states: Dict) -> bool:
        return states['is_grasped']
    
    def visualize(self, motion):
        return np.zeros_like(motion)[..., :3]   # Since the motion dim will always be larger than 3 due to the existence of other primitives

@register_primitive("place_w_gripper_action", GroundingTypes.BACKGROUND_ONLY, motion_dim=6)
class PlaceWGripperAction(Place):
    def execute(self, location, motion, **kwargs):
        # Peform the normal place
        raw_obs, all_rewards, done, info = super().execute(location, motion[1:], **kwargs)

        # Perform the gripper motion
        if not done:
            if motion[0] > 0:
                raw_obs, rewards, done, info = self.lift_gripper(open_gripper=True)
                all_rewards.append(rewards)
        
        return raw_obs, all_rewards, done, info

@register_primitive("place_n_open_gripper", GroundingTypes.BACKGROUND_ONLY, motion_dim=5)
class PlaceNOpenGripper(Place):
    def execute(self, location, motion, **kwargs):
        # Peform the normal place
        raw_obs, all_rewards, done, info = super().execute(location, motion, **kwargs)

        # Perform the gripper motion
        if not done:
            raw_obs, rewards, done, info = self.lift_gripper(open_gripper=True)
            all_rewards.append(rewards)
        
        return raw_obs, all_rewards, done, info

@register_primitive("place_from_top", GroundingTypes.BACKGROUND_ONLY, motion_dim=5)
class PlaceFromTop(Place):
    def execute(self, location, motion, **kwargs):
        all_rewards = []
        # Move to the target location
        # motion = np.array([0, 0, 1])
        offset = motion[:3] * self.env.get_object_dim()
        target_pos = location + offset
        z_rot = np.arctan2(motion[-1], motion[-2])
        euler = np.array([np.pi, 0, z_rot])

        # Move to precontact position
        precontact_pos = np.copy(target_pos)
        precontact_pos[2] += 0.05
        raw_obs, rewards, done, info = self.move_to(
            precontact_pos, speed=7., gripper_action=-1.,
            max_steps=20, euler=euler, force_euler=False)
        all_rewards.append(rewards)

        if not done:
            raw_obs, all_rewards2, done, info = super().execute(location, motion, **kwargs)
            all_rewards.extend(all_rewards2)

        return raw_obs, all_rewards, done, info

@register_primitive("place_n_insert", GroundingTypes.BACKGROUND_ONLY, motion_dim=3)
class PlaceNInsert(Place):
    def execute(self, location, motion, **kwargs):
        # Place the object
        raw_obs, all_rewards, done, info = super().execute(location, motion, **kwargs)
        # Insert the object
        euler = np.array([np.pi, 0, self.env.get_default_z_rot()])
        location = self.env.get_object_pose().p + np.array([0.00, 0.0, 0.0])
        if not done:
            try:
                raw_obs, rewards, done, info = self.move_to(
                    location, speed=3., gripper_action=-1.,
                    max_steps=20, euler=euler, force_euler=True)
                # raw_obs, rewards, done, info = self.post_contact_motion(motion, max_steps=40)
                all_rewards.append(rewards)
            except:
                pass
        return raw_obs, all_rewards, done, info

@register_primitive("place_n_insert_oracle", GroundingTypes.BACKGROUND_ONLY, motion_dim=3)
class PlaceNInsertOracle(PlaceNInsert):
    def execute(self, location, motion, **kwargs):
        # location = self.env.get_goal_pose().p + np.array([0.00, -0.3, -0.005])
        location = self.env.get_goal_pose().p + np.array([-0.06, 0.0, -0.005])
        motion = 0.0 * motion
        # Place the object
        raw_obs, all_rewards, done, info = super().execute(location, motion, **kwargs)
        return raw_obs, all_rewards, done, info