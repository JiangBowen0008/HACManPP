import numpy as np
import copy
import torch
import time
from typing import Any, Callable, List, Optional, Sequence, Tuple, Type, Union, Dict
from scipy.spatial.transform import Rotation
import os
import pickle
import gym
import robosuite
robosuite.utils.macros.SIMULATION_TIMESTEP = 0.002
import traceback
from hacman.utils.primitive_utils import GroundingTypes, Primitive, register_primitive
from hacman_bin.utils.point_cloud_utils import convert_depth, get_point_cloud, add_additive_noise_to_xyz, dropout_random_ellipses
from robosuite.utils.control_utils import orientation_error
from hacman_bin.utils.transformations import to_pose_mat, decompose_pose_mat

VERBOSE=False

class BinPrimitive(Primitive):    
    '''
    Utility primitives
    '''
    def move_to(self, pos, quat=None, set_qpos=False, use_euler=False, euler = None, run_simulation_time=None,
                suppress_collision_warning=False, gripper_action=None,
                pos_p_control=None):
        # try:
        # Designed to move in free space
        self.env.set_marker(pos)    # Show the target location

        # Save original object pose
        obj_pos = np.array(self.env.sim.data.body_xpos[self.env.cube_body_id])
        obj_mat = np.array(self.env.sim.data.body_xmat[self.env.cube_body_id].reshape([3, 3]))

        # Visualize target eef pose
        target_eef_id = self.env.sim.model.body_name2id('target_eef')
        self.env.sim.model.body_pos[target_eef_id] = pos
        if quat is not None:
            self.env.sim.model.body_quat[target_eef_id] = quat
        self.env.sim.forward()
        
        site_id = self.env.sim.model.site_name2id('target_eef:grip_site')
        ee_pos_target = np.array(self.env.sim.data.site_xpos[site_id])
        ee_ori_mat_target = np.array(self.env.sim.data.site_xmat[site_id].reshape([3, 3]))
        
        # Calculate the target rotation
        target_rot = Rotation.from_euler('xyz', np.array([0, np.pi, 0]))

    
        if quat is not None:

            target_rot = Rotation.from_quat(quat.copy()[[1,2,3,0]])
            target_euler = target_rot.as_euler('XYZ')
            self.env.env.env.target_euler = target_euler
            
        
        if set_qpos:
        
            assert self.env.robots[0].name == 'Virtual'

            # Set the orientation
            self.env.sim.data.set_joint_qpos('robot0_hinge_joint0', target_euler[0])
            self.env.sim.data.set_joint_qpos('robot0_hinge_joint1', target_euler[1])
            self.env.sim.data.set_joint_qpos('robot0_hinge_joint2', target_euler[2] + np.pi/2)
        
            # Set the floating gripper to the location directly (ik)
            eef_body_id = self.env.sim.model.body_name2id('gripper0_eef')
            offset = -target_rot.apply(self.env.sim.model.body_pos[eef_body_id])
            qpos = ee_pos_target + offset
        
            self.env.sim.data.set_joint_qpos('robot0_slide_joint0', qpos[0])
            self.env.sim.data.set_joint_qpos('robot0_slide_joint1', qpos[1])
            self.env.sim.data.set_joint_qpos('robot0_slide_joint2', qpos[2])
        

            # Set the gripper finger to the desired action
            if gripper_action is not None:
                self.env.robots[0].grip_action(gripper=self.env.robots[0].gripper, gripper_action=[gripper_action])
        
            self.env.sim.forward()
            self.env.sim.step()
            self.env.robots[0].controller.update(force=True)
            self.env.maybe_render()
        else:
            if pos_p_control is not None:
                
                for step in range(15):
                    # Calculate the position control signal
                    pos_diff = ee_pos_target - self.env.robots[0].controller.ee_pos
                    reached_pos = np.linalg.norm(pos_diff) < self.pos_tolerance  #0.02
                    pos_control = pos_diff * pos_p_control

                    # Normalize the control signal to be within 1
                    if max_control:= np.max(np.abs(pos_control)) > 1:
                        pos_control /= max_control

                    # Calculate the euler control signal
                    reached_euler, euler_control = True, []

                    current_euler = np.zeros(3)
                    current_euler[0] = self.env.sim.data.get_joint_qpos('robot0_hinge_joint0')
                    current_euler[1] = self.env.sim.data.get_joint_qpos('robot0_hinge_joint1')
                    current_euler[2] = self.env.sim.data.get_joint_qpos('robot0_hinge_joint2') - np.pi/2
                    # print('current_euler', current_euler/np.pi*180)

                    # ## modify the current target_pos based on the current ee pose
      
                    euler_control = target_euler - current_euler
                    euler_control[1:] = - euler_control[1:]
                    
                    reached_euler = np.linalg.norm(euler_control) < self.rot_tolerance # 0.1 = ~5.7 deg

                    # Normalize the control signal to be within 1
                    euler_control *= pos_p_control / 2.
                    if max_control:= np.max(np.abs(euler_control)) > 1:
                        euler_control /= max_control
 
                    if reached_pos and reached_euler:
                        
                        break
                    
                    control_signal = np.concatenate([pos_control, euler_control])
                    self.env.run_simulation(action=control_signal, gripper_action=gripper_action)
            else:
                waypoints = self.generate_waypoints(self.env.robots[0].controller.ee_pos, ee_pos_target)
                for waypoint in waypoints:
                    self.env.run_simulation(ee_pos=waypoint, ee_ori_mat=ee_ori_mat_target,
                                    total_execution_time=run_simulation_time,
                                    gripper_action=gripper_action)
            
        # Check if the gripper reaches the pose
        reachable = True
        ee_pos_after = self.env.robots[0].controller.ee_pos
        ee_ori_mat_after = self.env.robots[0].controller.ee_ori_mat.reshape(3,3)
        pos_diff = np.linalg.norm(ee_pos_after - ee_pos_target)
        ori_diff = np.abs(orientation_error(ee_ori_mat_after, ee_ori_mat_target)/np.pi*180)
        if pos_diff > 0.02 or np.any(ori_diff > 5):
            if VERBOSE:
                print(f"fail: cannot reach the ee pose.\t Current pose: {ee_pos_after} \t"
                    f"Desired Pose: {ee_pos_target}\t Diff:{pos_diff}\t{ori_diff}")
            reachable = False

        # Check if the object has moved
        collision_free = True
        obj_pos_after = np.array(self.env.sim.data.body_xpos[self.env.cube_body_id])
        obj_mat_after = np.array(self.env.sim.data.body_xmat[self.env.cube_body_id].reshape(3,3))
        pos_diff = np.linalg.norm(obj_pos_after - obj_pos)
        ori_diff = np.abs(orientation_error(obj_mat_after, obj_mat)/np.pi*180)
        if pos_diff > 0.02 or np.any(ori_diff > 5):
            if VERBOSE and not suppress_collision_warning:
                print(f"fail: object moved during initialization.\t {pos_diff}\t{ori_diff}")
            collision_free = False

        self.env.hide_marker()
            # Return success: the location is reachable and the object didn't move



        return reachable, collision_free
    

    def move_to_from_top(self, location, quat, start_location=None, gripper_action=None):
        # First move to the start location
        
        s1, s2 = self.move_to(start_location, quat=quat, set_qpos=True, gripper_action=gripper_action)
       
        self.env.maybe_render()
        if not (s1 and s2):
            return False
        
        # Generating several waypoints to move the gripper from the top
        # Calls move_to to move between waypoints later
        site_id = self.env.sim.model.site_name2id('target_eef:grip_site')
        ee_pos = np.array(self.env.sim.data.site_xpos[site_id])
        ee_ori_mat = np.array(self.env.sim.data.site_xmat[site_id].reshape([3, 3]))
        ee_z = ee_pos[2]
        loc_z = location[2]

        # Create the waypoints
        success = np.isclose(ee_pos[:2], location[:2], atol=0.01).all() 


        # Move to each waypoint
        if success:
            reachable, collision_free = self.move_to(
                location, quat=quat, set_qpos=False, gripper_action=gripper_action,
                pos_p_control=5)
        else:
            Warning("Cannot move to top of the target location directly.")
        
        return success
    
    def move_to_contact(self, location, normal, quat, gripper_action=None):
        ee_pos = self.env.robots[0].controller.ee_pos
        precontact = location + normal * 0.02

        if self.env.robots[0].name == 'Virtual' and self.env.ik_precontact:
            # Set the floating gripper to the location directly (ik)
            preprecontact = location + normal * 0.04
            s1, s2 = self.move_to(
                pos=preprecontact, quat=quat, set_qpos=True, gripper_action=gripper_action)
            if not (s1 and s2):
                return False
            self.move_to(pos=precontact, quat=quat, gripper_action=gripper_action)
        else:
            # Gradually move to the contact point using the low-level controller
            midpoint = (precontact - ee_pos)*2/3 + ee_pos
            
            waypoints = 5
            for i in range(waypoints):
                new_pos = (midpoint - ee_pos)/waypoints*(i+1) + ee_pos
                s1, s2 = self.move_to(pos=new_pos, quat=quat, gripper_action=gripper_action)
                if not (s1 and s2):
                    return False
            
            waypoints = 5
            ee_pos = self.env.robots[0].controller.ee_pos
            for i in range(waypoints):
                new_pos = (precontact - ee_pos)/waypoints*(i+1) + ee_pos
                s1, s2 = self.move_to(pos=new_pos, quat=quat, gripper_action=gripper_action)
                if not (s1 and s2):
                    return False

        s1, s2 = self.move_to(pos=location, quat=quat, suppress_collision_warning=True, gripper_action=gripper_action) # it's ok if the object moves a little during contact
        return s1
    
    def generate_waypoints(self, start_location, end_location, max_distance=0.02):
        waypoints = []
        start_location = np.array(start_location)
        end_location = np.array(end_location)
        distance = np.linalg.norm(start_location - end_location)
        num_waypoints = int(distance / max_distance)
        for i in range(num_waypoints):
            new_location = (end_location - start_location)/num_waypoints*(i+1) + start_location
            waypoints.append(new_location)
        return waypoints
    
    def euler_to_gripper_quat(self, euler):
        base_rot = Rotation.from_euler('xyz', np.array([0, np.pi, 0]))
        rot = Rotation.from_euler('zyx', euler)
        quat = Rotation.as_quat(base_rot * rot)
        quat = quat[[3, 0, 1, 2]]
        return quat
    
    def quat_to_gripper_quat(self, quat, wxyz=True):
        base_rot = Rotation.from_euler('xyz', np.array([0, np.pi, 0]))
        if wxyz:
            rot = Rotation.from_quat(quat[[1, 2, 3, 0]])
        else:
            rot = Rotation.from_quat(quat)
        quat = Rotation.as_quat(base_rot * rot)
        quat = quat[[3, 0, 1, 2]]
        return quat

    def convert_motion_to_rotation(self, motion, rotation_type):
        
        ## four types of rotation
        ## 0 : use motion[-1], motion[-2] to calculate arctan2(motion[-1], motion[-2])
        ## 1: use motion[-2] and scale it from (-1,1) to (-pi/2, pi/2)
        ## 2: use motion[-2] and scale it from (-1, 1) to (0, pi)
        ## 3: use motion[-2] and scale it from (-1, 1) to (-pi, pi)
        ## 4: use motion[-2] and scale it from (-1, 1) to (0, 2pi)
        # Calculate the angle
        if rotation_type == 0:
            z_rot = np.arctan2(motion[-1], motion[-2])
        elif rotation_type == 1:
            z_rot = motion[-2] * np.pi / 2
        elif rotation_type == 2:
            z_rot = (motion[-2] + 1) * np.pi / 2
        elif rotation_type == 3:    
            z_rot = motion[-2] * np.pi
        elif rotation_type == 4:
            z_rot = (motion[-2] + 1) * np.pi
    
        quat = self.euler_to_gripper_quat([z_rot, 0, 0])    # euler in zyx
        return quat

    '''
    Low-level motor controls
    '''
    

@register_primitive("bin-poke", GroundingTypes.OBJECT_ONLY, motion_dim=5)
class Poke(BinPrimitive):
    def execute(self, location, motion, compute_return=True,**kwargs):
        action_repeat=3
        start_location=None
        '''
        Set normal to None to directly move to locations.
        '''
        # Save original object pose
        normal = kwargs.get('normal', None)
        rotation_type = kwargs.get('rotation_type', 0)
        quat = self.convert_motion_to_rotation(motion, rotation_type)

        
        try:
            # Contiguous movements
            if location is None:
                pass    # No additional movement for contiguous movements

            # Regress locations and actions
            elif normal is None:
                # assert self.collision_check(start_location)
                self.env.robots[0].reset(deterministic=True)
                self.env.sim.forward()
                start_location = np.copy(location)
                start_location[2] += 0.11

                # Approach from start location (should be top)
                assert self.move_to_from_top(location=location, quat=quat, start_location=start_location)

            # Per-point actions
            else:
                self.env.robots[0].reset(deterministic=True)
                self.env.sim.forward()

                noise = (np.random.rand(3)*2-1)*self.env.location_noise
                location += noise
                success = self.move_to_contact(location=location, quat=quat, normal=normal)
                assert success

        except:
            success = False

        else:
            # Execute action parameters
            success = True
            for _ in range(action_repeat):
                self.env.run_simulation(motion[:3])
    
        # Reset robot if using location
        if location is not None:
            self.env.robots[0].reset(deterministic=True)
            self.env.sim.forward()
            self.env.run_simulation(total_execution_time=self.env.resting_time)
        
        info = {"poke_success": success} 
        # Calculate the step outcome
        if compute_return:
            obs, reward, done, info = self.env.unwrapped.compute_step_return(info)
            all_rewards = [[reward,],]  # For consistency with other primitives
            return obs, all_rewards, done, info
        return 
    
    def is_valid(self, states: Dict) -> bool:
        return not (states['is_grasped'] or states['is_lifted'])
    
    def visualize(self, motion):
        return super().visualize(motion) * 0.1

class Pick(BinPrimitive):
    def execute(self, location, motion, **kwargs):
        self.env.robots[0].reset(deterministic=True)
        self.env.sim.forward()
        rotation_type = kwargs.get('rotation_type', 0)
        quat = self.convert_motion_to_rotation(motion, rotation_type) 

        # Evaluate the pcd before hands
        initial_reward = self.env.unwrapped.get_prev_reward()
        
        # Move to the top of the cube
        pregrasp_location = location + np.array([0, 0, 0.1])
    
        grasp_location = location + np.array([0, 0, -0.04])
        
        success = self.move_to_from_top(
            location=grasp_location, quat=quat, start_location=pregrasp_location, gripper_action=-1.)
        

        return success, initial_reward
    
    def is_valid(self, states: Dict) -> bool:
        return not (states['is_grasped'] or states['is_lifted'])


@register_primitive("bin-pick_n_lift", GroundingTypes.OBJECT_ONLY, motion_dim=5)
## motion definition: # [Move: 3, (Pick: 2)]
class Pick_N_Lift(Pick):

    def get_error_info(self, exception, location, motion):
        print(f'Exception at primitive execution pick_n_lift step: {exception}, {exception.__traceback__}')

        ## create a pickle file and dump all the debug information into the pickle file
        ## add error string
        error_str = traceback.format_exception(etype = type(exception), value = exception, tb = exception.__traceback__)
        print('error message', error_str)
        all_rewards = [[-1.]]
        raw_obs, reward, done, info = self.env.unwrapped.compute_step_return()
        return raw_obs, all_rewards, done, info
     
     
    def execute(self, location, motion, compute_return=True, reward_before_lift = False, **kwargs):
        # Reach the object
        initial_reward = -1
        try:
            success, initial_reward = super().execute(location, motion[3:], **kwargs)
            
            assert success

            # Grasp the cube
            action_repeat = 5
            for _ in range(action_repeat):
                self.env.run_simulation(gripper_action=1.)
        
            ## lift the object x + k
            start_location = self.env.get_gripper_pos() # location
            target_pos = start_location + np.array([0, 0, 0.15]) +motion[:3]*0.15  #np.array([0, 0, motion[0]*0.15])#
            success_1, success_2 = self.move_to(pos = target_pos, set_qpos=False, gripper_action = 1.0 )
            assert success_1
            self.env.run_simulation(gripper_action=1., total_execution_time=self.env.resting_time * 2)
            
            
        except Exception as e:
            ## add log error function here to catch the mujoco error that leads to the failure of the reset 
            raw_obs, all_rewards, done, info = self.get_error_info(exception=e, location = location, motion = motion)
            try:
                self.env.robots[0].reset(deterministic=True)
                self.env.sim.forward()
                self.env.run_simulation(total_execution_time=self.env.resting_time)
            except:
                if compute_return:
                    return raw_obs, all_rewards, done, info
                return

        if compute_return:
            final_obs, final_reward, done, info = self.env.unwrapped.compute_step_return()
            if reward_before_lift:
                return final_obs, [[initial_reward,],], done, info
            return final_obs, [[final_reward]], done, info
        return
    
    def visualize(self, motion):
        motion_ = motion[..., :3] * 0.15
        motion_[2] += 0.15
        return motion_

@register_primitive("bin-pick_n_lift_fixed", GroundingTypes.OBJECT_ONLY, motion_dim=5)
## motion definition: # [Move: 3, (Pick: 2)]
class Pick_N_Lift_Fixed(Pick_N_Lift):
    def execute(self, location, motion, compute_return=True, **kwargs):
        # Reach the object
        motion[:3] *= 0
        return super().execute(location, motion, compute_return=compute_return, **kwargs)
        
    def visualize(self, motion):
        motion = np.zeros_like(motion)[..., :3]
        return super().visualize(motion)


@register_primitive("bin-pick_n_lift_fixed_reward_before_lift", GroundingTypes.OBJECT_ONLY, motion_dim=5)
## motion definition: # [Move: 3, (Pick: 2)]
class Pick_N_Lift_Fixed_Reward_Before_Lift(Pick_N_Lift):
    def execute(self, location, motion, compute_return=True, **kwargs):
        # Reach the object
        motion[:3] *= 0
        return super().execute(location, motion, compute_return=compute_return, reward_before_lift= True, **kwargs)
        
    def visualize(self, motion):
        motion = np.zeros_like(motion)[..., :3]
        return super().visualize(motion)
    
@register_primitive("bin-pick_n_lift_z_only", GroundingTypes.OBJECT_ONLY, motion_dim=5)
## motion definition: # [Move: 3, (Pick: 2)]
class Pick_N_Lift_Z_only(Pick_N_Lift):
    def execute(self, location, motion, compute_return=True, **kwargs):
        # Reach the object
        motion[:2] *= 0
        return super().execute(location, motion, compute_return=compute_return, **kwargs)
        
    def visualize(self, motion):
        motion_ = motion[..., :3]
        motion_[..., :2] *= 0
        return super().visualize(motion_)

    
@register_primitive("bin-place", GroundingTypes.BACKGROUND_ONLY, motion_dim=5)
class Place(BinPrimitive):
    def execute(self, location, motion, compute_return = True, **kwargs):
        target_pos = location + motion[:3] * np.max(self.env.get_cube_size())

        # Calculate the angle
        rotation_type = kwargs.get('rotation_type', 0)
        quat = self.convert_motion_to_rotation(motion, rotation_type)

        # Move to the target location
        telerport = False
        if telerport:
            # Telerport
            ee_pos = self.env.unwrapped.get_gripper_pos()
            delta_pos = target_pos - ee_pos
            success = self.move_to(
                pos=target_pos, quat=quat, set_qpos=True, gripper_action=1., 
                suppress_collision_warning=True)
            
            obj_pos = self.env.get_object_pose().p
            target_obj_pos = obj_pos + delta_pos
            self.env.set_cube_pos(target_obj_pos)
        else:
            success = self.move_to(
                pos=target_pos, quat=quat, set_qpos=False, gripper_action=1., 
                suppress_collision_warning=True, pos_p_control=7)
          

        # Calculate the step outcome
        if compute_return:
            obs, reward, done, info = self.env.unwrapped.compute_step_return()
            all_rewards = [[reward,],]  # For consistency with other primitives
            return obs, all_rewards, done, info
        return 

    def is_valid(self, states: Dict) -> bool:
        return states['is_grasped'] or states['is_lifted']
    
    def visualize(self, motion):
        motion_ = np.copy(motion)[..., :3] * np.max(self.env.get_cube_size())
        return motion_


@register_primitive("bin-place_from_top", GroundingTypes.BACKGROUND_ONLY, motion_dim=5)
class PlaceFromTop(Place):
    def execute(self, location, motion, compute_return=True, **kwargs):
        target_pos = location + motion[:3] * np.max(self.env.get_cube_size())
        target_pos[2] = 0.2

        # Calculate the angle
        rotation_type = kwargs.get('rotation_type', 0)
        quat = self.convert_motion_to_rotation(motion, rotation_type)

        # Move to the target location
        success = self.move_to(
            pos=target_pos, quat=quat, set_qpos=False, gripper_action=1., 
            suppress_collision_warning=True, pos_p_control=7)
        if compute_return:
            obs, all_rewards, done, info = super().execute(location, motion, **kwargs)
            return obs, all_rewards, done, info
        return 

    def visualize(self, motion):
        return super().visualize(motion)

@register_primitive("bin-open_gripper", GroundingTypes.BACKGROUND_ONLY, motion_dim=1)
class OpenGripper(BinPrimitive):
    def lift_gripper(self, max_steps=10, open_gripper=True):
        rewards = []
        gripper_action = -1.
        for i in range(max_steps):
            if open_gripper and i <=4:
                up_motion = 0.
            else:
                up_motion = 0.4
            self.env.run_simulation(action=np.array([0., 0., up_motion])/4., gripper_action=gripper_action)
            

        # Calculate the step outcome
        raw_obs, reward, done, info = self.env.unwrapped.compute_step_return()
        rewards.append(reward)
        return raw_obs, rewards, done, info

    def execute(self, location, motion, **kwargs):
        all_rewards = []
        raw_obs, rewards, done, info = self.lift_gripper()
        all_rewards.append(rewards)
        return raw_obs, all_rewards, done, info

    def is_valid(self, states: Dict) -> bool:
        return states['is_grasped'] or states['is_lifted']
    
    def visualize(self, motion):
        n = motion.shape[:-1]
        new_shape = tuple(list(n)+ [3])
        return np.zeros(new_shape)
      

@register_primitive("bin-place_from_top_n_open_gripper", GroundingTypes.BACKGROUND_ONLY, motion_dim=5)
class PlaceFromTopOpenGripper(PlaceFromTop):
    def lift_gripper(self, max_steps=10, open_gripper=True):
        rewards = []
        gripper_action = -1. if open_gripper else 1.
        for i in range(max_steps):
            if open_gripper and i <=4:
                up_motion = 0.
            else:
                up_motion = 0.4
            self.env.run_simulation(action=np.array([0., 0., up_motion])/4., gripper_action=gripper_action)
          
        # Calculate the step outcome
        raw_obs, reward, done, info = self.env.unwrapped.compute_step_return()
        rewards.append(reward)
        return raw_obs, rewards, done, info

    def execute(self, location, motion, compute_return = True, **kwargs):
        super().execute(location, motion, compute_return=False, **kwargs)

         # Open the gripper
        for _ in range(5):
            self.env.run_simulation(gripper_action=-1.)

        ## open gripper & lift gripper

        final_obs, final_rewards, final_done, final_info = self.lift_gripper()

        return final_obs, [final_rewards], final_done, final_info

    def visualize(self, motion):
        return super().visualize(motion)

@register_primitive("bin-place_at_bottom", GroundingTypes.BACKGROUND_ONLY, motion_dim=4)
class PlaceAtBottom(PlaceFromTop):
    def execute(self, location, motion, compute_return=True, **kwargs):
        # Compute the object bottom location
        obj_pcd = self.env.prev_obs['object_pcd_points']
        min_z = torch.min(obj_pcd[:, 2])
        obj_botm_points = obj_pcd[obj_pcd[:, 2] < (min_z + 0.015)]   # 1.5 cm above the bottom
        obj_botm_min, _ = torch.min(obj_botm_points, dim=0)
        obj_botm_max, _ = torch.max(obj_botm_points, dim=0)
        obj_botm_location = (obj_botm_min + obj_botm_max) / 2
        obj_botm_location[2] = min_z
        obj_botm_location = obj_botm_location.cpu().numpy()

        # Compute the gripper-bottom offset
        eef_pos = self.env.get_gripper_pose().p
        offset = eef_pos - obj_botm_location

        # Apply xy offset
        xy_offset = np.array([*motion[:2], 0]) * 0.01   # 1cm is the voxel size 
        location += xy_offset

        # Compute the target location
        target_location = location + offset

        visualize = False
        if visualize:
            import open3d as o3d

            # Visualize position of the object bottom
            obj_o3d = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(obj_pcd.cpu().numpy()))
            obj_o3d.paint_uniform_color([1, 0, 0])
            obj_botm_o3d = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(obj_botm_points.cpu().numpy()))
            obj_botm_o3d.paint_uniform_color([0, 1, 0])
            bot_center_o3d = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(obj_botm_location.reshape(1, 3)))
            bot_center_o3d.paint_uniform_color([0, 0, 1])
            o3d.visualization.draw_geometries([obj_o3d, obj_botm_o3d, bot_center_o3d])

            # Visualize the place
            place_offset = location - obj_botm_location
            obj_o3d = obj_o3d.translate(place_offset)
            bot_center_o3d = bot_center_o3d.translate(place_offset)
            bg_pcd = self.env.prev_obs['background_pcd_points']
            bg_o3d = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(bg_pcd.cpu().numpy()))
            bg_o3d.paint_uniform_color([0, 1, 0])
            bg_location_o3d = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(location.reshape(1, 3)))
            bg_location_o3d.paint_uniform_color([0, 0.5, 1])
            o3d.visualization.draw_geometries([obj_o3d, bot_center_o3d, bg_o3d, bg_location_o3d])

        # Execute the motion
        motion = np.concatenate([np.zeros(3), motion])  # Only keep the rotation
        obs, rewards, done, info = super().execute(target_location, motion, compute_return=True, **kwargs)

        return obs, rewards, done, info
    
    def visualize(self, motion):
        motion_ = np.copy(motion)[..., :3]
        motion_[..., :2] *= 0.01
        motion_[..., 2] *= 0
        return super().visualize(motion)

@register_primitive("bin-move", GroundingTypes.OBJECT_ONLY, motion_dim=5)
class Move(BinPrimitive):
    def execute(self, location, motion, **kwargs):
        rewards = []

        # Calculate the angle
        rotation_type = kwargs.get('rotation_type', 0)
        quat = self.convert_motion_to_rotation(motion, rotation_type)


        current_location = self.env.robots[0].controller.ee_pos
        target_location = current_location + motion[:3] * 0.2   # Max 20 cm

        s1, s2 = self.move_to(
            pos = target_location, quat = quat, gripper_action=1.)

        # Calculate the step outcome
        raw_obs, reward, done, info = self.env.unwrapped.compute_step_return()
        rewards.append(reward)

        return raw_obs, [rewards], done, info

    def is_valid(self, states: Dict) -> bool:
        return states['is_grasped'] or states['is_lifted']
    
    def visualize(self, motion):
        return motion[..., :3] * 0.3 