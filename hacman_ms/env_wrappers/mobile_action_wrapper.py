import gym
import gym.spaces as spaces
from copy import deepcopy
from scipy.spatial.transform import Rotation
import numpy as np

from hacman.utils.transformations import transform_point_cloud, to_pose_mat, sample_idx, inv_pose_mat, decompose_pose_mat
from hacman.envs.sim_envs.base_env import BaseEnv, RandomLocation
from hacman.envs.env_wrappers.base_obs_wrapper import BaseObsWrapper

import mani_skill2.envs
from mani_skill2.sensors.camera import CameraConfig
# from mani_skill2.utils.registration import register_env
from mani_skill2.vector.vec_env import VecEnvObservationWrapper
from mani_skill2.utils.common import flatten_dict_space_keys, flatten_state_dict
from mani_skill2.utils.sapien_utils import look_at
from mani_skill2.sensors.camera import parse_camera_cfgs
from sapien.core import Pose



class HACManMobileActionWrapper(BaseObsWrapper):
    def __init__(self, env,
                 object_pcd_size=400, 
                 background_pcd_size=400,
                 record_video=False):
        super().__init__(env, object_pcd_size, background_pcd_size, record_video)
        self.action_space = gym.spaces.Box(-1, 1, (31,))

    def observation(self, obs):
        extra_obs = {}

        p, q = np.array([0, 0, 0]), np.array([1, 0, 0, 0])
        extra_obs['goal_pose'] = to_pose_mat(p, q, input_wxyz=True)  
        extra_obs['object_pose'] = to_pose_mat(p, q, input_wxyz=True)
        extra_obs['poke_idx'] = np.array([-1])

        seg_ids = self.get_seg_ids()
        extra_obs['object_ids'] = seg_ids['object_ids']
        extra_obs['background_ids'] = seg_ids['background_ids']

        obs['extra'].update(extra_obs)
        self.prev_obs = obs
        
        return obs
    
    def get_seg_ids(self):
        # Point cloud segementation ids
        actors = self.get_actors()
        actor_ids = {actor.name: actor.id for actor in actors}
        
        bg_ids = []
        for name in ['visual_ground', 'ground']:
            if name in actor_ids:
                bg_ids.append(actor_ids[name])
        bg_ids = np.array(bg_ids, dtype=np.int32)
        bg_ids = np.pad(bg_ids, (0, 5 - len(bg_ids)), 'constant', constant_values=-1)

        obj_ids = np.array([23, 25, 0, 0, 0])

        return {"object_ids": obj_ids, "background_ids": bg_ids}

    def step(self, action):
        # Take the environment step
        # Use the previous observation to get the location
        points = self.prev_obs['object_pcd_points']
        idx = self.prev_obs['poke_idx'][0]
        location = points[idx].numpy()

        self.contact_site.set_pose(Pose(location + np.array([0, 0, 0])))  # Add visualization

        # 1. Move to the front
        self._move_to_front(location)

        # 2. Grasp from the front
        z_angle = (np.pi / 2) * action[0]
        raw_obs, reward, done, info = self._grasp_from_front(location, z_angle=z_angle)

        # Given the robot-object contact location "location", the approach
        # parameters and the continuous paramter "motion", Execute the action
        # in simulation.
   
        action = np.copy(action)

        # Post contact motion
        motion_steps = action[1:].reshape(-1, 6)
        for motion in motion_steps:

            m_pos, m_euler = motion[:3], motion[3:]
            m_pos *= np.array([1., 1., 1.]) * 0.3
            control = np.concatenate([m_pos, m_euler, np.array([-1])])  # Set the gripper to be closed
            for _ in range(20):
                raw_obs, reward, done, info = self._sim_step(control)
        
        # Compute the step outcome
        obs = self.observation(raw_obs)

       

        step_info = {"is_success": info["success"],
                "action_param": action,
                "action_location": location,
                "cam_frames": deepcopy(self.cam_frames),}
        info.update(step_info)
        self.cam_frames.clear()

        done = True

        return obs, reward, done, info

    def move_to_front(self, location, distance=-0.05):
        location = np.copy(location)
        precontact_location = location[:2] + np.array([distance, 0])

        tcp_offset = np.array([-0.5824338, 0.92549074])

        new_base_location = precontact_location + tcp_offset
        self.agent.set_base_pose(new_base_location, -2.2029610e-01) #

    def grasp_from_front(self, location, z_angle=0):
        # Move to the top of the object
        precontact_location = location + np.array([-0.05, 0, 0])
        precontact_euler = np.array([0, np.pi/2, z_angle])
        for _ in range(85): # max 50 steps
            result = self.move_gripper_to(precontact_location, euler=precontact_euler, speed=1., gripper_action=1.)
            if self.gripper_reached_pos(precontact_location):
                break
        
        # Move down and grasp
        grasp_location = location + np.array([0.05, 0, 0])
        for _ in range(20):
            result = self.move_gripper_to(grasp_location, euler=precontact_euler, speed=1., gripper_action=1.)
        for _ in range(15):
            result = self.move_gripper_to(grasp_location, euler=precontact_euler, speed=1., gripper_action=-1.) # Grasp            
        
        return result
    
    def grasp_from_top(self, location):
        # Move to the top of the object
        precontact_location = location + np.array([0, 0, 0.05]) 
        for _ in range(50): # max 50 steps
            result = self.move_gripper_to(precontact_location, speed=10., gripper_action=1.)
            if self.gripper_reached_pos(precontact_location):
                break
        
        # Move down and grasp
        grasp_location = location + np.array([0, 0, -0.02])
        for _ in range(15):
            result = self.move_gripper_to(grasp_location, speed=5., gripper_action=1.)
        for _ in range(10):
            result = self.move_gripper_to(grasp_location, speed=3., gripper_action=-1.) # Grasp            
        
        return result

    def move_gripper_to(self, pos, euler=None, speed=3., gripper_action=0.):
        delta_pos = pos - self.tcp.pose.p
        pos_control = delta_pos * speed
        
        # Calculate the delta euler required
        if euler is None:
            euler_control = np.zeros(3)
        else:
            # Correct
            euler_rot = Rotation.from_euler('XYZ', euler)
            tcp_pose = self.tcp.pose
            to_tcp_mat = tcp_pose.inv().to_transformation_matrix()
            to_tcp_rot = Rotation.from_matrix(to_tcp_mat[:3, :3])
            delta_rot = to_tcp_rot * euler_rot
            euler_control = delta_rot.as_euler('XYZ')


        control = np.concatenate([pos_control, euler_control, [gripper_action]])
        
        raw_obs, reward, success, info = self.sim_step(control)
        return raw_obs, reward, success, info
    
    def gripper_reached_pos(self, pos):
        gripper_pos = self.tcp.pose.p
        return np.linalg.norm(gripper_pos - pos) < 0.01
    
    def sim_step(self, action):
        if self.control_mode.endswith("pd_ee_delta_pose"):
            action_pos = action[:3]
            action_euler = action[3:6]
            action_gripper = action[6]
        elif self.control_mode.endswith("pd_ee_delta_pos"):
            action_pos = action[:3]
            action_euler = np.zeros(3)
            action_gripper = action[6]
        else:
            raise NotImplementedError

        # Convert the action pos to the gripper frame
        tcp_pose = self.tcp.pose
        to_tcp_mat = tcp_pose.inv().to_transformation_matrix()
        to_tcp_rot = Rotation.from_matrix(to_tcp_mat[:3, :3])
        pos_control = to_tcp_rot.apply(action_pos)

        # base movements
        base_control = np.array([0, 0, 0, 0])

        control = [base_control, pos_control, action_euler, [action_gripper]]
        control = np.concatenate(control)

        output = self.step(control)
        self.record_cam_frame()
        return output

    
    

# Test the wrapper
if __name__ == "__main__":
    from mani_skill2.utils.wrappers import RecordEpisode
    from mani_skill2.vector import VecEnv, make as make_vec_env
    # env = gym.make(
    #         "PickCube-v0", obs_mode="pointcloud", 
    #         control_mode="pd_ee_delta_pose",
    #         # control_mode="pd_ee_target_delta_pose",
    #         # control_mode="pd_ee_target_delta_pos",
    #         # control_mode="pd_ee_delta_pos",
    #         camera_cfgs={"add_segmentation": True}
    # )
    wrappers = [HACManMobileActionWrapper, ]
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
            action = np.zeros((num_envs, 31))
            action[:, i] = 0.1
            print(action)
            # action[5] = -0.5
            obs, reward, done, info = env.step(action)
            print(obs.keys())
    env.close()
    # print(obs)