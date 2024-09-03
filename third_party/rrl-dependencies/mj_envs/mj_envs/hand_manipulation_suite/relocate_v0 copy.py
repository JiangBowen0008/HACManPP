import numpy as np
from gym import utils
from mjrl.envs import mujoco_env
from mujoco_py import MjViewer
import os

from gym.envs.registration import register
from mjrl.envs.mujoco_env import MujocoEnv

ADD_BONUS_REWARDS = True
USE_SPARSE_REWARDS = True

from scipy.spatial.transform import Rotation as R
import quaternion

def xyzw2wxyz(quat : np.ndarray) -> np.ndarray:
    assert len(quat) == 4, f'quaternion size must be 4, got {len(quat)}'
    return np.asarray([quat[3], quat[0], quat[1], quat[2]])

def wxyz2xyzw(quat : np.ndarray) -> np.ndarray:
    assert len(quat) == 4, f'quaternion size must be 4, got {len(quat)}'
    return np.asarray([quat[1], quat[2], quat[3], quat[0]])


def get_pose_from_matrix(matrix, pose_size : int = 7, adjust_rot_order : bool=False) -> np.ndarray:

    mat = np.array(matrix)
    assert mat.shape == (4, 4), f'pose must contain 4 x 4 elements, but got {mat.shape}'
    
    pos = matrix[:3, 3]
    rot = None

    if pose_size == 6:
        rot = R.from_matrix(matrix[:3, :3]).as_euler('XYZ')
    elif pose_size == 7:
        rot = R.from_matrix(matrix[:3, :3]).as_quat()
        if adjust_rot_order:
            rot = xyzw2wxyz(rot)
    elif pose_size == 9:
        rot = (matrix[:3, :2].T).reshape(-1)
            
    pose = list(pos) + list(rot)

    return np.array(pose)

def get_matrix_from_pose(pose, adjust_rot_order : bool=False) -> np.ndarray:
    assert len(pose) == 6 or len(pose) == 7 or len(pose) == 9, f'pose must contain 6 or 7 elements, but got {len(pose)}'
    pos_m = np.asarray(pose[:3])
    rot_m = np.identity(3)

    if len(pose) == 6:
        rot_m = R.from_euler('XYZ', pose[3:]).as_matrix()
    elif len(pose) == 7:
        if adjust_rot_order:
            quat = wxyz2xyzw(pose[3:])
        rot_m = R.from_quat(quat).as_matrix()
    elif len(pose) == 9:
        rot_xy = pose[3:].reshape(2, 3)
        rot_m = np.vstack((rot_xy, np.cross(rot_xy[0], rot_xy[1]))).T
            
    ret_m = np.identity(4)
    ret_m[:3, :3] = rot_m
    ret_m[:3, 3] = pos_m

    return ret_m

class RelocateEnvV0(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):

        # for adjusting wrist pose
        # wrist_world_pos   = [    0, -0.7, 0.2]
        # wrist_world_euler = [-1.57,    0, 3.14]
        # self.wrist_world_pose = wrist_world_pos + wrist_world_euler
        # self.wrist_world_trans = get_matrix_from_pose(self.wrist_world_pose)

        self.first_call = True

        self.target_obj_sid = 0
        self.S_grasp_sid = 0
        self.obj_bid = 0
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        mujoco_env.MujocoEnv.__init__(self, curr_dir+'/assets/DAPG_relocate.xml', 5)
        print(curr_dir+'/assets/DAPG_relocate.xml')
        # exit(0)
        
        # change actuator sensitivity
        self.sim.model.actuator_gainprm[self.sim.model.actuator_name2id('A_WRJ1'):self.sim.model.actuator_name2id('A_WRJ0')+1,:3] = np.array([10, 0, 0])
        self.sim.model.actuator_gainprm[self.sim.model.actuator_name2id('A_FFJ3'):self.sim.model.actuator_name2id('A_THJ0')+1,:3] = np.array([1, 0, 0])
        self.sim.model.actuator_biasprm[self.sim.model.actuator_name2id('A_WRJ1'):self.sim.model.actuator_name2id('A_WRJ0')+1,:3] = np.array([0, -10, 0])
        self.sim.model.actuator_biasprm[self.sim.model.actuator_name2id('A_FFJ3'):self.sim.model.actuator_name2id('A_THJ0')+1,:3] = np.array([0, -1, 0])

        # print(self.model.actuator_ctrlrange[:,0])
        # print(self.model.actuator_ctrlrange[:,1])
        # exit(0)

        self.target_obj_sid = self.sim.model.site_name2id("target")
        self.S_grasp_sid = self.sim.model.site_name2id('S_grasp')
        self.obj_bid = self.sim.model.body_name2id('Object')
        utils.EzPickle.__init__(self)
        self.act_mid = np.mean(self.model.actuator_ctrlrange, axis=1)
        self.act_rng = 0.5*(self.model.actuator_ctrlrange[:,1]-self.model.actuator_ctrlrange[:,0])
        self.action_space.high = np.ones_like(self.model.actuator_ctrlrange[:,1])
        self.action_space.low  = -1.0 * np.ones_like(self.model.actuator_ctrlrange[:,0])

        self.control_range_high = self.model.actuator_ctrlrange[:,1]
        self.control_range_low  = self.model.actuator_ctrlrange[:,0]
        self.control_range_mid = (self.model.actuator_ctrlrange[:,1] + self.model.actuator_ctrlrange[:,0]) / 2

    #     self.set_init_qpos()

    # [Debug] [State]
    def step(self, a):
        
        # in gym.make(), it will call this function ones by passing a all-zero "a" as action

        # # Debug: only joint angle are relative angles
        # a = np.clip(a, -1.0, 1.0)
        # try:
        #     a = self.act_mid + a*self.act_rng # mean center and scale
        # except:
        #     a = a                             # only for the initialization phase

        # # adjust pose
        # target_pose_wrist = get_matrix_from_pose(a[:6], adjust_rot_order=True)
        # target_trans_world = np.linalg.inv(self.wrist_world_trans) @ target_pose_wrist
        # target_pose_world = get_pose_from_matrix(target_trans_world, pose_size=6)
        # a[:6] = target_pose_world
        # print(f'action: {np.round(a[:6] * 100) / 100}')
        # exit(0)

        # cange the scale of the action
        # if not self.first_call:
        #     # print(self.control_range_mid[:6])
        #     print(self.control_range_high[:6])
        #     print(self.control_range_low[:6])
        #     exit(0)

        # convertion using regression method
        a_convert = np.zeros(3)
        a_convert[0] = -0.19 / 0.477 * a[0] + (0.1  + 0.246 * (-0.19 / 0.477))
        a_convert[1] =  0.06 / 0.175 * a[2] + (0.11 - 0.112 * (0.06 / 0.175))
        a_convert[2] =  0.3  / 0.764 * a[1] + (0.2  - 0.279 * (0.3 / 0.764))
        a_convert = np.clip(a_convert, [-0.09, 0.11, -0.1], [0.1, 0.17, 0.2])
        a[:3] = a_convert
        
        self.do_simulation(a, self.frame_skip)
        ob = self.get_obs()
        obj_pos  = self.data.body_xpos[self.obj_bid].ravel()
        gripper_pos = self.data.site_xpos[self.S_grasp_sid].ravel()
        target_pos = self.data.site_xpos[self.target_obj_sid].ravel()

        reward = -0.1*np.linalg.norm(gripper_pos-obj_pos)              # take hand to object
        if obj_pos[2] > 0.04:                                       # if object off the table
            reward += 1.0                                           # bonus for lifting the object
            reward += -0.5*np.linalg.norm(gripper_pos-target_pos)      # make hand go to target
            reward += -0.5*np.linalg.norm(obj_pos-target_pos)       # make object go to target

        if ADD_BONUS_REWARDS:
            if np.linalg.norm(obj_pos-target_pos) < 0.1:
                reward += 10.0                                          # bonus for object close to target
            if np.linalg.norm(obj_pos-target_pos) < 0.05:
                reward += 20.0                                          # bonus for object "very" close to target

        if USE_SPARSE_REWARDS:
            reward = 0
            if np.linalg.norm(obj_pos-target_pos) < 0.1:
                reward += 10.0 
            if np.linalg.norm(obj_pos-target_pos) < 0.05:
                reward += 20.0

        is_successed = True if np.linalg.norm(obj_pos-target_pos) < 0.05 else False
        # is_successed = True if np.linalg.norm(obj_pos-target_pos) < 0.1 else False

        # self.first_call = False
        # print(dict(is_successed=is_successed))

        return ob, reward, is_successed, dict(is_successed=is_successed)
    
    def get_joint_pos(self):
        qp = self.data.qpos.ravel()
        return qp[:-6]

    def get_obs(self):
        # qpos for hand
        # xpos for obj
        # xpos for target
        qp = self.data.qpos.ravel()
        # print(f'state: {np.round(qp[:3], 3)}')
        obj_pos  = self.data.body_xpos[self.obj_bid].ravel()
        palm_pos = self.data.site_xpos[self.S_grasp_sid].ravel()
        target_pos = self.data.site_xpos[self.target_obj_sid].ravel()
        return np.concatenate([qp[:-6], palm_pos-obj_pos, palm_pos-target_pos, obj_pos-target_pos])

    # [Note] [Reset] the base class "reset() will call this method"
    # will be called by reset() in mujoco_env.py
    def reset_model(self): 
        qp = self.init_qpos.copy()
        qv = self.init_qvel.copy()
        self.set_state(qp, qv)
        self.model.body_pos[self.obj_bid,0] = self.np_random.uniform(low=-0.15, high=0.15)
        self.model.body_pos[self.obj_bid,1] = self.np_random.uniform(low=-0.15, high=0.3)
        self.model.site_pos[self.target_obj_sid, :3] = [0.0, 0.0, 0.3]
        # self.model.site_pos[self.target_obj_sid, 0] = self.np_random.uniform(low=-0.2, high=0.2)
        # self.model.site_pos[self.target_obj_sid,1] = self.np_random.uniform(low=-0.2, high=0.2)
        # self.model.site_pos[self.target_obj_sid,2] = 0.25 
        self.sim.forward()
        return self.get_obs()

    def get_env_state(self):
        """
        Get state of hand as well as objects and targets in the scene
        """
        qp = self.data.qpos.ravel().copy()
        qv = self.data.qvel.ravel().copy()
        hand_qpos = qp[:30]
        obj_pos  = self.data.body_xpos[self.obj_bid].ravel()
        palm_pos = self.data.site_xpos[self.S_grasp_sid].ravel()
        target_pos = self.data.site_xpos[self.target_obj_sid].ravel()
        return dict(hand_qpos=hand_qpos, obj_pos=obj_pos, target_pos=target_pos, palm_pos=palm_pos,
            qpos=qp, qvel=qv)

    def set_env_state(self, state_dict):
        """
        Set the state which includes hand as well as objects and targets in the scene
        """
        qp = state_dict['qpos']
        qv = state_dict['qvel']
        obj_pos = state_dict['obj_pos']
        target_pos = state_dict['target_pos']
        self.set_state(qp, qv)
        self.model.body_pos[self.obj_bid] = obj_pos
        self.model.site_pos[self.target_obj_sid] = target_pos
        self.sim.forward()

    def mj_viewer_setup(self):
        self.viewer = MjViewer(self.sim)
        self.viewer.cam.azimuth = 90
        self.sim.forward()
        self.viewer.cam.distance = 1.5

    def evaluate_success(self, paths):
        num_success = 0
        num_paths = len(paths)
        # success if object close to target for 25 steps
        for path in paths:
            if np.sum(path['env_infos']['goal_achieved']) > 25:
                num_success += 1
        success_percentage = num_success*100.0/num_paths
        return success_percentage
    
# # Relcoate an object to the target
# register(
#     id='relocate-v0',
#     entry_point='mj_envs.hand_manipulation_suite:RelocateEnvV0',
#     max_episode_steps=200,
# )
