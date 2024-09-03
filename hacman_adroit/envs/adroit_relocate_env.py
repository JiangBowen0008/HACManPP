import numpy as np
import os
import cv2
import gym
import matplotlib.pyplot as plt
import open3d as o3d

from scipy.spatial.transform import Rotation
from tqdm import tqdm
from PIL import Image
import imageio 

from mj_envs.hand_manipulation_suite.relocate_v0 import RelocateEnvV0
from hacman_adroit.envs.hacman_utility_base import HACManUtilityBase
from hacman.utils.transformations import transform_point_cloud, to_pose_mat

from sapien.core import Pose

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
        rot = R.from_matrix(matrix[:3, :3]).as_euler('XYZ')
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
        rot_m = R.from_euler('XYZ', pose[3:]).as_matrix()
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

class HACManAdroitRelocate(HACManUtilityBase, RelocateEnvV0, gym.Env):

    # 'front', 'bird', 'agent', 'side'
    def __init__(self, record_video=False, **kwargs):
        default_kwargs = dict(
            height=512, 
            width=512,
            has_renderer=False,
            render_camera="agent",
            has_offscreen_renderer=True,
            use_camera_obs=True,
            camera_names=['front', 'bird', 'side'],
            camera_depths=True,
            camera_segmentations="instance",
        )
        hacman_kwargs = dict()
        for k, v in kwargs.items():
            if k in default_kwargs:
                default_kwargs[k] = v
            else:
                hacman_kwargs[k] = v
        
        RelocateEnvV0.__init__(self, **default_kwargs)
        HACManUtilityBase.__init__(self, **hacman_kwargs)

        # [Note] add from BasicAdroitEnv
    
    # def reward(self, action):
    #     r = super().reward(action)
    #     if self.reward_shaping:
    #         r -= 4
    #     else:
    #         r -= 2
    #     return r
        
    # [Bowen]
    def reward(self, action=None):

        r = super().reward(action)
        # r -= 30

        return r
    
    # [Dubug]
    def get_segmentation_ids(self):
        object_ids = np.array([48])
        background_ids = np.array([0])  # The table is 0
        # background_ids = np.array([i for i in range(28)])  # The table is 0
        return {"object_ids": object_ids, "background_ids": background_ids}
    
    # [Debug]
    def get_goal_pose(self, format="pose"):
        obj_pos  = self.sim.data.site_xpos[self.sim.model.site_name2id("target")].ravel()
        # obj_quat = self.sim.data.site_xquat[self.sim.model.site_name2id("target")].ravel()
        obj_rot = self.sim.data.site_xmat[self.sim.model.site_name2id("target")].reshape(3, 3)
        obj_rot = Rotation.from_matrix(obj_rot)
        obj_quat = obj_rot.as_quat()
        obj_quat = wxyz2xyzw(obj_quat)
        if format == "mat":
            return to_pose_mat(obj_pos, obj_quat, input_wxyz=True)
        elif format == "vector":
            return np.concatenate([obj_pos, obj_quat])
        elif format == "pose":
            return Pose(obj_pos, obj_quat)

    # [Debug]
    def get_object_pose(self, format="pose"):
        obj_pos  = self.sim.data.body_xpos[self.sim.model.body_name2id("Object")].ravel()
        obj_quat = self.sim.data.body_xquat[self.sim.model.body_name2id("Object")].ravel()
        if format == "mat":
            return to_pose_mat(obj_pos, obj_quat, input_wxyz=True)
        elif format == "vector":
            return np.concatenate([obj_pos, obj_quat])
        elif format == "pose":
            return Pose(obj_pos, obj_quat)
    
    # [Debug] 
    def get_gripper_pose(self, format="pose"):
        gripper_pos  = self.sim.data.site_xpos[self.sim.model.site_name2id("S_grasp")].ravel()
        gripper_rot = self.sim.data.site_xmat[self.sim.model.site_name2id("S_grasp")].reshape(3, 3)
        gripper_rot = Rotation.from_matrix(gripper_rot)
        gripper_quat = gripper_rot.as_quat()
        gripper_quat = wxyz2xyzw(gripper_quat)
        if format == "mat":
            return to_pose_mat(gripper_pos, gripper_quat, input_wxyz=True)
        elif format == "vector":
            return np.concatenate([gripper_pos, gripper_quat])
        elif format == "pose":
            return Pose(gripper_pos, gripper_quat)
    
    # [Debug]
    def get_primitive_states(self):
        
        count = 0
        for i in range(self.sim.data.ncon):
            con = self.sim.data.contact[i]
            for j in self.hand_geom_names:
                if (con.geom1 == self.sim.model.geom_name2id(j) and con.geom2 == self.sim.model.geom_name2id("sphere")) or \
                    (con.geom1 == self.sim.model.geom_name2id("sphere") and con.geom2 == self.sim.model.geom_name2id(j)):
                    count += 1
        
        if count > 3: # Magic number
            return {"is_grasped": True}
        return {"is_grasped": False}
    
    # [Debug]
    def get_object_dim(self):
        return 0.035

    # [Debug]
    def get_default_z_rot(self):
        return 0
    
    # [TODO]
    def get_arena_bounds(self):
        return np.array([[-0.25, -0.3, 0], [0.25, 0.5, 0.3]])
    
    # ============================== #

    def reset(self, reconfigure=False, **kwargs):

        super().reset() # return images and joint set
        obs_all = self.get_camera_observation()
        # obs_all = self._get_observation()
        obs_all['pointcloud'] = self.get_point_cloud(obs_all)

        if obs_all["pointcloud"] is None:
            return self.reset()
        # print(obs_all['goal_pose'])

        return obs_all

    def get_observation(self):
        
        # obs_all = self._get_observation()
        # Visual entry
        target_pos = self.sim.data.site_xpos[self.sim.model.site_name2id("target")].ravel()
        target_rot = self.sim.data.site_xmat[self.sim.model.site_name2id("target")].reshape(3, 3)
        target_mat = np.identity(4)
        target_mat[:3, 3] = target_pos
        target_mat[:3, :3] = target_rot

        obj_pos = self.sim.data.body_xpos[self.sim.model.body_name2id("Object")].ravel()
        obj_rot = self.sim.data.body_xmat[self.sim.model.body_name2id("Object")].reshape(3, 3)
        obj_mat = np.identity(4)
        obj_mat[:3, 3] = obj_pos
        obj_mat[:3, :3] = obj_rot

        S_grasp_pos = self.sim.data.site_xpos[self.sim.model.site_name2id("S_grasp")].ravel()
        S_grasp_rot = self.sim.data.site_xmat[self.sim.model.site_name2id("S_grasp")].reshape(3, 3)
        S_grasp_mat = np.identity(4)
        S_grasp_mat[:3, 3] = S_grasp_pos
        S_grasp_mat[:3, :3] = S_grasp_rot

        obs_all = self.get_camera_observation()
        obs_all['pointcloud'] = self.get_point_cloud(obs_all)
        if obs_all["pointcloud"] is None:
            return self.reset()

        # [Find] get_obs
        obs_all['object_pose'] = obj_mat
        obs_all['goal_pose'] = target_mat
        obs_all['gripper_pose'] = S_grasp_mat
    
        return obs_all

def fig2image(fig):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    canvas = fig.canvas
    canvas.draw()  # Draw the canvas, cache the renderer

    image_flat = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')  # (H * W * 3,)
    # NOTE: reversed converts (W, H) from get_width_height to (H, W)
    image = image_flat.reshape(*reversed(canvas.get_width_height()), 3)  # (H, W, 3)

    return image

def make_reach_joint_pos(wrist_pos):
    joint_pos = np.zeros(30)
    joint_pos[:3] = wrist_pos
    joint_pos[7]  = 0
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
    joint_pos[26] = -0.2
    joint_pos[27] = 0
    joint_pos[28:30] = -0.2

    return joint_pos

def make_grasp(wrist_pos):
    joint_pos = np.zeros(30)
    joint_pos[:3] = wrist_pos
    joint_pos[7]  = 0.1
    # forefinger
    joint_pos[8] = 0.1
    joint_pos[9:12] = 1
    # middle finger
    joint_pos[12] = 0
    joint_pos[13:16] = 1
    # ring finger
    joint_pos[16] = -0.1
    joint_pos[17:21] = 1
    # little finger
    joint_pos[21] = -0.2
    joint_pos[22:25] = 1
    # thumb
    joint_pos[25] = 0.4
    joint_pos[26] = 1
    joint_pos[27] = 1
    joint_pos[28:30] = 1

    return joint_pos

def make_move_joint_pos(wrist_pos):
    joint_pos = np.zeros(30)
    joint_pos[:3] = wrist_pos
    joint_pos[7]  = -0.2
    # forefinger
    # forefinger
    joint_pos[8] = 0.1
    joint_pos[9:12] = 1
    # middle finger
    joint_pos[12] = 0
    joint_pos[13:16] = 1
    # ring finger
    joint_pos[16] = -0.1
    joint_pos[17:21] = 1
    # little finger
    joint_pos[21] = -0.2
    joint_pos[22:25] = 1
    # thumb
    joint_pos[25] = 0.4
    joint_pos[26] = 1
    joint_pos[27] = 1
    joint_pos[28:30] = 1

    return joint_pos

def make_release_joint_pos(wrist_pos):
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

def test_env():
    import time
    
    # camera_names = ['front', 'bird', 'agent', 'side']
    env = HACManAdroitRelocate(
        has_renderer=True,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        # render_camera="side",
        render_camera="agent",
        camera_names=['front', 'bird', 'agent', 'side'],
        camera_depths=False,
        camera_segmentations=None,
    )
    env.reset()
    env.render()
    print(env.get_segmentation_ids())
    print(env.get_primitive_states())
    print(env.get_goal_pose())
    print(env.get_object_pose())
    print(env.get_gripper_pose())
    print(env.get_object_dim())
    print(env.get_default_z_rot())

    gif_frames = []
    action_dim = 30
    test_list = [-0.25 + i*0.01 for i in range(50)]
    test_res = []
    first = True

    env.reset()
    # actions[7] = make_release_joint_pos(env.get_goal_pose().p)

    # for t in tqdm(test_list):
    #     xyz = [0, 0, t]
    #     actions = np.zeros(action_dim)
    #     actions[:3] = xyz

    #     for _ in range(100):

    #         obs, reward, done, info = env.step(actions)

    #         img = env.render()
    #         cv2.imshow('show_adroit', img)
    #         cv2.waitKey(1)
        
    #     obs = env.get_observation()
    #     print('gripper_pose: {}'.format(obs['gripper_pose'][:3,3]))
    #     test_res.append(obs['gripper_pose'][:3,3])

    # f = open('map_z.txt', 'w')
    # for (test, res) in zip(test_list, test_res):
    #     print(f'{np.round(test, 3)} => {res}')
    #     f.write(f'{np.round(test, 3)} => {res}\n')
    # f.close()
    s = []
    for index in range(100):

        env.reset()
        actions = np.zeros((9, action_dim))
        actions[0] = make_reach_joint_pos(env.get_object_pose().p + np.array([0, -0.04,  0.03]))
        # actions[0] = make_reach_joint_pos(env.get_object_pose().p + np.array([0, -0.04,  0.1]))
        actions[1] = make_reach_joint_pos(env.get_object_pose().p + np.array([0, -0.04, -0.03]))
        actions[2] = make_reach_joint_pos(env.get_object_pose().p + np.array([0, -0.04, -0.03]))
        actions[3] = make_grasp(env.get_object_pose().p - np.array([0, -0.04, -0.1]))
        actions[4] = make_move_joint_pos(env.get_goal_pose().p)
        actions[5] = make_move_joint_pos(env.get_goal_pose().p)
        actions[6] = make_move_joint_pos(env.get_goal_pose().p)
        actions[7] = make_move_joint_pos(env.get_goal_pose().p)
        actions[8] = make_move_joint_pos(env.get_goal_pose().p)
        
        imgs = []
        t_list = []
        for action in actions:

            bs, reward, done, info  = None, None, None, None

            env.get_observation()

            for _ in range(100):

                start_time = time.time()
                obs, reward, done, info = env.step(np.copy(action))
                t_list.append(time.time() - start_time)
                
                im = env.render()
                cv2.imshow('show_adroit', im)
                cv2.waitKey(1)

        print(env.reward())

            # img = env.render()
            # cv2.imshow('show_adroit', img)
            # cv2.waitKey(0)
        
        reward = env.reward()
        s.append(1 if info['is_successed'] else 0)  
        print(reward, done, info)
        print("FPS: ", 1 / np.mean(t_list))
    
    print(f'Success rate: {np.mean(s)}')    
            
        # imageio.mimsave(f'test_exec_{index}.gif', imgs, duration=100, loop=0)
        # cv2.destroyAllWindows()
        # env.close()
        # end_time = time.time()
        # print("Time taken: ", end_time - start_time)

    # f = open('map_z.txt', 'w')
    # for (test, res) in zip(test_list, test_res):
    #     print(f'{np.round(test, 3)} => {res}')
    #     f.write(f'{np.round(test, 3)} => {res}\n')
    # f.close()

    # imageio.mimsave(f'test_pcd.gif', gif_frames, duration=100, loop=0)
    # imageio.mimsave('test_exec_after.gif', imgs, duration=100, loop=0)
    # cv2.destroyAllWindows()
    # env.close()
    # end_time = time.time()
    # print("Time taken: ", end_time - start_time)

def test_primitive():
    from hacman.utils.primitive_utils import GroundingTypes, Primitive, get_primitive_class, MAX_PRIMITIVES
    import hacman_adroit.env_wrappers.primitives

    # [TODO] What is the relationship between has_offscreen_renderer and render_camera?
    env = HACManAdroitRelocate(
        has_renderer=True,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        render_camera="agent",
    )

    import time
    start_time = time.time()

    success = []
    for i in range(100):
        
        obs = env.reset()

        # print('==============================================================')
        # print('============================ poke ============================')
        # print('==============================================================')
        # primitive_name = 'adroit-poke'
        # prim = get_primitive_class(primitive_name)(env, end_on_reached=False)
        # obs = env.get_observation()
        # object_pos = obs['object_pose'][:3, 3]
        # motion = np.array([0, 1.0, 0])
        # obs, all_rewards, done, info = prim.execute(object_pos, motion)

        # print('====================================================================')
        # print('============================ move dummy ============================')
        # print('====================================================================')
        # primitive_name = 'adroit-dummy'
        # prim = get_primitive_class(primitive_name)(env, end_on_reached=True)
        # obs = env.get_observation()
        # gripper_pos = obs['gripper_pose'][:3, 3]
        # object_pos = obs['object_pose'][:3, 3] + np.array([0, 0, 0.1])
        # delta = (object_pos - gripper_pos) / np.linalg.norm(object_pos - gripper_pos) 
        # obs, all_rewards, done, info = prim.execute(None, delta)

        # obs = env.get_observation()
        # gripper_pos = obs['gripper_pose'][:3, 3]
        # object_pos = obs['object_pose'][:3, 3] + np.array([0, 0, 0.1])
        # delta = (object_pos - gripper_pos) / np.linalg.norm(object_pos - gripper_pos) 
        # obs, all_rewards, done, info = prim.execute(None, delta)
        
        # obs = env.get_observation()
        # plt.imshow(obs['front_segmentation'][-1::-1])
        # plt.show()
        
        # pcd = obs['pointcloud'][:,:3]
        # o3d_pcd = o3d.geometry.PointCloud()
        # o3d_pcd.points = o3d.utility.Vector3dVector(pcd)
        # o3d.visualization.draw_geometries([o3d_pcd])

        # grasp
        print('===============================================================')
        print('============================ grasp ============================')
        print('===============================================================')
        primitive_name = 'adroit-pick_n_lift_fixed'
        prim = get_primitive_class(primitive_name)(env, end_on_reached=False)
        obs = env.get_observation()
        object_pos = obs['object_pose'][:3, 3]
        obs, all_rewards, done, info = prim.execute(object_pos, None)

        # move delta
        print('====================================================================')
        print('============================ move delta ============================')
        print('====================================================================')
        primitive_name = 'adroit-move'
        prim = get_primitive_class(primitive_name)(env, end_on_reached=True)
        obs = env.get_observation()
        object_pos = obs['object_pose'][:3, 3]
        goal_pos = obs['goal_pose'][:3, 3]
        # delta = (goal_pos - object_pos) / np.linalg.norm(goal_pos - object_pos) * 0.6
        delta = (goal_pos - object_pos)
        obs, all_rewards, done, info = prim.execute(None, delta)

        # print('==============================================================')
        # print('============================ move ============================')
        # print('==============================================================')
        # primitive_name = 'adroit-move'
        # prim = get_primitive_class(primitive_name)(env, end_on_reached=True)
        # obs = env.get_observation()
        # motion = np.random.randn(3)
        # location = np.random.randn(3)
        # location[2] = 0
        # # print(f'delta: {delta}')
        # obs, all_rewards, done, info = prim.execute(location, motion)

        # print('=================================================================')
        # print('============================ open_gripper ============================')
        # print('=================================================================')
        # primitive_name = 'adroit-open_gripper'
        # prim = get_primitive_class(primitive_name)(env, end_on_reached=True)
        # obs, all_rewards, done, info = prim.execute(None, None)

        # # move delta
        # print('====================================================================')
        # print('============================ move dummy ============================')
        # print('====================================================================')
        # obs = env.get_observation()
        # gripper_pos = obs['gripper_pose'][:3, 3]
        # object_pos = obs['object_pose'][:3, 3]
        # delta = (object_pos - gripper_pos) / np.linalg.norm(object_pos - gripper_pos) 
        # obs, all_rewards, done, info = prim.execute(None, delta)

        print('reward: {}, success: {}'.format(all_rewards[-1][-1], 1 if info['success'] else 0))
        # print(info)
        success.append(1 if info['success'] else 0)
    
    print(f'Success rate: {np.mean(success)}')

    
    cv2.destroyAllWindows()
    end_time = time.time()

    print("Time taken: ", end_time - start_time)

if __name__ == "__main__":
    # test_env()
    test_primitive()


