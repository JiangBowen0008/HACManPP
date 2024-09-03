import numpy as np
import gym
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt

from hacman_suite.envs.hacman_utility_base import HACManUtilityBase
from hacman.utils.transformations import transform_point_cloud, to_pose_mat

from robosuite.environments.manipulation.stack import Stack
from robosuite import load_controller_config
from sapien.core import Pose

class HACManSuiteStack(HACManUtilityBase, Stack, gym.Env):
    def __init__(self, record_video=False, **kwargs):
        default_kwargs = dict(
            has_renderer=False,
            render_camera="agentview",
            has_offscreen_renderer=True,
            use_camera_obs=False,
            camera_names=['agentview', 'birdview'],
            camera_depths=True,
            camera_segmentations="instance",
            controller_configs = load_controller_config(default_controller="OSC_POSE")
        )
        hacman_kwargs = dict()
        for k, v in kwargs.items():
            if k in default_kwargs:
                default_kwargs[k] = v
            else:
                hacman_kwargs[k] = v
        Stack.__init__(self, 
                       robots="Panda",
                       reward_shaping=True,
                       **default_kwargs)
        HACManUtilityBase.__init__(self, **hacman_kwargs)
        pass
    
    def reward(self, action):
        r = super().reward(action)
        if self.reward_shaping:
            r -= 4
        else:
            r -= 2
        return r
        
    def get_segmentation_ids(self):
        object_ids = np.array([1])
        background_ids = np.array([2])  # The table is 0
        return {"object_ids": object_ids, "background_ids": background_ids}
        print("get_segmentation_ids")
    
    def get_primitive_states(self):
        grasping_cubeA = self._check_grasp(gripper=self.robots[0].gripper, object_geoms=self.cubeA)
        return {"is_grasped": grasping_cubeA}
    
    def get_goal_pose(self, format="pose"):
        cubeA_pos = self.sim.data.body_xpos[self.cubeA_body_id]
        goal_pos = cubeA_pos + np.array([0, 0, 0.05])
        cubeA_quat = self.sim.data.body_xquat[self.cubeA_body_id]
        if format == "mat":
            return to_pose_mat(goal_pos, cubeA_quat, input_wxyz=True)
        elif format == "vector":
            return np.concatenate([goal_pos, cubeA_quat])
        elif format == "pose":
            return Pose(goal_pos, cubeA_quat)

    def get_object_pose(self, format="pose"):
        cubeA_pos = self.sim.data.body_xpos[self.cubeA_body_id]
        cubeA_quat = self.sim.data.body_xquat[self.cubeA_body_id]
        if format == "mat":
            return to_pose_mat(cubeA_pos, cubeA_quat, input_wxyz=True)
        elif format == "vector":
            return np.concatenate([cubeA_pos, cubeA_quat])
        elif format == "pose":
            return Pose(cubeA_pos, cubeA_quat)
    
    def get_gripper_pose(self, format="pose"):
        gripper_pos = self.sim.data.site_xpos[self.robots[0].eef_site_id]
        gripper_quat = self.sim.data.get_body_xquat(self.robots[0].robot_model.eef_name)    # wxyz
        if format == "mat":
            return to_pose_mat(gripper_pos, gripper_quat, input_wxyz=True)
        elif format == "vector":
            return np.concatenate([gripper_pos, gripper_quat])
        elif format == "pose":
            return Pose(gripper_pos, gripper_quat)
    
    def get_object_dim(self):
        return 0.025

    def get_default_z_rot(self):
        return 0
    
    def get_arena_bounds(self):
        return np.array([[-1, -1, 0], [1, 1, 1]])


def test_env():
    import time
    from scipy.spatial.transform import Rotation
    
    env = HACManSuiteStack(
        # has_renderer=True,
        # has_offscreen_renderer=False,
        use_camera_obs=False,
        render_camera="frontview",
        # camera_names=["agentview"],
        # camera_depths=False,
        # camera_segmentations=None,
    )
    env.reset()
    # env.render()
    # print(env.get_segmentation_ids())
    # print(env.get_primitive_states())
    # print(env.get_goal_pose())
    # print(env.get_object_pose())
    # print(env.get_gripper_pose())
    # print(env.get_object_dim())
    # print(env.get_default_z_rot())

    start_time = time.time()
    
    for _ in range(4):
        target_euler = np.array([np.pi, 0, np.pi * np.random.rand()])
        target_rot = Rotation.from_euler('xyz', target_euler)
        target_loc = np.random.randn(3) * 0.1 + np.array([0, 0, 1])
        for _ in range(200):
            grp_pose = env.get_gripper_pose(format='mat')
            grp_rot = Rotation.from_matrix(grp_pose[:3, :3])
            # grp_euler = grp_rot.as_euler('xyz')
            delta_rot = target_rot * grp_rot.inv()
            control = delta_rot.as_euler('XYZ') / np.pi
            for i in range(3):
                if abs(control[i]) > 1: # Opposite direction
                    control[i] = control[i] - np.sign(control[i]) * 2

            action = np.zeros(7)
            action[:3] = (target_loc - grp_pose[:3, 3]) * 0.2
            action[3:6] = control
            action[3]

            start_time = time.time()
            obs, reward, done, info = env.step(action)
            end_time = time.time()
            print("Time taken: ", end_time - start_time)
            # obs = env.get_observation()
            # print(obs)
            env.render_viewer()
            # print(grp_rot.as_euler('XYZ', degrees=True), action[3:6])
            # print(grp_rot.as_euler('XYZ', degrees=True), target_rot.as_euler('XYZ', degrees=True))

        # print(img)
    # print(obs.keys())
    # env.close()
    
    

def test_primitive():
    from hacman.utils.primitive_utils import GroundingTypes, Primitive, get_primitive_class, MAX_PRIMITIVES
    import hacman_suite.env_wrappers.primitives

    env = HACManSuiteStack(
        # has_renderer=True,
        # has_offscreen_renderer=False,
        # use_camera_obs=False,
        # render_camera="frontview",
    )
    obs = env.reset()

    import time
    start_time = time.time()
    primitive_names = ['suite-poke', 'suite-pick_n_lift_fixed']
    for p in primitive_names:
        prim = get_primitive_class(p)(env, end_on_reached=True)
        for _ in range(4):
            # loc = np.random.randn(3)
            loc = obs['cubeA_pos']
            # action = np.random.randn(prim.param_dim)
            action = np.zeros(prim.param_dim)
            action[-1] = np.random.rand()
            obs, all_rewards, done, info = prim.execute(loc, action)
    end_time = time.time()
    print("Time taken: ", end_time - start_time)

def test_completion():
    from hacman.utils.primitive_utils import GroundingTypes, Primitive, get_primitive_class, MAX_PRIMITIVES
    import hacman_suite.env_wrappers.primitives

    env = HACManSuiteStack(
        has_renderer=True,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        render_camera="frontview",
    )
    obs = env.reset()

    import time
    start_time = time.time()
    grasp = get_primitive_class('suite-pick_n_lift_fixed')(env, end_on_reached=True)
    place = get_primitive_class('suite-place')(env, end_on_reached=True)
    open_gripper = get_primitive_class('suite-open_gripper')(env, end_on_reached=True)
    for _ in range(4):
        # loc = np.random.randn(3)
        loc = obs['cubeA_pos']
        action = np.zeros(grasp.param_dim)
        obs, all_rewards, done, info = grasp.execute(loc, action)

        loc = obs['cubeB_pos'] + np.array([0, 0, 0.02])
        action = np.array([0, 0, 1, 0])
        obs, all_rewards, done, info = place.execute(loc, action)

        action = np.array([0])
        obs, all_rewards, done, info = open_gripper.execute(loc, action)
        print(done, info, env._check_success())
    end_time = time.time()
    print("Time taken: ", end_time - start_time)

if __name__ == "__main__":
    # test_env()
    test_primitive()
    # test_completion()


