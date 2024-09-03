import numpy as np

from mani_skill2.envs.assembly.peg_insertion_side import PegInsertionSideEnv
from mani_skill2.sensors.camera import CameraConfig
from mani_skill2.utils.registration import register_env
from mani_skill2.utils.sapien_utils import look_at, hex2rgba
from transforms3d.euler import euler2quat
from sapien.core import Pose
import sapien

from hacman.utils.transformations import transform_point_cloud, to_pose_mat
from .hacman_utility_base import HACManUtilityBase

@register_env("HACMan-PegInsertionSideEnv-v0", max_episode_steps=200, override=True)
class HACManPegInsertionSideEnv(PegInsertionSideEnv, HACManUtilityBase):
    '''
    Task adjustments
    '''
    _clearance = 0.01
    def set_episode_rng(self, seed):
        """Set the random generator for current episode."""
        seed=0
        if seed is None:
            self._episode_seed = self._main_rng.randint(2**32)
        else:
            self._episode_seed = seed
        self._episode_rng = np.random.RandomState(self._episode_seed)

    def _initialize_actors(self):
        xy = self._episode_rng.uniform([-0.1, -0.3], [0.1, 0])
        xy = np.array([0.0, -0.15])
        pos = np.hstack([xy, self.peg_half_size[2]])
        ori = np.pi / 2 + self._episode_rng.uniform(-np.pi / 3, np.pi / 3)
        ori = np.pi / 2
        quat = euler2quat(0, 0, ori)
        self.peg.set_pose(Pose(pos, quat))

        xy = self._episode_rng.uniform([-0.05, 0.2], [0.05, 0.4])
        xy = np.array([0.0, 0.3])
        pos = np.hstack([xy, self.peg_half_size[0]])
        ori = np.pi / 2 + self._episode_rng.uniform(-np.pi / 8, np.pi / 8)
        ori = np.pi / 2
        quat = euler2quat(0, 0, ori)
        self.box.set_pose(Pose(pos, quat))
    
    def _load_actors(self):
        super()._load_actors()
        self.build_contact_site()
    
    def _register_cameras(self):
        base_camera = super()._register_cameras()
        pose = look_at([0.4, 0.6, 0.06], [0, 0, 0.1])
        left_camera = CameraConfig(
            "left_camera", pose.p, pose.q, 128, 128, np.pi / 2, 0.01, 10
        )

        pose = look_at([0.3, -0.3, 0.06], [0, 0, 0.1])
        right_camera = CameraConfig(
            "right_camera", pose.p, pose.q, 128, 128, np.pi / 2, 0.01, 10
        )
        pose = look_at([0.4, 0.0, 0.06], [0, 0, 0.2])
        front_camera = CameraConfig(
            "front_camera", pose.p, pose.q, 128, 128, np.pi / 2, 0.01, 10
        )
        pose = look_at([-0.5, 0, 0.06], [0, 0, 0.2])
        back_camera = CameraConfig(
            "back_camera", pose.p, pose.q, 128, 128, np.pi / 2, 0.01, 10
        )
        pose = look_at([0.0, 0, 2.0], [0, 0, 0.2])
        top_camera = CameraConfig(
            "top_camera", pose.p, pose.q, 128, 128, np.pi / 2, 0.01, 10
        )
        return [base_camera, top_camera, left_camera, right_camera, back_camera]
    
    def has_peg_inserted(self):
        # Only head position is used in fact
        peg_head_pose = self.peg.pose.transform(self.peg_head_offset)
        box_hole_pose = self.box_hole_pose
        peg_head_pos_at_hole = (box_hole_pose.inv() * peg_head_pose).p
        # x-axis is hole direction
        x_flag = -0.015 <= peg_head_pos_at_hole[0]
        x_flag = x_flag and (peg_head_pos_at_hole[0] <= self.peg_half_size[0])
        y_flag = (
            -self.box_hole_radius <= peg_head_pos_at_hole[1] <= self.box_hole_radius
        )
        z_flag = (
            -self.box_hole_radius <= peg_head_pos_at_hole[2] <= self.box_hole_radius
        )
        return (x_flag and y_flag and z_flag), peg_head_pos_at_hole
    
    def compute_dense_reward(self, info, **kwargs):
        reward = super().compute_dense_reward(info, **kwargs)
        reward -= 25.
        return reward
    
    def compute_dense_reward_v5(self, info, **kwargs):
        reward = -10

        if info["success"]:
            reward = 0
        else:
            self.peg_half_length = self.peg_half_size[0]
            self.peg_radius = self.peg_half_size[1]
            
            # reaching reward
            gripper_pos = self.tcp.pose.p
            peg_head_pose = self.peg.pose.transform(self.peg_head_offset)
            head_pos, center_pos = peg_head_pose.p, self.peg.pose.p
            grasp_pos = center_pos - (head_pos - center_pos) * ((0.015+self.peg_radius)/self.peg_half_length) # hack a grasp point
            gripper_to_peg_dist = np.linalg.norm(gripper_pos - grasp_pos)
            reaching_reward = 1 - np.tanh(10.0 * gripper_to_peg_dist)
            reward += reaching_reward

            # grasp reward
            is_grasped = self.agent.check_grasp(self.peg) and (gripper_to_peg_dist < self.peg_radius * 0.9)
            if is_grasped:
                reward += 0.25

            # insertion reward
            if is_grasped:
                box_hole_pose = self.box_hole_pose
                peg_head_pos_at_hole = (box_hole_pose.inv() * peg_head_pose).p

                insertion_reward = 1 - np.tanh(5.0 * abs(self.peg_half_length - peg_head_pos_at_hole[0])) # (0, 1)
                align_reward_y = 1 - np.tanh(10.0 * abs(peg_head_pos_at_hole[1])) # (0, 1)
                align_reward_z = 1 - np.tanh(10.0 * abs(peg_head_pos_at_hole[2])) # (0, 1) 
                
                reward += insertion_reward * 2 + align_reward_y + align_reward_z

                peg_normal = self.peg.pose.transform(Pose([0,0,1])).p - self.peg.pose.p
                hole_normal = box_hole_pose.transform(Pose([0,0,1])).p - box_hole_pose.p
                cos_normal = abs(np.dot(hole_normal, peg_normal) / np.linalg.norm(peg_normal) / np.linalg.norm(hole_normal)) # (0, 1)
                reward += cos_normal

                peg_axis = self.peg.pose.transform(Pose([1,0,0])).p - self.peg.pose.p
                hole_axis = box_hole_pose.transform(Pose([1,0,0])).p - box_hole_pose.p
                cos_axis = abs(np.dot(hole_axis, peg_axis) / np.linalg.norm(peg_axis) / np.linalg.norm(hole_axis)) # (0, 1)
                reward += cos_axis

        return reward

    def compute_dense_reward_v4(self, info, **kwargs):
        reward = -10

        if info["success"]:
            reward = 0
        else:
            self.peg_half_length = self.peg_half_size[0]
            self.peg_radius = self.peg_half_size[1]

            # reaching reward
            gripper_pos = self.tcp.pose.p
            peg_head_pose = self.peg.pose.transform(self.peg_head_offset)
            head_pos, center_pos = peg_head_pose.p, self.peg.pose.p
            grasp_pos = center_pos - (head_pos - center_pos) * ((0.015+self.peg_radius)/self.peg_half_length) # hack a grasp point
            gripper_to_peg_dist = np.linalg.norm(gripper_pos - grasp_pos)
            reaching_reward = 1 - np.tanh(10.0 * gripper_to_peg_dist)
            reward += reaching_reward

            # grasp reward
            is_grasped = self.agent.check_grasp(self.peg) and (gripper_to_peg_dist < self.peg_radius * 0.9)
            if is_grasped:
                reward += 0.25

            # insertion reward
            if is_grasped:
                offset = np.copy(self.box_hole_offset.p)
                offset[0] = -self.peg_half_size[0]  # peg length
                hole_pose = self.box.pose.transform(Pose(offset))
                
                box_hole_pose = hole_pose # self.box_hole_pose
                peg_head_pos_at_hole = (box_hole_pose.inv() * peg_head_pose).p
                peg_pos_at_hole = (box_hole_pose.inv() * self.peg.pose).p

                # Align the center of the peg to the hole through yz
                center_align_reward_y = 1 - np.tanh(10.0 * abs(peg_pos_at_hole[1])) # (0, 1)
                center_align_reward_z = 1 - np.tanh(10.0 * abs(peg_pos_at_hole[2])) # (0, 1)
                reward += center_align_reward_y + center_align_reward_z

                peg_normal = self.peg.pose.transform(Pose([0,0,1])).p - self.peg.pose.p
                hole_normal = box_hole_pose.transform(Pose([0,0,1])).p - box_hole_pose.p
                cos_normal = abs(np.dot(hole_normal, peg_normal) / np.linalg.norm(peg_normal) / np.linalg.norm(hole_normal)) # (0, 1)
                reward += 0.5 * cos_normal

                peg_axis = self.peg.pose.transform(Pose([1,0,0])).p - self.peg.pose.p
                hole_axis = box_hole_pose.transform(Pose([1,0,0])).p - box_hole_pose.p
                cos_axis = abs(np.dot(hole_axis, peg_axis) / np.linalg.norm(peg_axis) / np.linalg.norm(hole_axis)) # (0, 1)
                reward += 0.5 * cos_axis

                if peg_head_pos_at_hole[0] < 0.01:
                    # pre-insert reward, encouraging te peg head to get close to the hole
                    pre_insertion_pos = np.copy(peg_head_pos_at_hole)
                    pre_insertion_pos[0] = np.clip(pre_insertion_pos[0], None, 0)
                    pre_insertion_reward = 1 - np.tanh(5.0 * np.linalg.norm(pre_insertion_pos))
                    reward += 3 * pre_insertion_reward
                    # reward = pre_insertion_pos[0]
                    # print(f"pre_insertion_x_distance: {pre_insertion_pos}")
                elif peg_head_pos_at_hole[0] < self.peg_half_length:
                    # pre-insert reward, encouraging te peg head to get close to the goal
                    pre_insertion_reward = 3
                    # reward = 2
                    insertion_reward = 1 - np.tanh(5.0 * abs(self.peg_half_length - peg_head_pos_at_hole[0])) # (0, 1)
                    reward += pre_insertion_reward + 2 * insertion_reward

        return reward

    def compute_dense_reward_v0(self, info, **kwargs):
        # reward =  super().compute_dense_reward(info, **kwargs)
        reward = 0.0

        if info["success"]:
            return 0.0  # 25

        # grasp pose rotation reward
        tcp_pose_wrt_peg = self.peg.pose.inv() * self.tcp.pose
        tcp_rot_wrt_peg = tcp_pose_wrt_peg.to_transformation_matrix()[:3, :3]
        gt_rot_1 = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        gt_rot_2 = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
        grasp_rot_loss_fxn = lambda A: np.arcsin(
            np.clip(1 / (2 * np.sqrt(2)) * np.sqrt(np.trace(A.T @ A)), 0, 1)
        )
        grasp_rot_loss = np.minimum(
            grasp_rot_loss_fxn(gt_rot_1 - tcp_rot_wrt_peg),
            grasp_rot_loss_fxn(gt_rot_2 - tcp_rot_wrt_peg),
        ) / (np.pi / 2)
        rotated_properly = grasp_rot_loss < 0.2
        # reward += 1 - grasp_rot_loss

        gripper_pos = self.tcp.pose.p
        tgt_gripper_pose = self.peg.pose
        offset = Pose(
            [-0.06, 0, 0]
        )  # account for panda gripper width with a bit more leeway
        tgt_gripper_pose = tgt_gripper_pose.transform(offset)
        if rotated_properly:
            # reaching reward
            # gripper_to_peg_dist = np.linalg.norm(gripper_pos - tgt_gripper_pose.p)
            # reaching_reward = 1 - np.tanh(
            #     4.0 * np.maximum(gripper_to_peg_dist - 0.015, 0.0)
            # )
            # # reaching_reward = 1 - np.tanh(10.0 * gripper_to_peg_dist)
            # reward += reaching_reward

            # grasp reward
            is_grasped = self.agent.check_grasp(
                self.peg, max_angle=20
            )  # max_angle ensures that the gripper grasps the peg appropriately, not in a strange pose
            # if is_grasped:
            #     reward += 2.0

            # pre-insertion award, encouraging both the peg center and the peg head to match the yz coordinates of goal_pose
            pre_inserted = False
            if is_grasped:
                peg_head_wrt_goal = self.goal_pose.inv() * self.peg_head_pose
                peg_head_wrt_goal_yz_dist = np.linalg.norm(peg_head_wrt_goal.p[1:])
                peg_wrt_goal = self.goal_pose.inv() * self.peg.pose
                peg_wrt_goal_yz_dist = np.linalg.norm(peg_wrt_goal.p[1:])
                if peg_head_wrt_goal_yz_dist < 0.01 and peg_wrt_goal_yz_dist < 0.01:
                    pre_inserted = True
                    reward += 3.0
                pre_insertion_reward = 3 * (
                    1
                    - np.tanh(
                        0.5 * (peg_head_wrt_goal_yz_dist + peg_wrt_goal_yz_dist) +
                        4.5 * np.maximum(peg_head_wrt_goal_yz_dist, peg_wrt_goal_yz_dist)
                    )
                )
                reward += pre_insertion_reward

            # insertion reward
            if is_grasped and pre_inserted:
                peg_head_wrt_goal_inside_hole = (
                    self.box_hole_pose.inv() * self.peg_head_pose
                )
                insertion_reward = 5 * (
                    1 - np.tanh(5.0 * np.linalg.norm(peg_head_wrt_goal_inside_hole.p))
                )
                reward += insertion_reward
        else:
            reward = reward - 10 * np.maximum(
                self.peg.pose.p[2] + self.peg_half_size[2] + 0.01 - self.tcp.pose.p[2],
                0.0,
            )
            reward = reward - 10 * np.linalg.norm(
                tgt_gripper_pose.p[:2] - self.tcp.pose.p[:2]
            )

        reward -= 12
        return reward
    
    '''
    HACMan utilities
    '''
    def get_segmentation_ids(self):       
        actors = self.get_actors()
        actor_ids = {actor.name: actor.id for actor in actors}
        background_id, peg_id, box_id = actor_ids["ground"], actor_ids["peg"], actor_ids["box_with_hole"]

        object_ids = np.array([peg_id])
        # background_ids = np.array([cubeB_id, background_id])
        background_ids = np.array([box_id])
        return {"object_ids": object_ids, "background_ids": background_ids}

    def get_primitive_states(self):
        is_grasped = self.agent.check_grasp(self.peg) # , max_angle=20
        return {"is_grasped": is_grasped}
    
    def get_goal_pose(self, format="mat"):
        offset = np.copy(self.box_hole_offset.p)
        offset[0] = -self.peg_half_size[0]  # peg length
        hole_pose = self.box.pose.transform(Pose(offset))
        p = hole_pose.p
        q = hole_pose.q
        if format == "mat":
            return to_pose_mat(p, q, input_wxyz=True)
        elif format == "vector":
            return np.concatenate([p, q])
        elif format == "pose":
            return Pose(p, q)
    
    def get_object_pose(self, format="mat"):
        peg_head_pose = self.peg_head_pose
        p = peg_head_pose.p
        q = peg_head_pose.q
        if format == "mat":
            return to_pose_mat(p, q, input_wxyz=True)
        elif format == "vector":
            return np.concatenate([p, q])
        elif format == "pose":
            return Pose(p, q)
    
    def get_gripper_pose(self, format="mat"):
        if format == "pose":
            return self.tcp.pose
        else:
            p = self.tcp.pose.p
            q = self.tcp.pose.q
            if format == "mat":
                return to_pose_mat(p, q, input_wxyz=True)
            elif format == "vector":
                return np.concatenate([p, q])
    
    def get_object_dim(self):
        return np.max(self.peg_half_size) * 2
    
    def get_default_z_rot(self):
        return -np.pi / 2
    
    def render(self, mode="human"):
        self.contact_site.unhide_visual()
        ret = super().render(mode)
        self.contact_site.hide_visual()
        return ret

if __name__ == "__main__":
    env = HACManPegInsertionSideEnv()
    env.reset()
    env.get_segmentation_ids()
    frame = env.render(mode="cameras")
    import matplotlib.pyplot as plt
    plt.imshow(frame)
    plt.show()
    env.close()