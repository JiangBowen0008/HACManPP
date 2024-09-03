import numpy as np

from mani_skill2.envs.pick_and_place.pick_cube import PickCubeEnv, LiftCubeEnv
from mani_skill2.sensors.camera import CameraConfig
from mani_skill2.utils.registration import register_env
from mani_skill2.utils.sapien_utils import look_at
from transforms3d.euler import euler2quat
from sapien.core import Pose

from hacman.utils.transformations import transform_point_cloud, to_pose_mat
from .hacman_utility_base import HACManUtilityBase

@register_env("HACMan-LiftCube-v0", max_episode_steps=200, override=True)
class HACManLiftCubeEnv(LiftCubeEnv, HACManUtilityBase):
    '''
    Task adjustments
    '''
    goal_height = 0.05
    def _register_cameras(self):
        base_camera = super()._register_cameras()
        pose = look_at([0.2, 0.2, 0.06], [0, 0, 0.1])
        left_camera = CameraConfig(
            "left_camera", pose.p, pose.q, 128, 128, np.pi / 2, 0.01, 10
        )

        pose = look_at([0.2, -0.2, 0.06], [0, 0, 0.1])
        right_camera = CameraConfig(
            "right_camera", pose.p, pose.q, 128, 128, np.pi / 2, 0.01, 10
        )
        pose = look_at([0.4, 0.0, 0.06], [0, 0, 0.2])
        front_camera = CameraConfig(
            "front_camera", pose.p, pose.q, 128, 128, np.pi / 2, 0.01, 10
        )
        pose = look_at([-0.4, 0, 0.06], [0, 0, 0.2])
        back_camera = CameraConfig(
            "back_camera", pose.p, pose.q, 128, 128, np.pi / 2, 0.01, 10
        )
        return [base_camera, left_camera, right_camera, back_camera]
    
    
    def compute_dense_reward(self, info, **kwargs):
        reward = super().compute_dense_reward(info, **kwargs)
        reward -= 2.25
        return reward

    def evaluate(self, **kwargs):
        is_obj_placed = self.check_obj_placed()
        is_robot_static = self.check_robot_static()
        return dict(
            is_obj_placed=is_obj_placed,
            is_robot_static=is_robot_static,
            success=is_obj_placed,
        )


    
    def _load_actors(self):
        super()._load_actors()
        self.build_contact_site()
    
    '''
    HACMan utilities
    '''
    def get_segmentation_ids(self):       
        actors = self.get_actors()
        actor_ids = {actor.name: actor.id for actor in actors}
        cube_id, background_id = actor_ids["cube"], actor_ids["ground"]

        object_ids = np.array([cube_id])
        background_ids = np.array([background_id])
        return {"object_ids": object_ids, "background_ids": background_ids}

    def get_primitive_states(self):
        is_cube_grasped = self.agent.check_grasp(self.obj, max_angle=30)
        return {"is_grasped": is_cube_grasped}
    
    def get_goal_pose(self, format="mat"):
        p = self.goal_pos
        q = self.obj.pose.q
        if format == "mat":
            return to_pose_mat(p, q, input_wxyz=True)
        elif format == "vector":
            return np.concatenate([p, q])
    
    def get_object_pose(self, format="mat"):
        p = self.obj.pose.p
        q = self.obj.pose.q
        if format == "mat":
            return to_pose_mat(p, q, input_wxyz=True)
        elif format == "vector":
            return np.concatenate([p, q])
    
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
        return 0.05
    

    
    def render(self, mode="human"):
        self.contact_site.unhide_visual()
        ret = super().render(mode)
        self.contact_site.hide_visual()
        return ret
    
    


if __name__ == "__main__":
    env = HACManPickCubeEnv()
    env.reset()
    frame = env.render(mode="cameras")
    import matplotlib.pyplot as plt
    plt.imshow(frame)
    plt.show()
    env.close()