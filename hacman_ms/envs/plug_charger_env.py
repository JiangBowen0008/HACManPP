import numpy as np

from mani_skill2.envs.assembly.plug_charger import PlugChargerEnv
from mani_skill2.sensors.camera import CameraConfig
from mani_skill2.utils.registration import register_env
from mani_skill2.utils.sapien_utils import look_at, hex2rgba
from transforms3d.euler import euler2quat
from sapien.core import Pose

from hacman.utils.transformations import transform_point_cloud, to_pose_mat

@register_env("HACMan-PlugCharger-v0", max_episode_steps=200, override=True)
class HACManPlugChargerEnv(PlugChargerEnv):
    def set_episode_rng(self, seed):
        """Set the random generator for current episode."""
        seed=0
        if seed is None:
            self._episode_seed = self._main_rng.randint(2**32)
        else:
            self._episode_seed = seed
        self._episode_rng = np.random.RandomState(self._episode_seed)
    
    def _load_actors(self):
        super()._load_actors()
        self.contact_site = self._build_sphere_site(0.02, color=(0.6, 0.6, 0), name="goal_site")
    
    def _build_sphere_site(self, radius, color=(0, 1, 0), name="goal_site"):
        """Build a sphere site (visual only). Used to indicate goal position."""
        builder = self._scene.create_actor_builder()
        builder.add_sphere_visual(radius=radius, color=color)
        sphere = builder.build_static(name)
        # NOTE(jigu): Must hide after creation to avoid pollute observations!
        sphere.hide_visual()
        return sphere
    
    def render(self, mode="human", **kwargs):
        # self.contact_site.set_pose(self.get_goal_pose(format="pose"))
        self.contact_site.unhide_visual()
        ret = super().render(mode, **kwargs)
        self.contact_site.hide_visual()
        return ret

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

    def compute_dense_reward(self, info, **kwargs):
        reward =  super().compute_dense_reward(info, **kwargs)
        reward -= 50
        return reward
    
    def get_primitive_states(self):
        is_grasped = self.agent.check_grasp(self.charger) # , max_angle=20
        return {"is_grasped": is_grasped}
    
    def get_segmentation_ids(self):       
        actors = self.get_actors()
        actor_ids = {actor.name: actor.id for actor in actors}
        background_id, charger_id, recep_id = actor_ids["ground"], actor_ids["charger"], actor_ids["receptacle"]

        object_ids = np.array([charger_id])
        # background_ids = np.array([cubeB_id, background_id])
        background_ids = np.array([recep_id])
        return {"object_ids": object_ids, "background_ids": background_ids}
    
    def get_goal_pose(self, format="mat"):
        p = self.goal_pose.p
        q = self.goal_pose.q
        if format == "mat":
            return to_pose_mat(p, q, input_wxyz=True)
        elif format == "vector":
            return np.concatenate([p, q])
        elif format == "pose":
            return Pose(p, q)
    
    def get_object_pose(self, format="mat"):
        p = self.charger.pose.p
        q = self.charger.pose.q
        if format == "mat":
            return to_pose_mat(p, q, input_wxyz=True)
        elif format == "vector":
            return np.concatenate([p, q])
        elif format == "pose":
            return Pose(p, q)
    
    def get_object_dim(self):
        return 0.06

    def get_default_z_rot(self):
        return 0.0

if __name__ == "__main__":
    env = HACManPlugChargerEnv()
    env.reset()
    env.get_segmentation_ids()
    frame = env.render(mode="cameras")
    # import matplotlib.pyplot as plt
    # plt.imshow(frame)
    # plt.show()
    env.close()