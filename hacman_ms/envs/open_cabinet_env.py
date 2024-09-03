from mani_skill2.envs.ms1.open_cabinet_door_drawer import OpenCabinetDoorEnv, OpenCabinetDrawerEnv
from mani_skill2.sensors.camera import CameraConfig
from mani_skill2.utils.registration import register_env
from mani_skill2.utils.sapien_utils import look_at
from transforms3d.euler import euler2quat
from sapien.core import Pose

import numpy as np

@register_env("HACMan-OpenCabinetDoor-v1", max_episode_steps=200, override=True)
class HACManOpenDoorEnv(OpenCabinetDoorEnv):
   

    def _load_actors(self):
        super()._load_actors()
        self.contact_site = self._build_sphere_site(0.02, color=(0.6, 0.6, 0), name="contact_site")
    
    def _build_sphere_site(self, radius, color=(0, 1, 0), name="goal_site"):
        """Build a sphere site (visual only). Used to indicate goal position."""
        builder = self._scene.create_actor_builder()
        builder.add_sphere_visual(radius=radius, color=color)
        sphere = builder.build_static(name)
        # NOTE(jigu): Must hide after creation to avoid pollute observations!
        sphere.hide_visual()
        return sphere
    
    def render(self, mode="human"):
        self.contact_site.unhide_visual()
        ret = super().render(mode)
        self.contact_site.hide_visual()
        return ret


if __name__ == "__main__":
    import gym
    pose = look_at([-1, -1, 0.5], [0, 0, 0.3])
    env = gym.make(
        "HACMan-OpenCabinetDoor-v1", obs_mode="image", 
        control_mode="base_pd_joint_vel_arm_pd_ee_delta_pose",
        # control_mode="pd_ee_target_delta_pose",
        # control_mode="pd_ee_target_delta_pos",
        # control_mode="pd_ee_delta_pos",
        camera_cfgs={"add_segmentation": True},
        render_camera_cfgs=dict(p=pose.p, q=pose.q),
    )
    for i in range(10):
        obs = env.reset()
        # frame = env.render(mode="cameras")
        # plt.imshow(frame)
        # plt.show()
        # Display the color image
        import matplotlib.pyplot as plt
        color = obs["image"]["overhead_camera_0"]["Color"]
        seg = obs["image"]["overhead_camera_0"]["Segmentation"]
        plt.figure()
        plt.subplot(3, 1, 1)
        plt.imshow(color)

        for i in range(0,2):
            plt.subplot(3, 1, i+2)
            plt.imshow(seg[:, :, i])

        plt.show()
    env.close()