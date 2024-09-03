import numpy as np
import itertools
from math import sqrt
from sapien.core import Pose

class HACManUtilityBase():
    def get_segmentation_ids(self):       
        raise NotImplementedError
    
    def get_primitive_states(self):
        raise NotImplementedError
    
    def get_goal_pose(self, format="mat"):
        raise NotImplementedError

    def get_object_pose(self, format="mat"):
        raise NotImplementedError
    
    def get_gripper_pose(self, format="mat"):
        raise NotImplementedError
    
    def get_object_dim(self):
        raise NotImplementedError

    def get_default_z_rot(self):
        raise NotImplementedError
    
    def map_location(self, loc, mode="bbox"):
        '''
        Map from [-1, 1] to the workspace. Used for regressed location baselines.
        '''
        # Centered around the workspace
        if mode == "ws":
            lower_bound = np.array([-0.3, -0.3, 0.0])
            upper_bound = np.array([0.3, 0.3, 0.2])
            center = np.array([0, 0, 0])

        # Centered around the target cube
        elif mode == "recentered_ws":
            lower_bound = np.array([-0.2, -0.2, 0.0])
            upper_bound = np.array([0.2, 0.2, 0.2])
            center = self.get_goal_pose("vector")[:3]
        
        elif mode == "bbox":
            half_obj_dim = self.get_object_dim() / 2.
            lower_bound = np.ones(3) * -half_obj_dim * sqrt(3)
            upper_bound = np.ones(3) * half_obj_dim * sqrt(3)
            center = self.get_goal_pose("vector")[:3]
        
        else:
            raise ValueError(f"Unknown mapping mode: {mode}")
        
        mapped = (loc + 1) / 2.        # map to [0, 1]
        mapped = mapped * (upper_bound - lower_bound) + lower_bound + center

   
        return mapped
    
    def build_contact_site(self):
        self.contact_site = self._build_sphere_site(0.02, color=(0.6, 0.6, 0), name="contact_site")

    def _build_sphere_site(self, radius, color=(0, 1, 0), name="goal_site"):
        """Build a sphere site (visual only). Used to indicate goal position."""
        builder = self._scene.create_actor_builder()
        builder.add_sphere_visual(radius=radius, color=color)
        sphere = builder.build_static(name)
        # NOTE(jigu): Must hide after creation to avoid pollute observations!
        sphere.hide_visual()
        return sphere