import numpy as np
import open3d as o3d
import torch
import itertools
import gym
from math import sqrt
from copy import deepcopy
import matplotlib.pyplot as plt

from hacman_bin.utils.point_cloud_utils import convert_depth, get_point_cloud, add_additive_noise_to_xyz, dropout_random_ellipses
from hacman.utils.transformations import to_pose_mat, transform_point_cloud, decompose_pose_mat, sample_idx

class HACManUtilityBase():
    def __init__(self, 
                 voxel_downsample_size=0.005,
                 background_downsample_scale=1,
                 object_pcd_size=400,
                 background_pcd_size=400,
                 clip_arena=False,
                 clip_object=None,  # radius of clipping, set to None to not clip
                 clip_goal=None,    # radius of clipping, set to None to not clip
                 use_flow_reward=False,
                 record_video=False,
                 record_from_cam="agent",
                 segmentation=None,
                 use_color=False,
                 img_size=(300, 300)):
        
        self.voxel_downsample_size = voxel_downsample_size
        self.background_downsample_scale = background_downsample_scale
        self.object_pcd_size = object_pcd_size
        self.background_pcd_size = background_pcd_size
        self.pcd_size = object_pcd_size + background_pcd_size
        self.clip_arena = clip_arena
        self.clip_object = clip_object
        self.clip_goal = clip_goal

        # updated
        self.use_color = use_color
        self.img_size = img_size

        self.use_flow_reward = use_flow_reward
        self.record_video = record_video
        self.record_from_cam = record_from_cam
        # [Note] has_offscreen_renderer 
        self.has_offscreen_renderer = True
        # if self.has_offscreen_renderer:
        #     self.disable_cameras()

        self.observation_space = gym.spaces.Dict(
            spaces={
                "object_pose": gym.spaces.Box(-np.inf, np.inf, (4,4)),
                "goal_pose": gym.spaces.Box(-np.inf, np.inf, (4,4)),
                "gripper_pose": gym.spaces.Box(-np.inf, np.inf, (4,4)),
                "pointcloud": gym.spaces.Box(-np.inf, np.inf, (self.pcd_size, 10)), # only used when
            }
        )
        self.action_space = gym.spaces.Box(-1, 1, (30,))

        # updated

        self.segmentation = segmentation
        if segmentation is not None:
            self.seg_mapping = {}
            self.name2id = {}
            for id, name in self.sim.model._geom_id2name.items():
                if name is None:
                    continue
                name = name.split("_")[0]
                self.add_seg_label(name, id)
        else:
            self.seg_mapping = None
        
    def add_seg_label(self, name, id, force_update=False):
        if self.seg_mapping is None:
            return
        
        if len(self.seg_mapping) == 0:
            self.name2id[name] = 0
            self.seg_mapping[id] = 0
            self.max_id = 0
            return
        
        if id in self.seg_mapping and (not force_update):
            return
        else:
            if not (name in self.name2id):
                self.max_id = self.max_id + 1
                self.name2id[name] = self.max_id
            self.seg_mapping[id] = self.name2id[name]

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
    
    def get_arena_bounds(self):
        raise NotImplementedError
    
    # [Note] function conflicted in 
    # /home/bowen/Desktop/hacman_cleanup-multi-netowrks-primitives/third_party/rrl-dependencies/mjrl/mjrl/envs/mujoco_env.py", line 71
    # """
    # Helper functions
    # """
    # def seed(self, seed):
    #     np.random.seed(seed)
    #     torch.manual_seed(seed)

    # def disable_cameras(self):
    #     for o_name in self.observation_names:
    #         if "view" in o_name:
    #             self._observables[o_name].set_active(False)
    
    # def enable_cameras(self):
    #     print(self.observation_names)
    #     for o_name in self.observation_names:
    #         if "view" in o_name:
    #             self._observables[o_name].set_active(True)
    
    # # [Note] [Reset] debug
    # def reset(self, reconfigure=False, **kwargs):

    #     # Old code
    #     # obs = self.get_observation()

    #     # Note
    #     # obs_dict = {
    #     #     'image': obs_rgbd,
    #     #     'agent_pos': obs_sensor
    #     # }
    #     print('reset in base hacman')
    #     obs_dict = super().reset() # return images and joint set
    #     self.observation_space['object_pose'] = obs_dict['object_pose']
    #     self.observation_space['goal_pose'] = obs_dict['goal_pose']
    #     self.observation_space['gripper_pose'] = obs_dict['gripper_pose']
    #     self.observation_space['pointcloud'] = self.get_point_cloud(obs_dict)

    #     return obs_dict

    # def get_observation(self):
    #     # Try 5 times
    #     for _ in range(3):
    #         obs = self._get_observations()
    #         obs.update(self.get_camera_observation())
    #         obs["pointcloud"] = None
    #         pcd = self.get_point_cloud(obs, clip_arena=self.clip_arena)
    #         if pcd is not None:
    #             obs["pointcloud"] = pcd
    #             break
    #     if obs["pointcloud"] is None:
    #         return self.reset()
    #     return obs
    
    def get_camera_observation(self):
        obs = {}
        for c in self.camera_names:
            if self.use_color:
                img = self.sim.render(
                    width=self.img_size[0],
                    height=self.img_size[1],
                    mode='offscreen',
                    camera_name=c,
                )
                obs[c + "_image"] = img

            seg, depth = self.sim.render(
                width=self.img_size[0],
                height=self.img_size[1],
                mode='offscreen',
                segmentation=True,
                depth=True,
                camera_name=c,
            )
            # depth
            depth_name = c + "_depth"
            obs[depth_name] = depth

            # segmentation
            seg_name = c + "_segmentation"
            seg = seg[:, :, -1]
            if self.seg_mapping is not None:
                seg = np.fromiter(map(lambda x: self.seg_mapping.get(x, 0), seg.flatten()), dtype=np.int32) \
                          .reshape(self.img_size[1], self.img_size[0], 1)
            obs[seg_name] = seg
        return obs
    
    def get_point_cloud(self, obs, compute_normals=False):
        box_in_the_view = True
        origin_coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        pc_list = []
        pc_color_list = []
        object_mask_list = []
        bg_mask_list = []
        # gripper_mask_list = []
        seg_ids = self.get_segmentation_ids()
        obj_seg_id, bg_seg_id = seg_ids["object_ids"], seg_ids["background_ids"]
        for cam in self.camera_names:

            depth = obs[cam + '_depth'][-1::-1]
            if self.use_color:
                color = obs[cam + '_image'][-1::-1]
            else:
                color = np.zeros((depth.shape[0], depth.shape[1], 3))

            # color = obs[cam + '_image'][-1::-1]
            # depth = obs[cam + '_depth'][-1::-1]
            seg = obs[cam + '_segmentation'][-1::-1]
            obj_mask = np.isin(seg, obj_seg_id).reshape(color.shape[0], color.shape[1])
            bg_mask = np.isin(seg, bg_seg_id).reshape(color.shape[0], color.shape[1])

            # self.debug_segmentation(seg, color, obj_mask, cam)
            depth = convert_depth(self, depth)
            pc = get_point_cloud(self, depth, camera_name=cam)
            pc_color = color.reshape(-1, 3).astype(np.float64)/256

            pc_list.append(pc)
            pc_color_list.append(pc_color)
            object_mask_list.append(obj_mask.reshape(-1))
            bg_mask_list.append(bg_mask.reshape(-1))
            # gripper_mask_list.append(gripper_mask.reshape(-1))
            # background_mask_list.append(background_mask.reshape(-1))
        
        pc = np.concatenate(pc_list)
        pc_color = np.concatenate(pc_color_list)
        object_mask = np.concatenate(object_mask_list)
        bg_mask = np.concatenate(bg_mask_list)
        # pc_gripper_mask = np.concatenate(gripper_mask_list)
        # background_mask = np.concatenate(background_mask_list)
     
        # Remove unnecessary points outside of the arena
        if self.clip_arena:
            mask = self.within_arena(pc, margin=np.array([0.01, 0.01, 0]), check_z=True)

            pc = pc[mask]
            pc_color = pc_color[mask]
            object_mask = object_mask[mask]
            bg_mask = bg_mask[mask]

        # Extract the object points
        try:
            assert object_mask.sum() > 10, "No object points in the view"
            object_pcd = o3d.geometry.PointCloud()
            object_pcd.points = o3d.utility.Vector3dVector(pc[object_mask])
            object_pcd.colors = o3d.utility.Vector3dVector(pc_color[object_mask])
            object_pcd = object_pcd.voxel_down_sample(self.voxel_downsample_size)
            if compute_normals:
                object_pcd.estimate_normals()
                object_pcd.orient_normals_consistent_tangent_plane(10)
        
        except :
            obj_in_the_view = False
            return None
        
        # Extract the background points
        try:
            assert bg_mask.sum() > 10, "No background points in the view"
            bg_pcd = o3d.geometry.PointCloud()
            bg_pcd.points = o3d.utility.Vector3dVector(pc[bg_mask])
            bg_pcd.colors = o3d.utility.Vector3dVector(pc_color[bg_mask])
            # o3d.visualization.draw_geometries([bg_pcd])
            bg_voxel_size = self.voxel_downsample_size * self.background_downsample_scale
            bg_pcd = bg_pcd.voxel_down_sample(bg_voxel_size)
            if compute_normals:
                bg_pcd.estimate_normals()
                bg_pcd.orient_normals_consistent_tangent_plane(10)
        except:
            bg_in_the_view = False
            return None
        
        # o3d.visualization.draw_geometries([object_pcd, bg_pcd, origin_coord])

        # Convert to np array & sample to the desired size
        
        obj_idx = sample_idx(len(object_pcd.points), self.object_pcd_size)
        object_pcd_points = np.asarray(object_pcd.points)[obj_idx]
        object_pcd_colors = np.asarray(object_pcd.colors)[obj_idx]
        if compute_normals:
            object_pcd_normals = np.asarray(object_pcd.normals)[obj_idx]
        
        bg_idx = sample_idx(len(bg_pcd.points), self.background_pcd_size)
        background_pcd_points = np.asarray(bg_pcd.points)[bg_idx]
        background_pcd_colors = np.asarray(bg_pcd.colors)[bg_idx]
        if compute_normals:
            background_pcd_normals = np.asarray(bg_pcd.normals)[bg_idx]

        # Add the segmentation labels
        object_pcd_seg = np.ones((len(object_pcd_points), 1)) * obj_seg_id[0]
        background_pcd_seg = np.ones((len(background_pcd_points), 1)) * bg_seg_id[0]

        points = np.vstack([object_pcd_points, background_pcd_points])
        colors = np.vstack([object_pcd_colors, background_pcd_colors])
        if compute_normals:
            normals = np.vstack([object_pcd_normals, background_pcd_normals])
        else:
            normals = np.zeros((len(points), 3))
            normals[:, 2] = 1.
        segs = np.vstack([object_pcd_seg, background_pcd_seg])
        pcd = np.hstack([points, colors, normals, segs])
        return pcd
    
    def sim_step(self, action):
        return super().step(action) 
    
    # def evaluate_obs(self, obs, info):
    #     # Compute the flow reward
    #     seg_ids = self.get_segmentation_ids()
    #     obj_ids = seg_ids['object_ids']
    #     pointcloud = obs['pointcloud']
    #     current_pcd = pointcloud[np.isin(pointcloud[:,-1], obj_ids)][..., :3]
        
    #     current_pose = self.get_object_pose(format="mat")
    #     goal_pose = self.get_goal_pose(format="mat")

    #     goal_pcd = transform_point_cloud(current_pose, goal_pose, current_pcd)
    #     flow = np.linalg.norm(goal_pcd - current_pcd, axis=-1)
    #     mean_flow = np.mean(flow, axis=-1)
    #     reward = -mean_flow
    #     success = mean_flow < self.success_threshold

    def render_viewer(self):
        if self.has_renderer:
            super().render()
    
    def render(self, mode='offscreen', **kwargs):
        if self.has_offscreen_renderer:
            # current_data = deepcopy(self.sim.data)
            # current_states = self.get_full_states()
            # print(f'mode: {mode}')
            img = self.sim.render(
                camera_name=self.record_from_cam,
                # width=self.width, 
                # height=self.height, 
                width=640,
                height=480,
                mode=mode, 
            )
            img = np.flipud(img)
            # post_states = self.get_full_states()
            # self.set_states(current_states)
            # self.sim.forward() # Optionally, not sure if it's necessary
            return img
    
    def get_actuator_states(self):
        states = {}
        for key in {'actuator_force', 'actuator_length', 'actuator_moment', 'actuator_velocity', 'act', 'act_dot'}:
            states[key] = deepcopy(getattr(self.sim.data, key))
        return states
    
    def get_full_states(self):
        states = {}
        for key in dir(self.sim.data):
            if key.startswith("_"):
                continue
            try:
                val = getattr(self.sim.data, key)
                if isinstance(val, np.ndarray):
                    states[key] = deepcopy(val)
            except:
                pass
        return states
    
    def set_states(self, states):
        for key, val in states.items():
            try:
                setattr(self.sim.data, key, val)
            except:
                pass

    def compare_states(self, states1, states2):
        # Show which keys are different, and the indices of the differences
        for key in states1.keys():
            if key not in states2:
                print(f"Key {key} not in states2")
            else:
                diff = states1[key] - states2[key]
                if diff.any():
                    print(f"Key {key} has differences at indices {np.where(diff)}")
    
    # def compare_states_simple(states1, states2):
    #     # Show which keys are different, and the indices of the differences
    #     for key in states1.keys():
    #         if key not in states2:
    #             print(f"Key {key} not in states2")
    #         else:
    #             if not np.allclose(states1[key], states2[key]):
    #                 print(f"Key {key} has differences")
    
    def within_arena(self, pc, margin=np.array([0.01, 0.01, 0.01]), check_z=True):
        arena_bounds = self.get_arena_bounds()
        mask = np.all(pc > arena_bounds[0] + margin, axis=1) & np.all(pc < arena_bounds[1] - margin, axis=1)
        if check_z:
            mask &= pc[:, 2] > 0
        return mask
    
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

        # Display the vertices of the space
        # bounds = np.vstack([lower_bound, upper_bound])
        # for i, coord in enumerate(itertools.product(*(bounds.T))):
        #     sphere = self._build_sphere_site(0.01, coord, name=f"vertex_{i}")
        #     sphere.set_pose(Pose(coord + center))
        #     sphere.unhide_visual()
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

def sample_pcd(pcd, size):
    idx = sample_idx(pcd.shape[0], size)
    return pcd[idx]