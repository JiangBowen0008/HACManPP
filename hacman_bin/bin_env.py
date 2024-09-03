import os, sys
import pickle
import numpy as np
import copy
from scipy.spatial.transform import Rotation
import imageio 
import os
from hacman_bin.base_env import BaseEnv
import open3d as o3d
from hacman_bin.utils.point_cloud_utils import convert_depth, get_point_cloud, add_additive_noise_to_xyz, dropout_random_ellipses
from robosuite.utils.control_utils import orientation_error
from hacman_bin.utils.transformations import to_pose_mat, inv_pose_mat, decompose_pose_mat
from robosuite.controllers import load_controller_config

VERBOSE = False

class BinEnv(BaseEnv):
    def __init__(
            self,
            # Specific to PokeEnv
            ik_precontact=True,
            transparent_gripper_base=True,
            planar_action=False,
            resting_time=2, 
            goal_mode='fixed',
            goal_on_oppo_side=False,
            enable_rotation=True,
            ignore_position_limits=True,
            renderer='offscreen',
            # Will be passed to BaseEnv
            robots='Virtual',
            close_gripper=True,
            object_dataset='housekeep_all',
            object_name='rubiks_cube_077_rubiks_cube_M',
            control_freq=2,
            ignore_done=True,
            render_camera='agentview',
            location_noise=0.,
            pcd_noise=None,
            voxel_downsample_size=0.01,
            exclude_wall_pcd=False,
            exclude_gripper_pcd=False,
            move_gripper_to_object_pcd=False,
            friction_config='default',
            record_video=False,
            record_from_cam=None,
            set_camera_names = ['agentview'],
            camera_poses = [np.array([1.  ,  0.  ,  0.715])],
            camera_quats = [np.array([0.653, 0.271 ,0.271 ,0.653])],
            **kwargs
    ):

        """
        Setup parameters for PokeEnv
        """
        self.planar_action = planar_action
        self.ik_precontact = ik_precontact
        self.transparent_gripper_base = transparent_gripper_base
        self.record_video = record_video
        self.record_from_cam = render_camera if record_from_cam is None else record_from_cam
        self.cam_frames = []
        self.goal = None
        self.resting_time = resting_time
        self.location_noise = location_noise
        self.pcd_noise = pcd_noise # [2, 30, 2, 0.005]
        self.exclude_wall_pcd = exclude_wall_pcd
        self.exclude_gripper_pcd = exclude_gripper_pcd
        self.move_gripper_to_object_pcd = move_gripper_to_object_pcd
        self.voxel_downsample_size = voxel_downsample_size
        self.enable_rotation = enable_rotation
        if self.enable_rotation:
            self.controller = "osc_6d_002_no_ori_limit.json"
        else:
            self.controller = "osc_no_ori.json"
        
        current_folder = os.path.dirname(os.path.abspath(__file__))
        
        # Setup goal sampling
        self.goal_mode = goal_mode
        self.goal_on_oppo_side = goal_on_oppo_side
        self.goal_list = None
        if object_dataset == 'cube':
            if goal_mode == 'any-old':
                pose_file = os.path.join(current_folder, 'data/Dataset0004/states.npy')
            elif goal_mode == 'any-old-eval':
                pose_file = os.path.join(current_folder, 'data/Dataset0004_eval/states.npy')
            elif goal_mode == 'oop-old':
                pose_file = os.path.join(current_folder, 'data/Dataset0004/states_out_of_plane.npy')
            elif goal_mode == 'oop-old-eval':
                pose_file = os.path.join(current_folder, 'data/Dataset0004_eval/states_out_of_plane.npy')
            elif goal_mode in ['fixed', 'translation', 'upright', 'fixed_off_center', 'off_center']:
                pose_file = None
            else:
                raise ValueError("Unknown goal_mode: {}".format(goal_mode))
            
            if pose_file is not None:
                self.goal_list = {'cube': {'poses':np.load(pose_file)}}
        
        elif object_dataset in {'housekeep', 'housekeep_all'}:
            if goal_mode == 'any':
                if 'object_scale_range' in kwargs.keys() and kwargs['object_scale_range'] is not None:
                    assert kwargs['object_scale_range'][0] == kwargs['object_scale_range'][1], "object_scale_range should be a single value!"
                pose_file = os.path.join(current_folder, f'data/{object_dataset}/poses_fixed_size.pk')
        
            

            elif goal_mode == 'any_var_size':
                pose_file = os.path.join(current_folder, f'data/{object_dataset}/poses_variable_size_v1.pk')

            
            elif goal_mode == 'any_var_size_with_wall':
                pose_file = os.path.join(current_folder, f'data/{object_dataset}/poses_variable_size_with_wall.pk')

            elif goal_mode in ['fixed', 'translation', 'upright', 'fixed_off_center', 'off_center', 'lifted']:
                pose_file = None

            else:
                raise ValueError("Unknown goal_mode: {}".format(goal_mode))
            
            if pose_file is not None:
                assert os.path.exists(pose_file), "Pose file does not exist: {}".format(pose_file)
                with open(pose_file, 'rb') as f:
                    self.goal_list = pickle.load(f)
        
        else:
            ValueError("Unknown object_dataset: {}".format('object_dataset'))

        """
        Setup configurations for BaseEnv
        """
        if robots == 'Virtual':
            kwargs['initial_qpos'] = np.array([0.45, 0., 0.4, 0., np.pi, np.pi/2])
        else:
            kwargs['initial_qpos'] = np.array([0, -0.2743, 0, -2.263, 0, 2.008, -0.7946])
        kwargs['initialization_noise'] = None
        
    
        if friction_config == 'default':
            kwargs['table_friction']=(0.5, 0.005, 0.0001)
        elif friction_config == 'low_0.3':
            kwargs['table_friction']=(0.3, 0.005, 0.0001)
        elif friction_config == 'low_0.4':
            kwargs['table_friction'] =(0.4, 0.005, 0.0001)
        elif 'low' in friction_config:
            kwargs['table_friction'] = (float(friction_config.split('_')[1]), 0.005, 0.0001)
        elif friction_config == 'high':
            kwargs['table_friction']=(0.95, 0.05, 0.001)
        elif friction_config == 'robosuite':
            kwargs['table_friction']=(0.95, 0.3, 0.1)
        else:
            raise NotImplementedError    
        
        kwargs['table_solref'] = (0.01, 1)
        kwargs['hard_reset'] = object_name is None and object_dataset != 'cube'

        # Setup Controller
        if '.json' in self.controller:
            controller_configs = load_controller_config(custom_fpath=os.path.join(current_folder, 'controller_config', self.controller))
        else:
            controller_configs = load_controller_config(default_controller='OSC_POSE')
            
        if ignore_position_limits:
            controller_configs['position_limits'] = None
        kwargs.update(controller_configs=controller_configs)
        
        # Setup Renderer
        if renderer == 'onscreen':
            # Use onscreen renderer
            render_config = dict(has_renderer=True,
                                has_offscreen_renderer=False,
                                use_camera_obs=False)
            kwargs.update(render_config)
        elif renderer == 'offscreen':
            # Use offscreen renderer
            render_config = dict(has_renderer=False,
                                has_offscreen_renderer=True,
                                use_camera_obs=True,
                                camera_names=['leftview', 'rightview', 'agentview'],
                                camera_depths=True,
                                camera_segmentations='instance'
                                )
            kwargs.update(render_config)

        super().__init__(robots=robots,
            close_gripper=close_gripper,
            object_dataset=object_dataset,
            object_name=object_name,
            control_freq=control_freq,
            ignore_done=ignore_done,
            render_camera=render_camera,
            set_camera_names= set_camera_names,
            camera_poses = camera_poses,
            camera_quats = camera_quats, 
            **kwargs)
                
        # Action location range (after BaseEnv init)
        self.location_scale = None
        self.location_center = None
        return
    
    def _reset_internal(self):
        super()._reset_internal()
        
        # Update location scale
        table_size = self.get_table_full_size()
        self.location_scale = table_size/2 + np.array([-0.03, -0.03, 0])
        # make the range smaller for x and y, but not for z
        self.location_center = self.table_offset + np.array([0, 0, table_size[2]/2])
        # self.table_offset is at the center of the bottom of the bin.
        # self.location_center is at the center of the entire free space of the bin.
        
        if self.transparent_gripper_base:
            gid = self.sim.model.geom_name2id('gripper0_hand_collision')
            self.sim.model.geom_conaffinity[gid] = 0
            self.sim.model.geom_contype[gid] = 0    
            gid = self.sim.model.geom_name2id('gripper0_hand_visual')
            self.sim.model.geom_rgba[gid] = np.array([1, 1, 1, 0.3])
        
        if self.robots[0].gripper_type == 'default':
            eef_body_id = self.sim.model.body_name2id('gripper0_eef')
            if self.close_gripper:
                self.sim.model.body_pos[eef_body_id] = np.array([0, 0, 0.105])
            else:
                self.sim.model.body_pos[eef_body_id] = np.array([-0.053, 0, 0.105])
        elif 'panda_festo' in self.robots[0].gripper_type:
            eef_body_id = self.sim.model.body_name2id('gripper0_eef')
            self.sim.model.body_pos[eef_body_id] = np.array([0, 0, 0.135])
        
        # Hide global coordinate markers
        for site_name in ['xaxis', 'yaxis', 'zaxis']:
            sid = self.sim.model.site_name2id(site_name)
            self.sim.model.site_rgba[sid] = np.array([1, 1, 1, 0])

        return
    

    def run_simulation(self, action=None, ee_pos=None, ee_ori_mat=None, total_execution_time=None, gripper_action=None):
        # Run simulation without taking a env.step
        action_ = np.zeros(self.action_dim)
        if action is not None:
            action_[:len(action)] = action
        action = action_
        total_execution_time = self.control_timestep if total_execution_time is None else total_execution_time
        for i in range(int(total_execution_time/self.control_timestep)):
            self.robots[0].controller.set_goal(action, set_pos=ee_pos, set_ori=ee_ori_mat)
            for _ in range(int(self.control_timestep / self.model_timestep)):
                self.sim.forward()
                torques = self.robots[0].controller.run_controller()
                low, high = self.robots[0].torque_limits
                torques = np.clip(torques, low, high)
                if gripper_action is None:
                    if self.close_gripper:
                        gripper_action = 1
                    else:
                        gripper_action = -1
                self.robots[0].grip_action(gripper=self.robots[0].gripper, gripper_action=[gripper_action])
                self.sim.data.ctrl[self.robots[0]._ref_joint_actuator_indexes] = torques
                self.sim.step()
            self.maybe_render()
    
    def show_goal(self):
        goal = self.goal
        if goal is not None:
            body_id = self.sim.model.body_name2id(self.cube_target.root_body)
            self.sim.model.body_pos[body_id] = goal[:3]
            self.sim.model.body_quat[body_id] = goal[3:]
            self.sim.forward()
        return
    
    
    def hide_goal(self):
        body_id = self.sim.model.body_name2id(self.cube_target.root_body)
        self.sim.model.body_pos[body_id][2] = -0.2
        self.sim.forward()
        return

    def reset(self, object_pose=None, object_size=None, goal=None, hard_reset=None, object_name=None, start_gripper_above_obj=False,
              attempt=0, **kwargs):
        if object_size is not None:
            self.object_size_x_min = object_size[0]
            self.object_size_x_max = object_size[0]
            self.object_size_y_min = object_size[1]
            self.object_size_y_max = object_size[1]
            self.object_size_z_min = object_size[2]
            self.object_size_z_max = object_size[2]
        
        if object_name is not None:
            self.object_sampler.set_object(object_name)

        previous_hard_reset = self.hard_reset
        # Override hard_reset if specified
        self.hard_reset = hard_reset if hard_reset is not None else self.hard_reset
        super().reset()
        self.hard_reset = previous_hard_reset
        
        self.goal = goal
        # Update mujoco visual of the goal
        if self.goal is not None:
            body_id = self.sim.model.body_name2id(self.cube_target.root_body)
            self.sim.model.body_pos[body_id] = self.goal[:3]
            self.sim.model.body_quat[body_id] = self.goal[3:]
            self.sim.forward()
        else:
            body_id = self.sim.model.body_name2id(self.cube_target.root_body)
            self.sim.model.body_pos[body_id] = np.array([0,0,-1]) # hide it
            self.sim.forward()
                
        if object_pose is not None:
            self.sim.data.set_joint_qpos(self.cube.joints[0], object_pose)
            self.sim.forward()
            self.robots[0].controller.run_controller()
        
        if start_gripper_above_obj:
            obj_size = np.max(self.get_cube_size())
            
            gripper_pos = self.sim.data.get_joint_qpos(self.cube.joints[0])[:3]
            gripper_pos[2] = obj_size + self.table_offset[2] + 0.05
            self.move_to(gripper_pos)

        # Run simulation to rest the object.
        self.run_simulation(total_execution_time=self.resting_time)

        self.visualize(vis_settings=dict(env=False, grippers=True, robots=False))
        obs = self._get_observations(force_update=True)
        self.goal = self.sample_goal()
        
        if 'pointcloud' in obs.keys() and obs['pointcloud'] is None:
            assert attempt < 5, "Failed to get object point cloud after 5 attempts."
            print(f'no object point cloud during reset. try to reset again. attempt {attempt}')
            obs = self.reset(hard_reset=False, attempt=attempt+1)
        
        # Clear the frames
        self.cam_frames.clear()
        self.maybe_render()

        return obs
    
    @property
    def is_cube_grasped(self):
        return self._check_grasp(self.robots[0].gripper, object_geoms=self.cube)
    
    def get_cube_size(self):
        if hasattr(self.cube, 'scaled_size'):
            cube_size = self.cube.scaled_size
        elif hasattr(self.cube, 'size'):
            cube_size = np.array(self.cube.size) * 2
        else:
            raise ValueError("Unknown cube size.")
        
        return copy.copy(cube_size)
    
    def get_cube_pose(self):
        return self.sim.data.get_joint_qpos(self.cube.joints[0]).copy()    # [:3] pos, [3:] quat
    
    def get_gripper_pos(self):
        return self.robots[0].controller.ee_pos
    
    def get_gripper_quat(self):
        ori_mat = self.robots[0].controller.ee_ori_mat
        quat = Rotation.as_quat(Rotation.from_matrix(ori_mat))[[3, 0, 1, 2]]
        return quat

    def set_cube_pos(self, pos, quat=None):
        quat = np.array([1, 0, 0, 0]) if quat is None else quat
        pose = np.concatenate([pos, quat])
        self.sim.data.set_joint_qpos(self.cube.joints[0], pose)
        self.sim.forward()
        return
    
    def set_cube_pose(self, pose):
        self.sim.data.set_joint_qpos(self.cube.joints[0], pose)
        self.sim.forward()
        return
    
    def set_marker(self, pos):
        # body_id = self.sim.model.body_name2id('marker')
        # self.sim.model.body_pos[body_id] = pos
        # self.sim.forward()
        pass
        return
    
    def hide_marker(self):
        # body_id = self.sim.model.body_name2id('marker')
        # self.sim.model.body_pos[body_id][2] = 0
        # self.sim.forward()
        pass
        return

    """
    Movement related
    """
    def sample_random_poke(self, obs=None, points=None, normals=None):
        # Take a dictionary of obs as input, or directly take points and normals as input
        if obs is not None:
            assert points is None and normals is None
            points = obs['pointcloud'][:, :3]
            normals = obs['pointcloud'][:, 3:6]
        idx = np.random.randint(len(points))
        location = points[idx]
        normal = normals[idx]
        action = np.random.rand(self.action_dim)*2-1
        if self.planar_action:
            action[-1] = 0
        if action.dot(normal) > 0:
            action *= -1
        return location, normal, action, idx
    
    def collision_check(self, location):
        scaled_location = (location - self.location_center)/self.location_scale
        return np.all(np.abs(scaled_location) <= 1.01)
    
    def render_offscreen(self):
        assert self.has_offscreen_renderer, "No offscreen renderer is available."
        img = self.sim.render(
            camera_name=self.record_from_cam,
            width=640,
            height=480)
            # width = 1920, 
            # height = 1080)
        img = np.flipud(img)
        frame_info = {}
        return (img, frame_info)

    def maybe_render(self):
        if self.has_offscreen_renderer:
            if self.record_video:
                self.show_goal()
                frame = self.render_offscreen()
                self.cam_frames.append(frame)
                self.hide_goal()
            
        elif self.has_renderer:
            # Always render intermediate steps if there is an onscreen renderer.
            # Turn this off by setting renderer to None.
            self.show_goal()
            super().render()
            self.hide_goal()
            
    """
    Point cloud related
    """
    def _get_observations(self, force_update=False, compute_normals=True):
        obs = super()._get_observations(force_update=force_update)

        if self.has_offscreen_renderer:
            for i in range(5):
                pcd = self.get_point_cloud(obs, compute_normals=compute_normals)
                if pcd is not None or self.pcd_noise is None:
                    break
                print(f'resample pcd: {i}')

            obs.update({"pointcloud": pcd})            
        return obs
    
    def debug_segmentation(self, seg, color, obj_mask, cam):
        seg_img = np.zeros_like(color)
        color_choice = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 204),(128, 0, 128)]#red, green, blue, yellow, purple
        seg_mask = seg.reshape(color.shape[0], color.shape[1])
        for id in range(np.max(seg)):
            seg_img[seg_mask==id] = color_choice[id]
        folder_path = 'debug_double_bin_version1' + self.cube.mesh_name
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        seg_all_file = os.path.join(folder_path, cam+'_seg_all_scene_'+ self.cube.mesh_name + '.png')
        seg_file = os.path.join(folder_path, cam+'_seg_'+ self.cube.mesh_name+'.png')
        img_file = os.path.join(folder_path,cam + '_image_'+self.cube.mesh_name + '.png' )
        imageio.imsave(seg_file, (255* obj_mask).astype(np.uint8))
        imageio.imsave(img_file, color.astype(np.uint8))
        imageio.imsave(seg_all_file, seg_img.astype(np.uint8))
        print('save camera image for '+cam)

    def get_point_cloud(self, obs, compute_normals=True):
        box_in_the_view = True
        pc_list = []
        pc_color_list = []
        object_mask_list = []
        # gripper_mask_list = []    
        for cam in self.camera_names:
            color = obs[cam + '_image'][-1::-1]
            depth = obs[cam + '_depth'][-1::-1][:, :, 0]
            if self.pcd_noise is not None:
                depth = dropout_random_ellipses(depth_img=depth, dropout_mean=self.pcd_noise[0], 
                                                gamma_shape=self.pcd_noise[1], gamma_scale=self.pcd_noise[2])

            # Segmentation from MuJoCo
            seg = obs[cam + '_segmentation_instance'][-1::-1]
            obj_mask = (seg == 1).reshape(color.shape[0], color.shape[1])
        

            depth = convert_depth(self, depth)
            pc = get_point_cloud(self, depth, camera_name=cam)
            if self.pcd_noise is not None:
                pc = add_additive_noise_to_xyz(pc.reshape(color.shape), gaussian_scale_range=[0.0, self.pcd_noise[3]]).reshape(-1, 3)
            pc_color = color.reshape(-1, 3).astype(np.float64)/256

            pc_list.append(pc)
            pc_color_list.append(pc_color)
            object_mask_list.append(obj_mask.reshape(-1))

        pc = np.concatenate(pc_list)
        pc_color = np.concatenate(pc_color_list)
        object_mask = np.concatenate(object_mask_list)
     
        # Remove unnecessary points outside of the bin
        if self.exclude_wall_pcd:
            mask = self.within_bin(pc, margin=np.array([0.01, 0.01, 0]), check_z=True)
        else:
            mask = self.within_arena(pc, margin=np.array([-0.01, -0.01, -0.03]), check_z=True)
        pc = pc[mask]
        pc_color = pc_color[mask]
        object_mask = object_mask[mask]

        # Remove points outside of the bounding box
        obj_bbox_mask = self.within_boundingbox(pc)
        obj_mask = obj_bbox_mask * object_mask
      
        


        if obj_mask.sum() <= 10:
            box_in_the_view = False
       
        if box_in_the_view:
            try:
                # Downsample object pcd
                if self.move_gripper_to_object_pcd:
                    finger1 = self.sim.data.get_joint_qpos('gripper0_finger_joint1')
                    
                    if  'panda_festo' in self.robots[0].gripper_type:
                        gripper_open_thresh = 0.002
                    elif self.robots[0].gripper_type=='default':
                        gripper_open_thresh = 0.001
                    # if finger1 > gripper_open_thresh:
                    if self.is_cube_grasped: ## only when grasped will include gripper pcds as object pcds
                        gripper_bbox_mask = self.within_gripper_bbox(pc)
                        obj_mask = gripper_bbox_mask + obj_mask
                object_pcd = o3d.geometry.PointCloud()
                object_pcd.points = o3d.utility.Vector3dVector(pc[obj_mask])
                object_pcd.colors = o3d.utility.Vector3dVector(pc_color[obj_mask])
        
                object_pcd = object_pcd.voxel_down_sample(self.voxel_downsample_size)
                if compute_normals:
                    object_pcd.estimate_normals()
                    object_pcd.orient_normals_consistent_tangent_plane(10)
    
            except :
                box_in_the_view = False
        
        # Downsample background pcd
        bg_pcd = o3d.geometry.PointCloud()
        bg_mask = ~obj_mask
        bin_lower_z_mask = pc[:, 2] >= self.table_offset[2]
        ## exclude the error projection point beneath the bin and exclude those object points that are not correctly segmented
        bg_mask *= ~obj_bbox_mask * bin_lower_z_mask 
        if self.exclude_gripper_pcd:
            gripper_bbox_mask = self.within_gripper_bbox(pc)
            bg_mask *= ~gripper_bbox_mask
        bg_pcd.points = o3d.utility.Vector3dVector(pc[bg_mask])
        bg_pcd.colors = o3d.utility.Vector3dVector(pc_color[bg_mask])
        if compute_normals:
            bg_pcd.estimate_normals()
            bg_pcd.orient_normals_consistent_tangent_plane(10)
        bg_voxel_size = self.voxel_downsample_size * 2
        if self.exclude_wall_pcd:
            bg_voxel_size /= 2  # Allow denser background pcd if wall is removed
        bg_pcd = bg_pcd.voxel_down_sample(bg_voxel_size)
        if box_in_the_view:
            object_pcd_points = np.asarray(object_pcd.points)
            object_pcd_colors = np.asarray(object_pcd.colors)
            if compute_normals:
                object_pcd_normals = np.asarray(object_pcd.normals)
            object_pcd_seg = np.ones((len(object_pcd_points), 1))
        else:
            return None
        background_pcd_points = np.asarray(bg_pcd.points)
        background_pcd_colors = np.asarray(bg_pcd.colors)
        if compute_normals:
            background_pcd_normals = np.asarray(bg_pcd.normals)
        background_pcd_seg = np.zeros((len(background_pcd_points), 1))

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

    def within_arena(self, points, margin=None, check_z=False):
        # Check if the points are within the box on the table
        # Input: N x 3
        if margin is None:
            margin = np.zeros(3)
        table_size = self.get_table_full_size()
        x_min = self.table_offset[0] - table_size[0]/2 + margin[0]
        x_max = self.table_offset[0] + table_size[0]/2 - margin[0]
        y_min = self.table_offset[1] - table_size[1]/2 + margin[1]
        y_max = self.table_offset[1] + table_size[1]/2 - margin[1]
        valid = (points[:, 0] >= x_min) * (points[:, 0] <= x_max) * \
                (points[:, 1] >= y_min) * (points[:, 1] <= y_max)
        if check_z:
            z_min = self.table_offset[2] + margin[2]
            valid *= (points[:, 2] >= z_min) # * (points[:, 2] <= z_max)
        return valid

    def within_bin(self, points, margin=None, check_z=False):
        # Check if the points are within either bin
        # Input: N x 3
        if margin is None:
            margin = np.zeros(3)
        
        bin_half_sizes = self.get_bin_half_sizes()
        bin_centers = self.get_bin_centers()
        bin_centers += self.table_offset
        bin_centers[:, 2] -= self.get_bin_height()   # bin_centers are in the world frame, converted to pc frame
        
        valid = np.zeros(len(points), dtype=np.bool)
        for bin_center, bin_half_size in zip(bin_centers, bin_half_sizes):
            bin_min = bin_center - bin_half_size + margin
            bin_max = bin_center + bin_half_size - margin

            bin_max[2] = np.inf
            if not check_z:
                bin_min[2] = -np.inf
            valid += np.all((points >= bin_min) * (points <= bin_max), axis=-1)
        return valid
    
    def get_object_pose(self, format="mat"):
        # Notice: the cube pos and quat may not be updated in the env 
        # if _get_observation is not called
        pose = self.get_cube_pose.copy()
        p, q = pose[:3], pose[3:]
        if format == "mat":
            return to_pose_mat(p, q, input_wxyz=True)
        elif format == "vector":
            return np.concatenate([p, q])
    
   

        
    def get_obj_dim(self):
        # return np.max(self.get_cube_size()) * 2
        return self.get_cube_size()*2
    
    def within_gripper_bbox(self, points):
        gripper_pose = self.get_gripper_pose(format='mat')
        transform = inv_pose_mat(gripper_pose)
        transformed_pcd = np.concatenate([points, np.ones((len(points), 1))], axis=1)
        transformed_pcd = np.dot(transform, transformed_pcd.T).T[:, :3]
        
        # gripper_dims = [0.8, 0.2, 0.3]
        if self.robots[0].gripper_type == 'default':
            finger1 = self.sim.data.get_joint_qpos('gripper0_finger_joint1')
            gripper_dims=[0.03 + np.abs(finger1) , 0.014, 0.03]
            obj_x_dim = gripper_dims[0]
            obj_y_dim = gripper_dims[1]
            obj_z_dim = gripper_dims[2]
            z_offset = 0.028
        elif 'panda_festo' in self.robots[0].gripper_type:
            finger1 = self.sim.data.get_joint_qpos('gripper0_finger_joint1')
            gripper_dims=[0.025 + np.abs(finger1) , 0.014, 0.038]
            obj_x_dim = gripper_dims[0]
            obj_y_dim = gripper_dims[1]
            obj_z_dim = gripper_dims[2]
            z_offset = 0.037
        mask = (np.abs(transformed_pcd[:, 0]) <= obj_x_dim) * \
                (np.abs(transformed_pcd[:, 1]) <= obj_y_dim) * \
                (transformed_pcd[:, 2] <= obj_z_dim - z_offset) *\
                (transformed_pcd[:, 2] >= - obj_z_dim - z_offset)
        return mask
        
    def within_boundingbox(self, points):
        # Transform the points into the object frame
        obj_pose = self.get_object_pose(format='mat')
        transform = inv_pose_mat(obj_pose)
        transformed_pcd = np.concatenate([points, np.ones((len(points), 1))], axis=1)
        transformed_pcd = np.dot(transform, transformed_pcd.T).T[:, :3]
        
        obj_dims = self.get_obj_dim()
        obj_x_dim = obj_dims[0]/2 + 0.002
        obj_y_dim = obj_dims[1]/2 + 0.002
        obj_z_dim = obj_dims[2]/2 + 0.002
        mask = (np.abs(transformed_pcd[:, 0]) <= obj_x_dim) * \
                (np.abs(transformed_pcd[:, 1]) <= obj_y_dim) * \
                (np.abs(transformed_pcd[:, 2]) <= obj_z_dim) 
        return mask
    
    """
    Others
    """
    def reward(self, action=None):
        return 0
    
    def sample_goal(self):
        """
        Returns goal as a vector of size 7:
        [x, y, z, quat_w, quat_x, quat_y, quat_z]
        """
        scale = None
        physics_check = True
        if self.goal_list is not None:
            if hasattr(self, 'cube') and hasattr(self.cube, 'mesh_name'):
                mesh_name = self.cube.mesh_name
                scale = self.cube.scale
                # Find the goals with the same scale
                goal_scales = self.goal_list[mesh_name]['scales']
                scale_mask = np.isclose(np.linalg.norm(goal_scales - scale, axis=1), 0, rtol=1e-2)
                if not np.any(scale_mask):
                    # raise ValueError('No goals with the same scale!')
                    # print('No goals with the same scale!')
                    scale = np.min(goal_scales)
                    scale_mask = np.isclose(np.linalg.norm(goal_scales - np.array([scale, scale, scale]), axis=1), 0, rtol=1e-2)
                goals = self.goal_list[mesh_name]['poses'][scale_mask]
            else:
                mesh_name = 'cube'
                goals = self.goal_list[mesh_name]['poses']
            
            goal_idx = np.random.choice(len(goals))
            goal = goals[goal_idx]
            
            
            # default bin size when generating the dataset
            original_scale = np.array([0.45, 0.54, 0.107])
            new_scale = self.get_table_full_size()
            if mesh_name == 'cube' and not np.all(np.isclose(original_scale, new_scale)):
                goal[:2] = ((goal[:3] - self.table_offset)*new_scale/original_scale*0.5 + self.table_offset)[:2]

            # add variation to the goals when there is no wall interactions
            if self.goal_mode in {"any", "any_var_size"}:
                bin_centers, bin_scales = self.get_bin_centers(), self.get_bin_half_sizes()
                
                if self.goal_on_oppo_side and len(bin_centers) == 2:
                    cube_y = self.get_cube_pose()[1]
                    bin_idx = 0 if cube_y > 0 else 1
                else:
                    bin_idx = np.random.randint(len(bin_centers))
                
                bin_center, bin_scale = bin_centers[bin_idx], bin_scales[bin_idx]
                max_obj_dim  = np.linalg.norm(self.get_cube_size())
                pos_range_lb = bin_center + self.table_offset + 0.8*np.clip(-bin_scale + max_obj_dim, a_min = None, a_max = 0)# 0.75
                pos_range_ub = bin_center + self.table_offset + 0.8*np.clip( bin_scale  - max_obj_dim, a_min = 0, a_max = None)
                goal_xy = np.random.uniform(pos_range_lb, pos_range_ub)[:2]         
                goal[:2] = goal_xy
                goal[1] = bin_center[1] + self.table_offset[1]
                

            # goal = to_pose_mat(goal[:3], goal[3:])

        elif self.goal_mode == 'fixed':
            # goal_pos = self.location_center
            bin_centers, bin_scales = self.get_bin_centers(), self.get_bin_half_sizes()
            if self.goal_on_oppo_side and len(bin_centers) == 2:
                    cube_y = self.get_cube_pose()[1]
                    bin_idx = 0 if cube_y > 0 else 1
            else:
                    bin_idx = np.random.randint(len(bin_centers))
                
            bin_center, bin_scale = bin_centers[bin_idx], bin_scales[bin_idx]
            goal_pos = bin_center + self.table_offset 

            goal_pos[2] = self.get_cube_pose()[2] + self.table_offset[2]
            goal_ori = np.array([1., 0., 0., 0.])
            goal = np.concatenate([goal_pos, goal_ori])
            # goal = to_pose_mat(goal_pos, goal_ori)
        
        elif self.goal_mode == 'fixed_off_center':
            cube_z = self.get_cube_pose()[2]
            goal = np.array([0.5, 0.1, cube_z, 1., 0., 0., 0.])
            # goal = to_pose_mat(np.array([0.5, 0.1, obs['cube_pos'][2]]), np.array([1., 0., 0., 0.]))
        
        elif self.goal_mode == 'off_center':
            y = np.random.uniform(0.0, 0.2)
            cube_z = self.get_cube_pose()[2]
            goal = np.array([0.5, y, cube_z, 1., 0., 0., 0.])
            # goal = to_pose_mat(np.array([0.5, y, obs['cube_pos'][2]]), np.array([1., 0., 0., 0.]))
            
        elif self.goal_mode == 'translation':
            # Same pose as the object, with x, y randomized 
            # in the range of the bin (scaled by 0.6)
            cube_pose = self.get_cube_pose()
            bin_centers, bin_scales = self.get_bin_centers(), self.get_bin_half_sizes()
            
            if self.goal_on_oppo_side and len(bin_centers) == 2:
                cube_y = self.get_cube_pose()[1]
                bin_idx = 0 if cube_y > 0 else 1
            else:
                bin_idx = np.random.randint(len(bin_centers))
            
            bin_center, bin_scale = bin_centers[bin_idx], bin_scales[bin_idx]
            pos_range_lb = bin_center - bin_scale * 0.75 + self.table_offset
            pos_range_ub = bin_center + bin_scale * 0.75 + self.table_offset
                    
            goal_pos = np.random.uniform(pos_range_lb, pos_range_ub)
            goal_pos[2] = cube_pose[2]
            goal_ori = cube_pose[3:]
            goal = np.concatenate([goal_pos, goal_ori])
            # goal = to_pose_mat(goal_pos, goal_ori)
        
        elif self.goal_mode == 'upright':
            # Sample a x,y location
            pos_range_lb = self.location_center - self.location_scale * 0.6
            pos_range_ub = self.location_center + self.location_scale * 0.6
            goal_pos = np.random.uniform(pos_range_lb, pos_range_ub)
            goal_pos[2] = self.get_cube_size()[2] + self.table_offset[2]
            
            # Sample a planar rotation
            # quat=cos(a/2),sin(a/2)⋅(x,y,z)
            z_angle = np.random.uniform(360)
            goal_ori = np.array([np.cos(z_angle/2), 0, 0, np.sin(z_angle/2)])
            # goal = to_pose_mat(goal_pos, goal_ori)
            goal = np.concatenate([goal_pos, goal_ori])

        elif self.goal_mode == "lifted":
            cube_pose = self.get_cube_pose()
            goal_pos = cube_pose[:3]
            goal_pos[2] += 0.1
            goal_ori = cube_pose[3:]
            goal = np.concatenate([goal_pos, goal_ori])

            physics_check = False
            
        else:
            raise NotImplementedError

        # Run simulation to set the goal to a physically feasible pose
        if physics_check:
            current_pose = self.get_cube_pose()
            self.set_cube_pose(goal)
            self.run_simulation(total_execution_time=self.resting_time)
            goal = self.get_cube_pose()

            self.set_cube_pose(current_pose)    # set back the pose

        return goal

def visualize_point_mask(points, mask):
    inlier_points = points[mask]
    inlier_points = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(inlier_points))
    inlier_points.paint_uniform_color([0.967, 0.643, 0.31])
    inlier_points.estimate_normals()
    outlier_points = points[~mask]
    outlier_points = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(outlier_points))
    outlier_points.paint_uniform_color([0.31, 0.867, 0.967])
    outlier_points.estimate_normals()

    o3d.visualization.draw_geometries([inlier_points, outlier_points])

def draw_boundingbox( obj_dim, obj_pos):
    z_offset = 0.037
    points = [
        [obj_dim[0], -obj_dim[1], obj_dim[2]-z_offset],
        [obj_dim[0], obj_dim[1], obj_dim[2]-z_offset],
        [-obj_dim[0] ,obj_dim[1], obj_dim[2]-z_offset],
        [-obj_dim[0], -obj_dim[1], obj_dim[2]- z_offset ],
        [-obj_dim[0], -obj_dim[1], -obj_dim[2]- z_offset],
        [-obj_dim[0], obj_dim[1], -obj_dim[2] - z_offset],
        [obj_dim[0], obj_dim[1], -obj_dim[2] - z_offset],
        [obj_dim[0], -obj_dim[1], -obj_dim[2] - z_offset]
    ]
    transformed_pcd = np.concatenate([points, np.ones((len(points), 1))], axis=1)
    transformed_pcd = np.dot(obj_pos, transformed_pcd.T).T[:, :3]
    lines = [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,7],[1,6],[2,5],[3,4]]
    colors = [[0,1,0 ] for i in range(len(lines))]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(transformed_pcd)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set

if __name__ == "__main__":
    env = PokeEnv(object_dataset="housekeep_all")
