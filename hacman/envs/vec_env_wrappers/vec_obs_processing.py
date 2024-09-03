import numpy as np
from typing import Optional, Sequence
import gym
import gym.spaces as spaces
from copy import deepcopy
import torch as th
import open3d as o3d

from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from hacman.utils.transformations import transform_point_cloud, to_pose_mat, sample_idx, voxel_downsample

class VecObsProcessing():
    """
    VecObsWrapper defines the observation space to be the same as HACMan's.
    """
    def __init__(self,
                 observation_space,
                 object_pcd_size=400, 
                 background_pcd_size=400,
                 background_clip_radius=2.0,
                 downsample = False,
                 voxel_downsample_size=0.01,
                 compute_normals=False,
                 skip_processing=False,
                 primitives=[]):
        self.object_pcd_size = object_pcd_size
        self.background_pcd_size = background_pcd_size
        self.compute_normals = compute_normals
        self.observation_space = self.init_observation_space(
            observation_space, object_pcd_size, background_pcd_size, primitives)
        self.background_clip_radius = background_clip_radius
        self.downsample = downsample
        self.skip_processing = skip_processing
        self.voxel_downsample_size = voxel_downsample_size
    
    def init_observation_space(self, obs_space, object_pcd_size, background_pcd_size, primitives):
        # Add the HACMan space
        new_obs_space = deepcopy(obs_space)
        if "pointcloud" in new_obs_space.spaces.keys():
            new_obs_space.spaces.pop("pointcloud")
        space_dict = new_obs_space.spaces
        extra_spaces = {
            "object_pcd_points": gym.spaces.Box(-np.inf, np.inf, (object_pcd_size, 3)),
            "background_pcd_points": gym.spaces.Box(-np.inf, np.inf, (background_pcd_size, 3)),
        }
        if self.compute_normals:
            extra_spaces["object_pcd_normals"] = gym.spaces.Box(-np.inf, np.inf, (object_pcd_size, 3))
            extra_spaces["background_pcd_normals"] = gym.spaces.Box(-np.inf, np.inf, (background_pcd_size, 3))
        # Create the new observation space
        space_dict.update(extra_spaces)
        return spaces.Dict(space_dict)

    def observation(self, raw_obs):
        """
        Processes the observation before returning it.

        Args:
            raw_obs: The raw observation from the environment.
            downsample: downsample includes voxel downsample followed by random sampling.
        """
        # Keep part of the original observations
        # {
        #     "poke_idx": raw_obs["poke_idx"],
        #     "prim_idx": raw_obs["prim_idx"],
        #     "object_pose": raw_obs["object_pose"],
        #     "goal_pose": raw_obs["goal_pose"],
        #     "prim_groundings": raw_obs["prim_groundings"]}
        if self.skip_processing:
            return raw_obs
        
        all_pcds = raw_obs.pop("pointcloud")
        # Assuming the input to be torch tensor
        if not isinstance(all_pcds, th.Tensor):
            all_pcds = th.from_numpy(all_pcds)
        
        all_object_ids, all_background_ids = raw_obs["object_ids"], raw_obs["background_ids"]
        if not isinstance(all_object_ids, th.Tensor):
            all_object_ids = th.from_numpy(all_object_ids)
            all_background_ids = th.from_numpy(all_background_ids)
            all_goal_poses = th.from_numpy(raw_obs["goal_pose"])
        
        # Handle the case of non-batch input (happens when input comes from terminal observation)
        non_batch_input = (all_pcds.ndim == 2)
        if non_batch_input:
            all_pcds = all_pcds.unsqueeze(0)
            all_object_ids = all_object_ids.unsqueeze(0)
            all_background_ids = all_background_ids.unsqueeze(0)
            all_goal_poses = all_goal_poses.unsqueeze(0)
        
        # Segment object points and bacground
        device = all_pcds.device
        num_envs = all_pcds.shape[0]
        final_object_pcds, final_bg_pcds = [], []
        for i in range(num_envs):
            pcd = all_pcds[i]
            object_ids = all_object_ids[i].to(device=device)
            background_ids = all_background_ids[i].to(device=device)

            # Skip downsampling if requested
            if not self.downsample:
                object_points = pcd[:self.object_pcd_size]
                bg_points = pcd[self.object_pcd_size:]
                final_object_pcds.append(object_points)
                final_bg_pcds.append(bg_points)
                continue

            downsampled_pcd = voxel_downsample(pcd, self.voxel_downsample_size)
            points, seg = downsampled_pcd[:, :-1], downsampled_pcd[:, -1]
            point_dim = points.shape[-1]
            object_points = points[th.isin(seg, object_ids)]
            bg_points = points[th.isin(seg, background_ids)]
            
            # Filter out far away points
            if self.background_clip_radius is not None:
                goal_pos = all_goal_poses[i, :3, 3].to(device=device)
                mask = (th.linalg.norm(bg_points[..., :3] - goal_pos, dim=-1) < self.background_clip_radius)
                bg_points = bg_points[mask]

            if len(object_points) < 2 or len(bg_points) < 2:
                if len(object_points) < 2:
                    Warning(f"Object points are too few: {len(object_points)}.")
                    object_points = th.zeros((self.object_pcd_size, point_dim), dtype=bg_points.dtype, device=device)
                if len(bg_points) < 2:
                    Warning(f"Background points are too few: {len(bg_points)}.")
                    bg_points = th.zeros((self.background_pcd_size, point_dim), dtype=bg_points.dtype, device=device)

                if "image" in raw_obs.keys():   # works for MS only
                    cameras = raw_obs["image"].keys()
                    pictures = {
                        "colors": [raw_obs["image"][cam]["Color"][i].cpu().numpy() for cam in cameras],
                        "segs": [raw_obs["image"][cam]["Segmentation"][i].cpu().numpy() for cam in cameras],
                    }
                    np.savez("debug_scene.npz", **pictures)
                    # raise ValueError(f"Object points are too few: {len(object_points)}. See debug_scene.npz for details.")
                    Warning("See debug_scene.npz for details.")

            # Sample points to a fixed length
            sampled_idx = sample_idx(len(object_points), self.object_pcd_size)
            final_object_pcds.append(object_points[sampled_idx])
            sampled_idx = sample_idx(len(bg_points), self.background_pcd_size)
            final_bg_pcds.append(bg_points[sampled_idx])
        
        final_object_pcds = th.stack(final_object_pcds, dim=0)
        final_bg_pcds = th.stack(final_bg_pcds, dim=0)

        if non_batch_input:
            final_object_pcds = final_object_pcds.squeeze(0)
            final_bg_pcds = final_bg_pcds.squeeze(0)
        # NOTE: SB3 does not support GPU tensors, so we transfer them to CPU.
        # For other RL frameworks that natively support GPU tensors, this step is not necessary.
        # Process the point cloud
        if isinstance(all_pcds, th.Tensor):
            final_object_pcds = final_object_pcds.to(device="cpu", non_blocking=True)
            final_bg_pcds = final_bg_pcds.to(device="cpu", non_blocking=True)
        
        raw_obs["object_pcd_points"] = final_object_pcds[..., :3]
        raw_obs["object_pcd_normals"] = final_object_pcds[..., 6:9]
        raw_obs["background_pcd_points"] = final_bg_pcds[..., :3]
        raw_obs["background_pcd_normals"] = final_bg_pcds[..., 6:9]
        
        return raw_obs
    
    def info(self, infos):
        # When using SB3 SubprocVecEnv, the terminal observation is not processed by the observation function.
        # We process it here. 
        term_env_idxs = []
        stacked_term_obs = []
        for i in range(len(infos)):
            if "terminal_observation" in infos[i].keys():
                # Non-batched version
                # terminal_obs = self.observation(infos[i]["terminal_observation"])
                # infos[i]["terminal_observation"] = terminal_obs

                # Batched version
                term_env_idxs.append(i)
                stacked_term_obs.append(infos[i]["terminal_observation"])
        
        if len(term_env_idxs) > 0:
            stacked_term_obs = stack_obs(stacked_term_obs, self.sub_observation_space)
            processed_term_obs = self.observation(stacked_term_obs)
            obs_keys = processed_term_obs.keys()
            for i, env_idx in enumerate(term_env_idxs):
                sub_term_obs = {key: processed_term_obs[key][i] for key in obs_keys}
                infos[env_idx]["terminal_observation"] = sub_term_obs
        
        return infos

def stack_obs(obs: Sequence, space: spaces.Space, buffer: Optional[np.ndarray] = None):
    if isinstance(space, spaces.Dict):
        ret = {}
        for key in space:
            _obs = [o[key] for o in obs]
            _buffer = None if buffer is None else buffer[key]
            ret[key] = stack_obs(_obs, space[key], buffer=_buffer)
        return ret
    elif isinstance(space, spaces.Box):
        # print(obs)
        return np.stack(obs, out=buffer)
    else:
        raise NotImplementedError(type(space))

class PCDSubprocVecEnv(VecObsProcessing, SubprocVecEnv):
    """
    A vectorized environment that processes the individual sub-environments in parallel using the Stable Baselines 3
    SubprocVecEnv.

    This environment wrapper inherits from the `SubprocVecEnv` class and adds observation processing functionality.

    Args:
        env_fns (list): A list of functions that create the individual sub-environments.
        start_method (str, optional): The multiprocessing start method to use. Defaults to "fork".
        object_pcd_size (int, optional): The size of the point cloud data for objects. Defaults to 400.
        background_pcd_size (int, optional): The size of the point cloud data for the background. Defaults to 400.

    Methods:
        reset(self): Resets the environment and returns the processed observation.
        step_wait(self): Waits for the asynchronous step to complete and returns the processed observation, reward, done
            flag, and info dictionary. Note: SubprocVecEnv calls step_async before step_wait.
    """
    def __init__(self, env_fns,
                 start_method=None,
                 object_pcd_size=400, 
                 background_pcd_size=400,
                 background_clip_radius=0.8,
                 voxel_downsample_size=0.01,
                 primitives=[],):
        SubprocVecEnv.__init__(self, env_fns, start_method)
        self.sub_observation_space = deepcopy(self.observation_space)

        VecObsProcessing.__init__(self,
                                self.sub_observation_space,
                                object_pcd_size=object_pcd_size, 
                                background_pcd_size=background_pcd_size,
                                background_clip_radius=background_clip_radius,
                                downsample=False,
                                voxel_downsample_size=voxel_downsample_size,
                                primitives=primitives)
    
    def step_wait(self):
        # Note: SubprocVecEnv calls step_async before step_wait.
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        raw_obs, rews, dones, infos = zip(*results)
        stacked_obs = stack_obs(raw_obs, self.sub_observation_space)
        obs = self.observation(stacked_obs)
        return obs, np.stack(rews), np.stack(dones), self.info(infos)
    
    def reset(self):
        for remote in self.remotes:
            remote.send(("reset", None))
        raw_obs = [remote.recv() for remote in self.remotes]
        stacked_obs = stack_obs(raw_obs, self.sub_observation_space)
        obs = self.observation(stacked_obs)
        return obs


class PCDDummyVecEnv(VecObsProcessing, DummyVecEnv):
    """
    A vectorized environment that processes the individual sub-environments in parallel using the Stable Baselines 3
    DummyVecEnv.

    This environment wrapper inherits from the `DummyVecEnv` class and adds observation processing functionality.

    Args:
        env_fns (list): A list of functions that create the individual sub-environments.
        object_pcd_size (int, optional): The size of the point cloud data for objects. Defaults to 400.
        background_pcd_size (int, optional): The size of the point cloud data for the background. Defaults to 400.

    Methods:
        reset(self): Resets the environment and returns the processed observation.
        step_wait(self): Waits for the asynchronous step to complete and returns the processed observation, reward, done
            flag, and info dictionary. Note: SubprocVecEnv calls step_async before step_wait.
    """
    def __init__(self, env_fns,
                 object_pcd_size=400, 
                 background_pcd_size=400,
                 background_clip_radius=0.8,
                 voxel_downsample_size=0.01,
                 skip_processing=False,
                 primitives=[],):
        DummyVecEnv.__init__(self, env_fns)
        self.sub_observation_space = deepcopy(self.observation_space)
        VecObsProcessing.__init__(self,
                                self.sub_observation_space,
                                object_pcd_size=object_pcd_size, 
                                background_pcd_size=background_pcd_size,
                                background_clip_radius=background_clip_radius,
                                downsample=False,
                                skip_processing=skip_processing,
                                voxel_downsample_size=voxel_downsample_size,
                                primitives=primitives)
    
    def step_wait(self):
        for env_idx in range(self.num_envs):
            obs, self.buf_rews[env_idx], self.buf_dones[env_idx], self.buf_infos[env_idx] = self.envs[env_idx].step(
                self.actions[env_idx]
            )
            if self.buf_dones[env_idx]:
                # save final observation where user can get it, then reset
                if "terminal_observation" not in self.buf_infos[env_idx].keys():
                    self.buf_infos[env_idx]["terminal_observation"] = obs
                    obs = self.envs[env_idx].reset()
            self._save_obs(env_idx, obs)
        stacked_obs = self._obs_from_buf()
        processed_obs = self.observation(stacked_obs)
        return (processed_obs, np.copy(self.buf_rews), np.copy(self.buf_dones), self.info(deepcopy(self.buf_infos)))
    
        # stacked_obs = stack_obs(raw_obs, self.sub_observation_space)
        # obs = self.observation(stacked_obs)
        # return obs, np.stack(rews), np.stack(dones), infos
    
    def reset(self):
        for env_idx in range(self.num_envs):
            obs = self.envs[env_idx].reset()
            self._save_obs(env_idx, obs)
        stacked_obs = self._obs_from_buf()
        processed_obs = self.observation(stacked_obs)
        return processed_obs