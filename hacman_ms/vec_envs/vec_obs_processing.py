import numpy as np
import gym
import gym.spaces as spaces
import torch as th
from copy import deepcopy

from hacman.utils.transformations import transform_point_cloud, to_pose_mat, sample_idx, voxel_downsample
from hacman.envs.vec_env_wrappers.vec_obs_processing import VecObsProcessing

class MSVecObsProcessing(VecObsProcessing):
    """
    MSVecObsWrapper defines the observation space to be the same as HACMan's.
    It contains implementation to process the vectorized observations (stacked) from ManiSkill2.
    """
    def __init__(self, observation_space, **kwargs):
        # Remove the dictionary obs keys from the observation space
        # Since sb3 does not support nested dictionary observation space
        obs_space = deepcopy(observation_space)
        obs_space.spaces.pop("agent")
        obs_space.spaces.pop("extra")
        obs_space.spaces.pop("pointcloud")
        super().__init__(obs_space, **kwargs)

    def observation(self, raw_obs):
        # Remove unnecessary keys
        raw_obs.pop("agent")
        raw_obs.pop("extra") 
        # PCDs
        pointcloud = raw_obs.pop("pointcloud")
        xyzw, rgb = pointcloud["xyzw"], pointcloud["rgb"]
        segs = pointcloud["Segmentation"][..., 1]
        # Assuming the input to be torch tensor
        if not isinstance(xyzw, th.Tensor):
            xyzw = th.from_numpy(xyzw)
            rgb = th.from_numpy(rgb)
            segs = th.from_numpy(segs.astype(np.int64))
        xyz = xyzw[..., :3]
        all_pcds = th.cat([xyz, rgb, segs.unsqueeze(-1)], dim=-1)
        raw_obs.update({"pointcloud": all_pcds,})

        return super().observation(raw_obs)