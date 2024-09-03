import gym
import gym.spaces as spaces
from copy import deepcopy
from scipy.spatial.transform import Rotation
import numpy as np

from hacman.envs.sim_envs.base_env import BaseEnv, RandomLocation
from hacman.utils.transformations import transform_point_cloud, to_pose_mat

class BaseObsWrapper(gym.ObservationWrapper):
    """
    Base class for observation wrappers.
    IMPORTANT: It assumes that the observation DOES not contain PCDs!

    Args:
        env (gym.Env): The environment to wrap.

    Methods:
        process_observation(self, observation): Processes the observation before returning it.
    """

    def __init__(self, env,
                 object_pcd_size=400, 
                 background_pcd_size=400,
                 record_video=False,
                 reward_scale=1.0,
                 reward_aggregation='final',    # Options: None, "average", "stage_final", "stage_average"
                 **kwargs,
                ):
        super().__init__(env)
        self.object_pcd_size = object_pcd_size
        self.background_pcd_size = background_pcd_size
        # self.prims = prims
        self.observation_space = self.init_observation_space(self.env.observation_space)
        self.reward_scale = reward_scale
        self.reward_aggregation = reward_aggregation
        self.action_space = gym.spaces.Box(-1, 1, (3,))
        self.deterministic = False

        self.cam_frames = []
        self.record_video = record_video 

        if len(kwargs) > 0:
            Warning("Unused kwargs: {}".format(kwargs))

    def init_observation_space(self, obs_space: spaces.Dict):
        space_dict = obs_space.spaces
        # Add the HACMan spaces
        extra_spaces = {
            "object_pose": gym.spaces.Box(-np.inf, np.inf, (4,4)),
            "goal_pose": gym.spaces.Box(-np.inf, np.inf, (4,4)),
            "gripper_pose": gym.spaces.Box(-np.inf, np.inf, (4,4)),
            "object_ids": gym.spaces.Box(-np.inf, np.inf, (5,)),
            "background_ids": gym.spaces.Box(-np.inf, np.inf, (5,)),
            "object_dim": gym.spaces.Box(-np.inf, np.inf, (1,)),
            }
        space_dict.update(extra_spaces)

        assert hasattr(self.env, 'get_goal_pose'), "Environment must have a get_goal_pose function"
        assert hasattr(self.env, 'get_object_pose'), "Environment must have a get_object_pose function"
        assert hasattr(self.env, 'get_gripper_pose'), "Environment must have a get_gripper_pose function"
        assert hasattr(self.env, 'get_object_dim'), "Environment must have a get_object_dim function"
        assert hasattr(self.env, 'get_segmentation_ids'), "Environment must have a get_segmentation_ids function"

        # Create the new observation space
        return spaces.Dict(space_dict)
    
    def observation(self, obs):
        """
        Processes the observation before returning it.

        Args:
            observation (np.ndarray): The observation to process.

        Returns:
            np.ndarray: The processed observation.
        """
        extra_obs = {}
        
        extra_obs['goal_pose'] = self.unwrapped.get_goal_pose('mat')   
        extra_obs['object_pose'] = self.unwrapped.get_object_pose('mat')
        extra_obs['gripper_pose'] = self.unwrapped.get_gripper_pose('mat')
        extra_obs['object_dim'] = np.array([self.unwrapped.get_object_dim()])

        seg_ids = self.format_seg_ids()
        extra_obs['object_ids'] = seg_ids['object_ids']
        extra_obs['background_ids'] = seg_ids['background_ids']

        obs.update(extra_obs)
        # self.prev_obs = obs
        
        return obs
    
    def aggregate_rewards(self, all_rewards, mode):
        if "post_contact" in mode:
            all_rewards = all_rewards[1:]
        
        # Aggregate method
        if "final" in mode:
            aggregate_fn = lambda rewards: rewards[-1]
        elif "average" in mode:
            aggregate_fn = np.mean
        elif "max" in mode:
            aggregate_fn = max
        else:
            raise NotImplementedError
        
        if "stage" in mode: # Aggregate by stages
            stage_rewards = []
            for rewards in all_rewards:
                stage_rewards.append(aggregate_fn(rewards))
            return np.mean(stage_rewards)
        else:               # Aggregate all rewards
            flat_all_rewards = []
            for rewards in all_rewards:
                flat_all_rewards.extend(rewards)
            return aggregate_fn(flat_all_rewards)


    def format_seg_ids(self):
        seg_ids = self.unwrapped.get_segmentation_ids()
        object_ids = seg_ids['object_ids']
        background_ids = seg_ids['background_ids']

        # Pad the ids to be of length 5
        padded_object_ids, padded_background_ids = np.ones(5) * -1, np.ones(5) * -1
        padded_object_ids[:len(object_ids)] = object_ids
        padded_background_ids[:len(background_ids)] = background_ids
        return {"object_ids": padded_object_ids, "background_ids": padded_background_ids}

    def evaluate_flow(self, obs):
        # Compute the flow
        object_pos = self.unwrapped.get_object_pose(format="vector")[:3]
        goal_pos = self.unwrapped.get_goal_pose(format="vector")[:3]
        mean_flow = np.linalg.norm(object_pos - goal_pos) 

        reward = -mean_flow
        success = mean_flow < 0.05

        return success, reward

    def reset(self, **kwargs):
        raw_obs = self.env.reset(**kwargs)
        obs = self.observation(raw_obs)
        self.record_cam_frame()
        return obs

    def set_prev_obs(self, prev_obs):
        self.prev_obs = prev_obs
        return True
    
    def set_deterministic(self, deterministic):
        self.deterministic = deterministic

    def record_cam_frame(self):
        if self.record_video:
            frame = self.render(mode="cameras")
            frame = np.flipud(frame)
            self.cam_frames.append(frame)