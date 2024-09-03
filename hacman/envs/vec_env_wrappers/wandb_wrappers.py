"""
TODO:
- Make a unified plotly plotting function
- Show more than one env?
"""

import os, sys
from typing import Any, Callable, List, Optional, Sequence, Tuple, Type, Union, Dict
import numpy as np
import imageio
import pickle
import gym
from copy import deepcopy
# from bin_env import PokeEnv
from hacman.utils.transformations import to_pose_mat, transform_point_cloud, decompose_pose_mat

import wandb
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvObs, VecEnvStepReturn, VecEnvWrapper
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv, _flatten_obs
from hacman.utils.plotly_utils import plot_pcd, plot_action, plot_pcd_with_score, plot_actions
# from bin_env.util import angle_diff
import plotly.graph_objects as go
from typing import Any, Callable, List, Optional, Sequence, Tuple, Type, Union, Dict
from hacman.algos.location_policy import RandomLocation
from hacman.utils.primitive_utils import GroundingTypes

    
class WandbPointCloudRecorder(VecEnvWrapper):
    """
    Modified based on VecVideoRecorder.
    """

    def __init__(self, venv, global_step_counter=None,
        save_plotly=False, foldername=None, 
        record_video=False, upload_video=False,
        log_primitive = False, ## whether to include per primitive heatmap in the plotly
        real_robot=False, log_plotly_once=False):

        VecEnvWrapper.__init__(self, venv)
        self.real_robot = real_robot
        # Log plotly once per evaluate_policy call. This is to save space during training.
        self.log_plotly_once = log_plotly_once
        self.log_primitive = log_primitive
        self.env = venv
        # Temp variable to retrieve metadata
        temp_env = venv
        self.prim_mapping =  {
            1: 'Grasp',
            4: 'Open Gripper',
            3: 'Move delta',
            2: 'Move to',
            0: 'Poke'
        }

        # Unwrap to retrieve metadata dict
        # that will be used by gym recorder
        while isinstance(temp_env, VecEnvWrapper):
            temp_env = temp_env.venv

        if isinstance(temp_env, DummyVecEnv) or isinstance(temp_env, SubprocVecEnv):
            metadata = temp_env.get_attr("metadata")[0]
        else:
            metadata = temp_env.metadata

        self.episode_count = -1
        self.global_step_counter = global_step_counter

        self.env.metadata = metadata
        self.save_plotly = save_plotly
        assert not (foldername is None and self.save_plotly), "Need to specify a folder for WandbPointCloudRecorder."
        self.foldername = os.path.join(foldername, 'plotly')
        os.makedirs(self.foldername, exist_ok=True)

        self.vis = []
        self.goal_poses = []
        self.recording = False
        self.all_prim_scores = []
        self.points_pcds = []
        self.title = None
        self.step_count = 0
        self.plot_count = 0

        # Video recording
        self.vid_count = 0
        self.record_video = record_video
        self.upload_video = upload_video
        self.vid_dir = os.path.join(foldername, 'video')
        self.cam_frames = []

    def reset(self) -> VecEnvObs:
        # This reset will only be called once per evaluate_policy
        # regardless of how many eval epsiodes. This is because
        # VecEnv will reset themselves.
        obs = self.venv.reset()
        self.reset_plotly_logging(obs)
        self.reset_video_recording()
        return obs
    
    def reset_video_recording(self):
        if len(self.cam_frames) == 0:
            return

        for i in range(len(self.cam_frames)):
            if len(self.cam_frames[i]) == 0:
                continue
            global_step = self.get_global_step()
            is_success = "fail"
            filename = f"video_{self.vid_count}_{global_step}_{i}_{is_success}.mp4"
            self.save_video(self.cam_frames[i], filename)
            self.cam_frames[i].clear()
    
    def plot_obs(self, obs, step, idx=slice(None)):
        # Plot goals
        if self.real_robot:
            # Ground truth goal pcd
            goal_pcd_o3d = self.venv.envs[0].unwrapped.obj_reg.get_last_goal_pcd()
            goal_pcd = np.asarray(goal_pcd_o3d.points)
            self.vis.append(plot_pcd(f'goal', goal_pcd, 'blue'))
            # Estimated goal pcd
            goal_pcd = transform_point_cloud(obs['object_pose'][idx], obs['goal_pose'][idx], obs['object_pcd_points'][idx])
            self.vis.append(plot_pcd(f'Goal_{step}', goal_pcd, 'lightblue'))
        else:
            goal_pcd = transform_point_cloud(obs['object_pose'][idx], obs['goal_pose'][idx], obs['object_pcd_points'][idx])
            self.vis.append(plot_pcd(f'Goal_{step}', goal_pcd, np.array([139, 123, 135]))) # blue
            self.goal_poses.append(goal_pcd)

        # Plot active points with action scores (if available)
        points, inactive_points, action_scores = None, None, None
        if step == "final":
            points = obs['object_pcd_points'][idx]
        elif 'action_location_score' in obs.keys():
            prim_idx = obs['prim_idx'][idx]
            prim_grounding = obs['prim_groundings'][idx][prim_idx]
            if prim_grounding == GroundingTypes.OBJECT_ONLY.value:
                points = obs['object_pcd_points'][idx]
                inactive_points = obs['background_pcd_points'][idx]
                action_scores = obs['action_location_score'][idx][:points.shape[0]]
            elif prim_grounding == GroundingTypes.BACKGROUND_ONLY.value:
                points = obs['background_pcd_points'][idx]
                num_obj_points = obs['object_pcd_points'][idx].shape[idx]
                offseted_points = points + obs['action_params'][idx][num_obj_points:] * 0.05
                inactive_points = np.concatenate([obs['object_pcd_points'][idx], offseted_points], axis=0)
                action_scores = obs['action_location_score'][idx][num_obj_points:]
            elif prim_grounding == GroundingTypes.OBJECT_AND_BACKGROUND.value:
                points = np.concatenate([obs['object_pcd_points'][idx], obs['background_pcd_points'][idx]], axis=0)
                action_scores = obs['action_location_score'][idx]
            elif prim_grounding == GroundingTypes.NONE.value:
                inactive_points = np.concatenate([obs['object_pcd_points'][idx], obs['background_pcd_points'][idx]], axis=0)
            else:
                raise ValueError  
        else:
            points = obs['object_pcd_points'][idx]
            inactive_points = obs['background_pcd_points'][idx]
        self.points_pcds.append(np.concatenate([obs['object_pcd_points'][idx], obs['background_pcd_points'][idx]], axis=0))

        if points is not None:
            if action_scores is not None:
                self.vis.append(plot_pcd_with_score(f'Observation_{step}', points, action_scores))
            else:
                self.vis.append(plot_pcd(f'Observation_{step}', points, 'purple'))
        if inactive_points is not None:
            self.vis.append(plot_pcd(f'inactive_{step}', inactive_points, np.array([238, 236, 238]), size=2))## light orange # lightgrey #np.array([240, 177, 136])

        # Plot the previous points
        if step == "final":
            prev_step = self.step_count - 1
            self.vis.append(plot_pcd(f'o_next_{prev_step}', obs['object_pcd_points'][idx], 'yellow')) ## yellow
        elif step > 0:
            prev_step = step - 1
            self.vis.append(plot_pcd(f'o_next_{prev_step}', obs['object_pcd_points'][idx], 'yellow')) ## yellow

    def get_prim_info(self, obs, idx): 
        if 'action_location_score_all' in obs.keys():
            prim_location_score_all = obs['action_location_score_all'][idx] ## num points x num_prims
            max_prim_score =  np.max(prim_location_score_all, axis=0)
            prim_prob_sum_score =np.sum(prim_location_score_all, axis=0)
            return prim_location_score_all, max_prim_score, prim_prob_sum_score
        else:
            return None, None, None
            
    def reset_plotly_logging(self, obs):
        self.recording = True
        self.vis = []
        self.titles = [] 
        self.prim_all_scores = []    
        self.points_pcds = []  
        self.step_count = 0
        self.episode_count += 1
        self.goal_poses = []
        self.plot_obs(obs, 0, idx=0)
        if self.log_primitive:
            prim_all_scores, prim_max_score, prim_sum_score = self.get_prim_info(obs, 0)
            self.prim_all_scores.append(prim_all_scores)
            self.prev_prim_max_score = prim_max_score
            self.prev_prim_sum_score = prim_sum_score
        return
    
    def enable_video_recording(self):
        self.record_video = True
    
    def disable_video_recording(self):
        self.record_video = False

    def get_global_step(self):
        if self.global_step_counter is not None:
            global_step = self.global_step_counter()
        else:
            global_step = self.episode_count
        return global_step

    def step_wait(self) -> VecEnvStepReturn:
        obs, rews, dones, infos = self.venv.step_wait()
        
        if self.recording:
            new_line = "<br>"
            if self.log_primitive:
                prim_max_string =  f"{new_line.join(f'{self.prim_mapping[key]}:{self.prev_prim_max_score[key]}' for key in range(self.prev_prim_max_score.shape[0]))}"
                prim_sum_string =  f"{new_line.join(f'{self.prim_mapping[key]}:{self.prev_prim_sum_score[key]}' for key in range(self.prev_prim_sum_score.shape[0]))}"
                self.titles.append( f"Final reward: {rews[0]:.2f}, Success: {infos[0]['is_success']},\
                <br>Primitives: {infos[0]['executed_prims']}<br> Primitive_max_score: <br> {prim_max_string} <br> Primitive_sum_score: <br>{prim_sum_string}")
            else:
                self.titles.append( f"Final reward: {rews[0]:.2f}, Success: {infos[0]['is_success']}")
            
            self.step_count += 1
            action_location = infos[0]['action_location']
            if self.log_primitive:
                prim_all_scores, prim_max_score, prim_sum_score = self.get_prim_info(obs, 0)
                self.prim_all_scores.append(prim_all_scores)
                self.prev_prim_max_score = prim_max_score
                self.prev_prim_sum_score = prim_sum_score
            if action_location is None:
                # When not predicting action location (contiguous action),
                # plot from previous gripper position
                if (not hasattr(self, "prev_obs")): # Issue with gym0.24 running one step before initialized
                    action_location = np.zeros(3)
                elif self.prev_obs is None:
                    action_location = np.zeros(3)
                else:
                    prev_gripper_pose = self.prev_obs['gripper_pose']
                    action_location, _ = decompose_pose_mat(prev_gripper_pose[0])
            
            if 'poke_success' in infos[0].keys() and not infos[0]['poke_success']:
                self.vis.append(plot_action(f'Action_{self.step_count-1}', action_location, infos[0]['action_param'], color='orange'))
            else:
                self.vis.append(plot_action(f'Action_{self.step_count-1}', action_location, infos[0]['action_param']))

                if "sampled_points" in infos[0].keys():
                    sampled_locations = infos[0]['sampled_points']
                    sampled_params = infos[0]['sampled_action_params']
                    self.vis.append(plot_actions(f'sampled_actions_{self.step_count-1}', sampled_locations, sampled_params, color='grey', size=1))             
                
            if dones[0]:
                # DummyVecEnv will reset venv automatically and replace the returned obs
                terminal_obs = infos[0]['terminal_observation']
                self.plot_obs(terminal_obs, 'final')
                self.close_video_recorder()
                if not self.log_plotly_once:
                    self.reset_plotly_logging(obs)                
            else:
                self.plot_obs(obs, self.step_count, idx=0)
                
        
        # Camera recording
        if self.record_video and len(infos[0]['cam_frames']) > 0:
            if len(self.cam_frames) == 0:
                 # Initialize cam_frames
                self.cam_frames = [[] for _ in range(len(infos))]
            
            # Append the poke frames to the corresponding cam_frames
            for i, info in enumerate(infos):
                if len(info['cam_frames']) > 0:
                    self.cam_frames[i].extend(info['cam_frames'])

                if dones[i]:
                    global_step = self.get_global_step()
                    is_success = "success" if infos[i]['is_success'] else "fail"
                    filename = f"video_{self.vid_count}_{global_step}_{i}_{is_success}.mp4"
                    self.save_video(self.cam_frames[i], filename)
                    self.cam_frames[i].clear()

        return obs, rews, dones, infos
    

    def save_video(self, frames, filename, fps=30) -> None:
        assert len(frames) > 0, "No frames to save"
        global_step = self.get_global_step()
        vid_path = os.path.join(self.vid_dir, filename)
        os.makedirs(self.vid_dir, exist_ok=True)

        # Save video locally
        with imageio.get_writer(vid_path, mode='I', fps=fps) as writer:
            for im in frames:
                writer.append_data(im)
        
        self.vid_count += 1

        
        return

    def close_video_recorder(self) -> None:
        if self.recording:
            global_step = self.get_global_step()
            fig = self.get_plotly_with_slidebar()
            # fig.show()  # Visualize locally
            wandb.log({"visualizations": fig, 'global_steps': global_step})

            if self.save_plotly:
                filename = f"plotly_{global_step}_{self.plot_count}.html"
                fig.write_html(os.path.join(self.foldername, filename), auto_play=False)  # Save as html locally
                score_filename = f"score_{global_step}_{self.plot_count}.pkl"
                data = {'heatmap_score': self.prim_all_scores, 'points': self.points_pcds, 'goal_pose': self.goal_poses} 
                with open(os.path.join(self.foldername, score_filename), 'wb') as f:
                    pickle.dump(data, f)
                self.plot_count += 1

        self.recording = False
        self.vis = []
        self.titles = []
        self.step_count = 0
    
    def get_plotly_with_slidebar(self):
        vis_name2id = {}
        for i in range(len(self.vis)):
            vis_name2id[self.vis[i].name] = i

        fig = go.Figure(self.vis)
        
        # Default
        for dt in fig.data:
            dt.visible = False
        
        try:
            if f'goal' in vis_name2id.keys():
                fig.data[vis_name2id['goal']].visible = True
            fig.data[vis_name2id['inactive_0']].visible = True  
            fig.data[vis_name2id[f'Goal_0']].visible = True
            fig.data[vis_name2id[f'Observation_0']].visible = True
            fig.data[vis_name2id[f'Action_0']].visible = True
            fig.data[vis_name2id[f'o_next_0']].visible = True
            fig.data[vis_name2id[f'sampled_actions_0']].visible = True
        except:
            pass

        fig.update_scenes(aspectmode='data',xaxis_visible=False, yaxis_visible=False,zaxis_visible=False)
        fig.update_layout(title=self.titles[0])

        steps = []
        for i in range(self.step_count):
            step = dict(
                method="update",
                args=[{"visible": [False] * len(fig.data)},
                    {"title": self.titles[i]}],
            )
            try:
                step["args"][0]["visible"][vis_name2id[f'inactive_{i}']] = True
                if f'goal' in vis_name2id.keys():
                    step["args"][0]["visible"][vis_name2id['goal']] = True
                # if f'g_{i}' in vis_name2id.keys():
                step["args"][0]["visible"][vis_name2id[f'Goal_{i}']] = True         
                step["args"][0]["visible"][vis_name2id[f'Observation_{i}']] = True
                step["args"][0]["visible"][vis_name2id[f'Action_{i}']] = True
                if self.log_primitive:
                    for key in self.prim_mapping.keys():
                        step["args"][0]["visible"][vis_name2id[f'prim_{self.prim_mapping[key]}_{i}']] = True
                if i != self.step_count - 1:
                    step["args"][0]["visible"][vis_name2id[f'o_next_{i}']] = True
                step["args"][0]["visible"][vis_name2id[f'sampled_actions_{i}']] = True
            except:
                Warning(f"Step {i} missing info.")
            finally:
                steps.append(step)
        
        if len(steps) > 0:
            steps[-1]["args"][0]["visible"][vis_name2id[f'Observation_final']] = True
        else:
            fig.data[vis_name2id[f'Observation_final']].visible = True

        sliders = [dict(
            active=0,    
            pad={"t": 50},
            steps=steps
        )]
        fig.update_layout(sliders=sliders)
        return fig
    
    def get_plotly_with_slidebar_v2(self):
        vis_name2id = {}
        for i in range(len(self.vis)):
            vis_name2id[self.vis[i].name] = i
        
        traces = {}
        for i in range(self.step_count):
            traces[i] = []
            traces[i].append(self.vis[vis_name2id[f'background']])
            if f'goal' in vis_name2id.keys():
                traces[i].append(self.vis[vis_name2id['goal']])
            if f'g_{i}' in vis_name2id.keys():
                traces[i].append(self.vis[vis_name2id[f'g_{i}']])
            traces[i].append(self.vis[vis_name2id[f'o_{i}']])
            traces[i].append(self.vis[vis_name2id[f'a_{i}']])
            
            if i == self.step_count - 1:
                traces[i].append(self.vis[vis_name2id[f'o_final']])
            else:
                traces[i].append(self.vis[vis_name2id[f'o_next_{i}']])
            
        # Final step
        i = self.step_count        
        traces[i] = []
        traces[i].append(self.vis[vis_name2id[f'background']])
        if f'goal' in vis_name2id.keys():
            traces[i].append(self.vis[vis_name2id['goal']])
        if f'g_{i-1}' in vis_name2id.keys():
            traces[i].append(self.vis[vis_name2id[f'g_{i-1}']])
        traces[i].append(self.vis[vis_name2id[f'o_final']])
        traces[i].append(self.vis[vis_name2id[f'a_{i-1}']])
            
        fig = go.Figure(
            data=traces[0],
            frames=[
                go.Frame(data=traces[k], name=str(k))
                for k in range(self.step_count+1)
            ]
        )
        fig.frames[-1].data[-1].visible = False
        # Every frame needs to have the same number of traces
        
        # Layout
        fig.update_layout(
            title=self.title,
            width=600,
            height=600,
            scene1=dict(aspectmode='data'),
            updatemenus=[
                {
                    "buttons": [
                        {
                            "args": [None, self.frame_args(500)],
                            "label": "&#9654;",  # play symbol
                            "method": "animate",
                        },
                        {
                            "args": [[None], self.frame_args(0)],
                            "label": "&#9724;",  # pause symbol
                            "method": "animate",
                        },
                    ],
                    "direction": "left",
                    "pad": {"r": 10, "t": 70},
                    "type": "buttons",
                    "x": 0.1,
                    "y": 0,
                }
            ],
            sliders= [
                {
                    "pad": {"b": 10, "t": 60},
                    "len": 0.9,
                    "x": 0.1,
                    "y": 0,
                    # "steps": steps,
                    "steps": [
                        {
                            "args": [[f.name], self.frame_args(0)],
                            "label": f'step {str(k)}',
                            "method": "animate",
                        }
                        for k, f in enumerate(fig.frames)
                    ],
                }
            ],
        )
        
        return fig
    
    def frame_args(self, duration):
        return {
            "frame": {"duration": duration},
            "mode": "immediate",
            "fromcurrent": True,
            "transition": {"duration": duration, "easing": "linear"},
        }
    

    def close(self) -> None:
        self.close_video_recorder()

    # def __del__(self):
    #     self.close()
