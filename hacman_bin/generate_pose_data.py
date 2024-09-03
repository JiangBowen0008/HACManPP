import os, sys
from tqdm import tqdm
import time
import multiprocessing as mp
import pickle
import json
import numpy as np
from scipy.spatial.transform import Rotation
import itertools
import open3d as o3d
from collections import defaultdict

# from rl.env_wrapper import GymPokeEnv
from hacman_bin_env import HACManBinEnv
from hacman_bin.assets import HousekeepSampler
from hacman_bin.util import angle_diff

curr_path = os.path.dirname(os.path.abspath(__file__))
EXPORTED_POSE_ROOT = os.path.join(curr_path, "data")
# HOUSEKEEP_OBJECT_INFO_ROOT = 'dex_env/assets/housekeep/object_info'
# NUM_STEPS_FOR_OBJECT_TYPE = {
#     "bottle": 10,
#     "cup": 10,
#     "mug": 25,
#     "pillbottle": 10,
#     default: 10
# }

def generate_pose_data(visualize=False, dataset_name="poses_fixed_size", poses_per_file=100, dataset="housekeep_all"):
    """
    Generating stable poses by dropping objects into the bin
    The result is a dictionary with the following format: {
        object_name (mesh file name) : {
            "object_type": string, object_type
            "poses": ndarray N*7, stable poses
            "scales": ndarray N*3, scales applied to the object in the simulation
        }
    }
    """
    start_time = time.time()
    debug = False

    # Iterate through each file
    object_sampler = HousekeepSampler(use_full_set=(dataset=="housekeep_all"))
    object_set = object_sampler.get_object_set()

    if debug:
        visualize = True
        poses_per_file = poses_per_file
       

    rollout_kwargs = []
    for object_name in object_set:
        object_type = object_name.split('_')[0]

        # Debug
        if object_name !="box_pink_M":
            continue
        else:
            print("Found box_pink_M")

        env_configs = dict(robots='Virtual', close_gripper=True, ik_precontact=True, randomize_shape=True)
        # env_configs['use_full_set'] = full_set
        env_configs['object_name'] = object_name
        env_configs['object_dataset'] = dataset
        env_configs['object_scale_range'] = [1.0, 1.0]
        env_configs['arena_type'] = 'double_bin'
        env_configs['use_object_obs'] = True
        sim_steps = 80

        kwarg = (env_configs, visualize, poses_per_file, sim_steps)
        rollout_kwargs.append(kwarg)

        print(f"Object {object_name} added")

    if debug:
        # Make a progress bar for easier tracking
        global pbar
        pbar = tqdm(total=len(rollout_kwargs) * poses_per_file)
        results = []
        for kwarg in rollout_kwargs:
            results.append(rollout(*kwarg))
    
    else:
        pbar = tqdm(total=len(rollout_kwargs) * poses_per_file)
        with mp.Pool(initializer=init_worker, initargs=(pbar,), processes=20) as p:
            results = list(p.starmap(rollout, rollout_kwargs))
    
    # Collect the results
    # all_data = {}
    pose_dir = os.path.join(EXPORTED_POSE_ROOT, env_configs['object_dataset'])
    # os.makedirs(pose_dir, exist_ok=True)

    pose_file = os.path.join(pose_dir, dataset_name + ".pk")
    with open(pose_file, 'rb') as f:  
        all_data = pickle.load(f)

    unstable_objects = {}
    for i, data in enumerate(results):
        env_configs = rollout_kwargs[i][0]
        object_name = env_configs['object_name']
        object_type = object_name.split('_')[0]
        
        data['object_type'] = object_type
        all_data[object_name] = data
        if data['unstable_count'] > 0:
            unstable_objects[object_name] = data['unstable_count']
    
    if not debug:
        # Write to a pickle file
        # pose_dir = os.path.join(EXPORTED_POSE_ROOT, env_configs['object_dataset'])
        # os.makedirs(pose_dir, exist_ok=True)
        dataset_name += '_with_box'
        pose_file = os.path.join(pose_dir, dataset_name + ".pk")
        with open(pose_file, 'wb') as f:
            pickle.dump(all_data, f)
        unstable_objects_file = os.path.join(pose_dir, "unstable_objects.json")
        with open(unstable_objects_file, 'w') as f:
            json.dump(unstable_objects, f, indent=2)
    
    else:
        print("Unstable objects:")
        print(unstable_objects)

    # Report execution time
    print(f"--- {(time.time() - start_time)} seconds ---")
    return

def init_worker(pbar_):
    global pbar
    pbar = pbar_

def rollout(env_configs, visualize, num_poses, max_sim_steps=100, pos_delta_threshold=0.005, angle_delta_threshold=0.1):
    """
    Set onscreen_env to None to disable onscreen preview.
    """
    object_name = env_configs['object_name']
    renderer = "onscreen" if visualize else "offscreen"
    env_configs.update(
        renderer=renderer, render_camera='agentview',
        action_mode='regress_location_and_action')
    # poke_env = GymPokeEnv(**env_configs)
    poke_env = HACManBinEnv(**env_configs)
    poke_env.goal = np.zeros(7)
    env = poke_env

    # Read the distance/scale constants
    table_offset = env.table_offset
    table_size = env.get_table_full_size()
    drop_range = (table_size / 2.0) * 0.65
    drop_range[2] = 0     # we only randomize x & y

    poses = []
    scales = []     # A list of same values for now (since sizes do not vary)
    unstable_count = 0

    for i in range(num_poses):
        """
        Procedure:
        1.  Estimate the vertical distance between the object and the bin. Lower
            the *pre_pose* to *intial_pose* such that it just touches the bin.
        2.  Run simulation. Stabilizes at *final_pose*.
        """
        env.reset(hard_reset=True)
        if i == 0:
            # Use a fixed pose for the first pose
            rot = Rotation.identity()
        else:
            # Randomize the rotation
            rot = Rotation.random()
        quat = rot.as_quat()  # scipy uses [x, y, z, w], same as robotsuite

        # Method 1: (relies on offscreen rendering)
        # # Lift the object to be sufficiently far from the bottom
        # pre_offset = table_offset.copy()
        # # pre_offset[2] += 0.1
        # pre_pose = np.concatenate([pre_offset, quat], axis=0)

        # # Read the points from the offscreen env
        # obs = env.reset(object_pose=pre_pose, fast_reset=True)
        # points = obs['object_pcd_points']

        # # Find the min vertical distance between the object and the bin bottom
        # min_z = np.min(points[:, 2])
        # min_z_dist =  min_z - table_offset[2]
        
        # Alternative method: (works with onscreen env as well)
        # Estimate the min_z from the object bounding box
        obj_scale = env.cube.scale
        obj_size = env.cube.scaled_size
        obj_origin = env.cube.scaled_origin

        # sa, sb = obj_size, np.zeros_like(obj_size)
        sa, sb = obj_size/2.0, -obj_size/2.0
        vertices = np.stack([sa, sb]) - obj_origin
        vertices = list(itertools.product(*vertices.T))
        vertices = np.array(vertices)

        # Transform
        vertices = rot.apply(vertices)
        vertices += obj_origin

        min_z = np.min(vertices[:, 2])
        min_z_dist = min_z

        # Construct initial pose
        pos_lb = table_offset - drop_range
        pos_ub = table_offset + drop_range
        pos = np.random.uniform(low=pos_lb, high=pos_ub)
        initial_pose = np.concatenate([pos, quat], axis=0)
        initial_pose[2] += -min_z_dist + 0.02
        # print(min_z_dist)

        # Method 1 simulation
        # if visualize:
        #     cube = env.cube
        #     test_env.cube = cube
        # else:
        #     test_env = env

        env.reset(object_pose=initial_pose, hard_reset=False)
        if visualize: env.render()

        obj_pos_buffer = []
        obj_quat_buffer = []
        for _ in range(max_sim_steps):
            obs, _, _, _ = env.step(np.zeros(env.action_dim))
            if visualize: env.render()

            # obj_pos, obj_quat = obs['cube_pos'], obs['cube_quat']
            obj_pose= env.get_cube_pose()
            obj_pos = obj_pose[:3]
            obj_quat = obj_pose[3:]
            obj_pos_buffer.append(obj_pos)
            obj_quat_buffer.append(obj_quat)

            if len(obj_pos_buffer) > 5:
                old_obj_pos = obj_pos_buffer[-5]
                old_obj_quat = obj_quat_buffer[-5]

                pos_delta = np.linalg.norm(old_obj_pos - obj_pos)
                angle_delta = angle_diff(old_obj_quat, obj_quat) / np.pi * 180.0
                obj_stable = (pos_delta < pos_delta_threshold) and (angle_delta < angle_delta_threshold)

                if obj_stable:
                    break
            
        if obj_stable:
            obj_pose = np.concatenate([obj_pos, obj_quat])
    
            poses.append(obj_pose)
            scales.append(obj_scale)
        
        else:
            unstable_count += 1
            print(f"Unstable object: {object_name}!")

        # Update the progress bar as well
        global pbar
        pbar.update(1)

        # obs = poke_env.step(np.zeros(poke_env.action_space.shape))
    
    del poke_env

    try:
        poses = np.stack(poses)
        scales = np.stack(scales)
    except:
        poses = np.array([[]])
        scales = np.array([[]])

    data = {
        "poses": poses,
        "scales": scales,
        "unstable_count": unstable_count
    }
    return data


def test_read_data():
    with open("data/housekeep_all/poses_fixed_size.pk", "rb") as f:
        data = pickle.load(f)

    print(data['flashlight_HeavyDuty_Flashlight_M']['scales'].shape)
    # keys = list(data.keys())
    # print(f"Objects count: {len(keys)}")
    # print(f"Objects: {keys}")
    # print(f"Pose array shape: {data[keys[0]]['poses'].shape}")
    # print(data['flashlight_HeavyDuty_Flashlight_M']['scales'].shape)

if __name__ == '__main__':
    generate_pose_data()
    test_read_data()