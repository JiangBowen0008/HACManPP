from collections import OrderedDict
import numpy as np
import pandas as pd

from robosuite.models.objects import MujocoXMLObject
from robosuite.utils.mjcf_utils import xml_path_completion, array_to_string

import h5py
import tempfile
import json
import shutil
import os, random
from lxml import etree

from hacman_bin.assets.objects import MujocoMeshObject

curr_path = os.path.dirname(os.path.abspath(__file__))

class HousekeepSampler:
    def __init__(self, use_full_set=False, object_scale_range=None, object_size_limit=[0.05, 0.05], convex_decomposed=False, **kwargs):
        if use_full_set:
            self.housekeep_root = os.path.join(curr_path, 'housekeep_all')
        else:
            self.housekeep_root = os.path.join(curr_path, 'housekeep')
        
        self.objects_root = os.path.join(self.housekeep_root, 'models_processed')
        self.object_info_root = os.path.join(self.housekeep_root, 'object_info')
        self.texture_root = os.path.join(curr_path, 'textures')
            
        self.object_set, self.all_object_info = self.load_all_objects(**kwargs)
        self.object_scale_range = object_scale_range
        self.object_size_limit = object_size_limit
        self.convex_decomposed = convex_decomposed


    def load_all_objects(self, object_types=None, object_name=None, object_split=None, **kwargs):
        object_df = pd.read_csv(os.path.join(self.housekeep_root, 'metainfo.csv'))
        object_set = []
        all_object_info = {}
        
        # Found the set of specified objects
        if object_name is None:            
            # Sample from a split configuration
            if object_split is not None and (object_split != 'no_split'):
                with open(os.path.join(self.housekeep_root, 'dataset_split.json')) as f:
                    split = json.load(f)
                object_set = split[object_split]
            
            # Sample from specific types
            elif object_types is not None:
                object_set = object_df[object_df['type'].isin(object_types)]['name'].tolist()
            
            # Sample from all objects
            else:
                object_set = object_df['name'].values.tolist()
            
        else:
            assert object_name in object_df['name'].values.tolist(), f"Specified object {object_name} does not exist."
            object_set = [object_name]
            
        assert len(object_set) > 0, "No objects found in the specified set."
        
        # Load all object info
        for object_name in object_set:
            info_path = os.path.join(self.object_info_root, object_name+'.h5')
            assert os.path.exists(info_path), f"Specified object {object_name} does not exist."
            all_object_info[object_name] = h5py.File(info_path, "r")

        print(f"Loaded {len(object_set)} objects from {self.housekeep_root}.")
        
        return object_set, all_object_info
    

    def set_object(self, object_name):
        assert object_name in self.all_object_info.keys(), f"Specified object {object_name} does not exist."
        self.object_set = [object_name]
    

    def get_object_set(self):
        return self.object_set

    def generate_housekeep_object(self):
        object_info = self.sample_housekeep_object()

        obj = MujocoMeshObject(object_info=object_info, 
                            name="object", 
                            stl_dir=self.objects_root,
                            texture_dir=self.texture_root, 
                            joints=[dict(type="free", damping="0.0005")], 
                            obj_type="all", 
                            duplicate_collision_geoms=True, 
                            size_limit=self.object_size_limit,
                            object_scale_range=self.object_scale_range,
                            convex_decomposed=self.convex_decomposed)
        
        obj_target = MujocoMeshObject(object_info=object_info, 
                                    name="object_target", 
                                    stl_dir=self.objects_root, 
                                    texture_dir=self.texture_root, 
                                    joints=None, 
                                    obj_type="visual",
                                    duplicate_collision_geoms=True, 
                                    size_limit=self.object_size_limit,
                                    object_scale_range=self.object_scale_range,
                                    object_scale=obj.scale,
                                    convex_decomposed=self.convex_decomposed)
        
        return obj, obj_target

    def sample_housekeep_object(self):        
        object_name = np.random.choice(self.object_set).item()
        object_info = self.all_object_info[object_name]

        return object_info

    

if __name__ == '__main__':
    sampler = HousekeepSampler(object_types='all', object_name=None, object_split=None, object_eval_set=False)
    for _ in range(15):
        obj, target_obj = sampler.generate_housekeep_object()
        print(obj.mesh_name)