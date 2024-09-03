from typing import Dict, Type, Tuple
import functools
from enum import IntEnum
import numpy as np
import gym

MAX_PRIMITIVES = 10

class GroundingTypes(IntEnum):
    BACKGROUND_ONLY = 0
    OBJECT_ONLY = 1
    OBJECT_AND_BACKGROUND = 2
    NONE = 3

class Primitive():
    def __init__(self,
                 env: gym.ObservationWrapper, # Union[HACManActionWrapper, HACManMobileActionWrapper]
                 grounding_type: GroundingTypes,
                 traj_steps: int = 1,
                 motion_dim: int = 5,
                 end_on_reached: bool = False,
                 end_on_collision: bool = False,
                 use_oracle_rotation: bool = False,
                 pad_rewards: bool = False,
                 penalize_collision: bool = False,
                 free_move: bool = False,
                 pos_tolerance: float = 0.002,
                 rot_tolerance: float = 0.05):
        self.env = env
        self.traj_steps = traj_steps
        self.motion_dim = motion_dim
        self.use_oracle_rotation = use_oracle_rotation
        self.grounding_type = grounding_type
        self.end_on_reached = end_on_reached
        self.end_on_collision = end_on_collision
        self.pad_rewards = pad_rewards
        self.penalize_collision = penalize_collision
        self.free_move = free_move
        self.pos_tolerance = pos_tolerance
        self.rot_tolerance = rot_tolerance

    def execute(self, location, motion, **kwargs):
        raise NotImplementedError

    def is_valid(self, states: Dict) -> bool:
        return True

    def visualize(self, motion):
        return motion[..., :3]
    
    @property
    def param_dim(self):
        return self.traj_steps * self.motion_dim


REGISTERED_PRIMITIVES: Dict[str, Tuple[Primitive, Dict]] = {}

def get_primitive_class(name) -> Type["Primitive"]:
    cls, kwargs = REGISTERED_PRIMITIVES[name]
    return functools.partial(cls, **kwargs)

def register_primitive(name, grounding_type, **kwargs):
    """ Decorator for registering primitives """
    def decorator(cls):
        # cls_name = cls.__name__
        if name in REGISTERED_PRIMITIVES:
            raise ValueError(f"Primitive {name} already registered")
        kwargs['grounding_type'] = grounding_type
        REGISTERED_PRIMITIVES[name] = (cls, kwargs)
        return cls
    return decorator


def sample_prim_idx(grounding_types, num_obj_points, num_bg_points):
    # Sample a primitive index for each batch element
    batched = (grounding_types.ndim > 1)
    if batched:
        bs = grounding_types.shape[0]
        prim_idxs, poke_idxs = [], []
        for i in range(bs):
            prim_idx, poke_idx = _sample_prim_idx(grounding_types[i], num_obj_points, num_bg_points)
            prim_idxs.append(prim_idx)
            poke_idxs.append(poke_idx)
        return prim_idxs, poke_idxs
    else:
        return _sample_prim_idx(grounding_types, num_obj_points, num_bg_points)

def _sample_prim_idx(grounding_types, num_obj_points, num_bg_points):
    # First select the grounding type
    non_none_grounding = (grounding_types != GroundingTypes.NONE)
    p = non_none_grounding / np.sum(non_none_grounding)
    prim_idx = np.random.choice(len(grounding_types), p=p,)
    grounding_type = grounding_types[prim_idx]

    if grounding_type == GroundingTypes.OBJECT_ONLY:
        poke_idx = np.random.randint(num_obj_points)
    elif grounding_type == GroundingTypes.BACKGROUND_ONLY:
        poke_idx = np.random.randint(num_bg_points) + num_obj_points
    elif grounding_type == GroundingTypes.OBJECT_AND_BACKGROUND:
        poke_idx = np.random.randint(num_bg_points + num_obj_points)
    else:
        raise ValueError
    
    return prim_idx, poke_idx