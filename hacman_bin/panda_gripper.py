"""
Gripper for Franka's Panda (has two fingers).
"""
import numpy as np
from robosuite.utils.mjcf_utils import xml_path_completion
from robosuite.models.grippers.gripper_model import GripperModel

import os

## the panda gripper with replaced festo finger used by the panda robot
class PandaGripperBase(GripperModel):
    """
    Gripper for Franka's Panda (has two fingers).

    Args:
        idn (int or str): Number or some other unique identification string for this gripper instance
    """

    def __init__(self, idn=0):
        xml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets/grippers', 'panda_gripper.xml')
        super().__init__(xml_path, idn=idn)

    def format_action(self, action):
        return action

    @property
    def init_qpos(self):
        return np.array([0.020833, -0.020833])

    @property
    def _important_geoms(self):
        return {
            "left_finger": ["finger1_collision", "finger1_pad_collision", 'finger1_tip_outer_collision'],
            "right_finger": ["finger2_collision", "finger2_pad_collision", "finger2_tip_outer_collision"],
            "left_fingerpad": ["finger1_pad_collision"],
            "right_fingerpad": ["finger2_pad_collision"],
        }


class PandaGripper(PandaGripperBase):
    """
    Modifies PandaGripperBase to only take one action.
    """

    def format_action(self, action):
        """
        Maps continuous action into binary output
        -1 => open, 1 => closed

        Args:
            action (np.array): gripper-specific action

        Raises:
            AssertionError: [Invalid action dimension size]
        """
        assert len(action) == self.dof
        self.current_action = np.clip(self.current_action + np.array([-1.0, 1.0]) * self.speed * np.sign(action), -1.0, 1.0)
        return self.current_action

    @property
    def speed(self):
        return 0.01

    @property
    def dof(self):
        return 1

    

## the gripper has a smaller control range and there is inner piece attached to the gripper to set the different friction for grasping and poking
class PandaGripperNarrowBase(GripperModel):
    """
    Gripper for Franka's Panda (has two fingers).

    Args:
        idn (int or str): Number or some other unique identification string for this gripper instance
    """

    def __init__(self, idn=0):
        xml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets/grippers', 'panda_gripper_narrow.xml')
        super().__init__(xml_path, idn=idn)

    def format_action(self, action):
        return action

    @property
    def init_qpos(self):
        return np.array([0.020833, -0.020833])

    @property
    def _important_geoms(self):
        return {
            "left_finger": ["finger1_collision", "finger1_pad_collision", 'finger1_pad_outer_collision'],
            "right_finger": ["finger2_collision", "finger2_pad_collision", "finger2_pad_outer_collision"],
            "left_fingerpad": ["finger1_pad_outer_collision"],
            "right_fingerpad": ["finger2_pad_outer_collision"],
        }

class PandaGripperNarrow(PandaGripperNarrowBase):
    """
    Modifies PandaGripperBase to only take one action.
    """

    def format_action(self, action):
        """
        Maps continuous action into binary output
        -1 => open, 1 => closed

        Args:
            action (np.array): gripper-specific action

        Raises:
            AssertionError: [Invalid action dimension size]
        """
        assert len(action) == self.dof
        self.current_action = np.clip(self.current_action + np.array([-1.0, 1.0]) * self.speed * np.sign(action), -1.0, 1.0)
        return self.current_action

    @property
    def speed(self):
        return 0.01

    @property
    def dof(self):
        return 1
    
