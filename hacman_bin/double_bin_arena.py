import numpy as np
from robosuite.models.arenas import Arena
from robosuite.utils.mjcf_utils import xml_path_completion
from robosuite.utils.mjcf_utils import array_to_string
import os


class DoubleBinArena(Arena):
    """
    Workspace that contains one bin. The table underneath the bin is just for visualization purpose.

    Args:
        bin_pos (3-tuple): (x,y,z) position to place bin (center of bottom plate, upper surface)
        bin_full_size (3-tuple): (L,W,H) full dimensions of the bin (inner dimension)
        bin_thickness (float): thickness of the walls
        bin_friction (3-tuple): (sliding, torsional, rolling) friction parameters of the table
        bin_height (float): height of the bin relative to the ground
        hidden_walls (str): hide some of the walls if neccessary. i.e.: "FL" will hide front and left walls.
    """

    def __init__(
            self,
            xml_filename="double_bin_arena.xml",
            bin_pos=(0, 0, 0),
            bin_base_size=(0.4, 0.24, 0.06),
            bin_friction=(0.3, 0.005, 0.0001),
            bin_solref=(0.02, 1.),
            bin_solimp=(0.9, 0.95, 0.001), 
            bin_thickness=0.003,
            bin_height=0.8,
            bin_distance=0.135,
            table_base_size=(0.4, 0.52, 0.107),
            arena_rand_size=(0, 0, 0),
            hidden_walls="",
    ):
        xml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets/arenas', xml_filename)
        super().__init__(xml_path)

        # Samples a random size
        arena_rand_size = np.array(arena_rand_size)
        sampled_size = (1 + arena_rand_size * np.random.uniform(-1, 1, size=3))
        self.table_full_size = np.array(table_base_size) * sampled_size
        self.bin_full_size = np.array(bin_base_size) * sampled_size

        self.bin_half_size = self.bin_full_size / 2
        self.bin_friction = bin_friction
        self.bin_solref = bin_solref
        self.bin_solimp = bin_solimp
        self.bin_thickness = bin_thickness
        self.bin_distance = bin_distance
        self.hidden_walls = str.upper(hidden_walls)

        self.bin_height = bin_height
        self.bins = {}
        self.bin_visuals = {}
        self.bin_collisions = {}
        self.bin_planes = {}
        self.bin_planes["bin_plane"] = self.worldbody.find(f"./geom[@name='bin_bottom_plane']")
        for bin_idx in range(2):
            bin_name = f'bin_{bin_idx}'
            bin_body = self.worldbody.find(f"./body[@name='{bin_name}']")
            self.bins[bin_name] = bin_body
            for name in ['bottom', 'left', 'right', 'front', 'back']:
                obj_name = f'{name}_{bin_idx}'
                self.bin_visuals[obj_name] = bin_body.find(f"./geom[@name='{bin_name}_" + name + "_visual']")
                self.bin_collisions[obj_name] = bin_body.find(f"./geom[@name='{bin_name}_" + name + "_collision']")
                self.bin_planes[obj_name] = bin_body.find(f"./geom[@name='{bin_name}_" + name + "_plane']")

        self.configure_location()

        self.set_origin((bin_pos[0], bin_pos[1], bin_pos[2] - bin_height))
        self.table_offset = np.array(bin_pos)  # For compatibility

    def configure_location(self):
        """Configures correct locations for this arena"""
        self.floor.set("pos", array_to_string(self.bottom_pos))

        # Set bin locations
        self.bins["bin_0"].set("pos", array_to_string(np.array([0, -self.bin_distance, self.bin_height])))
        self.bins["bin_1"].set("pos", array_to_string(np.array([0, self.bin_distance, self.bin_height])))

        # Set bin sides
        for bin_idx, bin_name in enumerate(self.bins.keys()):
            # Bottom
            bottom_name = f'bottom_{bin_idx}'
            size = np.array([self.bin_half_size[0], self.bin_half_size[1], self.bin_thickness])
            size += np.array([self.bin_thickness*2, self.bin_thickness*2, 0])  # To cover the edges
            pos = np.array([0, 0, -self.bin_thickness])
            self.bin_visuals[bottom_name].set("size", array_to_string(size))
            self.bin_visuals[bottom_name].set("pos", array_to_string(pos))
            # Replace bin collision with plane
            if self.bin_planes[bottom_name] is not None:
                self.bin_planes[bottom_name].set("pos", array_to_string(np.array([0, 0, 0])))
                self.bin_collisions[bottom_name].set("pos", array_to_string(np.array([0, 0, -1]))) # Hide

            # Left
            left_name = f'left_{bin_idx}'
            size = np.array([self.bin_half_size[0], self.bin_thickness, self.bin_half_size[2]])
            size += np.array([self.bin_thickness*2, 0, 0])  # To cover the edges
            pos = np.array([0, -self.bin_half_size[1] - self.bin_thickness, self.bin_half_size[2]])
            if "L" in self.hidden_walls: pos = np.array([0, 0, -1])
            self.bin_visuals[left_name].set("size", array_to_string(size))
            self.bin_visuals[left_name].set("pos", array_to_string(pos))
            self.bin_collisions[left_name].set("size", array_to_string(size))
            self.bin_collisions[left_name].set("pos", array_to_string(pos))
            if self.bin_planes[left_name] is not None:
                self.bin_planes[left_name].set("pos", array_to_string(np.array([0, -self.bin_half_size[1], self.bin_half_size[2]])))
                self.bin_collisions[left_name].set("pos", array_to_string(np.array([0, 0, -1]))) # Hide
                        
            # Right
            right_name = f'right_{bin_idx}'
            size = np.array([self.bin_half_size[0], self.bin_thickness, self.bin_half_size[2]])
            size += np.array([self.bin_thickness*2, 0, 0])  # To cover the edges
            pos = np.array([0, self.bin_half_size[1] + self.bin_thickness, self.bin_half_size[2]])
            if "R" in self.hidden_walls: pos = np.array([0, 0, -1])
            self.bin_visuals[right_name].set("size", array_to_string(size))
            self.bin_visuals[right_name].set("pos", array_to_string(pos))
            self.bin_collisions[right_name].set("size", array_to_string(size))
            self.bin_collisions[right_name].set("pos", array_to_string(pos))
            if self.bin_planes[right_name] is not None:
                self.bin_planes[right_name].set("pos", array_to_string(np.array([0, self.bin_half_size[1], self.bin_half_size[2]])))
                self.bin_collisions[right_name].set("pos", array_to_string(np.array([0, 0, -1]))) # Hide
                        
            # Front
            front_name = f'front_{bin_idx}'
            size = np.array([self.bin_thickness, self.bin_half_size[1], self.bin_half_size[2]])
            size += np.array([0, self.bin_thickness*2, 0])  # To cover the edges
            pos = np.array([self.bin_half_size[0] + self.bin_thickness, 0, self.bin_half_size[2]])
            if "F" in self.hidden_walls: pos = np.array([0, 0, -1])
            self.bin_visuals[front_name].set("size", array_to_string(size))
            self.bin_visuals[front_name].set("pos", array_to_string(pos))
            self.bin_collisions[front_name].set("size", array_to_string(size))
            self.bin_collisions[front_name].set("pos", array_to_string(pos))
            if self.bin_planes[front_name] is not None:
                self.bin_planes[front_name].set("pos", array_to_string(np.array([self.bin_half_size[0], 0, self.bin_half_size[2]])))
                self.bin_collisions[front_name].set("pos", array_to_string(np.array([0, 0, -1]))) # Hide
                        
            # Back
            back_name = f'back_{bin_idx}'
            size = np.array([self.bin_thickness, self.bin_half_size[1], self.bin_half_size[2]])
            size += np.array([0, self.bin_thickness*2, 0])  # To cover the edges
            pos = np.array([-self.bin_half_size[0] - self.bin_thickness, 0, self.bin_half_size[2]])
            if "B" in self.hidden_walls: pos = np.array([0, 0, -1])
            self.bin_visuals[back_name].set("size", array_to_string(size))
            self.bin_visuals[back_name].set("pos", array_to_string(pos))
            self.bin_collisions[back_name].set("size", array_to_string(size))
            self.bin_collisions[back_name].set("pos", array_to_string(pos))
            if self.bin_planes[back_name] is not None:
                self.bin_planes[back_name].set("pos", array_to_string(np.array([-self.bin_half_size[0], 0, self.bin_half_size[2]])))
                self.bin_collisions[back_name].set("pos", array_to_string(np.array([0, 0, -1]))) # Hide
            
        # Set friction
        for key, value in self.bin_collisions.items():
            if value is None:
                continue
            value.set("friction", array_to_string(self.bin_friction))
            value.set("solref", array_to_string(self.bin_solref))
            value.set("solimp", array_to_string(self.bin_solimp))
            value.set("priority", str(1))
            value.set("condim", str(4))
            
        # Set friction
        for key, value in self.bin_planes.items():
            # Note that planes are all infinite.
            if value is not None:
                value.set("friction", array_to_string(self.bin_friction))
                value.set("solref", array_to_string(self.bin_solref))
                value.set("solimp", array_to_string(self.bin_solimp))
                value.set("priority", str(1))
                value.set("condim", str(4))
                value.set("rgba", array_to_string(np.array([0, 0, 0, 0]))) # hide

        # Set table locations
        table_full_size = np.array([self.table_full_size[0], self.table_full_size[1], 0.05])
        table_offset = self.bin_height
        table_offset -= self.bin_thickness*2
        table_half_size = table_full_size / 2
        center_pos = self.bottom_pos + np.array([0, 0, -table_half_size[2] + table_offset])

        table_body = self.worldbody.find("./body[@name='table']")
        table_collision = table_body.find("./geom[@name='table_collision']")
        table_visual = table_body.find("./geom[@name='table_visual']")
        table_top = table_body.find("./site[@name='table_top']")

        table_legs_visual = [
            table_body.find("./geom[@name='table_leg1_visual']"),
            table_body.find("./geom[@name='table_leg2_visual']"),
            table_body.find("./geom[@name='table_leg3_visual']"),
            table_body.find("./geom[@name='table_leg4_visual']"),
        ]

        table_body.set("pos", array_to_string(center_pos))
        table_collision.set("size", array_to_string(table_half_size))
        table_visual.set("size", array_to_string(table_half_size))
        table_top.set("pos", array_to_string(np.array([0, 0, table_half_size[2]])))

        # Set leg locations
        delta_x = [0.1, -0.1, -0.1, 0.1]
        delta_y = [0.1, 0.1, -0.1, -0.1]
        for leg, dx, dy in zip(table_legs_visual, delta_x, delta_y):
            # If x-length of table is less than a certain length, place leg in the middle between ends
            # Otherwise we place it near the edge
            x = 0
            if table_half_size[0] > abs(dx * 2.0):
                x += np.sign(dx) * table_half_size[0] - dx
            # Repeat the same process for y
            y = 0
            if table_half_size[1] > abs(dy * 2.0):
                y += np.sign(dy) * table_half_size[1] - dy
            # Get z value
            z = (table_offset - table_half_size[2]) / 2.0
            # Set leg position
            leg.set("pos", array_to_string([x, y, -z]))
            # Set leg size
            leg.set("size", array_to_string([0.025, z]))
    
    def get_bin_centers(self):
        half_sizes = self.get_bin_half_sizes()
        bin_centers = np.array([
            [0, -self.bin_distance, self.bin_height],
            [0, self.bin_distance, self.bin_height]
        ])
        bin_centers[:, 2] += half_sizes[:, 2]
        return bin_centers
    
    def get_bin_half_sizes(self):
        """
        Half sizes of the bins
        """
        bin_half_sizes = np.array([
            [self.bin_half_size[0], self.bin_half_size[1], self.bin_half_size[2]],
            [self.bin_half_size[0], self.bin_half_size[1], self.bin_half_size[2]]
        ])
        return bin_half_sizes
    
    def get_table_full_size(self):
        return self.table_full_size
