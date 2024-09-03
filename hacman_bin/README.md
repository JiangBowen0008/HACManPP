# Double Bin

Double Bin is our customized environment for HACMan++. It is a simulated environment that consists of two bins and a floating robot gripper. The gripper is controlled by a controller that uses operational space control (OSC) to move the gripper to the desired goal pose. The goal of the task to move the spawned object from one bin to the desired pose in the other bin. 

- `assets`: Assets used by `HACManBinEnv`.
  - `housekeep`: Cylindrical housekeep objects.
  - `housekeep_all`: All housekeep objects.
  - ...
- `controller_configs`: Controller configuration files.
- `data`
  - `housekeep`: Goal pose data for cylindrical housekeep objects.
  - `housekeep_all`: Goal pose data for all housekeep objects.
- `base_env.py`: Base environment. It contains initialization functions for the environment and object loading functions.
- `double_bin_arena.py`: Double bin workspace.
- `osc.py`: Controller for controlling robot arm via operational space control.
- `poke_env.py`: Top-level sim environment directly used by `hacman_bin_env.py`. It contains functions for interacting with the environment, resetting the environment, and setting goals.
- `hacman_bin_env.py`: HACManBinEnv environment. It contains the environment class for the Double Bin environment.
- ...