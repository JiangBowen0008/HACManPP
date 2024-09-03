
Code Structure
===

- `docs`: Documentation Files
- `hacmanpp`: HACMan++ main package
  - `algos`: HACMan++ RL algorithm implementation
  - `envs`: Environment wrappers for the HACMan++ algorithm
    - `setup_env.py`: Environment (parallelized) setup functions
    - `setup_location_policy.py`: Location policy setup functions (see details below for explanation of location policy)
    - `env_wrappers`: Action environment wrappers (interface for executing the primitives)
      - `action_wrapper.py`: HACMan++-based action wrapper
      - `...`: Baseline-based action wrappers
    - `vec_env_wrappers`: Vectorized environment wrappers
      - `location_policy_vec_wrappers.py`: Location-policy embedded vectorized environment wrappers
      - `vec_obs_processing.py`: Vectorized observation processing
      - `deterministic_wrapper.py`: Deterministic environment wrapper (for eval)
      - `wandb_wrapper.py`: Logging environment wrapper
  - `networks`: Network modules
  - `utils`: Utility functions
  - `sb3_utils`: Stable Baselines3 utility functions
- `hacmanpp_*`: HACMan++-wrapped environments
  - `make_*_vec_env.py`: Parallelized environment setup functions
  - `env_wrappers/primitives.py`: Primitive implementations under the simulator
  - `envs`: Environments interfaced with HACMan++ wrappers
  - ... (See details below addressing **environment parallelization** and **implementation differences** specific to each environment)

### HACMan++-wrapped Environments
#### Robosuite
The robosuite version we use does not support parallelized environments. To use Robosuite with HACMan++, we launch multiple instances of the environment as specified in `make_suite_vec_env.py`.

#### ManiSkill2
ManiSkill2 has native support for parallelized environments. In `make_ms_vec_env.py`, we wrap the MS vectorized environments with vec wrappers needed for HACMan++.

#### DoubleBin
DoubleBin is a custom environment that we implemented modified from a Robosuite environment. We use `make_doublebin_vec_env.py` to set up the environment. Its implementation details can be found [here](../hacman_bin/README.md).

#### Adroit
We use the adroit environment from the `manipulation_suite` package. It does not support parallelized environments. We use `make_adroit_vec_env.py` to set up the environment.