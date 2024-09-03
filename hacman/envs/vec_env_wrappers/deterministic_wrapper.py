from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Type, Union
import numpy as np
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvObs, VecEnvStepReturn, VecEnvWrapper
VecEnvIndices = Union[None, int, Iterable[int]]

class DeterministicVecEnvWrapper(VecEnvWrapper):
    def reset(self) -> VecEnvObs:
        return self.venv.reset()
    
    def step_wait(self) -> VecEnvStepReturn:
        step_return = self.venv.step_wait()
        for i in range(self.venv.num_envs):
            self.env_method("set_deterministic", False, indices=i)
        return step_return
    
    def step_async(self, actions: np.ndarray) -> None:
        for i in range(self.venv.num_envs):
            self.env_method("set_deterministic", True, indices=i)
        return self.venv.step_async(actions)
    
    
