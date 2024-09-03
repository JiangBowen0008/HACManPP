# from hacman_bin_env.base_env import BaseEnv
# from hacman_bin_env.bin_env import BinEnv
from .hacman_bin_env import HACManBinEnv
from robosuite.environments.base import register_env
from .make_bin_vec_env import make_bin_venv

register_env(HACManBinEnv)
# register_env(BinEnv)
