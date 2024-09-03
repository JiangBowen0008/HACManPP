import os
import argparse
import numpy as np
import time
import hydra
from omegaconf import DictConfig, OmegaConf

HEADER = """#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=500:00:00
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=50
#SBATCH --exclude=compute-0-25
#SBATCH --mem=200gb
"""

CLUSTER_ADDRESS = "" # CLUSTER IP ADDRESS
CLUSTER_NAME = "" # NAME OF YOUR CLUSTER
SINGULAIRTY_FILE = ""#/PATH/TO/YOUR/SINGULARITY/FILE

PROJECT_DIRNAME = "hacman_cleanup"
PRECOMMAND = f"""
echo $HOSTNAME
module load singularity
cd ~/Projects/{PROJECT_DIRNAME}/
"""
COMMAND = f"""singularity exec --nv {SINGULAIRTY_FILE} bash -c \
'source ~/.bashrc && conda activate hacman && \
OPENBLAS_NUM_THREADS=1 \
OMP_NUM_THREADS=1 \
python scripts/run.py \
"""


def flatten_dict(d):
    """
    Flatten a nested dictionary.
    """
    flat_dict = {}
    for k, v in d.items():
        if isinstance(v, dict):
            flat_dict.update(flatten_dict(v))
        else:
            flat_dict[k] = v
    return flat_dict

@hydra.main(version_base=None, config_path="configs", config_name="config")
def launch_experiments(cfg: DictConfig):
    job_list = []
    if len(cfg.seed_list) > 0:
        seed_list = cfg.seed_list
    else:
        seed_list = np.arange(cfg.n_seeds)

    # Convert DictConfig to a regular dictionary
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    # Parse from the configuration file
    default_kwargs = dict(cfg_dict['experiments']['default_kwargs'])  # Convert to a regular dictionary
    variants = cfg_dict['experiments']['variants']
    skip_variants = cfg_dict['experiments']['skip_variants']
    
    # Flatten the config
    nested_keys = {"task", "primitives", "method"}
    exp_kwargs = {}
    for key in nested_keys:
        if key in cfg_dict['experiments'].keys():
            extra_kwargs = cfg_dict['experiments'].pop(key)
            extra_kwargs = flatten_dict(extra_kwargs)
            exp_kwargs.update(extra_kwargs)
    exp_kwargs.update(default_kwargs)
    
    # Load the _base_ config in variants if it exists
    original_cwd = os.path.join(hydra.utils.get_original_cwd(), "scripts", "configs", "experiments")
    for key, val in variants.items():
        base_kwargs = {}
        if "_base_" in val.keys():
            base = val.pop("_base_")
            config_file_path = f"{original_cwd}/{base}.yaml"
            base_kwargs = OmegaConf.load(config_file_path)
        base_kwargs.update(val)
        variants[key] = base_kwargs

  
    
    # Change the saved log location
    if cfg.cluster == CLUSTER_NAME:
        slurm_dir = f"/home/{cfg.user}/Projects/{PROJECT_DIRNAME}/slurm_output"
        log_dir = f"/home/{cfg.user}/Projects/{PROJECT_DIRNAME}/logs/hacman_seuss/"

    
    elif cfg.cluster == "local":
        slurm_dir = "slurm_output"
        log_dir = "logs/hacman"
        os.makedirs(slurm_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
    
    else:
        raise ValueError
    
    # Generate and launch individual experiments
    all_scripts = []
    task_commands = {}
    gpu_id = 0
    for i, variant_name in enumerate(variants.keys()):
        if i in skip_variants:
            continue

        for seed in seed_list:
            kwargs = exp_kwargs.copy()
            kwargs.update(variants[variant_name])
            # kwargs = OmegaConf.to_object(kwargs)

            # Add parsed names to kwargs
            if variant_name is None or variant_name == "default":
                parsed_names = {"name": "default"}
            else:
                parsed_names = {
                    "ExpGroup": f"Exp{cfg.ExpID:04d}-{cfg.ExpGroup}",                   # For wandb grouping
                    "ExpVariant": f"Exp{cfg.ExpID:04d}-{cfg.ExpGroup}-{variant_name}",  # For wandb grouping
                    "name": f"Exp{cfg.ExpID:04d}-{i}-{seed}-{cfg.ExpGroup}-{variant_name}",
                }

            kwargs.update(parsed_names)
            kwargs['dirname'] = log_dir
            kwargs['ExpID'] = f'{cfg.ExpID:04d}'
            kwargs['seed'] = seed
            if 'load_exp' in cfg_dict.keys():
                load_exp = cfg_dict['load_exp']
                kwargs['load_exp'] = load_exp

            # job_name is the unique name of the job (with id and seed)
            job_name = f"Exp{cfg.ExpID:04d}-{i}-{seed}_{cfg.ExpGroup}-{variant_name}"
            job_list.append(job_name)

            # Construct the command
            params = []
            for key, val in kwargs.items():
                if val is None:
                    continue
                elif isinstance(val, bool):
                    if val:
                        params.append(f'--{key}')
                elif isinstance(val, list):
                    params.append(f'--{key} {" ".join([str(v) for v in val])}')
                else:
                    params.append(f'--{key} {val}')
                    
            params = " ".join(params)
            
            current_command = COMMAND + " " + params
            current_command += f" > {slurm_dir}/{job_name}-{seed}.stdout ' & \n"
            
            if cfg.cluster == CLUSTER_NAME:
                # Add sbatch heading
                scripts = HEADER
                scripts += f"#SBATCH --job-name={cfg.ExpID:04d}-{i}\n"
                scripts += f"#SBATCH -o {slurm_dir}/{job_name}.out\n"
                scripts += f"#SBATCH -e {slurm_dir}/{job_name}.err\n"
                
                scripts += PRECOMMAND
                scripts += current_command

                scripts += "wait\n"
                file_suffix = "_sbatch"
                cluster_address = cfg.user + CLUSTER_ADDRESS

            
            elif cfg.cluster == "local":
                file_suffix = '.txt'
                cluster_address = None
                scripts = f"CUDA_VISIBLE_DEVICES={gpu_id} OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/run.py " + params # 
                if kwargs['env'] == "hacman_bin":
                    scripts = f"LD_PRELOAD='' MUJOCO_PY_FORCE_CPU=1 {scripts}"
                scripts += f" > {slurm_dir}/{job_name}.stdout"
                gpu_id += 1
                
            else:
                raise ValueError

            all_scripts.append(scripts)
            os.makedirs('sbatch_scripts', exist_ok=True)
            if cfg.cluster != "fair":
                filename = job_name + file_suffix
                with open(os.path.join('sbatch_scripts', filename), 'w') as f:
                    f.write(scripts)

            task_commands[job_name] = scripts
            print(scripts + "\n\n")

    # Sync job specifications & (optionally) code 
    if not cfg.dry_run and cfg.cluster== CLUSTER_NAME :
        # Only sync sbatch scripts
        os.system("rsync -avz --exclude={'.idea','*__pycache__*','logs','results','*.git*','slurm_output'} "
                    "sbatch_scripts "
                    f"{cluster_address}:/home/{cfg.user}/Projects/{PROJECT_DIRNAME}")

        if not cfg.quick:
            # Sync both sbatch scripts and code
            local_dir = os.path.expanduser("~/Projects/{PROJECT_DIRNAME}")
            os.system(f"rsync -avz --exclude={{'.idea','*__pycache__*','*.pkl','logs','results','*.git*','slurm_output','outputs'}}  {local_dir} {cluster_address}:/home/{cfg.user}/Projects/")
            
        time.sleep(5)

    # Launch all the jobs
    os_command = ""
    if cfg.cluster == CLUSTER_NAME:
        os_command += f"ssh {cluster_address} "
    
    launch_commands = []
    for job in job_list:
        if cfg.cluster == CLUSTER_NAME:
            launch_commands.append(f"sbatch /home/{cfg.user}/Projects/{PROJECT_DIRNAME}/sbatch_scripts/{job}_sbatch")
        elif cfg.cluster == "local":
            scripts = task_commands[job]
            launch_commands.append(f"screen -dm bash -c 'conda activate hacman; {scripts}'")
        elif cfg.cluster == "fair":
            pass
        else:
            raise KeyError
    
    if cfg.cluster == CLUSTER_NAME:
        os_command += '"' + ' & '.join(launch_commands) + '"' 
    elif cfg.cluster == "local":
        os_command = "\n& ".join(launch_commands)
    
    # Do not execute if dry run
    if cfg.dry_run or cfg.cluster in {"local"}:
        print(os_command)
    else:
        if os_command != '""':
            os.system(os_command)  
            


if __name__ == '__main__':
    launch_experiments()