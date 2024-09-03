import os
import numpy as np
import json
import logging
import torch
from tqdm import tqdm

def use_freer_gpu():
    """
    Set the program to use the freer GPU. Run this at the beginning of the program.
    """
    try:
        os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
        memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
        if len(memory_available) == 1:
            return os.environ['CUDA_VISIBLE_DEVICES']

        gpu_idx = np.argmax(memory_available).item()
        os.environ['CUDA_VISIBLE_DEVICES'] = f'{gpu_idx}'
        # torch.cuda.set_device(gpu_idx)
        print(f"Free GPU: {gpu_idx}")
        
    except:
        gpu_idx = 0
    
    return gpu_idx

def load_exp_config(parser, config):
    """
    Load the config from a previous exp.
    """
    # By default, load from the same directory
    config['load_dirname'] = config['dirname'] if config['load_dirname'] is None else config['load_dirname']

    # Search for the previous config
    ckpt_path, metadata_path, run_id = search_for_the_latest_run(config['load_dirname'], config['load_exp'])
    prev_config = parse_exp_config(parser, metadata_path)

    # Override the current config with the previous config
    config = override_exp_config(config, prev_config, config["override_args"])
    config['load_ckpt'] = ckpt_path     # make sure to load the new ckpt
    config['name'] = prev_config['name'] if config['name'] == "default" else config['name']

    # TODO: Add continue training when buffer is also found
    # Continue the previous run if it is not eval mode

    return config

def override_exp_config(config, override_config, override_args):
    """
    Override the config with the override_config. Keys in override_args will be ignored.
    """
    arg_avoid_list = [
        "name", "dirname",
        "load_dirname", "load_exp",
        # "seed",
        "initial_timesteps",
        "eval", "eval_n_envs",
        "record_video", "upload_video", "delete_video"]
    
    for key, value in override_config.items():
        if (key in arg_avoid_list and config[key] is not None) or key in override_args:
            continue
        config[key] = value
    
    return config

def parse_exp_config(parser, metadata_path):
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    args_list = metadata.get('args', [])

    # Create parser and parse arguments
    args = parser.parse_args(args_list)
    config = vars(args)
    return config



def search_for_the_latest_run(dir, keyword):
    """
    Search for the latest run in the directory.
    Return the ckpt path and the metadata path.
    """
    # Recursively walk through all directories and subdirectories
    ckpt_list = []
    for root, _, files in os.walk(dir):
        
        # If a keyword is provided, skip directories that don't contain the keyword
        if keyword and keyword not in root:
            continue
            
        for file in files:
            if file.endswith("rl_model_latest.zip"):
                ckpt_path = os.path.join(root, file)
                ckpt_list.append(ckpt_path)
                
    # Sort by modification time
    ckpt_list = sorted(ckpt_list, key=os.path.getmtime)
    assert len(ckpt_list) > 0, f"Cannot find any metadata in {dir}!"
    ckpt_path = ckpt_list[-1]
    print(f"Fount ckpt: {ckpt_path}")

    # Find the corresponding metadata file
    model_dir = os.path.dirname(ckpt_path)
    run_id = os.path.basename(model_dir).split('-')[-1]

    wandb_dir = os.path.join(os.path.dirname(model_dir), "wandb")
    for root, _, files in os.walk(wandb_dir):
        for file in files:
            if file.endswith("wandb-metadata.json") and run_id in root:
                metadata_path = os.path.join(root, file)
                print(f"Found metadata: {metadata_path}")
                break

    return ckpt_path, metadata_path, run_id

def upload_videos(folder_path, folder_name, delete=False):
    """
    Upload all the videos in the directory to Google drive.
    """
    # Add date to the folder name
    import datetime
    now = datetime.datetime.now()
    date = now.strftime("%m-%d")
    folder_name = f"{date}-{folder_name}"

    upload_to_google_drive(folder_path, folder_name)
    if delete:
        print(f"Deleting {folder_path}...")
        os.system(f"rm -rf {folder_path}")


from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

def upload_to_google_drive(folder_path, folder_name):
    gauth = GoogleAuth()
    shared_folder_id = "1PnE809jMtlEpCheIKlcwGldzrHNkb3_y"

    # Try to load saved client credentials
    try:
        gauth.LoadCredentialsFile("credentials.json")
        if gauth.access_token_expired:
            # Refresh them if expired
            gauth.Refresh()
        gauth.Authorize()
    
    except:
        # Delete the original credential and create a new one
        os.remove("credentials.json")
        gauth = GoogleAuth()
        gauth.CommandLineAuth()
        # Save the current credentials to a file
        gauth.SaveCredentialsFile("credentials.json")

    drive = GoogleDrive(gauth)
    
    # Create a new folder inside the shared folder
    new_folder = drive.CreateFile({
        'title': folder_name,
        'mimeType': 'application/vnd.google-apps.folder',
        'parents': [{'id': shared_folder_id}]
    })
    new_folder.Upload()
    new_folder_id = new_folder['id']
    
    # Upload files to the new folder
    print("Uploading files to Google Drive...")
    for filename in tqdm(os.listdir(folder_path)):
        file_path = os.path.join(folder_path, filename)
        file = drive.CreateFile({
            'title': filename,
            'parents': [{'id': new_folder_id}]
        })
        file.SetContentFile(file_path)
        file.Upload()

if __name__ == "__main__":
    metadata_path = search_for_the_latest_run('logs/hacman', 'Exp2136-0')
    # print(metadata_path)
    # upload_to_google_drive("scripts/results/Exp2053-test_loading-0/video", "test_upload_videos")
    pass