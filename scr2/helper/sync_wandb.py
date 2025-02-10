import os
import subprocess

# Directory where the model runs are stored
# base_directory = "/home/etatar/GraphPredCod2/scr/trained_models/pc/fully_connected/_no_sens2sens"  # Replace with your base directory path

tmp = ['_normal','_no_sens2sens_no_sens2sup', '_no_sens2sup/']

for t in tmp:
    base_directory = "/home/etatar/GraphPredCod2/scr/trained_models/pc/fully_connected/"+t
    # Loop through all directories in the base directory
    for folder_name in os.listdir(base_directory):
        folder_path = os.path.join(base_directory, folder_name)

        # Check if the folder is a directory and contains a 'wandb' folder
        if os.path.isdir(folder_path) and 'wandb' in os.listdir(folder_path):
            wandb_path = os.path.join(folder_path, 'wandb')

            # Look for subdirectories inside 'wandb' (e.g., 'run-xxxx' or 'offline-run-xxxx')
            for subfolder_name in os.listdir(wandb_path):
                subfolder_path = os.path.join(wandb_path, subfolder_name)
                
                # Check if it is a directory before syncing
                if os.path.isdir(subfolder_path):
                    try:
                        print(f"Syncing {subfolder_path}...")
                        subprocess.run(['wandb', 'sync', subfolder_path], check=True)
                    except subprocess.CalledProcessError as e:
                        print(f"Failed to sync {subfolder_path}: {e}")
