import wandb
import os
import logging


# Set up logging to local file in case of failures
logging.basicConfig(filename='eval_log_error.log', level=logging.ERROR)


def write_eval_log(eval_data, model_dir, log_to_wandb=True):
    """
    Logs evaluation metrics to both local storage and optionally to WandB.
    
    Parameters:
    - eval_data: A dictionary containing evaluation metrics to log.
    - model_dir: The directory where the local eval log file should be saved.
    - log_to_wandb: Boolean flag to decide whether to log to WandB. Default is True.
    """
    # Define the local log file path
    local_log_path = os.path.join(model_dir, "eval/eval_scores.txt")
    
    try:
        # Attempt to log to WandB
        if log_to_wandb:
            wandb.log(eval_data)
            print("Logged to WandB successfully.")
    except Exception as e:
        logging.error(f"Failed to log to WandB: {e}")
        print(f"Error logging to WandB: {e}. Logging to WandB failed.")

    try:
        # Attempt to write to the local log file
        with open(local_log_path, 'a') as file:  # 'a' to append to the file
            for section, values in eval_data.items():
                file.write(f"### {section} ###\n")
                if isinstance(values, dict):
                    for key, value in values.items():
                        file.write(f"{key}:\n")
                        if isinstance(value, (list, tuple)):
                            file.write(", ".join(map(str, value)) + "\n")
                        else:
                            file.write(f"{value}\n")
                else:
                    if isinstance(values, (list, tuple)):
                        file.write(", ".join(map(str, values)) + "\n")
                    else:
                        file.write(f"{values}\n")
                file.write("\n")
            print(f"Logged to local file: {local_log_path}")
    except Exception as e:
        logging.error(f"Failed to write to local file {local_log_path}: {e}")
        print(f"Error logging to local file: {e}. Data not logged to local file.")

