import wandb
import os

class TerminalColor:
    """Provides ANSI escape codes for colored terminal output."""
    RESET = '\033[0m'
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    BRIGHT_BLACK = '\033[90m'
    BRIGHT_RED = '\033[91m'
    BRIGHT_GREEN = '\033[92m'
    BRIGHT_YELLOW = '\033[93m'
    BRIGHT_BLUE = '\033[94m'
    BRIGHT_MAGENTA = '\033[95m'
    BRIGHT_CYAN = '\033[96m'
    BRIGHT_WHITE = '\033[97m'

    # Background colors
    BG_BLACK = '\033[40m'
    BG_RED = '\033[41m'
    BG_GREEN = '\033[42m'
    BG_YELLOW = '\033[43m'
    BG_BLUE = '\033[44m'
    BG_MAGENTA = '\033[45m'
    BG_CYAN = '\033[46m'
    BG_WHITE = '\033[47m'
    BG_BRIGHT_BLACK = '\033[100m'
    BG_BRIGHT_RED = '\033[101m'
    BG_BRIGHT_GREEN = '\033[102m'
    BG_BRIGHT_YELLOW = '\033[103m'
    BG_BRIGHT_BLUE = '\033[104m'
    BG_BRIGHT_MAGENTA = '\033[105m'
    BG_BRIGHT_CYAN = '\033[106m'
    BG_BRIGHT_WHITE = '\033[107m'

    # Text styles
    BOLD = '\033[1m'
    DIM = '\033[2m'
    ITALIC = '\033[3m'
    UNDERLINE = '\033[4m'
    BLINK = '\033[5m'
    REVERSE = '\033[7m'
    HIDDEN = '\033[8m'
    STRIKETHROUGH = '\033[9m'



# import logging


# # Set up logging to local file in case of failures
# logging.basicConfig(filename='eval_log_error.log', level=logging.ERROR)


# def write_eval_log(eval_data, model_dir, log_to_wandb=True):
#     """
#     Logs evaluation metrics to both local storage and optionally to WandB.
    
#     Parameters:
#     - eval_data: A dictionary containing evaluation metrics to log.
#     - model_dir: The directory where the local eval log file should be saved.
#     - log_to_wandb: Boolean flag to decide whether to log to WandB. Default is True.
#     """
#     # Define the local log file path
#     local_log_path = os.path.join(model_dir, "eval/eval_scores.txt")
    
#     try:
#         # Attempt to log to WandB
#         if log_to_wandb:
#             wandb.log(eval_data)
#             print("Logged to WandB successfully.")
#     except Exception as e:
#         logging.error(f"Failed to log to WandB: {e}")
#         print(f"Error logging to WandB: {e}. Logging to WandB failed.")

#     try:
#         # Attempt to write to the local log file
#         with open(local_log_path, 'a') as file:  # 'a' to append to the file
#             for section, values in eval_data.items():
#                 file.write(f"### {section} ###\n")
#                 if isinstance(values, dict):
#                     for key, value in values.items():
#                         file.write(f"{key}:\n")
#                         if isinstance(value, (list, tuple)):
#                             file.write(", ".join(map(str, value)) + "\n")
#                         else:
#                             file.write(f"{value}\n")
#                 else:
#                     if isinstance(values, (list, tuple)):
#                         file.write(", ".join(map(str, values)) + "\n")
#                     else:
#                         file.write(f"{values}\n")
#                 file.write("\n")
#             print(f"Logged to local file: {local_log_path}")
#     except Exception as e:
#         logging.error(f"Failed to write to local file {local_log_path}: {e}")
#         print(f"Error logging to local file: {e}. Data not logged to local file.")

