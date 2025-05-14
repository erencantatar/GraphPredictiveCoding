# Graph Predictive Coding

...
784 sensory verticestrained with 794 sensory vertices for classification
and generation tasks (784 pixels plus a 1-hot vector for the 10 labels),

We ommit the model for reconstruction and denoising: 
and 784 sensory vertices

## Changelog
- Major logic bug; weight_update is now only updating the internal nodes instead of the weights of all nodes.

## Training loop
1. Create graph topology by creating a 2d mask (Fully_connected, generative, discriminative, SBM, other (see `scr/graphbuilder.py`) )
2. Initialize the weights of the graph using the mask (either 2d matrix or 1d for sparse weights (MessagePassing))
3. Create dataloader with batched graphs 
4. Init the model, params, grads, optimizer
5. Training using clamped img (X) and img_label (Y)
   Test on val_set by removing either img for generation eval or remove img_label for classification task, depedining on the task and topology (FC can do both, hierachical (VanZwol only either one))
6. Eval on eval_set
7. Single MP that does all (pred, call error, dEdX)



## Installation
1. **todo** Install conda env. with Pytorch Geometric (PYG)
```bash
conda env create -f environment.yaml
conda activate PredCod  #(or source activate)
```
2. run `notebooks/training.ipynb` to train step by step   

3. TODO !!
conda env export --no-builds > environment.yaml
pip freeze > requirements.txt


## Snellius cheatsheat 

1. Interactive (debug) session: 
```bash
srun --partition=gpu_a100 --gpus=1 --ntasks=1 --cpus-per-task=18 --time=02:00:00 --pty bash -i
srun --partition=gpu_a100 --gpus=1 --ntasks=1 --cpus-per-task=18 --time=01:00:00 --pty bash -i

module purge
module load 2022
module load Anaconda3/2022.05
cd /home/etatar/GraphPredCod2/scr
source activate PredCod

module purge  
module load 2022
module load Anaconda3/2022.05
cd /home/etatar/GraphPredCod2/scr2
source activate PredCod

module purge
module load 2022
module load Anaconda3/2022.05
cd /home/etatar/GraphPredCod2/scr3
source activate PredCod

```
2. accinfo
3. myquota

4.  
from helper.validation import validate_messagePassing
validate_messagePassing()

<details>
  <summary>Grokfast</summary>

```python
### Imports
from collections import deque
from typing import Dict, Optional, Literal
import torch
import torch.nn as nn


### Grokfast
def gradfilter_ema(
    m: nn.Module,
    grads: Optional[Dict[str, torch.Tensor]] = None,
    alpha: float = 0.99,
    lamb: float = 5.0,
) -> Dict[str, torch.Tensor]:
    if grads is None:
        grads = {n: p.grad.data.detach() for n, p in m.named_parameters() if p.requires_grad}

    for n, p in m.named_parameters():
        if p.requires_grad:
            grads[n] = grads[n] * alpha + p.grad.data.detach() * (1 - alpha)
            p.grad.data = p.grad.data + grads[n] * lamb

    return grads
```
</details>

## Branches
- master
- dynamic graph; during training update the graph with new neurons/clusters

## TODO

- save artifact to wandb (model_type (class/generation/both), model_weights and edge_index)
- # validate_messagePassing()
- random_internal=True, inside:
model.query(method="pass", 
         random_internal=True,
- 
        # self.errors[self.internal_indices] += 0.1
        # print("!!!!!!!!!!!!!!!!!!!!!!!! IMPORTANT")    
-                 # model.pc_conv1.log_delta_w() --> not work if Van_Zwol
- use train, val, and test datasets
- remove prints in eval_tasks;
- in helper.plot if save_local=False, set model_dir=None, and ommit saving png/log locally. 
0. Weird bugg: tracing errors (preds are weird)
0. add energy drop in "ipc"
0. self.graph = self.use_old_graph(), for SBM --> make graph save folder name the **params
0. fix bug in wandb.watch(model)
0. Check for seed match when loading existing graph 
1. parser.add_argument('--connectivity_prob', type=float, default=0.08, help='Probability of connectivity.')
2. Find a clear goal (for the project/ for life) 
3. Clear my head
4. make wandb workbook available. 
5. see `compare_class_args(IPCGraphConv, PCGraphConv)` 
6. 
   if self.edge_type.numel() > 0:
      self.log_delta_w(adjusted_delta_w if self.adjust_delta_w else delta_w, self.edge_type, log=False)
7. - make 3: sens2Sup, sup2sep, sens2sens,
- add clipping of dEdX:
- Memory-Efficient aggregations [Advanced PyTorch Geometric Tutorial 6]

8. graphx cuda


## Sources:
- https://github.com/bjornvz/PRECO/tree/main 
- https://github.com/emptydiagram/pc-graphs 

## TODO

add 
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


tc = TerminalColor()

print(f"{tc.RED}This text is red.{tc.RESET}")
print(f"{tc.GREEN}{tc.BOLD}This text is bold green.{tc.RESET}")
print(f"{tc.BLUE}{tc.UNDERLINE}This text is blue and underlined.{tc.RESET}")
print(f"{tc.YELLOW}{tc.BG_BLACK}Yellow text on black background.{tc.RESET}")
print(f"{tc.BRIGHT_MAGENTA}Bright magenta text.{tc.RESET}")
print(f"{tc.CYAN}{tc.ITALIC}{tc.BG_WHITE}Cyan italic text on white background.{tc.RESET}")

# You can also combine styles and colors:
error_message = f"{tc.RED}{tc.BOLD}ERROR:{tc.RESET} {tc.RED}Something went wrong!{tc.RESET}"
print(error_message)

warning_message = f"{tc.YELLOW}{tc.BOLD}WARNING:{tc.RESET} {tc.YELLOW}Proceed with caution.{tc.RESET}"
print(warning_message)

success_message = f"{tc.GREEN}{tc.BOLD}SUCCESS:{tc.RESET} {tc.GREEN}Operation completed.{tc.RESET}"
print(success_message)



## Optimzation for later:
- Wandb sweep
- https://github.com/rapidsai/nx-cugraph 

## File Structure

- `dataset.py` - Contains code related to data processing and creation of custom datasets.
- `eval_tasks.py` - Includes evaluation tasks such as classification, denoising, and generation.
- `evaluate.py` - Script for evaluating the trained models.
- `graphbuilder.py` - Builds different types of graphs used in training.
- `helper/` - Contains helper functions such as argument parsing, activation functions, and plotting utilities.
- `models/` - Directory containing model definitions for predictive coding and inference-based models.
- `trained_models/` - Directory to store trained models and their configurations.
- `train.py` - Script to train the model using the defined datasets and parameters.


## Logs

1. [Wandb logs](https://wandb.ai/etatar-atdamen/PredCod?nw=nwuseretataratdamen)

## Example usage 


1. Load the model and graph params.
2. Training the model 
```python
              
   from models.PC import PCGNN

   model = PCGNN(**model_params,   
      log_tensorboard=False,
      wandb_logger=run if args.use_wandb in ['online', 'run'] else None,
      debug=False, device=device)

   model.pc_conv1.set_mode("training")

   history_epoch = model.learning(batch)

   # Log energy values for this batch/epoch to wandb
   wandb.log({...})

   model.pc_conv1.restart_activity()
   ```
3. Testing the model
```python
              
   from eval_tasks import classification, denoise, occlusion, generation #, reconstruction

   model.pc_conv1.restart_activity()


   model.pc_conv1.set_mode("testing", task="reconstruction")
   test_params = {
    "model_dir": model_dir,
    "T": 300,
    "supervised_learning":True, 
    "num_samples": 5,
    "add_sens_noise": False,
   }
   MSE_values_occ = occlusion(test_loader, model, test_params)

   #### 
   test_params = {
      "model_dir": model_dir,
      "T":300,
      "supervised_learning":False, 
      "num_samples": 30,
   }
   model.pc_conv1.set_mode("testing", task="classification")
   y_true, y_pred, accuracy_mean = classification(test_loader, model, test_params)

   ```


## Trainig parameters
1. ```python              
   python -u train.py \
    --mode \                  # training /experimenting to specify where to store the model,  
    --model_type PC \         # Specifies the model type (either "PC" or "IPC")
    --normalize_msg False \   # No normalization during message passing
    --dataset_transform none \  # No dataset transformations
    --numbers_list 0,1,3,4,5,6,7 \  # Specifies the classes of digits to be used
    --N 20 \                  # Use 20 instances per class
    --supervision_label_val 10 \  # Supervision label strength
    --num_internal_nodes 1500 \  # Number of internal nodes in the graph
    --graph_type fully_connected \  # Specifies the graph type (e.g., fully connected)
    --weight_init xavier \    # Weight initialization method (e.g., xavier, uniform)
    --T 40 \                  # Number of gradient descent iterations
    --lr_values 0.001 \       # Learning rate for model parameters
    --lr_weights 0.01 \       # Learning rate for weights
    --activation_func swish \ # Activation function (e.g., swish, relu, tanh)
    --epochs 20 \             # Number of training epochs
    --batch_size 32 \         # Batch size for training
    --seed 42 \               # Random seed for reproducibility
    --optimizer False \       # Use default optimizer
   ```

## Documentation page


## Contributing

Thank u Parva

