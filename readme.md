# Graph Predictive Coding

Foobar is a Python library for dealing with word pluralization.

## Snellius cheatsheat 

1. srun --partition=gpu --gpus=1 --ntasks=1 --cpus-per-task=18 --time=02:00:00 --pty bash -i
2. accinfo
3. myquota


## TODO


1. parser.add_argument('--connectivity_prob', type=float, default=0.08, help='Probability of connectivity.')
2. Find a clear goal (for the project/ for life) 
3. Clear my head
4. make wandb workbook available. 


## File Structure

- `dataset.py` - Contains code related to data processing and creation of custom datasets.
- `eval_tasks.py` - Includes evaluation tasks such as classification, denoising, and generation.
- `evaluate.py` - Script for evaluating the trained models.
- `graphbuilder.py` - Builds different types of graphs used in training.
- `helper/` - Contains helper functions such as argument parsing, activation functions, and plotting utilities.
- `models/` - Directory containing model definitions for predictive coding and inference-based models.
- `trained_models/` - Directory to store trained models and their configurations.
- `train.py` - Script to train the model using the defined datasets and parameters.



## Example usage 


1. Load the model and graph params.
2. ```python
              
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
3. 


## Usage
1. ```python              
srun python -u train.py \
    --model_type PC \  # Specifies the model type (either "PC" or "IPC")
    --normalize_msg False \  # No normalization during message passing
    --dataset_transform none \  # No dataset transformations
    --numbers_list 0,1,3,4,5,6,7 \  # Specifies the classes of digits to be used
    --N 20 \  # Use 20 instances per class
    --supervision_label_val 10 \  # Supervision label strength
    --num_internal_nodes 1500 \  # Number of internal nodes in the graph
    --graph_type fully_connected \  # Specifies the graph type (e.g., fully connected)
    --weight_init xavier \  # Weight initialization method (e.g., xavier, uniform)
    --T 40 \  # Number of gradient descent iterations
    --lr_values 0.001 \  # Learning rate for model parameters
    --lr_weights 0.01 \  # Learning rate for weights
    --activation_func swish \  # Activation function (e.g., swish, relu, tanh)
    --epochs 20 \  # Number of training epochs
    --batch_size 32 \  # Batch size for training
    --seed 42 \  # Random seed for reproducibility
    --optimizer False  # Use default optimizer
   ```


## Contributing

Thank u Parva

## License

[MIT](https://choosealicense.com/licenses/mit/)