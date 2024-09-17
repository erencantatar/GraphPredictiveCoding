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



1. Base Example: Get a free API Key at [https://example.com](https://example.com)
2. Clone the repo
   ```sh
   git clone https://github.com/github_username/repo_name.git
   ```
3. Install NPM packages
   ```sh
   npm install
   ```
4. Enter your API in `config.js`
   ```js
   const API_KEY = 'ENTER YOUR API';
   ```
5. Change git remote url to avoid accidental pushes to base project
   ```sh
   git remote set-url origin github_username/repo_name
   git remote -v # confirm the changes
   ```

## Usage
python train.py --batch_size 32 --epochs 5 --dataset_transform normalize_mnist_mean_std



## Contributing

Thank u Parva

## License

[MIT](https://choosealicense.com/licenses/mit/)