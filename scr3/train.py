import torch
import torchvision
from torch_geometric.data import Data
# from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader  # PyG's DataLoader

import matplotlib.pyplot as plt
from helper.plot import plot_model_weights, plot_energy_graphs


# args = {
#     "model_type": "IPC",
#     # "graph_type": "stochastic_block",  # Type of graph
#     "update_rules": "Van_Zwol",  # Update rules for learning
#     # "graph_type": "fully_connected",  # Type of graph
#     "graph_type": "single_hidden_layer",  # Type of graph
#     "discriminative_hidden_layers": [32, 16],  # Hidden layers for discriminative model
#     "generative_hidden_layers": [0],  # Hidden layers for generative model

#     # "discriminative_hidden_layers": [0],  # Hidden layers for discriminative model
#     # "generative_hidden_layers": [100, 50],  # Hidden layers for generative model

#     "delta_w_selection": "all",  # Selection strategy for weight updates
#     "weight_init": "fixed 0.001 0.001",  # Weight initialization method
#     "use_grokfast": True,  # Whether to use GrokFast
#     "optimizer": 1.0,  # Optimizer setting
#     "remove_sens_2_sens": False,  # Remove sensory-to-sensory connections
#     "remove_sens_2_sup": False,  # Remove sensory-to-supervised connections
#     "set_abs_small_w_2_zero": False,  # Set small absolute weights to zero
#     "mode": "experimenting",  # Mode of operation (training/experimenting)
#     "use_wandb": "online",  # WandB logging mode
#     "tags": "PC_vs_IPC",  # Tags for logging
#     "use_bias": False,  # Whether to use bias
#     "normalize_msg": False,  # Normalize message passing
#     "dataset_transform": ["normalize_mnist_mean_std"],  # Data transformations
#     "numbers_list": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],  # Classes to include
#     "N": "all",  # Number of samples per class
#     "supervision_label_val": 1,  # Value assigned for supervision
#     "num_internal_nodes": 500,  # Number of internal nodes in the network
#     "T": 5,  # Number of inference iterations
#     "lr_values": 0.01,  # Learning rate for value updates
#     "lr_weights": 0.00001,  # Learning rate for weight updates
#     "activation_func": "swish",  # Activation function
#     "epochs": 10,  # Number of training epochs
#     # "batch_size": 0,  # Batch size for training; fine for discriminative
#     # "batch_size": 50,  # Batch size for training
#     # "batch_size": 200,  # Batch size for training
#     "batch_size": 100,  # Batch size for training
#     "seed": 2,  # Random seed
# }


# class Args:
#     def __init__(self, **entries):
#         self.__dict__.update(entries)


# # Create an object from the dictionary
# args = Args(**args)


import os
import argparse
from helper.args import true_with_float, valid_str_or_float, valid_int_or_all, valid_int_list, str2bool, validate_weight_init
from helper.activation_func import activation_functions
from graphbuilder import graph_type_options

# allowed_tasks = ['classification', 'generation', 'denoise', 'occlusion']
allowed_tasks = ['classification', 'generation', 'denoise', 'occlusion']

# Parsing command-line arguments
parser = argparse.ArgumentParser(description='Train a model with specified parameters.')

# Training mode 
parser.add_argument('--mode', choices=['training', 'experimenting'], required=True,  help="Mode for training the model or testing new features.")
parser.add_argument('--tasks',nargs='+',choices=allowed_tasks,required=True,help=f"use allChoose one or more tasks from: {allowed_tasks}")

# -----dataset----- 
#data_dir default --data_dir default to $TMPDIR
parser.add_argument('--data_dir', type=str, default='../data', help='Path to the directory to store the dataset. Use $TMPDIR for scratch. or use ../data ')

parser.add_argument('--dataset_transform', nargs='*', default=[], help='List of transformations to apply.', choices=['normalize_min1_plus1', 'normalize_mnist_mean_std', 'random_rotation', 'none', 'None'])
parser.add_argument('--numbers_list', type=valid_int_list, default=[0, 1, 3, 4, 5, 6, 7], help="A comma-separated list of integers describing which distinct classes we take during training alone")
parser.add_argument('--N', type=valid_int_or_all, default=20, help="Number of distinct trainig images per class; greater than 0 or the string 'all' for all instances o.")

parser.add_argument('--supervision_label_val', default=10, type=int, required=True, help='An integer value.')


## -----graph-----  
parser.add_argument('--num_internal_nodes', type=int, default=1500, help='Number of internal nodes for fully connected graph. Otherwise, specify the number of internal nodes for other graph types.')
parser.add_argument('--graph_type', type=str, default="fully_connected", help='Type of Graph', choices=list(graph_type_options.keys()))
parser.add_argument('--remove_sens_2_sens', type=str2bool, required=True, help='Whether to remove sensory-to-sensory connections.')
parser.add_argument('--remove_sens_2_sup', type=str2bool, required=True, help='Whether to remove sensory-to-supervised connections.')
parser.add_argument('--discriminative_hidden_layers', type=valid_int_list, default=None, 
                    help="Optional: Comma-separated list of integers specifying the number of nodes in each discriminative hidden layer. Only used if graph_type is 'single_hidden_layer'.")
parser.add_argument('--generative_hidden_layers', type=valid_int_list, default=None, 
                    help="Optional: Comma-separated list of integers specifying the number of nodes in each generative hidden layer. Only used if graph_type is 'single_hidden_layer'.")

# --MessagePassing--
parser.add_argument('--normalize_msg', choices=['True', 'False'], required=True,  help='Normalize message passing, expected True or False')

# -----model-----  
parser.add_argument('--model_type', type=str, default="PC", help='Incremental_learning inside the PC model ', choices=["PC", "IPC"])
# parser.add_argument("--weight_init", type=str, default="fixed 0.001", help="Initialization method and params for weights")
parser.add_argument("--weight_init", type=validate_weight_init, default="fixed 0.001", help="Initialization method and params for weights")

parser.add_argument('--use_bias',  choices=['True', 'False'], required=True, help="....")
parser.add_argument("--bias_init", type=str, default="", required=False, help="ege. fixed 0.0 Initialization method and params for biases")

parser.add_argument('--T_train', type=int, default=5, help='Number of iterations for gradient descent.')
parser.add_argument('--T_test', type=int, default=10, help='Number of iterations for gradient descent.')

# TODO make T_train and T_test
# parser.add_argument('--T', type=int, default=40, help='Number of iterations for gradient descent.')
parser.add_argument('--lr_values', type=float, default=0.5, help='Learning rate values (alpha).')
parser.add_argument('--lr_weights', type=float, default=0.00001, help='Learning rate weights (gamma).')
parser.add_argument('--activation_func', default="swish", type=str, choices=list(activation_functions.keys()), required=True, help='Choose an activation function: tanh, relu, leaky_relu, linear, sigmoid, hard_tanh, swish')

# update rules
# parser str choices "Van_Zwol" or "salvatori", "vectorized"
parser.add_argument('--update_rules', type=str, default="vanZwol_AMB", choices=["vanZwol_AMB", "MP_AMB"], help="Choose the update rules for the model equations")

parser.add_argument('--delta_w_selection', type=str, required=True, choices=["all", "internal_only"], help="Which weights to optimize in delta_w")
parser.add_argument('--use_grokfast', type=str, default="False", choices=["True", "False"], help="GroKfast fast and slow weights before using the optimizer")

# -----training----- 
parser.add_argument('--epochs', type=int, default=5, help='Number of epochs to train.')
parser.add_argument('--batch_size', type=int, default=1, help='Batch size.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--optimizer', type=true_with_float, default=False,
                    help="Either False or, if set to True, requires a float value.")

# ---after training-----
parser.add_argument('--set_abs_small_w_2_zero',  choices=['True', 'False'], required=True, help="....")

# logging 
import wandb
parser.add_argument('--use_wandb', type=str, default="disabled", help='Wandb mode.', choices=['shared', 'online', 'run', 'dryrun', 'offline', 'disabled'])
parser.add_argument('--tags', type=str, default="", help="Comma-separated tags to add to wandb logging (e.g., 'experiment,PC,test')")

args = parser.parse_args()

# Using argparse values
torch.manual_seed(args.seed)

generator_seed = torch.Generator()
generator_seed.manual_seed(args.seed)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
print(f"Seed used", args.seed)
if torch.cuda.is_available():
    print("Device name: ", torch.cuda.get_device_name(0))



torch.set_default_dtype(torch.float32)  # Ensuring consistent precision

# Use compiled model for speed optimization (if PyTorch 2.0+)
USE_TORCH_COMPILE = True


# Make True of False bool
args.normalize_msg = args.normalize_msg == 'True'
args.use_bias = args.use_bias == 'True'
args.set_abs_small_w_2_zero = args.set_abs_small_w_2_zero == 'True'
args.grokfast = args.use_grokfast == 'True'

tags_list = args.tags.split(",") if args.tags else []
if 'all' in args.tasks:
    args.tasks = allowed_tasks


# Using argparse values
torch.manual_seed(args.seed)

generator_seed = torch.Generator()
generator_seed.manual_seed(args.seed)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
print(f"Seed used", args.seed)
if torch.cuda.is_available():
    print("Device name: ", torch.cuda.get_device_name(0))

print("---------------model.pc_conv1.log_delta_w() turned off-----------------")

# Make True of False bool
args.normalize_msg = args.normalize_msg == 'True'
args.use_bias = args.use_bias == 'True'
args.set_abs_small_w_2_zero = args.set_abs_small_w_2_zero == 'True'
args.grokfast = args.use_grokfast == 'True'

tags_list = args.tags.split(",") if args.tags else []

import torchvision.transforms as transforms
import numpy as np


# The ToTensor() transformation is the one responsible for scaling (MNIST) images to the range [0, 1].
transform_list = [
    transforms.ToTensor()
]

if args.dataset_transform:

    if "normalize_min1_plus1" in args.dataset_transform:
        transform_list.append(transforms.Normalize((0.5,), (0.5,)))

    if "normalize_mnist_mean_std" in args.dataset_transform:
        transform_list.append(transforms.Normalize((0.1307,), (0.3081,)))
    
    if "random_rotation" in args.dataset_transform:
        transform_list.append(transforms.RandomRotation(degrees=20))
    

# # Create the transform
print("TODO ADD COMPASE TRANSFORMS")
# transform = transforms.Compose(transform_list)

# mnist_trainset = torchvision.datasets.MNIST(root=args.data_dir, train=True, download=True, transform=transform)
# mnist_testset  = torchvision.datasets.MNIST(root=args.data_dir, train=False, download=True, transform=transform)



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')




######################################################################################################### 
####                                            Dataset                                             #####
######################################################################################################### 


# print("USSSSSING SUBSEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEET")

# CustomGraphDataset params
# dataset_params = {
#     "mnist_dataset":            mnist_trainset,
#     # "mnist_dataset":            mnist_train_subset,
#     "supervised_learning":      True,
#     "numbers_list":             args.numbers_list,
#     "same_digit":               False,
#     "add_noise":                False,
#     "noise_intensity":          0.0,
#     "N":                        args.N,     # taking the first n instances of each digit or use "all"

#     "edge_index":               None,
#     "supervision_label_val":    args.supervision_label_val,         # Strength of label signal within the graph. MNIST ~0-1, label_vector[label] = self.supervision_label_val
# } 

print("------------------Importing Graph Params ---------------- ")
from graphbuilder import graph_type_options

# Define the graph type
# Options: "fully_connected", "fully_connected_w_self", "barabasi", "stochastic_block", "fully_connected_no_sens2sup"
graph_params = {
    "internal_nodes": args.num_internal_nodes,  # Number of internal nodes
    "supervised_learning": True,  # Whether the task involves supervised learning
    "graph_type": {    
        "name": args.graph_type, # Options: "fully_connected", "fully_connected_w_self", "barabasi", "stochastic_block"
        "params": graph_type_options[args.graph_type]["params"], 
        # "params_general": {
        #     "remove_sens_2_sens": args.remove_sens_2_sens,  # Set from command line
        #     "remove_sens_2_sup": args.remove_sens_2_sup,    # Set from command line
        #     },
        },  
    "seed": args.seed,   
}

eval_generation, eval_classification, eval_denoise, eval_occlusion = True, True, 0, 0 


# add graph specific info: 
# print("zzz", args.remove_sens_2_sens, args.remove_sens_2_sup)
graph_params["graph_type"]["params"]["remove_sens_2_sens"] = args.remove_sens_2_sens  
graph_params["graph_type"]["params"]["remove_sens_2_sup"]  = args.remove_sens_2_sup 


if graph_params["graph_type"]["name"] == "stochastic_block":
    
    # override internal nodes if doing clustering
    graph_params["internal_nodes"] = (graph_params["graph_type"]["params"]["num_communities"] * graph_params["graph_type"]["params"]["community_size"])
#    self.INTERNAL = range(self.SENSORY_NODES, (self.SENSORY_NODES+sum(sizes)))
#     self.num_internal_nodes = sum(self.INTERNAL)

if graph_params["graph_type"]["name"] == "stochastic_block_hierarchy":
    raise ValueError("Not implemented yet")


if graph_params["graph_type"]["name"] in ["custom_two_branch","two_branch_graph"]:
    # Configure internal nodes for two_branch_graph
    # This assumes two branches with specified configurations
    branch1_layers, branch1_clusters_per_layer, branch1_nodes_per_cluster = graph_params["graph_type"]["params"]["branch1_config"]
    branch2_layers, branch2_clusters_per_layer, branch2_nodes_per_cluster = graph_params["graph_type"]["params"]["branch2_config"]
    
    # Calculate total internal nodes for both branches
    # Branch 1
    branch1_internal_nodes = branch1_layers * branch1_clusters_per_layer * branch1_nodes_per_cluster
    # Branch 2 (Reversed order)
    branch2_internal_nodes = branch2_layers * branch2_clusters_per_layer * branch2_nodes_per_cluster
    
    # The total number of internal nodes will be the sum of both branches
    graph_params["internal_nodes"] = branch1_internal_nodes + branch2_internal_nodes



if graph_params["graph_type"]["name"] in ["single_hidden_layer"]:

    
    # # discriminative_hidden_layers = [0]  # Adjust if layers change
    # generative_hidden_layers = [50, 100, 200] # Adjust if layers change
    
    # # Calculate total number of nodes
    # discriminative_hidden_layers = [200, 100, 50]  # Adjust if layers change
    # # generative_hidden_layers = [0] # Adjust if layers change

    discriminative_hidden_layers = args.discriminative_hidden_layers or [200, 100, 50]  # Default if not provided
    generative_hidden_layers = args.generative_hidden_layers or [50, 100, 200]  # Default if not provided


    num_discriminative_nodes = sum(discriminative_hidden_layers)
    num_generative_nodes = sum(generative_hidden_layers)

    graph_params["graph_type"]["params"]["discriminative_hidden_layers"] = discriminative_hidden_layers
    graph_params["graph_type"]["params"]["generative_hidden_layers"]  = generative_hidden_layers
   
    graph_params["internal_nodes"] = num_discriminative_nodes + num_generative_nodes

    # edge_index, N = test_single_hidden_layer(discriminative_hidden_layers, generative_hidden_layers,
    #                                         no_sens2sens=True, no_sens2supervised=True)

    if sum(discriminative_hidden_layers) == 0:
        eval_classification = False
    if sum(generative_hidden_layers) == 0:
        eval_generation = False

    # TODO ; still unsure about which graph does which task
    eval_generation, eval_classification, eval_denoise, eval_occlusion = True, True, 0, 0 

# if graph_params["graph_type"]["name"] not in ["single_hidden_layer"]:
#     # Ensure these arguments are not specified for other graph types
#     if "discriminative_hidden_layers" in args:
#         assert args.discriminative_hidden_layers is None, \
#             "The argument --discriminative_hidden_layers can only be used if graph_type is 'single_hidden_layer'."
#         assert args.generative_hidden_layers is None, \
#             "The argument --generative_hidden_layers can only be used if graph_type is 'single_hidden_layer'."

# if graph_params["graph_type"]["name"] in ["custom_two_branch", "two_branch_graph"]:
#     # Configure internal nodes for two_branch_graph
#     # This assumes two branches with specified configurations
#     branch1_config = graph_params["graph_type"]["params"]["branch1_config"]
#     branch2_config = graph_params["graph_type"]["params"]["branch2_config"]
    
#     # Calculate total internal nodes for both branches
#     # Sum up the total internal nodes for Branch 1
#     branch1_internal_nodes = sum([clusters * nodes_per_cluster for clusters, nodes_per_cluster in branch1_config])
    
#     # Sum up the total internal nodes for Branch 2 (Reversed order if required)
#     branch2_internal_nodes = sum([clusters * nodes_per_cluster for clusters, nodes_per_cluster in branch2_config])
    
#     # The total number of internal nodes will be the sum of both branches
#     graph_params["internal_nodes"] = branch1_internal_nodes + branch2_internal_nodes


# TASK = []
# if args.graph_type == "fully_connected" or args.graph_type == "stochastic_block":
#     TASK = ["classification", "generation"]

# if args.graph_type == "single_hidden_layer":
#     if sum(args.discriminative_hidden_layers) > 0:
#         TASK.append("classification")
#     else:
#         TASK.append("generation")
        

TASK = args.tasks

print("----------TASK------------------")
print("TASK", TASK)
        
import os 
# if not exist make folder trained_models/args.graph_type/
if not os.path.exists(f"trained_models/{args.graph_type}"):
    # create 
    os.makedirs(f"trained_models/{args.graph_type}")


print("graph_params 1 :", graph_params)

from graphbuilder import GraphBuilder
# from helper.plot import plot_adj_matrix

print("graph_params", graph_params)
graph = GraphBuilder(**graph_params)

# 🔹 Create Graph Structure (Assuming GraphBuilder is already defined)
graph = GraphBuilder(**graph_params)
single_graph = graph.edge_index  # Use the precomputed edge index


import wandb 
optimizer_str = str(args.optimizer) if isinstance(args.optimizer, float) else str(args.optimizer)

model_params_name = (
    f"{args.model_type}_"
    f"nodes_{graph_params['internal_nodes']}_" 
    f"T_{args.T_train}_"
    f"lr_vals_{args.lr_values}_"
    f"lr_wts_{args.lr_weights}_"
    f"bs_{args.batch_size}_"
    f"act_{args.activation_func}_"
    f"init_{args.weight_init}_"
    f"graph_{args.graph_type}_"
    f"sup_{args.supervision_label_val}_"
    f"norm_{args.normalize_msg}_"
    f"nums_{'_'.join(map(str, args.numbers_list))}_"
    f"N_{args.N}_"
    f"ep_{args.epochs}_"
    f"opt_{optimizer_str}_"
    f"trans_{'_'.join(args.dataset_transform) if args.dataset_transform else 'none'}"
)
model_params_name_full = (
    f"model_{args.model_type}_"
    f"num_internal_nodes_{graph_params['internal_nodes']}_"
    f"T_{args.T_train}_"
    f"lr_values_{args.lr_values}_"
    f"lr_weights_{args.lr_weights}_"
    f"batch_size_{args.batch_size}_"
    f"activation_{args.activation_func}_"
    f"weight_init_{args.weight_init}_"
    f"graph_type_{args.graph_type}_"
    f"supervision_val_{args.supervision_label_val}_"
    f"normalize_msg_{args.normalize_msg}_"
    f"numbers_list_{'_'.join(map(str, args.numbers_list))}_"
    f"N_{args.N}_"
    f"epochs_{args.epochs}_"
    f"optimizer_{optimizer_str}_"
    f"dataset_transform_{'_'.join(args.dataset_transform) if args.dataset_transform else 'none'}"
)
model_params_short = f"{args.model_type}_{args.graph_type}_T_{args.T_train}_lr_w_{args.lr_weights}_lr_val_{args.lr_values}"

from datetime import datetime
date_hour = datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
path = ""
# Initialize base path depending on mode (training or experimenting)
if args.mode == "experimenting":
    path += f"trained_models_experimenting/"
elif args.mode == "training":
    path += f"trained_models/"
else:
    raise ValueError("Invalid mode")

path += f"{args.model_type.lower()}/{graph_params['graph_type']['name']}/"
  
# Modify the path based on the graph configuration (removing sens2sens or sens2sup)
if graph_params["graph_type"]["params"]["remove_sens_2_sens"] and graph_params["graph_type"]["params"]["remove_sens_2_sup"]:
    graph_type_ = "_no_sens2sens_no_sens2sup"
elif graph_params["graph_type"]["params"]["remove_sens_2_sens"]:
    graph_type_ = "_no_sens2sens"
elif graph_params["graph_type"]["params"]["remove_sens_2_sup"]:
    graph_type_ = "_no_sens2sup"
else:
    graph_type_ = "_normal"  # If neither are removed, label the folder as 'normal'

path += graph_type_
# Append graph type, model parameters, and timestamp to the path
path += f"/{model_params_name}_{date_hour}/"
model_dir = path

config_dict = vars(args)  # Convert argparse Namespace to dictionary

config_dict.update({
    'graph_type_conn': graph_type_,  # Adding model type manually
    'remove_sens_2_sens_': args.remove_sens_2_sens,  # Custom boolean flag for data augmentation
    'remove_sens_2_sup_': args.remove_sens_2_sup,  # Custom boolean flag for data augmentation
    "checkpoint_dir": model_dir,  # Track where model checkpoints are saved
    **graph_params["graph_type"]["params"],  # Merge dynamic graph parameters
})

run = wandb.init(
    mode=args.use_wandb,
    # entity="Erencan Tatar", 
    project=f"PredCod",
    # name=f"T_{args.T}_lr_value_{args.lr_values}_lr_weights_{args.lr_weights}_",
    name=f"{model_params_short}_{date_hour}",
    # id=f"{model_params_short}_{date_hour}",
    tags=tags_list,  # Add tags list here

    dir=model_dir,
    # tags=["param_search", str(model_params["weight_init"]), model_params["activation"],  *learning_params['dataset_transform']], 
    # Track hyperparameters and run metadata
    config=config_dict,  # Pass the updated config dictionary to wandb.init
)







# from torch_geometric.data import Data
# from torch_geometric.loader import DataLoader


# def mnist_to_graph(image_tensor, label, graph):
#     """Convert MNIST image to a graph using a predefined edge_index from GraphBuilder."""
    
#     image = image_tensor.squeeze(0)  # (1, 28, 28) → (28, 28)

#     num_vertices = graph.num_vertices  # Get total node count from GraphBuilder
#     # num_sensory_nodes = graph.sensory_indices.shape[0]
#     # num_internal_nodes = graph.internal_indices.shape[0]
#     # num_supervision_nodes = graph.supervision_indices.shape[0]

#     # 🔹 Create node feature matrix
#     x = torch.zeros(num_vertices, 1)  # Initialize all as zeros
#     x[graph.sensory_indices] = image.view(-1, 1)  # Assign sensory values

#     # 🔹 One-hot encode the label
#     y = torch.zeros(10)
#     y[label] = 1

#     # 🔹 Assign supervision nodes the label encoding
#     for i, supervision_idx in enumerate(graph.supervision_indices):
#         x[supervision_idx] = y[i]

#     return Data(
#         x=x, edge_index=graph.edge_index, y=label,
#         # edge_attr=torch.ones(graph.edge_index.size(1)),  # Default edge weights as ones
#         sensory_indices=graph.sensory_indices,
#         internal_indices=graph.internal_indices,
#         supervision_indices=graph.supervision_indicesf
#     )


# from torch.utils.data import random_split



# DATASET_PATH = "../data"
# batch_size = args.batch_size

# # 🔹 Load MNIST dataset
# transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

# train_dataset = torchvision.datasets.MNIST(root="../data", train=True, transform=transform, download=True)
# test_set = torchvision.datasets.MNIST(root="../data", train=False, transform=transform, download=True)

# # train_set, val_set = torch.utils.data.random_split(train_dataset, [50000, 10000])
# # train_indices = train_subset_indices(train_set, 10, no_per_class=0)  # Set `no_per_class` as needed

# generator = torch.Generator().manual_seed(args.seed)  # Set a fixed seed
# train_set, val_set = random_split(train_dataset, [50000, 10000], generator=generator)

# # PYG DataLoaders
# NUM_WORKERS = 4    # higher number of workers can speed up data loading
# # NUM_WORKERS = 8    # higher number of workers can speed up data loading
# # NUM_WORKERS = 16    # higher number of workers can speed up data loading
# print("--------NUM_WORKERS----------", NUM_WORKERS)
# # train_loader = GeoDataLoader(train_graph_dataset, batch_size=batch_size, collate_fn=custom_collate_fn, shuffle=True, pin_memory=True, num_workers=4, drop_last=True)
# # val_loader = GeoDataLoader(val_graph_dataset, batch_size=batch_size, collate_fn=custom_collate_fn, shuffle=False,  pin_memory=True, num_workers=4, drop_last=True)
# # test_loader = GeoDataLoader(test_graph_dataset, batch_size=batch_size, collate_fn=custom_collate_fn, shuffle=False,  pin_memory=True, num_workers=4, drop_last=True)

# import multiprocessing

# # NUM_WORKERS = min(8, multiprocessing.cpu_count())  # Adjust based on CPU cores
# # NUM_WORKERS = min(16, os.cpu_count() // 2)  # Use half of CPU cores for optimal performance
# NUM_WORKERS = 4  # Use half of CPU cores for optimal performance


# print("Precompute all MNIST graphs using the predefined graph structure")
# # 🔹 Precompute all MNIST graphs using the predefined graph structure
# # graph_data = [mnist_to_graph(img, int(label), graph) for img, label in mnist]

# # subset_indices = list(range(1000))  # First 100 samples
# # train_set = torch.utils.data.Subset(train_set, subset_indices)
# # val_set = torch.utils.data.Subset(val_set, subset_indices)

# graph_data_train    = [mnist_to_graph(img, int(label), graph) for img, label in train_set]
# graph_data_val      = [mnist_to_graph(img, int(label), graph) for img, label in val_set]
# # graph_data_test     = [mnist_to_graph(img, int(label), graph) for img, label in mnist_subset]

# print("Number of graphs created:", len(graph_data_train))

# # 🔹 Create DataLoader for efficient batching
# train_loader = DataLoader(graph_data_train, batch_size=args.batch_size, shuffle=True, 
#     pin_memory=True, num_workers=NUM_WORKERS, drop_last=True, persistent_workers=True)
# val_loader = DataLoader(graph_data_val, batch_size=args.batch_size, shuffle=True, 
#     pin_memory=True, num_workers=NUM_WORKERS, drop_last=True, persistent_workers=True)
# # test_loader = DataLoader(graph_data, batch_size=args.batch_size, shuffle=True)




# for batch in train_loader:

#     # --- 🔹 Load a Batch ---
#     # batch = next(iter(graph_loader))
#     print("Batch Shapes:")
#     print("x shape:", batch.x.shape)  # (total_nodes_in_batch, 1)
#     # print("Edge Index shape:", batch.edge_index.shape)  # (2, total_edges)
#     # print("y shape:", batch.y.shape)  # (batch_size, 10)

#     test_img = batch.x[0:784].view(28, 28).detach().cpu().numpy()
#     plt.imshow(test_img, cmap='gray')
#     plt.title(f"digit: {batch.y[0].item()}")
#     plt.savefig("test_img.png")
#     break 




class GraphFormattedMNIST(torch.utils.data.Dataset):
    def __init__(self, dataset, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Precompute dataset transformation (instead of transforming on-the-fly)
        self.data = []
        self.labels = []

        one_hot_lookup = torch.eye(output_size)  # One-hot encoding lookup

        for img, label in dataset:
            image_flat = img.view(-1).to(torch.float32)  # Flattened
            internal_nodes = torch.zeros(hidden_size, dtype=torch.float32)  # Internal nodes
            one_hot_label = one_hot_lookup[label]

            # Precomputed graph data
            graph_data = torch.cat([image_flat, internal_nodes, one_hot_label])

            self.data.append(graph_data)
            self.labels.append(label)

        # Convert to tensors for fast access
        self.data = torch.stack(self.data)  # Shape: [N, input_size + hidden_size + output_size]
        self.labels = torch.tensor(self.labels, dtype=torch.long)  # Shape: [N]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]  # No transformations here!


DATASET_PATH = "../data"

from torch.utils.data import DataLoader, random_split

batch_size = args.batch_size

# Define dataset sizes
input_size = 784  # Flattened MNIST image
# hidden_size = 32+16  # Number of internal nodes in graph

print(graph_params["internal_nodes"] )
# print(graph.INTERNAL)
# print(graph.num_internal_nodes)

# if type(graph.num_internal_nodes) == int:
#     hidden_size = graph.num_internal_nodes
# if type(graph.num_internal_nodes) == list:
#     hidden_size = len(graph.num_internal_nodes)  # Number of internal nodes in graph

hidden_size = graph_params["internal_nodes"] 
output_size = 10  # Number of classes

# Load the raw MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

base_train_dataset = torchvision.datasets.MNIST(root=DATASET_PATH, train=True, transform=transform, download=True)
base_test_dataset = torchvision.datasets.MNIST(root=DATASET_PATH, train=False, transform=transform, download=True)

print("hidden_size", hidden_size)
# Wrap it in the GraphFormattedMNIST class
train_dataset = GraphFormattedMNIST(base_train_dataset, input_size, hidden_size, output_size)
test_dataset = GraphFormattedMNIST(base_test_dataset, input_size, hidden_size, output_size)

# Split the dataset
train_set, val_set = random_split(train_dataset, [50000, 10000])


# Create DataLoaders
train_loader = DataLoader(train_set, batch_size=args.batch_size, drop_last=False, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=4, pin_memory=True)


# graph_type [FC, generative, discriminative, SBM, (SBM_hierarchy, custom_two_branch, two_branch_graph) ]
# task       [classification, generation, denoise, occlusion]







print("done dataloader")


# for batch in train_loader:

#     # --- 🔹 Load a Batch ---
#     # batch = next(iter(graph_loader))
#     print("Batch Shapes:")
#     print("x shape:", batch.x.shape)  # (total_nodes_in_batch, 1)
#     # print("Edge Index shape:", batch.edge_index.shape)  # (2, total_edges)
#     # print("y shape:", batch.y.shape)  # (batch_size, 10)

#     test_img = batch.x[0:784].view(28, 28).detach().cpu().numpy()
#     plt.imshow(test_img, cmap='gray')
#     plt.title(f"digit: {batch.y[0].item()}")
#     plt.savefig("test_img.png")
#     break 



# from helper.vanZwol_optim import *

################################# discriminative model lr         ##########################################
################################# generative model lr         ##########################################


# Inference
f = "tanh"
# f = "relu"
lr_x = 0.5                  # inference rate                   # inference rate 
T_train = 5                 # inference time scale
T_test = 10              # unused for hierarchical model
incremental = True          # whether to use incremental EM or not
use_input_error = False     # whether to use errors in the input layer or not
# use_input_error = True      # whether to use errors in the input layer or not

# Learning
lr_w = 0.00001      
# Learning
# lr_w = 0.00001              # learning rate hierarchial model
# lr_w = 0.000001              # learning rate generative model



################################# fully connected model lr         ##########################################

# # GOOD FOR CLASSIFCATION
# lr_x = 0.01                  # inference rate                   # inference rate 
# lr_w = 0.00001              # learning rate hierarchial model
# lr_w = 0.0001      

# # # OKAY FOR GENRATION
# # lr_x = 0.5                  # inference rate                   # inference rate 
# # lr_x = 1                 # inference rate                   # inference rate 
# T_train = 10                 # inference time scale
# # T_train = 50                 # inference time scale
# T_test = 15              # unused for hierarchical model
# incremental = True          # whether to use incremental EM or not
# use_input_error = False     # whether to use errors in the input layer or not

# # Learning
# lr_w = 0.00001      
# # Learning
# lr_w = 0.00001              # learning rate hierarchial model
# # lr_w = 0.000001              # learning rate generative model
# weight_decay = 0             
# grad_clip = 1
# batch_scale = False

# lr_x = 0.1                  # inference rate                   # inference rate 
# lr_w = 0.000001              # learning rate hierarchial model


weight_decay = 0             
grad_clip = 1
batch_scale = False
 
# import helper.vanZwol_optim as optim

# vertices = [784, 48, 10] # input, hidden, output
# mask = get_mask_hierarchical([784,32,16,10])

from torch_geometric.utils import to_dense_adj
adj_matrix_pyg = to_dense_adj(graph.edge_index)[0]

print("adj_matrix_pyg", adj_matrix_pyg.shape)

from model import PCgraph

model_params = {
    "delta_w_selection": args.delta_w_selection,  # "all" or "internal_only"
    "use_bias": args.use_bias,
    "lr_params": (args.lr_values, args.lr_weights),
    #   (args.lr_gamma, args.lr_alpha), 
    "batch_size": train_loader.batch_size, 
 
 }



model_params = {

    "graph_type": args.graph_type,
    "task": TASK,

    "activation": args.activation_func,  
    "device": device,
    "num_vertices": graph.num_vertices,
    "num_internal": sum(graph.internal_indices),
    "adj": adj_matrix_pyg,             # 2d Adjacency matrix
    "edge_index": graph.edge_index,    # [2, num_edges] edge index
    "batch_size": batch_size,
    # "mask": mask,
    # "learning_rates": (lr_x, lr_w),
    "learning_rates": (args.lr_values, args.lr_weights),
    # "T_train": T_train,
    # "T_test": T_test,
    "T": (args.T_train, args.T_test),  # Number of iterations for gradient descent. (T_train, T_test)

    "incremental_learning": True if args.model_type == "IPC" else False, 
    "use_input_error": False,

    "update_rules": args.update_rules,  # "Van_Zwol" or "salvatori", "vectorized"
    "weight_init": args.weight_init,   # xavier, 'uniform', 'based_on_f', 'zero', 'kaiming'

    # "edge_type":  custom_dataset_train.edge_type,
    # "use_learning_optimizer": use_learning_optimizer,    # False or [0], [(weight_decay=)]
    # "use_grokfast": args.grokfast,  # False or True
    # "clamping": None , # (0, torch.inf) or 'None'
}

PCG = PCgraph(**model_params, 
                wandb_logging=run if args.use_wandb in ['online', 'run'] else None,
                debug=False)


# PCG = PCgraph(f,
#         device=device,
#         num_vertices=graph.num_vertices,
#         num_internal=sum(graph.internal_indices),
#         adj=adj_matrix_pyg,
#         edge_index=graph.edge_index,
#         batch_size=batch_size,
#         # mask=mask,
#         learning_rates=(lr_x, lr_w), 
#         T_train=T_train,
#         T_test=T_test,
#         incremental=incremental, 
#         use_input_error=use_input_error,
#         )

# optimizer = Adam(
#     PCG.params,
#     learning_rate=lr_w,
#     grad_clip=grad_clip,
#     batch_scale=batch_scale,
#     weight_decay=weight_decay,
# ) 
# PCG.set_optimizer(optimizer)

PCG.init_modes(graph_type=args.graph_type, graph=graph)
PCG.set_task(TASK)

model = PCG

plot_model_weights(model, args.graph_type, model_dir=None, save_wandb="before_training")

print("------------- compile model ------------- ")
# model = torch.compile(model, disable=True) 
# torch.compile(model, dynamic=True)
model = torch.compile(model, mode="max-autotune")
model.task = ""   # classification or generation, or both 

from datetime import datetime
from tqdm import tqdm
import torch

epochs = 100
start_time = datetime.now()

train_energy, train_loss, train_acc = [], [], []
val_loss, val_acc, val_acc2 = [], [], []
num_epochs = 40

# break_num = 150 
# break_num = 250 
# break_num = int(len(train_loader) -1 )
# break_num = 100

# break_num = 1200
# break_num = 200
# break_num = 100
break_num = 50
# break_num = 20

DEVICE = device 



TASK_config = {
    "classification": 
        {"batch_break": 100, "wandb": eval_classification},
}



with torch.no_grad():

    epoch_history = []

    for epoch in tqdm(range(num_epochs), desc="Epoch Progress"):
        model.train_(epoch)
        model.epoch = epoch
        total_loss = 0
        energy = 0


        print("\n-----train_supervised-----")
        print(len(train_loader))

        # for batch_no, batch in enumerate(tqdm(train_loader, total=min(break_num, len(train_loader)), desc=f"Epoch {epoch+1} - Training", leave=False)):
        for batch_no, (X_batch, y_batch) in enumerate(tqdm(train_loader, total=min(break_num, len(train_loader)), desc=f"Epoch {epoch+1} - Training", leave=False)):

            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)  # Move to GPU
            
            # batch = batch.to(model.device)

            history = model.train_supervised(X_batch)  # history is [..., ...]
            # history = model.train_supervised(batch)  # history is [..., ...]
            # append all items in history to epoch_history
            # for energy in history:
            #     epoch_history.append(energy)

            #     wandb.log({
            #         "epoch": epoch,
            #         "Training/internal_energy_mean": energy,
            #         # "Training/sensory_energy_mean": history_epoch["sensory_energy_mean"],

            #     })


            if batch_no >= break_num:
                break
    
        #### 
        loss, acc = 0, 0
        model.test_()
        cntr = 0

        break_num_eval = 20
        if "generation" in TASK:
            break_num_eval = 1
        # break_num_eval = 10
            
        print("\n----test_iterative-----")
        accs = []
        TASK_copy = TASK.copy()

        # for batch_no, batch in enumerate(tqdm(val_loader, total=min(len(val_loader)), desc=f"Epoch {epoch+1} - Validation", leave=False)):
        for batch_no, (X_batch, y_batch)  in enumerate(tqdm(val_loader, total=len(val_loader), desc=f"Epoch {epoch+1} - Validation | {TASK}", leave=False)):
        
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)  # Move to GPU
            
            # y_pred = PCG.test_iterative(batch, eval_types=TASK_copy, remove_label=True)

            for task in TASK_copy:
                y_pred = PCG.test_iterative( (X_batch, y_batch), 
                                            eval_types=[task], remove_label=True)

            # do generation once
            if "generation" in TASK_copy and "classification" in TASK_copy:
                TASK_copy = ["classification"]

            # print("y_pred", y_pred.shape)
            # print("y_pred", y_batch.shape)
            if "classification" in TASK_copy:
                mean_correct = torch.mean((y_pred == y_batch).float()).item()
                acc += mean_correct
                accs.append(mean_correct)

            if batch_no >= break_num_eval:
                break
        
        if "classification" in TASK:
            accuracy_mean = (sum(accs) / len(accs)) if accs else 0
            val_acc.append(accuracy_mean)

            print("epoch", epoch, "accuracy_mean", accuracy_mean)

            wandb.log({
                "epoch": epoch,
                "classification/accuracy_mean": accuracy_mean,
                "classification/size":  len(accs) * args.batch_size,
            })

        # save model weights plt.imshow to "trained_models/{TASK}/weights/model_{epoch}.png"
        # make folder if not exist
        import os 
        if not os.path.exists(f"trained_models/{args.graph_type}/weights/"):
            os.makedirs(f"trained_models/{args.graph_type}/weights/")
        
        if epoch % 10 == 0:
            plot_model_weights(model, args.graph_type, model_dir=None, save_wandb=str(epoch))

        # # save weights
        # w = PCG.w.detach().cpu().numpy()
        # print("mean w", w.mean())
        # plt.imshow(w)
        # plt.colorbar()
        # plt.savefig(f"trained_models/{args.graph_type}/weights/model_{epoch}.png")
        # plt.close()


        # if "classification" in TASK:
            
        #     # Corrected validation accuracy calculations
        #     val_acc.append(acc / len(val_loader))
        #     val_acc2.append(acc / cntr)
        #     val_loss.append(loss)

        
        #     print("val_acc2", val_acc2)
        #     print("Last prediction:", y_pred)
        #     print("Last y_batch:", y_batch)
        #     print("accs", accs)
        #     print("accs", sum(accs) / len(accs))

        #     print(f"\nEpoch {epoch+1}/{num_epochs} Completed")
        #     print(f"  Validation Accuracy: {val_acc[-1]:.3f}")
        #     print(f"  Validation Accuracy (limited): {val_acc2[-1]:.3f}")


        # plot history of energy
        # import matplotlib.pyplot as plt
        # print("epoch_history", len(epoch_history))
        # plt.plot(epoch_history)

        # plt.savefig(f"trained_models/{args.graph_type}/energy_history.png")
        # plt.close()





