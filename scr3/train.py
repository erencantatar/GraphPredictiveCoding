import torch
import torchvision
from torch_geometric.data import Data
# from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader  # PyG's DataLoader

import matplotlib.pyplot as plt
from helper.plot import plot_model_weights, plot_energy_graphs
from helper.log import TerminalColor



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

parser.add_argument('--dataset_transform', nargs='*', default=[], help='List of transformations to apply.', choices=['normalize_min1_plus1', 'normalize_mnist_mean_std', 'random_rotation', 'zoom_out', 'random_translation','None'])
parser.add_argument('--numbers_list', type=valid_int_list, default=[0, 1, 3, 4, 5, 6, 7], help="A comma-separated list of integers describing which distinct classes we take during training alone")
parser.add_argument('--N', type=valid_int_or_all, default=20, help="Number of distinct trainig images per class; greater than 0 or the string 'all' for all instances o.")

parser.add_argument('--supervision_label_val', default=1, type=int, required=True, help='An integer value.')


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

parser.add_argument('--use_bias',  default="False", choices=['True', 'False'], help="use bias in the model, expected True or False")
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
parser.add_argument('--grad_clip_lr_x_lr_w',type=str,default="False False",choices=["True True", "True False", "False True", "False False"], help='Enable/disable gradient clipping for lr_x and/or lr_w. Choices are: True True, True False, False True, False False.')
parser.add_argument('--init_hidden_values', type=float, default=0.001, help='Initial value for hidden nodes in the model. This is a small value for initialization of hidden nodes. Used as mean for normal distribution at the start of each trainig sample')
# optional init_hiddden_mu 
parser.add_argument('--init_hidden_mu', type=float, default=0.0, help='Mean for the normal distribution used to initialize hidden nodes. Default is 0.0')

# required use_input_error False or True 
parser.add_argument('--use_input_error', type=str, default="False", choices=["True", "False"], help="Use input error in the model")

# -----pre-training----- 
# TODO find appropriate  lr_values and lr_weights giv


# -----training----- 
parser.add_argument('--epochs', type=int, default=5, help='Number of epochs to train.')
parser.add_argument('--batch_size', type=int, default=1, help='Batch size.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
# parser.add_argument('--use_learning_optimizer', '--optimizer', type=true_with_float, default=False,  OLD OLD ---------------------------------
#                     help="Either False or, if set to True, requires a float value for weight decay.") 
parser.add_argument('--w_decay_lr_values', type=float, default=0.0, help="Weight decay for value updates (e.g., SGD on node activities). Set to 0.0 to disable.")
parser.add_argument('--w_decay_lr_weights', type=float, default=0.0, help="Weight decay for weight updates (e.g., Adam on synaptic weights). Set to 0.0 to disable.")
                    

parser.add_argument('--break_num_train', type=int, default=200,
                    help='Max number of training steps per epoch. If set to 0, runs for the full length of the train_loader.')


# ---post-training-----
parser.add_argument('--set_abs_small_w_2_zero',  choices=['True', 'False'], required=True, help="Remove edges with small weights in the graph before eval on test set")

# logging 
import wandb
parser.add_argument('--use_wandb', type=str, default="disabled", help='Wandb mode.', choices=['shared', 'online', 'run', 'dryrun', 'offline', 'disabled'])
parser.add_argument('--tags', type=str, default="", help="Comma-separated tags to add to wandb logging (e.g., 'experiment,PC,test')")


parser.add_argument('--train_mlp', type=lambda x: x.lower() == 'true', default=False,
                    help="Set to True to enable MLP training mode")

args = parser.parse_args()

# Using argparse values
# torch.manual_seed(args.seed)
print("FOR NOW ADD RANDOM SEED TO THE MODEL")
random_seed = torch.randint(0, 10, (1,)).item()
torch.manual_seed(random_seed)


generator_seed = torch.Generator()
generator_seed.manual_seed(random_seed)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
# print(f"Seed used", args.seed)
print(f"Seed used", random_seed)
if torch.cuda.is_available():
    print("Device name: ", torch.cuda.get_device_name(0))

import os
os.environ["NETWORKX_AUTOMATIC_BACKENDS"] = "cugraph"
import networkx as nx

if hasattr(nx, "__backend__"):
    print(f"NetworkX backend in use: {nx.__backend__.__name__}")
    if "cugraph" in nx.__backend__.__name__.lower():
        print("✅ NetworkX is using the CuGraph (GPU) backend.")
    else:
        print("⚠️ NetworkX is NOT using the CuGraph backend.")
else:
    print("❌ NetworkX backend attribute not found. You're likely not using a version with backend support (3.2+ required).")



torch.set_default_dtype(torch.float32)  # Ensuring consistent precision

# Use compiled model for speed optimization (if PyTorch 2.0+)
USE_TORCH_COMPILE = True


# Make True of False bool
args.normalize_msg = args.normalize_msg == 'True'
# args.use_bias = args.use_bias == 'True'
args.use_bias = args.use_bias.lower() == 'true'

args.set_abs_small_w_2_zero = args.set_abs_small_w_2_zero == 'True'
args.grokfast = args.use_grokfast == 'True'

grad_clip_lr_x_str, grad_clip_lr_w_str = args.grad_clip_lr_x_lr_w.split()
grad_clip_lr_x = grad_clip_lr_x_str == "True"
grad_clip_lr_w = grad_clip_lr_w_str == "True"

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


    graph_params["graph_type"]["params"]["add_residual"] = True

    # edge_index, N = test_single_hidden_layer(discriminative_hidden_layers, generative_hidden_layers,
    #                                         no_sens2sens=True, no_sens2supervised=True)

    if sum(discriminative_hidden_layers) == 0:
        eval_classification = False
    if sum(generative_hidden_layers) == 0:
        eval_generation = False

    # TODO ; still unsure about which graph does which task
    eval_generation, eval_classification, eval_denoise, eval_occlusion = True, True, 0, 0 



print("graph_params 1 :", graph_params)

from graphbuilder import GraphBuilder
# from helper.plot import plot_adj_matrix

print("graph_params", graph_params)
# graph = GraphBuilder(**graph_params)

# 🔹 Create Graph Structure (Assuming GraphBuilder is already defined)
graph = GraphBuilder(**graph_params)
single_graph = graph.edge_index  # Use the precomputed edge index

from torch_geometric.utils import to_dense_adj
adj_matrix_pyg = to_dense_adj(graph.edge_index)[0]

num_vertices = adj_matrix_pyg.shape[0]
graph.num_vertices = adj_matrix_pyg.shape[0]

if graph_params["graph_type"]["name"] in ["sbm_two_branch_chain", "two_branch_graph", "custom_two_branch", "dual_branch_sbm"]:

    # print(graph.num_vertices)
    # if "internal_indices" in graph.__dict__:
    #     print(graph_params["internal_nodes"])
    graph_params["internal_nodes"] = len(graph.internal_indices)  # Number of internal nodes
    # graph_params["internal_nodes"] = graph.num_vertices
    print(graph_params["internal_nodes"])

    print("----------1-----------")
    print(len(graph.internal_indices))
    print(graph.num_vertices)
    print()


print("debugggg------------------")
print()
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
# graph = GraphBuilder(**graph_params)

# 🔹 Create Graph Structure (Assuming GraphBuilder is already defined)
graph = GraphBuilder(**graph_params)
single_graph = graph.edge_index  # Use the precomputed edge index




import wandb 

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


if graph_params["graph_type"]["name"] in ["stochastic_block", "sbm_two_branch_chain", "two_branch_graph", "custom_two_branch"]:
    graph.log_hop_distribution_to_wandb()




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
    def __init__(self, dataset, input_size, hidden_size, output_size, label_value=1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.label_value = label_value  # <---- New parameter to control the value in one-hot vector

        # Precompute dataset transformation (instead of transforming on-the-fly)
        self.data = []
        self.labels = []

        one_hot_lookup = torch.eye(output_size)  # One-hot encoding lookup

        for img, label in dataset:
            image_flat = img.view(-1).to(torch.float32)  # Flattened
            internal_nodes = torch.zeros(hidden_size, dtype=torch.float32)  # Internal nodes
            one_hot_label = one_hot_lookup[label] * self.label_value  # <---- Changed here

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


class GraphFormattedMNISTGraphAware(torch.utils.data.Dataset):
    def __init__(self, dataset, graph, label_value=1.0):
        self.graph = graph
        self.label_value = label_value

        self.data = []
        self.labels = []

        for img, label in dataset:
            image_flat = img.view(-1).to(torch.float32)
            x = torch.zeros(graph.num_vertices, dtype=torch.float32)

            # Sensory nodes (set with the image)
            x[graph.sensory_indices] = image_flat

            # Set the correct label cluster to 1s
            if hasattr(graph, "label_cluster_map"):
                if label not in graph.label_cluster_map:
                    raise ValueError(f"⚠️ Label {label} not found in label_cluster_map")
                label_nodes = graph.label_cluster_map[label]
                x[label_nodes] = self.label_value
            else:
                raise ValueError("⚠️ Graph must define label_cluster_map for supervision nodes")

            self.data.append(x)
            self.labels.append(label)

        self.data = torch.stack(self.data)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

DATASET_PATH = "../data"

from torch.utils.data import DataLoader, random_split


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
# transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

# The ToTensor() transformation is the one responsible for scaling (MNIST) images to the range [0, 1].
transform_list = [
    transforms.ToTensor()
]

original_mnist = 28

if args.dataset_transform:

    if "normalize_min1_plus1" in args.dataset_transform:
        transform_list.append(transforms.Normalize((0.5,), (0.5,)))

    if "normalize_mnist_mean_std" in args.dataset_transform:
        transform_list.append(transforms.Normalize((0.1307,), (0.3081,)))

    if "random_translation" in args.dataset_transform:
        transform_list.append(transforms.RandomAffine(degrees=0, translate=(0.2, 0.2)))

    if "random_rotation" in args.dataset_transform:
        transform_list.append(transforms.RandomRotation(degrees=25))

    if "zoom_out" in args.dataset_transform:
        transform_list.append(transforms.Resize((20, 20)))

        # Compute padding to go from 20x20 back to 28x28
        total_padding = original_mnist - 20  # 8
        pad_left = total_padding // 2       # 4
        pad_right = total_padding - pad_left  # 4

        transform_list.append(transforms.Pad((pad_left, pad_left, pad_right, pad_right), fill=0, padding_mode='constant'))




# # Create the transform
# print("TODO ADD COMPASE TRANSFORMS")
composed_transform = transforms.Compose(transform_list)


base_train_dataset = torchvision.datasets.MNIST(root=DATASET_PATH, train=True, transform=composed_transform, download=True)
base_test_dataset = torchvision.datasets.MNIST(root=DATASET_PATH, train=False, transform=composed_transform, download=True)

print("hidden_size", hidden_size)
# Wrap it in the GraphFormattedMNIST class

if args.graph_type == "dual_branch_sbm": 
    if hasattr(graph, "label_cluster_map"):
        print("Using label-aware dataset (GraphFormattedMNISTGraphAware)")

    print("Using dual_branch_sbm graph type")
    print("making GraphFormattedMNISTGraphAware dataset")
    print(graph.num_vertices)
    train_dataset = GraphFormattedMNISTGraphAware(base_train_dataset, graph, label_value=args.supervision_label_val)     
    test_dataset = GraphFormattedMNISTGraphAware(base_test_dataset, graph, label_value=args.supervision_label_val)
    # test_dataset = GraphFormattedMNIST(base_test_dataset, input_size, hidden_size, output_size, label_value=args.supervision_label_val)

else:
    train_dataset = GraphFormattedMNIST(base_train_dataset, input_size, hidden_size, output_size, label_value=args.supervision_label_val)
    test_dataset = GraphFormattedMNIST(base_test_dataset, input_size, hidden_size, output_size, label_value=args.supervision_label_val)

# Split the dataset
train_set, val_set = random_split(train_dataset, [50000, 10000])


# Create DataLoaders
# QUICK FIX for now TODO, 
val_loader_batch_size = (args.batch_size if args.batch_size >= 10 else 10)
# val_loader_batch_size = args.batch_size

train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_set, batch_size=val_loader_batch_size, shuffle=True, drop_last=False, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=val_loader_batch_size, shuffle=True, drop_last=False, num_workers=4, pin_memory=True)


# graph_type [FC, generative, discriminative, SBM, (SBM_hierarchy, custom_two_branch, two_branch_graph) ]
# task       [classification, generation, denoise, occlusion]


next(iter(train_loader))  # Preload the first batch to ensure everything is set up correctly
# take a sample from the train_loader to get the input size
sample_batch = next(iter(train_loader))

# plot the values of the first item in the batch
import matplotlib.pyplot as plt
# using the sensory indices from the graph
# sensory_indices = graph.sensory_indices
# hidden_indices = graph.internal_indices
# output_indices = graph.supervision_indices

print("batch shape", sample_batch[0].shape)  # Should be [batch_size, input_size + hidden_size + output_size]
print("x shape", sample_batch[1].shape)  # Should be [batch_size

if args.graph_type == "dual_branch_sbm":

    # # flattten the item and round up to the nearest 100 and plot imshow of shape Nx100
    # graph_values = sample_batch  # [batch_size, num_vertices]

    # # first item batch
    # graph_values = graph_values[0]  # Take the first item in the batch
    # print("graph_values shape", graph_values.shape)  # Should be [num_vertices]
    
    # # plot imshow the sensory values using graph.sensory_indices
    # sensory_values = graph_values[graph.sensory_indices]  # Get sensory values
    # # reshape to (28, 28) for plotting
    # sensory_values = sensory_values.view(28, 28).detach().cpu().numpy()
    # plt.imshow(sensory_values, cmap='gray')
    # plt.title("Sensory Values")

    # plot the graph.supervision_indices values
    print("graph.supervision_indices", graph.supervision_indices)
    for i, supervision_indices in enumerate(graph.supervision_indices):
        print("i and value ", i, graph.supervision_indices[i], end="_")

    supervision_values = sample_batch[0][:, graph.supervision_indices]  # Get supervision
    supervision_values = supervision_values.view(-1, 10).detach().cpu().numpy()  # Reshape to (batch_size, 10)
    plt.imshow(supervision_values, cmap='hot', aspect='auto')
    plt.title("Supervision Values")
    plt.colorbar()
    # save to wandb monitor
    if args.use_wandb in ['online', 'run']:
        wandb.log({"Monitoring/supervision_values": wandb.Image(plt)})
    plt.close()






print("-------------IMPORTANT CHECK----------------")

# num_vertices_batch = sample_batch[0].shape[1]  # Get the number of vertices from the first batch

num_vertices_batch = graph.num_vertices
print("Model will use num_vertices =", num_vertices)


print("Number of vertices in the first batch:", num_vertices_batch)
hidden_size = len(graph.internal_indices)

print(input_size + hidden_size + output_size)
print(adj_matrix_pyg.shape)
print(graph.num_vertices)

################################# discriminative model lr         ##########################################
################################# generative model lr         ##########################################




print("adj_matrix_pyg", adj_matrix_pyg.shape)

# from model_latest import PCgraph
from model_latest2 import PCgraph
# from model import PCgraph

model_params = {
    "delta_w_selection": args.delta_w_selection,  # "all" or "internal_only"
    "batch_size": train_loader.batch_size, 
 
 }


# structure = only for single_hidden_layer (discriminative_hidden_layers, generative_hidden_layers): number of nodes in each layer
if args.graph_type == "single_hidden_layer":
    
    # give both layers as a list
    structure = {
        "discriminative_hidden_layers": args.discriminative_hidden_layers or [200, 100, 50],  # Default if not provided
        "generative_hidden_layers": args.generative_hidden_layers or [50, 100, 200],  # Default if not provided
    }

    # insert sensory nodes to the structure for both
    structure["discriminative_hidden_layers"].insert(0, input_size)  # Add input layer size
    structure["generative_hidden_layers"].insert(0, input_size)  # Add input layer size

else:
    structure = None
        

model_params = {

    "graph_type": args.graph_type,
    "structure": structure,
    "task": TASK,

    "use_bias": args.use_bias,


    "activation": args.activation_func,  
    "device": device,
    
    "init_hidden_values": args.init_hidden_values,  # Initial value for hidden nodes in the model. This is a small value for initialization of hidden nodes. Used as mean for normal distribution at the start of each training sample
    "init_hidden_mu": args.init_hidden_mu,  # Initial value for hidden nodes in the model. This is a small value for initialization of hidden nodes. Used as mean for normal distribution at the start of each training sample
    "delta_w_selection": args.delta_w_selection,  # "all" or "internal_only",

    # "num_vertices": (input_size + hidden_size + output_size) if args.graph_type != "dual_branch_sbm" else (graph.num_vertices),
    # "num_vertices": (input_size + hidden_size + output_size) if args.graph_type != "dual_branch_sbm" else (adj_matrix_pyg.shape[0]),
    "num_vertices": num_vertices_batch,
    # "num_vertices": graph.num_vertices,
    # "num_vertices": num_vertices,    

    "num_internal": sum(graph.internal_indices),
    "adj": adj_matrix_pyg,             # 2d Adjacency matrix
    "edge_index": graph.edge_index,    # [2, num_edges] edge index
    "batch_size": args.batch_size,
    # "learning_rates": (lr_x, lr_w),
    "learning_rates": (args.lr_values, args.lr_weights),
    "T": (args.T_train, args.T_test),  # Number of iterations for gradient descent. (T_train, T_test)

    "incremental_learning": True if args.model_type == "IPC" else False, 
    
    
    "use_input_error": args.use_input_error,

    "update_rules": args.update_rules,  # "Van_Zwol" or "salvatori", "vectorized"
    "weight_init": args.weight_init,   # xavier, 'uniform', 'based_on_f', 'zero', 'kaiming'

    # "edge_type":  custom_dataset_train.edge_type,
    "weight_decay": (args.w_decay_lr_values, args.w_decay_lr_weights),    # False or [0], [(weight_decay=)]
    "use_grokfast": args.grokfast,  # False or True
    "grad_clip_lr_x_lr_w": (grad_clip_lr_x, grad_clip_lr_w), 
}

PCG = PCgraph(**model_params, 
                wandb_logging=run if args.use_wandb in ['online', 'run'] else None,
                debug=False)

PCG.init_modes(graph_type=args.graph_type, graph=graph)
PCG.set_task(TASK)

model = PCG

plot_model_weights(model, args.graph_type, model_dir=None, save_wandb="before_training")

print("------------- compile model ------------- ")
# model = torch.compile(model, disable=True) 
# torch.compile(model, dynamic=True)
# TODO: torch.compile is not supported for all models yet, so we disable it for now
model = torch.compile(model, mode="max-autotune")
model.task = ""   # classification or generation, or both 

from datetime import datetime
from tqdm import tqdm
import torch

start_time = datetime.now()

train_energy, train_loss, train_acc = [], [], []
val_loss, val_acc, val_acc2 = [], [], []

if args.break_num_train == 0:
    break_num = len(train_loader)
else:
    break_num = args.break_num_train

DEVICE = device 



TASK_config = {
    "classification": 
        {"batch_break": 100, "wandb": eval_classification},
}

if args.graph_type in ["SBM", "stochastic_block", "scalable_barabasi", "scale_free_barabasi"]:
    # model.log_edge_connectivity_distribution_to_wandb(direction="both")
    model.log_node_connectivity_distribution_to_wandb(direction="both")

# model.log_edge_weight_distribution_to_wandb()

accuracy_means = []
training_error_means = []
num_of_trained_imgs = 0


tc = TerminalColor()

print(f"{tc.RED} ---------- Training ------- .{tc.RESET}")
weight_table = wandb.Table(columns=["epoch", "std", "mean"])

model_weight_std_mean = {
    "epoch": [],
    "std": [],
    "mean": [],
}

# initialize history buffer
weight_history = []

# load model weights if they exist
load_model_weights = False
# load scr3/trained_models/weight_matrix_pypc.npy
# if load_model_weights and os.path.exists("trained_models/weight_matrix_pypc.npy"):
#     print("Loading model weights from trained_models/weight_matrix_pypc.npy")

#     print(model.w.shape)

#     W = np.load("trained_models/weight_matrix_pypc.npy").T
#     print("W shape", W.shape)
#     W = torch.from_numpy(W).to(device).float()

#     with torch.no_grad():        # avoid tracking in autograd
#         PCG.w.copy_(W)           # copy *into* the existing Parameter

# TODO
print("fix error map do_log_error_map")



with torch.no_grad():

    epoch_history = []

    for epoch in tqdm(range(args.epochs), desc="Epoch Progress"):
        model.train_(epoch)
        model.epoch = epoch
        total_loss = 0
        energy = 0

        # if epoch == 2:
        #     # reduce by 10x
        #     model.lr_x /= 10 
        #     model.lr_w /= 10
        
        # log mean, min and max of weights to wandb
        w = PCG.w.detach().cpu().numpy()
        # collect weights
        mean = w.mean().item()
        std = w.std().item()

        wandb.log({
            "epoch": epoch,
            "Weights/mean": w.mean(),
            "Weights/min": w.min(),
            "Weights/max": w.max(),
            "Weights/std": w.std(),
        })

        weight_history.append(w.copy())  # shape: [epoch, N, N]





        print("\n-----train_supervised-----")
        print(len(train_loader))

        # model.do_log_error_map = True 
        model.do_log_error_map = False

        # --------------- EVAL -----------------
        if not load_model_weights:
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

                num_of_trained_imgs += X_batch.shape[0]
                if batch_no >= break_num:
                    break
                
                model.do_log_error_map = False

            model_weight_std_mean["epoch"].append(epoch)
            model_weight_std_mean["std"].append(std)
            model_weight_std_mean["mean"].append(mean)

            wandb.log({
                "epoch": epoch,
                "Training/num_of_trained_imgs": num_of_trained_imgs,
            })

        model.do_log_error_map = False

        if load_model_weights:
            model.reset_nodes(batch_size=val_loader_batch_size, force=True)
        #### 
        loss, acc = 0, 0
        model.test_(epoch)
        cntr = 0

        break_num_eval = 20
        # break_num_eval = len(val_loader)

        if TASK == ["generation"]:
            break_num_eval = 1
        #     break_num_eval = 1

        # if TASK == ["classification"]:

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

            # # do generation once
            if "generation" in TASK_copy and "classification" in TASK_copy:
                # pop generation from TASK_copy
                TASK_copy.remove("generation")
            
            # 

            # # do generation once
            if "occlusion" in TASK_copy and "classification" in TASK_copy:
                # pop generation from TASK_copy
                TASK_copy.remove("occlusion")
            
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
            accuracy_means.append(accuracy_mean)

            print("epoch", epoch, "accuracy_mean", accuracy_mean)

            if epoch % 10 == 0 or accuracy_mean > 0.90:
                print("delta pred ", y_pred - y_batch)

            wandb.log({
                "epoch": epoch,
                "classification/accuracy_mean": accuracy_mean,
                "classification/size":  len(accs) * args.batch_size,
            })

            # use_attention = (accuracy_mean > 0.5)  # Or epoch > X
            # model.updates.use_attention = use_attention


        # save model weights plt.imshow to "trained_models/{TASK}/weights/model_{epoch}.png"
        # make folder if not exist
        import os 
        if not os.path.exists(f"trained_models/{args.graph_type}/weights/"):
            os.makedirs(f"trained_models/{args.graph_type}/weights/")

       

        if epoch % 10 == 1 and args.use_wandb in ['online', 'run']:
            plot_model_weights(model, args.graph_type, model_dir=None, save_wandb=str(epoch))


        # if model.use_attention:
        #     if hasattr(model.updates, "alpha"):
        #         attn_matrix = model.updates.alpha.detach().cpu().numpy()  # [N, N]
        #         print(f"[Epoch {epoch}] Mean attention weight: {attn_matrix.mean():.4f}")

        #         fig, ax = plt.subplots(figsize=(6, 5))
        #         cax = ax.imshow(attn_matrix, cmap="viridis")
        #         ax.set_title(f"Attention Map (Epoch {epoch})")
        #         ax.set_xlabel("Source Nodes (j)")
        #         ax.set_ylabel("Target Nodes (i)")
        #         fig.colorbar(cax)

        #         # Save locally
        #         fig.savefig(f"attention/attention_map_epoch_{epoch}.png")

        #         # Log to wandb
        #         if args.use_wandb in ['online', 'run']:
        #             wandb.log({f"Monitoring/Attention_Heatmap_E{epoch}": wandb.Image(fig)})

        #         plt.close(fig)
            
        #     if hasattr(model.updates, "attn_param"):
        #         with torch.no_grad():
        #             attn_weights = torch.sigmoid(model.updates.attn_param).cpu().numpy()
        #         plt.figure(figsize=(6, 5))
        #         plt.imshow(attn_weights, cmap="viridis")
        #         plt.title("Learned Attention Weights")
        #         plt.colorbar()
        #         plt.xlabel("Source Nodes")
        #         plt.ylabel("Target Nodes")
        #         plt.savefig(f"attention/attention_weights_epoch_{epoch}.png")
        #         plt.close()


        #     # vanZwol_AMB_withTransformerAttentionHebbian
        #     if hasattr(model.updates, "attn_param"):
        #         with torch.no_grad():
        #             attn_weights = model.updates.attn_param.cpu().numpy()
        #             print(f"[Epoch {epoch}] Mean attention weight: {attn_weights.mean():.4f}")
        #         plt.figure(figsize=(6, 5))
        #         plt.imshow(attn_weights, cmap="viridis")
        #         plt.title(f"Learned Attention Weights (Epoch {epoch})")
        #         plt.colorbar()
        #         plt.xlabel("Source Nodes")
        #         plt.ylabel("Target Nodes")
        #         plt.savefig(f"attention/attention_weights_epoch_{epoch}.png")
        #         plt.close()




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



print("Training completed in:", datetime.now() - start_time)
print("save last model weights to wandb")

if args.use_wandb in ['online', 'run']:
        plot_model_weights(model, args.graph_type, model_dir=None, save_wandb="after_training")

        # plot_model_weights(model, args.graph_type, model_dir=None, save_wandb=str(int(args.epochs)))


from sklearn.decomposition import PCA

weight_history = np.array(weight_history)  # shape: [T, N, N]
flat_weights = weight_history.reshape(weight_history.shape[0], -1)

logs = {}

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import wandb
import numpy as np

# flat_weights: [T, N*N] from weight_history.reshape(T, -1)
pca = PCA(n_components=2)
pca_vals = pca.fit_transform(flat_weights)  # shape [T, 2]
x, y = pca_vals[:-1, 0], pca_vals[:-1, 1]
u, v = pca_vals[1:, 0] - x, pca_vals[1:, 1] - y

# Plot
plt.figure(figsize=(8, 6))
plt.quiver(x, y, u, v, angles='xy', scale_units='xy', scale=1, color='blue', alpha=0.7)
plt.plot(pca_vals[:, 0], pca_vals[:, 1], 'o--', color='gray', alpha=0.4)
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.title("PCA Trajectory of Weights Over Epochs")
plt.grid()

# Log to wandb
wandb.log({
    "Weights/PCA_quiver_plot": wandb.Image(plt)
})
plt.close()

# # ✅ 2. L2 distance from initial weights
# init = flat_weights[0]
# final = flat_weights[-1]
# l2 = np.linalg.norm(final - init)
# logs["Weights/L2_from_init"] = l2


# # ✅ 4. Loss landscape via interpolation
# if flat_weights.shape[0] >= 3:  # or just >= 2
#     W_a = flat_weights[0]       # early weight
#     W_b = flat_weights[-1]      # final weight

#     alphas = np.linspace(0, 1, 20)
#     interpolated = np.array([(1 - a) * W_a + a * W_b for a in alphas])
#     midpoint = 0.5 * (W_a + W_b)
#     losses = np.linalg.norm(interpolated - midpoint, axis=1) ** 2
#     logs["Weights/Loss_landscape_max"] = np.max(losses)
#     logs["Weights/Loss_landscape_min"] = np.min(losses)

wandb.log(logs)


# plot model_weight_std_mean plt.quiver x-axis std, y-axis mean, epoch over time with arrows
# Convert to numpy arrays for plotting
# Example data
stds = np.array(model_weight_std_mean["std"])
means = np.array(model_weight_std_mean["mean"])
epochs = np.arange(len(stds))

# Arrow origins (start of each arrow)
x = stds[:-1]
y = means[:-1]

# Arrow directions (difference to next point)
u = stds[1:] - stds[:-1]
v = means[1:] - means[:-1]

# Plot
plt.figure(figsize=(8, 6))
plt.quiver(x, y, u, v, angles='xy', scale_units='xy', scale=1, color='blue', alpha=0.6)
plt.plot(stds, means, 'o--', color='gray', alpha=0.3)  # optional path trace
plt.xlabel('Std Weights')
plt.ylabel('Mean Weights')
plt.title('Weight Evolution Over Epochs (Std vs Mean)')
plt.grid()
# plt.savefig(f"{model_dir}/model_weights_std_mean.png")
# save to wandb "weights/quiver_plot.png"
if args.use_wandb in ['online', 'run']:
    wandb.log({
        "Weights/quiver_plot": wandb.Image(plt),
    })
# Clear memory
plt.close() 


# Compute std/mean trajectory
stds = np.std(flat_weights, axis=1)
means = np.mean(flat_weights, axis=1)

import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


# WandB table
table = wandb.Table(columns=["run_id", "epoch", "PC1", "PC2", "std", "mean"])
run_id = wandb.run.id  # 👈 use short W&B ID
for i in range(len(flat_weights)):
    table.add_data(run_id, i, pca_vals[i, 0], pca_vals[i, 1], stds[i], means[i])

# Get run ID (use actual string if not using wandb)
# run_id = "pkt55489"  # or use timestamp etc.
run_id = wandb.run.id if args.use_wandb in ['online', 'run'] else "local_run"
save_dir = f"jobs/plotting/trajectories/{run_id}"
os.makedirs(save_dir, exist_ok=True)

# Create and save DataFrame
df = pd.DataFrame({
    "epoch": np.arange(len(pca_vals)),
    "PC1": pca_vals[:, 0],
    "PC2": pca_vals[:, 1],
    "std": stds,
    "mean": means,
})
df.to_csv(os.path.join(save_dir, "trajectory.csv"), index=False)
print(f"✓ Saved trajectory to {save_dir}/trajectory.csv")




# model.clear_memory()
model.trace_data = None  # Clear trace data to free memory
# clear memory
torch.cuda.empty_cache()
import gc
gc.collect()
print("Training completed")

try:
    print("done training")

    # torch no grad
    with torch.no_grad():

        if len(accuracy_means) > 0:
            print(max(accuracy_means))
            print(accuracy_means)


            if max(accuracy_means) > 0.90 or accuracy_means[-1] >= 0.85:

                # save weights; TODO

                # eval on test set

                
                # for batch_no, batch in enumerate(tqdm(val_loader, total=min(len(val_loader)), desc=f"Epoch {epoch+1} - Validation", leave=False)):
                for batch_no, (X_batch, y_batch)  in enumerate(tqdm(test_loader, total=len(test_loader), desc=f"Epoch {epoch+1} - Test Eval. | {TASK}", leave=False)):
                
                    X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)  # Move to GPU
                    
                    # y_pred = PCG.test_iterative(batch, eval_types=TASK_copy, remove_label=True)

                    for task in TASK_copy:
                        y_pred = PCG.test_iterative( (X_batch, y_batch), 
                                                    eval_types=[task], remove_label=True)

                    # # do generation once
                    if "generation" in TASK_copy and "classification" in TASK_copy:
                        TASK_copy = ["classification"]

                    # print("y_pred", y_pred.shape)
                    # print("y_pred", y_batch.shape)
                    if "classification" in TASK_copy:
                        mean_correct = torch.mean((y_pred == y_batch).float()).item()
                        acc += mean_correct
                        accs.append(mean_correct)
                    
                print("Done test eval")

                if "classification" in TASK:
                    accuracy_mean = (sum(accs) / len(accs)) if accs else 0
                    val_acc.append(accuracy_mean)
                    accuracy_means.append(accuracy_mean)

                    print("epoch", epoch, "accuracy_mean", accuracy_mean, "on size test set", len(accs) * args.batch_size)

                    # if epoch % 10 == 0 or accuracy_mean > 0.90:
                    #     print("delta pred ", y_pred - y_batch)

                    wandb.log({
                        "epoch": epoch,
                        "classification_test/accuracy_mean": accuracy_mean,
                        "classification_test/size":  len(accs) * args.batch_size,
                    })
                print("Test evaluation completed with accuracy:", accuracy_mean)
# Catch
except Exception as e:
    print("Error during training or evaluation:", e)
    print("Training might not have been successful. Check the logs for details.")
    # Optionally, you can raise the exception again if you want to stop execution
    # raise e







# only if we are doing discriminative training
if args.train_mlp and args.graph_type == "single_hidden_layer" and sum(args.discriminative_hidden_layers) > 0:
    # from helper.plot import plot_model_weights
    from helper.plot import train_mlp_classifier
    

    # hidden_layers = [512, 256, 128]          # or go deeper: [485, 176, 131, 122]
    # optimizer = torch.optim.Adam(
    #     mlp.parameters(),
    #     lr=1.3e-4,                           # ~0.00012885
    #     weight_decay=1e-4                   # small L2 regularization
    # )

    batch_size = 128
    epochs = 20
    print(args.discriminative_hidden_layers[1:])

    print("---------------- TRAIN MLP CLASSIFIER -----------------")
    # if args.discriminative_hidden_layers[1:] == hidden_layers:
    # else:
    #     PC_weights = None 
    PC_weights = model.w.cpu().detach().numpy()

    # if the first item in args.discriminative_hidden_layers is 784, don't include it
    # assert both first and second item are not 784

    if args.discriminative_hidden_layers[0] == 784:
        # remove the first item
        hidden_layers = args.discriminative_hidden_layers[1:]
    else:
        # keep the first item
        hidden_layers = args.discriminative_hidden_layers

    # assert (args.discriminative_hidden_layers[1] != 784) and (args.discriminative_hidden_layers[0] != 784), "First item in discriminative_hidden_layers should not be 784"

    
    # hidden_layers = args.discriminative_hidden_layers[1:]     
    print("hidden_layers", hidden_layers)

    train_mlp_classifier(run_id=run.id, 
                         PC_weights=PC_weights, 
                         hidden_layer=hidden_layers, activation_fn="relu", 
               base_train_dataset=base_test_dataset,
               base_test_dataset=base_test_dataset,
               batch_size=batch_size, 
               epochs=epochs,
                lr_w=1.3e-4, 
                weight_decay=1e-4,                   # small L2 regularization
                seed=args.seed, 
                graph_type="mlp_bp")
    


wandb.finish()