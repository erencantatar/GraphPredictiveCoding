import torch
import torchvision
import torchvision.transforms as transforms

import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch_geometric.data import Data

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

import torch.nn.init as init
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from icecream import ic


from torch_scatter import scatter_mean
from torch_geometric.data import Data
from torch.utils.data import Dataset

from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
import random

from torch_geometric.nn import MessagePassing
import torch.nn.functional as F
import os


######################################################################################################### 
####                                            Arguments                                           #####
######################################################################################################### 

# import wandb 

# make change here

import os
import argparse
from helper.args import true_with_float, valid_str_or_float, valid_int_or_all, valid_int_list, str2bool, validate_weight_init
from helper.activation_func import activation_functions
from graphbuilder import graph_type_options

# Parsing command-line arguments
# parser = argparse.ArgumentParser(description='Train a model with specified parameters.')

# # Training mode 
# parser.add_argument('--mode', choices=['training', 'experimenting'], required=True,  help="Mode for training the model or testing new features.")

# # -----dataset----- 
# #data_dir default --data_dir default to $TMPDIR
# parser.add_argument('--data_dir', type=str, default='../data', help='Path to the directory to store the dataset. Use $TMPDIR for scratch. or use ../data ')

# parser.add_argument('--dataset_transform', nargs='*', default=[], help='List of transformations to apply.', choices=['normalize_min1_plus1', 'normalize_mnist_mean_std', 'random_rotation', 'none', 'None'])
# parser.add_argument('--numbers_list', type=valid_int_list, default=[0, 1, 3, 4, 5, 6, 7], help="A comma-separated list of integers describing which distinct classes we take during training alone")
# parser.add_argument('--N', type=valid_int_or_all, default=20, help="Number of distinct trainig images per class; greater than 0 or the string 'all' for all instances o.")

# parser.add_argument('--supervision_label_val', default=10, type=int, required=True, help='An integer value.')


# ## -----graph-----  
# parser.add_argument('--num_internal_nodes', type=int, default=1500, help='Number of internal nodes.')
# parser.add_argument('--graph_type', type=str, default="fully_connected", help='Type of Graph', choices=list(graph_type_options.keys()))
# parser.add_argument('--remove_sens_2_sens', type=str2bool, required=True, help='Whether to remove sensory-to-sensory connections.')
# parser.add_argument('--remove_sens_2_sup', type=str2bool, required=True, help='Whether to remove sensory-to-supervised connections.')
# parser.add_argument('--discriminative_hidden_layers', type=valid_int_list, default=None, 
#                     help="Optional: Comma-separated list of integers specifying the number of nodes in each discriminative hidden layer. Only used if graph_type is 'single_hidden_layer'.")
# parser.add_argument('--generative_hidden_layers', type=valid_int_list, default=None, 
#                     help="Optional: Comma-separated list of integers specifying the number of nodes in each generative hidden layer. Only used if graph_type is 'single_hidden_layer'.")

# # --MessagePassing--
# parser.add_argument('--normalize_msg', choices=['True', 'False'], required=True,  help='Normalize message passing, expected True or False')

# # -----model-----  
# parser.add_argument('--model_type', type=str, default="PC", help='Predictive Model type: [PC,IPC] ', choices=["PC", "IPC"])
# # parser.add_argument("--weight_init", type=str, default="fixed 0.001", help="Initialization method and params for weights")
# parser.add_argument("--weight_init", type=validate_weight_init, default="fixed 0.001", help="Initialization method and params for weights")

# parser.add_argument('--use_bias',  choices=['True', 'False'], required=True, help="....")
# parser.add_argument("--bias_init", type=str, default="", required=False, help="ege. fixed 0.0 Initialization method and params for biases")

# parser.add_argument('--T', type=int, default=5, help='Number of iterations for gradient descent.')
# parser.add_argument('--T_test', type=int, default=10, help='Number of iterations for gradient descent.')
# # TODO make T_train and T_test
# # parser.add_argument('--T', type=int, default=40, help='Number of iterations for gradient descent.')
# parser.add_argument('--lr_values', type=float, default=0.001, help='Learning rate values (alpha).')
# parser.add_argument('--lr_weights', type=float, default=0.01, help='Learning rate weights (gamma).')
# parser.add_argument('--activation_func', default="swish", type=str, choices=list(activation_functions.keys()), required=True, help='Choose an activation function: tanh, relu, leaky_relu, linear, sigmoid, hard_tanh, swish')

# # update rules
# # parser str choices "Van_Zwol" or "salvatori", "vectorized"
# parser.add_argument('--update_rules', type=str, default="Salvatori", choices=["Van_Zwol", "Salvatori", "vectorized"], help="Choose the update rules for the model equations")


# parser.add_argument('--delta_w_selection', type=str, required=True, choices=["all", "internal_only"], help="Which weights to optimize in delta_w")
# parser.add_argument('--use_grokfast', type=str, default="False", choices=["True", "False"], help="GroKfast fast and slow weights before using the optimizer")

# # -----training----- 
# parser.add_argument('--epochs', type=int, default=5, help='Number of epochs to train.')
# parser.add_argument('--batch_size', type=int, default=1, help='Batch size.')
# parser.add_argument('--seed', type=int, default=42, help='Random seed.')
# parser.add_argument('--optimizer', type=true_with_float, default=False,
#                     help="Either False or, if set to True, requires a float value.")

# # ---after training-----
# parser.add_argument('--set_abs_small_w_2_zero',  choices=['True', 'False'], required=True, help="....")

# # logging 
# import wandb
# parser.add_argument('--use_wandb', type=str, default="disabled", help='Wandb mode.', choices=['shared', 'online', 'run', 'dryrun', 'offline', 'disabled'])
# parser.add_argument('--tags', type=str, default="", help="Comma-separated tags to add to wandb logging (e.g., 'experiment,PC,test')")

# args = parser.parse_args()



args = {
    # "model_type": "IPC",
    "model_type": "PC",
    "update_rules": "Salvatori",  # Van_Zwol Update rules for learning

    # "graph_type": "fully_connected",  # Type of graph
    "discriminative_hidden_layers": None,  # Hidden layers for discriminative model
    "generative_hidden_layers": None,  # Hidden layers for generative model

    "graph_type": "single_hidden_layer",  # Type of graph
    "discriminative_hidden_layers": [32, 16],  # Hidden layers for discriminative model
    "discriminative_hidden_layers": [100, 50],  # Hidden layers for discriminative model
    "generative_hidden_layers": [],  # Hidden layers for generative model

    "delta_w_selection": "internal_only",  # Selection strategy for weight updates
    "weight_init": "fixed 0.001 0.001",  # Weight initialization method
    "use_grokfast": False,  # Whether to use GrokFast
    "optimizer": False,  # Optimizer setting
    "remove_sens_2_sens": False,  # Remove sensory-to-sensory connections
    "remove_sens_2_sup": False,  # Remove sensory-to-supervised connections
    "set_abs_small_w_2_zero": False,  # Set small absolute weights to zero
    "mode": "experimenting",  # Mode of operation (training/experimenting)
    "use_wandb": "online",  # WandB logging mode
    "tags": "PC_vs_IPC",  # Tags for logging
    "use_bias": False,  # Whether to use bias
    "normalize_msg": False,  # Normalize message passing
    "dataset_transform": ["normalize_mnist_mean_std"],  # Data transformations
    "numbers_list": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],  # Classes to include
    "N": "all",  # Number of samples per class
    "supervision_label_val": 1,  # Value assigned for supervision
    "num_internal_nodes": 1000,  # Number of internal nodes in the network
    "T": 10,  # Number of inference iterations
    "T_test": 20, 
    "lr_values": 0.01,  # Learning rate for value updates
    "lr_weights": 0.00001,  # Learning rate for weight updates
    "activation_func": "swish",  # Activation function
    "epochs": 10,  # Number of training epochs
    "batch_size": 2,  # Batch size for training
    "seed": 2,  # Random seed
}




# Access the arguments just like you would with argparse
print(args['dataset_transform'])  # Example of accessing an argument


class Args:
    def __init__(self, **entries):
        self.__dict__.update(entries)


# Create an object from the dictionary
args = Args(**args)


# Make True of False bool
args.normalize_msg = args.normalize_msg == 'True'
args.use_bias = args.use_bias == 'True'
args.set_abs_small_w_2_zero = args.set_abs_small_w_2_zero == 'True'
args.grokfast = args.use_grokfast == 'True'

tags_list = args.tags.split(",") if args.tags else []

if isinstance(args.optimizer, (float, int)) and args.optimizer >= 0:
    # Optimizer enabled with explicit weight decay
    use_learning_optimizer = [args.optimizer]
    weight_decay = args.optimizer
    print(f"Optimizer enabled with weight decay: {weight_decay}")
elif args.optimizer is False:
    # Optimizer disabled
    use_learning_optimizer = False
    weight_decay = None
    print("Optimizer disabled (False).")
else:
    # Catch invalid or improperly parsed True cases
    raise ValueError(f"Invalid value for optimizer: {args.optimizer}. Expected False or a float value.")



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

# Checkpoints:
# - validate args
# - create graph topology
# - create dataloader
# - init model
# - training loop with train_set and val_set
# - full eval_set 








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


## Subset of the dataset (for faster development)
# subset_size = 100  # Number of samples to use from the training set
# indices = list(range(len(mnist_trainset)))
# random.shuffle(indices)
# subset_indices = indices[:subset_size]

# mnist_train_subset = torch.utils.data.Subset(mnist_trainset, subset_indices)
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

if graph_params["graph_type"]["name"] not in ["single_hidden_layer"]:
    # Ensure these arguments are not specified for other graph types
    assert args.discriminative_hidden_layers is None, \
        "The argument --discriminative_hidden_layers can only be used if graph_type is 'single_hidden_layer'."
    assert args.generative_hidden_layers is None, \
        "The argument --generative_hidden_layers can only be used if graph_type is 'single_hidden_layer'."

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




print("graph_params 1 :", graph_params)

from graphbuilder import GraphBuilder
from helper.plot import plot_adj_matrix

print("graph_params", graph_params)
graph = GraphBuilder(**graph_params)

# self.edge_type = 
# self.edge_index = loader.edge_index
# self.edge_index_tensor = self.edge_index


single_graph = graph.edge_index

adj_matrix_pyg = plot_adj_matrix(single_graph, model_dir=None, node_types=None)


from dataset import PCGraphDataset, train_subset_indices

from torch_geometric.data import Batch

def custom_collate_fn(batch):

    # batched_data = Data.from_batch(batch)  # Use the built-in PyG batching method
    batched_data = Batch.from_data_list(batch)

    sensory_indices = []
    internal_indices = []
    supervision_indices = []

    node_offset = 0  # Keeps track of node offset as we batch graphs

    for data in batch:
        # Adjust the indices by adding the current node offset
        sensory_indices.append(data.sensory_indices + node_offset)
        internal_indices.append(data.internal_indices + node_offset)
        supervision_indices.append(data.supervision_indices + node_offset)

        # Increment offset by the number of nodes in the current graph
        node_offset += data.x.size(0)

    # Concatenate indices across the batch
    batched_data.sensory_indices = torch.cat(sensory_indices)
    batched_data.internal_indices = torch.cat(internal_indices)
    batched_data.supervision_indices = torch.cat(supervision_indices)

    return batched_data


from torch_geometric.data import DataLoader as GeoDataLoader
import torchvision
from torchvision import transforms

# Dataset and DataLoader configurations

DATASET_PATH = "../data"
batch_size = args.batch_size

# # Initialize the GraphBuilder
# custom_dataset_train = PCGraphDataset(graph_params, **dataset_params)

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

train_dataset = torchvision.datasets.MNIST(root=DATASET_PATH, train=True, transform=transform, download=True)
test_set = torchvision.datasets.MNIST(root=DATASET_PATH, train=False, transform=transform, download=True)

train_set, val_set = torch.utils.data.random_split(train_dataset, [50000, 10000])
train_indices = train_subset_indices(train_set, 10, no_per_class=0)  # Set `no_per_class` as needed

# Initialize CustomGraphDataset for train, validation, and test sets
train_graph_dataset = PCGraphDataset(graph, train_set, supervised_learning=True, numbers_list=list(range(10)))
val_graph_dataset = PCGraphDataset(graph, val_set, supervised_learning=True, numbers_list=list(range(10)))
# test_graph_dataset = PCGraphDataset(graph, test_set, supervised_learning=True, numbers_list=list(range(10)))

# PYG DataLoaders
train_loader = GeoDataLoader(train_graph_dataset, batch_size=batch_size, collate_fn=custom_collate_fn, shuffle=True, drop_last=True)
val_loader = GeoDataLoader(val_graph_dataset, batch_size=batch_size, collate_fn=custom_collate_fn, shuffle=False, drop_last=True)
# test_loader = GeoDataLoader(test_graph_dataset, batch_size=batch_size, collate_fn=custom_collate_fn, shuffle=False, drop_last=True)



#### GET EXAMPLE BATCH #### 
import matplotlib.pyplot as plt

print(len(train_loader))
testing_remove_label = True
testing_remove_data  = False

for batch in train_loader:
    
    # Set the graph-level labels to zero
    # batch.y[:] = 0
    for i in range(len(batch)):
        
        ### batch.x[batch.supervision_indices, :] = 0 
        if testing_remove_label:
            sub_graph = batch[i]  # Access the subgraph
            sub_graph.x[sub_graph.supervision_indices, 0] = torch.zeros_like(sub_graph.x[sub_graph.supervision_indices, 0])  # Check all feature dimensions

    # train_loader.dataset.zero_img_in_graph  = True
    # train_loader.dataset.zero_y_in_graph    = True 

    print(batch)
    print(batch.x.shape)

    batch_item = 1  # Select a specific graph within the batch
    
    sub_graph = batch[batch_item]  # Access the subgraph
    sensory_indices = sub_graph.sensory_indices
    image_tensor = sub_graph.x[sensory_indices, 0].view(28, 28).detach().numpy()
    
    sup_tensor = sub_graph.x[sub_graph.supervision_indices, 0]  # Check all feature dimensions
    
    print("Supervision tensor after zeroing:")
    print(sup_tensor)
    
    batch_example = batch 
    # Display the image
    # plt.imshow(image_tensor, cmap='gray')
    # plt.title(f"Sensory Node Image Representation (Label set to {sub_graph.y})")

    print("---------------------")
    # print("Batched sensory indices:", batch.sensory_indices)
    # print("Batched internal indices:", batch.internal_indices)
    # print("Batched supervision indices:", batch.supervision_indices)

    print("-------Edge_index (single) vs (batched) ---------")

    single_graph = graph.edge_index
    print(single_graph.shape)
    print(batch.edge_index.shape)

    # plt.show()

    break  # Only process the first batch

###########################




# PCG = PC_graph_zwol_PYG(f,
#         device=device,
#         num_vertices=graph.num_vertices,
#         num_internal=sum(graph.internal_indices),
#         adj=adj_matrix_pyg,
#         edge_index=graph.edge_index,
#         batch_size=batch_size,
#         # mask=mask,
#         lr_x=lr_x, 
#         T_train=T_train,
#         T_test=T_test,
#         incremental=incremental, 
#         use_input_error=use_input_error,
#         )


model_params = {
    
    'num_vertices': graph.num_vertices,
    'sensory_indices': (sensory_indices), 
    'internal_indices': (graph.internal_indices), 
    "supervised_learning": (None),

    "update_rules": args.update_rules,  # "Van_Zwol" or "salvatori", "vectorized"
    "incremental_learning": True if args.model_type == "IPC" else False, 


    "delta_w_selection": args.delta_w_selection,  # "all" or "internal_only"

    "use_bias": args.use_bias,

    "normalize_msg": args.normalize_msg,

    "lr_params": (args.lr_values, args.lr_weights),
    #   (args.lr_gamma, args.lr_alpha), 
    "T": (args.T, args.T_test),  # Number of iterations for gradient descent. (T_train, T_test)
    "graph_structure": graph.edge_index, 
    "batch_size": train_loader.batch_size, 
    "edge_type":  graph.edge_type,

    "use_learning_optimizer": use_learning_optimizer,    # False or [0], [(weight_decay=)]
    "use_grokfast": args.grokfast,  # False or True
    
    # "weight_init": "uniform",   # xavier, 'uniform', 'based_on_f', 'zero', 'kaiming'
    "weight_init": args.weight_init,   # xavier, 'uniform', 'based_on_f', 'zero', 'kaiming'
    "activation": args.activation_func,  
    "clamping": None , # (0, torch.inf) or 'None' 

 }


import wandb

run = wandb.init(
    mode=args.use_wandb,
    # entity="Erencan Tatar", 
    project=f"PredCod",
    # name=f"T_{args.T}_lr_value_{args.lr_values}_lr_weights_{args.lr_weights}_",
    # name=f"{model_params_short}_{date_hour}",
    # id=f"{model_params_short}_{date_hour}",
    tags=tags_list,  # Add tags list here

    # dir=model_dir,
    # tags=["param_search", str(model_params["weight_init"]), model_params["activation"],  *learning_params['dataset_transform']], 
    # Track hyperparameters and run metadata
    # config=config_dict,  # Pass the updated config dictionary to wandb.init
)
from models.PC_graph import PCGNN


model = PCGNN(**model_params,   
    log_tensorboard=False,
    wandb_logger=run if args.use_wandb in ['online', 'run'] else None,
    debug=False, device=device)


print("Init model PC")

model.graph_type = args.graph_type
model.init_modes(batch_example=batch_example)
model.set_mode("training")



from tqdm import tqdm
train_energy, train_loss, train_acc = [], [], []
val_loss, val_acc, val_acc2 = [], [], []
num_epochs = 10

from datetime import datetime
from tqdm import tqdm
import torch

epochs = 100
start_time = datetime.now()

train_energy, train_loss, train_acc = [], [], []
val_loss, val_acc, val_acc2 = [], [], []
num_epochs = 10


break_num = 15

energy_vals = []

with torch.no_grad():

    for epochs in tqdm(range(num_epochs)):
        model.train()

        total_loss = 0
        energy = 0
        print("-----train_supervised-----")

        for batch_no, batch in enumerate(train_loader):

            # batch = Batch.from_data_list(batch)

            # batch_graph, batch_label, batch_edge_index = batch.x.to(model.device), batch.y.to(model.device), batch.edge_index.to(model.device)
            print(f"------------epochs/batch_nr---{epochs}/{batch_no}--------------")

            batch = batch.to(model.device)
            # Train model on the batch
            history = model.learning(batch)

            # for values in history:
            #   energy_vals.append(values)

            if batch_no >= 10:
                break
            # Compute loss (using last layer as prediction)
            # predictions = model.x[:, -10:]  # Supervision output (assumes 10 classes)
            # loss = criterion(predictions, batch_y)

            # energy += PCG.get_energy()

            # train_energy.append(energy/len(train_loader))

        loss, acc = 0, 0
        print("----test_iterative-----")
        model.test()
        cntr = 0
        for batch_no, batch in enumerate(val_loader):
            # y_pred = PCG.test_supervised(X_batch)
            y_batch = batch.y.clone()

            for i in range(len(batch)):
              sub_graph = batch[i]  # Access the subgraph
              sub_graph.x[sub_graph.supervision_indices, 0] = torch.ones_like(sub_graph.x[sub_graph.supervision_indices, 0])  # Check all feature dimensions

              # random
                # self.x[:,-do:] = torch.normal(0.5, self.node_init_std, size=(do,), device=DEVICE)

              sub_graph.x[sub_graph.supervision_indices, 0] = torch.torch.normal(0.5, 0.001, size=( sub_graph.x[sub_graph.supervision_indices, 0].shape))


            y_pred = model.test_class(batch)
            # loss += MSE(y_pred, onehot(y_batch, N=10) ).item()
            # print("y_pred", y_pred)
            a = y_pred == y_batch
            correct = torch.mean((a).float()).item()
            # print("correct", corret)
            acc += correct

            if batch_no >= 5:
                break
            cntr += 1

        val_acc.append(acc/len(val_loader))
        val_acc2.append(acc/cntr)
        val_loss.append(loss)
        print("val_acc", val_acc)
        print("val_acc2", val_acc2)
        print("val_acc2", val_acc2[-1])

        # print(f"\nEPOCH {epochs}/{num_epochs} \n #####################")
        # print(f"VAL acc:   {val_acc[i]:.3f}, VAL MSE:   {val_loss[i]:.3f}, TRAIN ENERGY:   {train_energy[i]:.3f}")

        # if early_stopper.early_stop(val_loss[i]):
        #     print(f"\nEarly stopping at epoch {i+1}")
        #     break

torch.cuda.empty_cache()


exit() 



















































































#########################################################################################################











# from models.PC_vanZwol import PC_graph_zwol 

# """ WITH MESSAGE_PASSING """
# from models.PC_vanZwolPYG import PC_graph_zwol_PYG

""" WITHOUT MESSAGE_PASSING """
# from models.PC_vanZwol_pyg_loader import PC_graph_zwol_PYG

from helper.vanZwol_optim import *

# Inference
f = tanh
lr_x = 0.5                  # inference rate 
T_train = 5                 # inference time scale
T_test = 10                 # unused for hierarchical model
incremental = True          # whether to use incremental EM or not
use_input_error = False     # whether to use errors in the input layer or not

# Learning
lr_w = 0.00001              # learning rate
weight_decay = 0             
grad_clip = 1
batch_scale = False
 
import helper.vanZwol_optim as optim

# vertices = [784, 48, 10] # input, hidden, output
# mask = get_mask_hierarchical([784,32,16,10])


PCG = PC_graph_zwol_PYG(f,
        device=device,
        num_vertices=graph.num_vertices,
        num_internal=sum(graph.internal_indices),
        adj=adj_matrix_pyg,
        edge_index=graph.edge_index,
        batch_size=batch_size,
        # mask=mask,
        lr_x=lr_x, 
        T_train=T_train,
        T_test=T_test,
        incremental=incremental, 
        use_input_error=use_input_error,
        )

optimizer = optim.Adam(
    PCG.params,
    learning_rate=lr_w,
    grad_clip=grad_clip,
    batch_scale=batch_scale,
    weight_decay=weight_decay,
)
PCG.set_optimizer(optimizer)

PCG.init_modes(batch_example=batch)


model = PCG


MSE = torch.nn.MSELoss()
from tqdm import tqdm

from datetime import datetime 

train_energy, train_loss, train_acc = [], [], []
val_loss, val_acc = [], []


epochs = 100

start_time = datetime.now()

from tqdm import tqdm
train_energy, train_loss, train_acc = [], [], []
val_loss, val_acc, val_acc2 = [], [], []
num_epochs = 10
model = PCG

from datetime import datetime
from tqdm import tqdm
import torch

epochs = 100
start_time = datetime.now()

train_energy, train_loss, train_acc = [], [], []
val_loss, val_acc, val_acc2 = [], [], []
num_epochs = 10
model = PCG  # Assuming PCG is defined elsewhere

break_num = 15

with torch.no_grad():
    for epoch in tqdm(range(num_epochs), desc="Epoch Progress"):
        model.train()
        total_loss = 0
        energy = 0
        print("\n-----train_supervised-----")
        print(len(train_loader))
        for batch_no, batch in enumerate(tqdm(train_loader, total=min(break_num, len(train_loader)), desc=f"Epoch {epoch+1} - Training", leave=False)):
            batch = batch.to(model.device)
            model.train_supervised(batch)

            if batch_no >= break_num:
                break

        loss, acc = 0, 0
        model.test()
        cntr = 0

        break_num_eval = 2
        print("\n----test_iterative-----")
        for batch_no, batch in enumerate(tqdm(val_loader, total=min(break_num_eval, len(val_loader)), desc=f"Epoch {epoch+1} - Validation", leave=False)):
            batch = batch.to(model.device)
            y_batch = batch.y.clone()
            y_pred = model.test_class(batch)
            correct = torch.mean((y_pred == y_batch).float()).item()
            acc += correct

            if batch_no >= break_num_eval:
                break
            cntr += 1

        # Corrected validation accuracy calculations
        val_acc.append(acc / min(break_num_eval, len(val_loader)))
        val_acc2.append(acc / cntr)
        val_loss.append(loss)

        print("Last prediction:", y_pred)
        print("Last y_batch:", y_batch)

        print(f"\nEpoch {epoch+1}/{num_epochs} Completed")
        print(f"  Validation Accuracy: {val_acc[-1]:.3f}")
        print(f"  Validation Accuracy (limited): {val_acc2[-1]:.3f}")



































# from dataset import CustomGraphDataset

# # Initialize the GraphBuilder
# custom_dataset_train = CustomGraphDataset(graph_params, **dataset_params)

# dataset_params["batch_size"] = args.batch_size
# dataset_params["NUM_INTERNAL_NODES"] = graph_params["internal_nodes"]
# # dataset_params["NUM_INTERNAL_NODES"] = (custom_dataset_train.NUM_INTERNAL_NODES)

# print("Device \t\t\t:", device)
# print("SUPERVISED on/off \t", dataset_params["supervised_learning"])


# from helper.plot import plot_adj_matrix

# single_graph = custom_dataset_train.edge_index


# print("--------------Init DataLoader --------------------")
# train_loader = DataLoader(custom_dataset_train, 
#                           batch_size=dataset_params["batch_size"], 
#                           shuffle=True, 
#                           generator=generator_seed,
#                           num_workers=1,
#                           pin_memory=True,
#                           drop_last=True,
#                           )


# NUM_SENSORY = 28*28  # 10

# ## TODO: FIX HOW TO DO THIS 
# #### ---------------------------------------------------------------------------------------------------------------
# # sensory_indices    = range(NUM_SENSORY)
# # internal_indices   = range(NUM_SENSORY, NUM_SENSORY + dataset_params["NUM_INTERNAL_NODES"])
# # num_vertices = NUM_SENSORY + dataset_params["NUM_INTERNAL_NODES"]  # Number of nodes in the graph
# # supervision_indices = None

# # if dataset_params["supervised_learning"]:
# #     label_indices     = range(NUM_SENSORY + dataset_params["NUM_INTERNAL_NODES"], NUM_SENSORY + dataset_params["NUM_INTERNAL_NODES"] + 10)
# #     supervision_indices = label_indices
# #     num_vertices += 10

# # print("sensory_indices\t\t:", len(sensory_indices), sensory_indices[0], "...", sensory_indices[-1])
# # print("internal_indices\t:", len(internal_indices), internal_indices[0], "...", internal_indices[-1])
# # print("num_vertices \t\t:", num_vertices)

# # if dataset_params["supervised_learning"]:
# #   assert num_vertices == len(sensory_indices) + len(internal_indices) + 10, "Number of vertices must match the sum of sensory and internal indices + labels"
# # else:
# #   assert num_vertices == len(sensory_indices) + len(internal_indices), "Number of vertices must match the sum of sensory and internal indices"
# #### ---------------------------------------------------------------------------------------------------------------

# num_vertices = custom_dataset_train.num_vertices
# sensory_indices = custom_dataset_train.sensory_indices
# internal_indices = custom_dataset_train.internal_indices
# supervision_indices = custom_dataset_train.supervision_indices



# print(train_loader.batch_size)
# for batch, clean_image in train_loader:
    
#     values, errors, predictions = batch.x[:, 0], batch.x[:, 1], batch.x[:, 2]
  
#     x, edge_index, y, edge_weight = batch.x, batch.edge_index, batch.y, batch.edge_attr
#     print("edge_index", edge_index.shape)

#     print(batch.x[:, 0].shape)
#     print(custom_dataset_train.edge_index.shape)
    

#     full_batch = edge_index

#     if graph_params["graph_type"]["name"] == "two_branch_graph":

#         number_of_internal_nodes = branch1_internal_nodes + branch2_internal_nodes

#         number_of_nodes = 784 + number_of_internal_nodes + 10

#         assert batch.x[:, 0] == number_of_nodes, f"Number of nodes in the graph must be {number_of_nodes} but is {batch.x[:, 0]}"

#     break




# ######################################################################################################### 
# ####                                            VALIDATION                                          #####
# ######################################################################################################### 
 
# from helper.validation import validate_messagePassing
# # validate_messagePassing()


# from helper.validation import compare_class_args

# from models.PC import PCGNN, PCGraphConv
# # from models.IPC import IPCGNN, IPCGraphConv

# # Usage example: comparing IPCGraphConv and PCGraphConv
# # compare_class_args(IPCGraphConv, PCGraphConv)


# ######################################################################################################### 
# ####                                            FIND OPTIMAL LR                                     #####
# ######################################################################################################### 
# """ 
# SKIPPING FOR NOW, see local  
# """





# ######################################################################################################### 
# ####                                              Model  (setup)                                    #####
# ######################################################################################################### 

# # lr_gamma, lr_alpha =  (0.1 ,  0.0001)
# # lr_gamma, lr_alpha =  (0.1, 0.00001)

# # Check if args.optimizer is valid
# if isinstance(args.optimizer, (float, int)) and args.optimizer >= 0:
#     # Optimizer enabled with explicit weight decay
#     use_learning_optimizer = [args.optimizer]
#     weight_decay = args.optimizer
#     print(f"Optimizer enabled with weight decay: {weight_decay}")
# elif args.optimizer is False:
#     # Optimizer disabled
#     use_learning_optimizer = False
#     weight_decay = None
#     print("Optimizer disabled (False).")
# else:
#     # Catch invalid or improperly parsed True cases
#     raise ValueError(f"Invalid value for optimizer: {args.optimizer}. Expected False or a float value.")


# # if args.update_rules == "Van_Zwol":

# #     # transpose adj
# #     custom_dataset_train.edge_index_tensor = torch.flip(custom_dataset_train.edge_index_tensor, dims=[0])

# #     print("DISREGARDING EDGE TYPES FOR NOW")
# #     # "edge_type":  custom_dataset_train.edge_type,



# model_params = {
    
#     'num_vertices': num_vertices,
#     'sensory_indices': (sensory_indices), 
#     'internal_indices': (internal_indices), 
#     "supervised_learning": (supervision_indices),

#     "update_rules": args.update_rules,  # "Van_Zwol" or "salvatori", "vectorized"
#     "incremental_learning": True if args.model_type == "IPC" else False, 


#     "delta_w_selection": args.delta_w_selection,  # "all" or "internal_only"

#     "use_bias": args.use_bias,

#     "normalize_msg": args.normalize_msg,

#     "lr_params": (args.lr_values, args.lr_weights),
#     #   (args.lr_gamma, args.lr_alpha), 
#     "T": (args.T, args.T_test),  # Number of iterations for gradient descent. (T_train, T_test)
#     "graph_structure": custom_dataset_train.edge_index_tensor, 
#     "batch_size": train_loader.batch_size, 
#     "edge_type":  custom_dataset_train.edge_type,

#     "use_learning_optimizer": use_learning_optimizer,    # False or [0], [(weight_decay=)]
#     "use_grokfast": args.grokfast,  # False or True
    
#     # "weight_init": "uniform",   # xavier, 'uniform', 'based_on_f', 'zero', 'kaiming'
#     "weight_init": args.weight_init,   # xavier, 'uniform', 'based_on_f', 'zero', 'kaiming'
#     "activation": args.activation_func,  
#     "clamping": None , # (0, torch.inf) or 'None' 

#  }

# # 

# learning_params = model_params.copy()
# learning_params['sensory_indices'] = list(learning_params['sensory_indices'])
# learning_params['internal_indices'] = list(learning_params['internal_indices'])
# learning_params['supervised_learning'] = list(learning_params['supervised_learning'])
# # learning_params['transform'] = transform.to_dict()["transform"]
# learning_params['dataset_transform'] = args.dataset_transform

# learning_params['graph_structure'] = (learning_params['graph_structure']).cpu().numpy().tolist()
# optimizer_str = str(args.optimizer) if isinstance(args.optimizer, float) else str(args.optimizer)

# # model_params_short = f"num_iternode_{args.num_internal_nodes}_T_{args.T}_lr_w_{args.lr_weights}_lr_val_{args.lr_values}_Bsize_{train_loader.batch_size}"
# model_params_short = f"{args.model_type}_{args.graph_type}_T_{args.T}_lr_w_{args.lr_weights}_lr_val_{args.lr_values}"
# print(len(model_params_short), model_params_short)
# model_params_name = (
#     f"{args.model_type}_"
#     f"nodes_{graph_params['internal_nodes']}_" 
#     f"T_{args.T}_"
#     f"lr_vals_{args.lr_values}_"
#     f"lr_wts_{args.lr_weights}_"
#     f"bs_{args.batch_size}_"
#     f"act_{args.activation_func}_"
#     f"init_{args.weight_init}_"
#     f"graph_{args.graph_type}_"
#     f"sup_{args.supervision_label_val}_"
#     f"norm_{args.normalize_msg}_"
#     f"nums_{'_'.join(map(str, args.numbers_list))}_"
#     f"N_{args.N}_"
#     f"ep_{args.epochs}_"
#     f"opt_{optimizer_str}_"
#     f"trans_{'_'.join(args.dataset_transform) if args.dataset_transform else 'none'}"
# )

# model_params_name_full = (
#     f"model_{args.model_type}_"
#     f"num_internal_nodes_{graph_params['internal_nodes']}_"
#     f"T_{args.T}_"
#     f"lr_values_{args.lr_values}_"
#     f"lr_weights_{args.lr_weights}_"
#     f"batch_size_{args.batch_size}_"
#     f"activation_{args.activation_func}_"
#     f"weight_init_{args.weight_init}_"
#     f"graph_type_{args.graph_type}_"
#     f"supervision_val_{args.supervision_label_val}_"
#     f"normalize_msg_{args.normalize_msg}_"
#     f"numbers_list_{'_'.join(map(str, args.numbers_list))}_"
#     f"N_{args.N}_"
#     f"epochs_{args.epochs}_"
#     f"optimizer_{optimizer_str}_"
#     f"dataset_transform_{'_'.join(args.dataset_transform) if args.dataset_transform else 'none'}"
# )

# def default(obj):
#     if type(obj).__module__ == np.__name__:
#         if isinstance(obj, np.ndarray):
#             return obj.tolist()
#         else:
#             return obj.item()
    
#     raise TypeError('Unknown type:', type(obj))

# # combi of learning params and dataset params
# params_dict = {**dataset_params, **learning_params}


# import json

# from datetime import datetime




# save_model_params = False

# GRAPH_TYPE = graph_params["graph_type"]["name"]    #"fully_connected"
# # GRAPH_TYPE = "test"    #"fully_connected"

# # Fetch the graph parameters based on the selected graph type

# date_hour = datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
# path = ""
# # Initialize base path depending on mode (training or experimenting)
# if args.mode == "experimenting":
#     path += f"trained_models_experimenting/"
# elif args.mode == "training":
#     path += f"trained_models/"
# else:
#     raise ValueError("Invalid mode")

# path += f"{args.model_type.lower()}/{graph_params['graph_type']['name']}/"
  
# # Modify the path based on the graph configuration (removing sens2sens or sens2sup)
# if graph_params["graph_type"]["params"]["remove_sens_2_sens"] and graph_params["graph_type"]["params"]["remove_sens_2_sup"]:
#     graph_type_ = "_no_sens2sens_no_sens2sup"
# elif graph_params["graph_type"]["params"]["remove_sens_2_sens"]:
#     graph_type_ = "_no_sens2sens"
# elif graph_params["graph_type"]["params"]["remove_sens_2_sup"]:
#     graph_type_ = "_no_sens2sup"
# else:
#     graph_type_ = "_normal"  # If neither are removed, label the folder as 'normal'

# path += graph_type_
# # Append graph type, model parameters, and timestamp to the path
# path += f"/{model_params_name}_{date_hour}/"
# model_dir = path

# # Define the directory path
# print("Saving model, params, graph_structure to :", model_dir)

# # # Ensure the directory exists
# # os.makedirs(model_dir, exist_ok=True)

# # # For saving, validation, re-creation 
# # os.makedirs(os.path.join(model_dir, "checkpoints"), exist_ok=True)
# # os.makedirs(os.path.join(model_dir, "parameter_info"), exist_ok=True)

# # # For testing the model
# # os.makedirs(os.path.join(model_dir, "eval"), exist_ok=True)
# # os.makedirs(os.path.join(model_dir, "eval/generation"), exist_ok=True)
# # # os.makedirs(os.path.join(model_dir, "reconstruction"), exist_ok=True)
# # os.makedirs(os.path.join(model_dir, "eval/classification"), exist_ok=True)
# # os.makedirs(os.path.join(model_dir, "eval/denoise"), exist_ok=True)
# # os.makedirs(os.path.join(model_dir, "eval/occlusion"), exist_ok=True)

# # # Monitor during training
# # os.makedirs(os.path.join(model_dir, "energy"), exist_ok=True)

# # plot_adj_matrix(single_graph, model_dir, 
# #                 node_types=(sensory_indices, internal_indices, supervision_indices))
# # plot_adj_matrix(full_batch, model_dir, node_types=None)


# config_dict = vars(args)  # Convert argparse Namespace to dictionary

# config_dict.update({
#     'graph_type_conn': graph_type_,  # Adding model type manually
#     'remove_sens_2_sens_': args.remove_sens_2_sens,  # Custom boolean flag for data augmentation
#     'remove_sens_2_sup_': args.remove_sens_2_sup,  # Custom boolean flag for data augmentation
#     "checkpoint_dir": model_dir,  # Track where model checkpoints are saved
#     **graph_params["graph_type"]["params"],  # Merge dynamic graph parameters
# })

# run = wandb.init(
#     mode=args.use_wandb,
#     # entity="Erencan Tatar", 
#     project=f"PredCod",
#     # name=f"T_{args.T}_lr_value_{args.lr_values}_lr_weights_{args.lr_weights}_",
#     name=f"{model_params_short}_{date_hour}",
#     # id=f"{model_params_short}_{date_hour}",
#     tags=tags_list,  # Add tags list here

#     dir=model_dir,
#     # tags=["param_search", str(model_params["weight_init"]), model_params["activation"],  *learning_params['dataset_transform']], 
#     # Track hyperparameters and run metadata
#     config=config_dict,  # Pass the updated config dictionary to wandb.init
# )

# # Contains graph edge matrix and other parameters so quite big to open.
# if save_model_params:
#     # Save the dictionary to a text file
#     with open(model_dir + "parameter_info/params_full.txt", 'w') as file:
#         json.dump(params_dict, file, default=default)
#     print('Done')

#     # Store the exact command-line arguments in a text file
#     import sys
#     command = ' '.join(sys.argv)
#     with open(model_dir +'parameter_info/command_log.txt', 'w') as f:
#         f.write(command)

#     with open('trained_models/current_running.txt', 'w') as f:
#         f.write(model_dir)

#     # Save the (small) dictionary to a text file
#     params_dict_small = {}
#     keys_to_copy = ['supervised_learning', 'numbers_list', 'NUM_INTERNAL_NODES',  
#                     'N',  'batch_size','use_learning_optimizer', 'weight_init', 'activation', ]
#     # copy value from params_dict to params_dict_small
#     for key in keys_to_copy:
#         params_dict_small[key] = params_dict[key]

#     if "dataset_transform" in params_dict:
#         params_dict_small["dataset_transform"] = params_dict["dataset_transform"]


#     if save_model_params:
#         # Save the dictionary to a text file
#         with open(model_dir + "parameter_info/params_small.txt", 'w') as file:
#             json.dump(params_dict_small, file, default=default)
#         print('Done')

# print(f"Using batch size of \t: {train_loader.batch_size}")
# print("Device \t\t\t:",          device)
# print("Model type", args.model_type.lower())


# model = PCGNN(**model_params,   
#     log_tensorboard=False,
#     wandb_logger=run if args.use_wandb in ['online', 'run'] else None,
#     debug=False, device=device)


# if args.model_type.lower() == "pc":
    
#     print("-----------Loading PC model without incremental -----------")

# if args.model_type.lower() == "ipc":
#     print("-----------Loading PC model with incremental -----------")


# model.pc_conv1.graph_type = args.graph_type


# # Magic
# wandb.watch(model, 
#             log="all",   # (str) One of "gradients", "parameters", "all", or None
#             log_freq=10)


# from helper.plot import plot_model_weights, plot_energy_graphs


# # save_path = os.path.join(model_dir, 'parameter_info/weight_matrix_visualization_beforeTraining.png')
# # # plot_model_weights(model, save_path)
# # plot_model_weights(model, GRAPH_TYPE, model_dir=save_path, save_wandb=True)



# ######################################################################################################### 
# ####                                            Eval setup (during training )                       #####
# ######################################################################################################### 



# # device = torch.device('cpu')
# from eval_tasks import classification, denoise, occlusion, generation #, reconstruction
# from eval_tasks import plot_digits_vertically, progressive_digit_generation

# # num_wandb_img_log = len(custom_dataset_train.numbers_list)   # Number of images to log to wandb
# num_wandb_img_log = 1   # Number of images to log to wandb
# model.pc_conv1.batchsize = 1

# ### Make dataloader for testing where we take all the digits of the number_list we trained on ###
# dataset_params_testing = dataset_params.copy()

# if "batch_size" in dataset_params_testing.keys():
#     # remove keys 
#     del dataset_params_testing["batch_size"]

# if "NUM_INTERNAL_NODES" in dataset_params_testing.keys():
#     # remove keys 
#     del dataset_params_testing["NUM_INTERNAL_NODES"]

# dataset_params_testing["edge_index"] = custom_dataset_train.edge_index

# dataset_params_testing["mnist_dataset"] = mnist_testset
# dataset_params_testing["N"] = "all"
# dataset_params_testing["supervision_label_val"] = dataset_params["supervision_label_val"]

# # --------------------------------------------
# model.trace(values=True, errors=True)
# model.pc_conv1.trace_activity_values = True 
# model.pc_conv1.trace_activity_preds = True 
# model.pc_conv1.batchsize = 1


# for key in dataset_params_testing:
#     print(key, ":\t ", dataset_params_testing[key])


# # CustomGraphDataset params
# custom_dataset_test = CustomGraphDataset(graph_params, **dataset_params_testing, 
#                                         indices=(num_vertices, sensory_indices, internal_indices, supervision_indices)
#                                         )
# # dataset_params_testing["batch_size"] = 2

# print("-----test_loader_1_batch---------")
# test_loader_1_batch = DataLoader(custom_dataset_test, 
#                          batch_size=1, 
#                          shuffle=True, 
#                          generator=generator_seed,
#                          drop_last=True,
#                         # num_workers=1,
#                         # pin_memory=True,
#                                     )
# print("-----test_loader---------")
# test_loader = DataLoader(custom_dataset_test, 
#                          batch_size=args.batch_size, 
#                          shuffle=True, 
#                          generator=generator_seed,
#                          drop_last=True,
#                         # num_workers=1,
#                         # pin_memory=True,
#                                     )

# from helper.eval import get_clean_images_by_label

# # want to see how good it can recreate the images it has seen during training
# clean_images = get_clean_images_by_label(mnist_trainset, num_images=20)

# import os
# import logging

# from helper.log import write_eval_log
# from helper.plot import plot_graph_with_edge_types, plot_updated_edges


# #  try except block for training
# plot_edge_types = False
# if args.mode == "training" and plot_edge_types:
#     try:
#         N = custom_dataset_train.num_vertices
#         edge_index = custom_dataset_train.edge_index
#         edge_types = custom_dataset_train.edge_type

#         edge_type_map = {
#             "Sens2Sens": 0, "Sens2Inter": 1, "Sens2Sup": 2,
#             "Inter2Sens": 3, "Inter2Inter": 4, "Inter2Sup": 5,
#             "Sup2Sens": 6, "Sup2Inter": 7, "Sup2Sup": 8
#         }

#         plot_graph_with_edge_types(N, edge_index, edge_types, edge_type_map)

#     except Exception as e:
#         print("Could not plot graph with edge types")
#         print(e)

#         # log error to wandb
#         wandb.log({"edge_type_plot_error": str(e)})



# ######################################################################################################### 
# ####                                              Model  (training)                                 #####
# ######################################################################################################### 
# model.pc_conv1.set_mode("training")


# # plot updated edges.
# try:
#     N = model.pc_conv1.num_vertices  # Number of nodes
#     edge_index = model.pc_conv1.edge_index_single_graph
#     edges_2_update = model.pc_conv1.edges_2_update
#     # Plot and save
#     plot_updated_edges(N, edge_index, edges_2_update, args.delta_w_selection, model_dir="plots", show=True, sample_size=15000)
# except Exception as e:
#     print("Could not plot updated edges")
#     print(e)

#     # log error to wandb
#     wandb.log({"updated_edges_plot_error": str(e)})


# import time 
# torch.cuda.empty_cache()

# print(model)      

# model = model.to(device)
# # assert train_loader.batch_size == 1
# print(len(train_loader))
# print("Starting training")


# # Initialize early stopping and history
# earlystop = False
# history = {
#     "internal_energy_per_epoch": [],
#     "sensory_energy_per_epoch": [],
# }

# wandb.watch(model.pc_conv1, log="all", log_freq=10)
# # wandb.watch(self.pc_conv1, log="all", log_freq=10)

# reduce_lr_weights = True
# print("Using reduce_lr_weights: ", reduce_lr_weights)

# model.train()

# # Define the early stopping threshold and OOM warning
# threshold_earlystop = 0.05
# max_energy_threshold = 1e6
# accuracy_mean = 0 

# items_per_epoch = 10 
# # because batch size is 1
# if args.update_rules == "Van_Zwol":
#     items_per_epoch = 50


# start_time = time.time()

# training_labels = [] 
# from collections import Counter

# for epoch in range(args.epochs):
#     total_loss = 0
#     last_loss = 1e10

#     if earlystop:
#         break
    
#     # close all open plt figures 
#     plt.close('all')
    
#     model.pc_conv1.batchsize = train_loader.batch_size
#     model.pc_conv1.trace_activity_values = False 
#     model.pc_conv1.trace_activity_preds = False
#     # model.pc_conv1.set_mode("training")

#     for idx, (batch, clean) in enumerate(train_loader):
#         torch.cuda.empty_cache()
#         # training_labels.append(int(batch.y.item()))

#         for label in batch.y:
#             training_labels.append(int(label.item()))

#         try:
#             # print("Label:", batch.y, "Input Shape:", batch.x.shape)
#             model.train()

#             batch = batch.to(device)
#             history_epoch = model.learning(batch)

#             # return predictions on supervised nodes
#             x_hat = model.pc_conv1.values.view(model.pc_conv1.batchsize, model.pc_conv1.num_vertices)
#             logits = x_hat[:, -10:]                       # batch_size x 10
#             y_pred = torch.argmax(logits, axis=1)         # batch_size
#             # print("--during training y_pred", y_pred)
#             # print("--during training y_true", batch.y)

#             if history_epoch:
#                 # Append energy values to history
#                 history["internal_energy_per_epoch"].append(history_epoch["internal_energy_mean"])
#                 history["sensory_energy_per_epoch"].append(history_epoch["sensory_energy_mean"])

#                 # Log energy values for this batch/epoch to wandb
#                 wandb.log({
#                     "epoch": epoch,
#                     "Training/internal_energy_mean": history_epoch["internal_energy_mean"],
#                     "Training/sensory_energy_mean": history_epoch["sensory_energy_mean"],
                    
#                 })

#                 model.pc_conv1.restart_activity()

#                 print(f"------------------ Epoch {epoch}: Batch {idx} / {len(train_loader)} ------------------")

#                 # if internal_energy_mean or sensory_energy_mean is nan or inf, break
#                 if not np.isfinite(history_epoch["internal_energy_mean"]) or not np.isfinite(history_epoch["sensory_energy_mean"]):
#                     print("Energy is not finite, stopping training")
#                     earlystop = True
#                     break

#                 # Early stopping based on loss change
#                 if abs(last_loss - history_epoch["internal_energy_mean"]) < threshold_earlystop:
#                     earlystop = True
#                     print(f"EARLY STOPPED at epoch {epoch}")
#                     print(f"Last Loss: {last_loss}, Current Loss: {history_epoch['internal_energy_mean']}")
#                     break

#                 # Early stopping based on high energy
#                 if history_epoch["internal_energy_mean"] > max_energy_threshold:
#                     print("energy :", history_epoch["internal_energy_mean"])
#                     print("Energy too high, stopping training")
#                     earlystop = True
#                     break

#             # PRINT MEAN OF THE WEIGHTS
#             w = model.pc_conv1.weights.cpu().detach().numpy()
#             print("----------------------Mean of weights: ----=====", np.mean(w))

#             if idx > 20:
#                 break

#         except RuntimeError as e:
#             if 'out of memory' in str(e):
#                 print('WARNING: CUDA ran out of memory, skipping batch...')
#                 torch.cuda.empty_cache()
#                 continue
#             else:
#                 torch.cuda.empty_cache()
#                 raise e

#     print(f"Epoch {epoch} / {args.epochs} completed")
    
#     # print mean of weights 
#     w = model.pc_conv1.weights.cpu().detach().numpy()

#     print("---Mean of weights: =====", np.mean(w))
    
   
#     acc = 0 

#     # Eval      
#     for batch_idx, (batch, clean) in enumerate(test_loader):
        
#         print("-----------------------  Eval  -----------------------")      
        
#         x_batch, y_batch = batch.x.to(device), batch.y.to(device)

#         y_pred = model.pc_conv1.test_class(batch.to(device))
#         print(y_batch, y_pred)

#         a = torch.mean(( y_pred == y_batch ).float()).item()
#         acc += a

#         if batch_idx > 10:
#             break
    
#     wandb.log({
#         "epoch": epoch,
#         "classification_new/y_true": acc,
#     })

#     print(y_batch, y_pred)
#     print(f"Accuracy {batch_idx}: ", a)
#     print(f"Accuracy {batch_idx}: ", acc/len(test_loader))
#     print(f"Accuracy {batch_idx}: ", acc/10)

   

#         # # log distribution of y_pred and y_true
#         # y_pred = y_pred.cpu().detach().numpy()
#         # y_batch = y_batch.cpu().detach().numpy()
#         # wandb.log({
#         #     "epoch": epoch,
#         #     "classification_new/y_pred": wandb.Histogram(y_pred),
#         #     "classification_new/y_true": wandb.Histogram(y_batch),
#         # })
        
      

       

    
#     if epoch % 5 == 0:

#         plot_model_weights(model, GRAPH_TYPE, model_dir=None, save_wandb=True)


# # Append to the appropriate file based on whether the training crashed or completed successfully
# if earlystop:
#     print("Stopping program-------")

#     # Log in WandB that the run crashed
#     wandb.log({"crashed": True})

#     # Finish WandB logging first
#     try:
#         wandb.finish()
#     except Exception as e:
#         print(f"WandB logging failed: {e}")

#     # Now remove the folder to save storage
#     import shutil
#     shutil.rmtree(model_dir)
    
#     print("----------------------------Removed run folder--------------------------- ")
#     print(model_dir)
    
#     exit()
# else:
#     wandb.log({"crashed": False})

#     element_counts = Counter(training_labels)

#     # Log a bar plot to WandB
#     wandb.log({"Training/element_counts_bar": wandb.plot.bar(
#         wandb.Table(data=[[k, v] for k, v in element_counts.items()], columns=["Label", "Count"]),
#         "Label",
#         "Count",
#         title="Element Counts"
#     )})

# # If training completed successfully, log to the finished runs file
# with open('trained_models/finished_training.txt', 'a') as file:
#     file.write(f"{model_dir}\n")

# save_path = os.path.join(model_dir, 'parameter_info')
# model.save_weights(path=save_path, overwrite=False)

# wandb.log({"run_complete": True})
# wandb.log({"model_dir": model_dir})
