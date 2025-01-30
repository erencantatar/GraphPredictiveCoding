# import torch
# import torchvision
# import torchvision.transforms as transforms

# import torchvision.transforms as transforms
# from torchvision.datasets import MNIST
# from torchvision.transforms import ToTensor
# from torch_geometric.data import Data

# from torch_geometric.nn import MessagePassing
# from torch_geometric.utils import add_self_loops, degree

# import torch.nn.init as init
# import matplotlib.pyplot as plt
# import networkx as nx
# import numpy as np
# from icecream import ic


# from torch_scatter import scatter_mean
# from torch_geometric.data import Data
# from torch.utils.data import Dataset

# from torch_geometric.loader import DataLoader
# from torch_geometric.data import Data
# import random

# from torch_geometric.nn import MessagePassing
# import torch.nn.functional as F
# import os

# ######################################################################################################### 
# ####                                      Post-Training model Eval                                  #####
# ######################################################################################################### 

# # import wandb 

# # NIET AF. 
# assert 1 == 2 


# import os
# import argparse
# from helper.args import true_with_float, valid_str_or_float, valid_int_or_all, valid_int_list
# from helper.activation_func import activation_functions

# # Parsing command-line arguments
# parser = argparse.ArgumentParser(description='Train a model with specified parameters.')

# # -----dataset----- 
# parser.add_argument('--dataset_transform', nargs='*', default=[], help='List of transformations to apply.', choices=['normalize_min1_plus1', 'normalize_mnist_mean_std', 'random_rotation', 'none'])
# parser.add_argument('--numbers_list', type=valid_int_list, default=[0, 1, 3, 4, 5, 6, 7], help="A comma-separated list of integers describing which distinct classes we take during training alone")
# parser.add_argument('--N', type=valid_int_or_all, default=20, help="Number of distinct trainig images per class; greater than 0 or the string 'all' for all instances o.")

# parser.add_argument('--supervision_label_val', default=10, type=int, required=True, help='An integer value.')



# # -----graph----- 
# from graphbuilder import graph_type_options
# parser.add_argument('--num_internal_nodes', type=int, default=1500, help='Number of internal nodes.')
# parser.add_argument('--graph_type', type=str, default="fully_connected", help='Type of Graph', choices=list(graph_type_options.keys()))

# # -----model----- 
# parser.add_argument('--model_type', type=str, default="PC", help='Predictive Model type: [PC,IPC] ', choices=["PC", "IPC"])
# parser.add_argument('--weight_init', default=0.001, type=valid_str_or_float, help='A float (e.g 0.01) or a string from the list: uniform, xavier')
# parser.add_argument('--T', type=int, default=40, help='Number of iterations for gradient descent.')
# parser.add_argument('--lr_values', type=float, default=0.001, help='Learning rate values (alpha).')
# parser.add_argument('--lr_weights', type=float, default=0.01, help='Learning rate weights (gamma).')
# parser.add_argument('--activation_func', default="swish", type=str, choices=list(activation_functions.keys()), required=True, help='Choose an activation function: tanh, relu, leaky_relu, linear, sigmoid, hard_tanh, swish')


# # -----learning----- 
# parser.add_argument('--epochs', type=int, default=5, help='Number of epochs to train.')
# parser.add_argument('--batch_size', type=int, default=1, help='Batch size.')
# parser.add_argument('--seed', type=int, default=42, help='Random seed.')
# parser.add_argument('--optimizer', type=true_with_float, default=False,
#                     help="Either False or, if set to True, requires a float value.")

# # logging 
# import wandb
# parser.add_argument('--use_wandb', type=str, default="disabled", help='Wandb mode.', choices=['shared', 'online', 'run', 'dryrun', 'offline', 'disabled'])

# args = parser.parse_args()

# # Using argparse values
# torch.manual_seed(args.seed)

# generator_seed = torch.Generator()
# generator_seed.manual_seed(args.seed)

# # Device configuration
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f'Using device: {device}')
# print(f"Seed used", args.seed)
# if torch.cuda.is_available():
#     print("Device name: ", torch.cuda.get_device_name(0))


# import torchvision.transforms as transforms
# import numpy as np


# transform_list = [
#     transforms.ToTensor()
# ]

# if args.dataset_transform:

#     if "normalize_min1_plus1" in args.dataset_transform:
#         transform_list.append(transforms.Normalize((0.5,), (0.5,)))

#     if "normalize_mnist_mean_std" in args.dataset_transform:
#         transform_list.append(transforms.Normalize((0.1307,), (0.3081,)))
    
#     if "random_rotation" in args.dataset_transform:
#         transform_list.append(transforms.RandomRotation(degrees=20))
    

# # Create the transform
# transform = transforms.Compose(transform_list)

# mnist_trainset = torchvision.datasets.MNIST(root='../data', train=True, download=True, transform=transform)
# mnist_testset  = torchvision.datasets.MNIST(root='../data', train=False, download=True, transform=transform)



# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')




# ######################################################################################################### 
# ####                                            Dataset                                             #####
# ######################################################################################################### 


# ## Subset of the dataset (for faster development)
# # subset_size = 100  # Number of samples to use from the training set
# # indices = list(range(len(mnist_trainset)))
# # random.shuffle(indices)
# # subset_indices = indices[:subset_size]

# # mnist_train_subset = torch.utils.data.Subset(mnist_trainset, subset_indices)
# # print("USSSSSING SUBSEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEET")

# # CustomGraphDataset params
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

# from graphbuilder import graph_type_options

# # Define the graph type
# # Options: "fully_connected", "fully_connected_w_self", "barabasi", "stochastic_block", "fully_connected_no_sens2sup"
# graph_params = {
#     "internal_nodes": args.num_internal_nodes,  # Number of internal nodes
#     "supervised_learning": True,  # Whether the task involves supervised learning
#     "graph_type": 

#       {    
#         "name": args.graph_type, # Options: "fully_connected", "fully_connected_w_self", "barabasi", "stochastic_block"
#         "params": graph_type_options[args.graph_type]["params"]
#       },      
  
# }

# if graph_params["graph_type"]["name"] == "stochastic_block":
#   assert graph_params["internal_nodes"] == (graph_params["graph_type"]["params"]["num_communities"] * graph_params["graph_type"]["params"]["community_size"])

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
#                           num_workers=1
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

#     break




# ######################################################################################################### 
# ####                                            VALIDATION                                          #####
# ######################################################################################################### 
 
# # from helper.validate_MP import validate_messagePassing
# # validate_messagePassing()

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


# model_params = {
    
#     'num_vertices': num_vertices,
#     'sensory_indices': (sensory_indices), 
#     'internal_indices': (internal_indices), 
#     "supervised_learning": (supervision_indices),

#     "lr_params": (args.lr_values, args.lr_weights),
#     #   (args.lr_gamma, args.lr_alpha), 
#     "T": args.T,
#     "graph_structure": custom_dataset_train.edge_index_tensor, 
#     "batch_size": train_loader.batch_size, 
#     "use_learning_optimizer": args.optimizer if not args.optimizer  else [args.optimizer],    # False or [0], [(weight_decay=)]
    
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

# model_params_name = f"num_internal_nodes_{args.num_internal_nodes}_T_{args.T}_lr_weights_{args.lr_weights}_lr_values_{args.lr_values}_batch_size_{train_loader.batch_size}"

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

# date_hour = datetime.now().strftime("%Y-%m-%d_%H_%M_%S")

# # model_dir = f"trained_models/test/{GRAPH_TYPE}/{model_params_name}_{date_hour}/"
# model_dir = args.


# run = True 

# if args.model_type.lower() == "pc":
        
#     from models.PC import PCGNN

#     model = PCGNN(**model_params,   
#         log_tensorboard=False,
#         wandb_logger=run if args.use_wandb in ['online', 'run'] else None,
#         debug=False, device=device)

#     print("-----------Loading PC model-----------")

# if args.model_type.lower() == "ipc":
        
#     from models.IPC import IPCGNN

#     model = IPCGNN(**model_params,   
#         log_tensorboard=False,
#         wandb_logger=run if args.use_wandb in ['online', 'run'] else None,
#         debug=False, device=device)
#     print("-----------Loading IPC model-----------")

# # Magic
# wandb.watch(model, 
#             log="all",   # (str) One of "gradients", "parameters", "all", or None
#             log_freq=10)


# # W = model_dir
# # model.load_weights(W, graph, b=None)




# # Save model weights 
# ######################################################################################################### 
# ####                                            Evaluation (setup)                                  #####
# ######################################################################################################### 
 
# # device = torch.device('cpu')
# from eval_tasks import classification, denoise, occlusion, generation #, reconstruction
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



# for key in dataset_params_testing:
#     print(key, ":\t ", dataset_params_testing[key])


# # CustomGraphDataset params
# custom_dataset_test = CustomGraphDataset(graph_params, **dataset_params_testing, 
#                                         indices=(num_vertices, sensory_indices, internal_indices, supervision_indices)
#                                         )
# # dataset_params_testing["batch_size"] = 2

# test_loader = DataLoader(custom_dataset_test, batch_size=1, shuffle=True, generator=generator_seed)


# model.pc_conv1.batchsize = 1

# ######################################################################################################### 
# ####                                            Evaluation (tasks)                                  #####
# ######################################################################################################### 
 
# test_params = {
#     "model_dir": model_dir,
#     "T": 300,
#     "supervised_learning":True, 
#     "num_samples": 5,
#     "add_sens_noise": False,
# }

# # model.pc_conv1.lr_values = 0.1 
# # model.pc_conv1.lr_values = model_params["lr_params"][0]

# MSE_values_occ = occlusion(test_loader, model, test_params)

# test_params["add_sens_noise"] = True
# MSE_values_occ_noise = occlusion(test_loader, model, test_params)

# ######################################################################################################### 

# model.pc_conv1.batchsize = 1
# test_params = {
#     "model_dir": model_dir,
#     "T":300,
#     "supervised_learning":False, 
#     "num_samples": 30,
# }

# # model.pc_conv1.lr_values = 0.1
# # model.pc_conv1.lr_values = model_params["lr_params"][0]

# accuracy_mean = classification(test_loader, model, test_params)
# ######################################################################################################### 

# test_params = {
#     "model_dir": model_dir,
#     "T": 300,
#     "supervised_learning":True, 
#     "num_samples": 6,
# }

# # model.pc_conv1.lr_values = 0.1
# # model.pc_conv1.lr_values = model_params["lr_params"][0]

# MSE_values_denoise_sup = denoise(test_loader, model, test_params)

# test_params["supervised_learning"] = False
# MSE_values_denoise = denoise(test_loader, model, test_params)
                            
# # MSE_values = denoise(test_loader, model, supervised_learning=True)
# # print("MSE_values", MSE_values)
# ######################################################################################################### 



# test_params = {
#     "model_dir": model_dir,
#     "T": 300,
#     "supervised_learning":True, 
#     "num_samples": 12,
# }

# # model.pc_conv1.lr_values = 0.1
# # model.pc_conv1.lr_values = model_params["lr_params"][0]

# generation(test_loader, model, test_params)
                            
# # MSE_values = denoise(test_loader, model, supervised_learning=True)
# # print("MSE_values", MSE_values)


# print("accuracy_mean", accuracy_mean)

# print("model_dir", model_dir)

# # write a text file with these 


# # Open the file in write mode
# with open(model_dir + "eval/eval_scores.txt", 'w') as file:
#     # Write each list to the file

#     file.write("MSE_values_denoise_sup:\n")
#     file.write(", ".join(map(str, MSE_values_denoise_sup)) + "\n\n")

#     file.write("MSE_values_denoise:\n")
#     file.write(", ".join(map(str, MSE_values_denoise)) + "\n\n")
    
#     file.write("MSE_values_occ_noise:\n")
#     file.write(", ".join(map(str, MSE_values_occ_noise)) + "\n\n")
    
#     file.write("MSE_values_occ:\n")
#     file.write(", ".join(map(str, MSE_values_occ)) + "\n\n")

#     file.write("accuracy_mean:\n")
#     file.write(str(accuracy_mean) + "\n\n")


# from datetime import datetime
# # Get the current date and time
# current_datetime = datetime.now()
# # Print the current date and time
# print("Current date and time:", current_datetime)



# wandb.finish()