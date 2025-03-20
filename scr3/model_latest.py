


#################################### IMPORTS ####################################
import numpy as np
import torch
from datetime import datetime
# import logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from torch import nn

import torch

import torch
from torch_geometric.nn import MessagePassing
import torch.nn.functional as F
import matplotlib.pyplot as plt
import wandb 

    
import matplotlib.pyplot as plt
import torch
import os
import wandb

start_time = datetime.now()
dt_string = start_time.strftime("%Y%m%d-%H.%M")

# Check if CUDA (GPU) is available
if torch.cuda.is_available():
    # Set the default device to the first available GPU (index 0)
    print("CUDA available, using GPU")
    torch.cuda.set_device(0)
    DEVICE = torch.device('cuda:0')
else:
    print("WARNING: CUDA not available, using CPU")
    DEVICE = torch.device('cpu')


#########################################################
# ACTIVATION FUNCTIONS
#########################################################

# NEW 
from helper.activation_func import set_activation
from helper.grokfast import gradfilter_ema, gradfilter_ma, gradfilter_ema_adjust



# OLD 
# relu = torch.nn.ReLU()
# tanh = torch.nn.Tanh()
# sigmoid = torch.nn.Sigmoid()
# silu = torch.nn.SiLU()
# linear = torch.nn.Identity()
# leaky_relu = torch.nn.LeakyReLU()

# @torch.jit.script
# def sigmoid_derivative(x):
#     return torch.exp(-x)/((1.+torch.exp(-x))**2)

# @torch.jit.script
# def relu_derivative(x):
#     return torch.heaviside(x, torch.tensor(0.))

# @torch.jit.script
# def tanh_derivative(x):
#     return 1-tanh(x)**2

# @torch.jit.script
# def silu_derivative(x):
#     return silu(x) + torch.sigmoid(x)*(1.0-silu(x))

# @torch.jit.script
# def leaky_relu_derivative(x):
#     return torch.where(x > 0, torch.tensor(1.), torch.tensor(0.01))

# def get_derivative(f):


#     print("f", f)

#     if f == "sigmoid":
#         f, f_prime = sigmoid, sigmoid_derivative
#     elif f == "relu":
#         f, f_prime = relu, relu_derivative
#     elif f == "tanh":
#         f, f_prime = tanh, tanh_derivative
#     elif f == "silu":
#         f, f_prime = silu, silu_derivative
#     elif f == "linear":
#         f, f_prime = 1, 1
#     elif f == "leaky_relu":
#         f, f_prime = leaky_relu, leaky_relu_derivative
#     else:
#         raise NotImplementedError(f"Derivative of {f} not implemented")

#     return f, f_prime

# Message passing layer
class PredictionMessagePassing(MessagePassing):
    def __init__(self, f):
        super().__init__(aggr='add', flow="target_to_source")  # Sum aggregation
        # super().__init__(aggr='add', flow="source_to_target")  # Sum aggregation
        # self.f = torch.tanh
        self.f = f

    def forward(self, x, edge_index, weight):
        # Start message passing
        return self.propagate(edge_index, x=x, weight=weight)
    
    def message(self, x_j, weight):
        # Apply activation to the source node's feature
        # return self.f(x_j) * weight.unsqueeze(-1) weight[:, None]
        # return self.f(x_j) * weight[:, None]
        return self.f(x_j) * weight.view(-1, 1)

        # return self.f(x_j) 
    
    def update(self, aggr_out):
        # No bias or additional transformation; return the aggregated messages directly
        return aggr_out
    


class GradientMessagePassing(MessagePassing):
    def __init__(self, dfdx):
        # super().__init__(aggr='add', flow="target_to_source")  # Reverse flow for backward pass
        super().__init__(aggr='add', flow="source_to_target")  # Reverse flow for backward pass

        self.dfdx = dfdx  # Activation derivative

    def forward(self, x, edge_index, error, weight):
        # Propagate messages backward using edge_index and node data
        return self.propagate(edge_index, x=x, error=error, weight=weight)

    def message(self, error_j, x_i, weight):
        # Compute the propagated error using the activation derivative and weight

        # error_j = error_j.view(-1, 1)  # Ensure shape consistency
        # x_i = x_i.view(-1, 1)  # Ensure shape consistency
        # weight = weight.view(-1, 1)  # Ensure shape consistency

        return self.dfdx(x_i) * error_j * weight

    def update(self, aggr_out, error):
        # Align with: e - dfdx(x) * torch.matmul(e, w.T)
        return error - aggr_out


    # def grad_x(self, x, e, w, b, train):
    #     lower = self.shape[0]
    #     upper = -self.shape[2] if train else sum(self.shape)
    #     return e[:,lower:upper] - self.dfdx(x[:,lower:upper]) * torch.matmul(e, w.T[lower:upper,:].T)




class UpdateRule:

    def __init__(self, update_type, batch_size, f, dfdx):
        self.model = update_type
        self.batch_size = batch_size
        self.f = f
        self.dfdx = dfdx

    def delta_w(self, errors, values):
        
        raise NotImplementedError("delta_w not implemented")
        # errors = errors.view(self.batch_size, self.num_vertices)
        # x = values.view(self.batch_size, self.num_vertices)
        # out = -torch.matmul(errors.T, self.f(x))
        # if self.mask is not None:
        #     out *= self.mask
        # return out 

class vanZwol_AMB(UpdateRule):
    def __init__(self, update_type, batch_size, f, dfdx):
        super().__init__(update_type, batch_size, f, dfdx)

    def pred(self, values, weights):

        mu = torch.matmul(self.f(values), weights.T)
        return mu 
    
    def grad_x(self, values, errors, weights):

        dEdx = errors - self.dfdx(values) * torch.matmul(errors, weights)

        return dEdx 



class MP_AMB(UpdateRule):

    def __init__(self, update_type, batch_size, f, dfdx, edge_index, batched_edge_index):
        super().__init__(update_type, batch_size, f, dfdx)
        """ 
        Message Passing for Predictive Coding:
        self.x | values  have shape [batch_size * num_vertices]
        
        """
        self.pred_mu_MP = PredictionMessagePassing(self.f)
        self.grad_x_MP = GradientMessagePassing(self.dfdx)
        self.edge_index = edge_index
        self.batched_edge_index = batched_edge_index


    def pred(self, x, w):
        # Gather 1D weights corresponding to connected edges
        # print("w shape", w.shape)
        # print("edge_index shape", self.edge_index.shape)


        weights_1d = w[self.edge_index[0], self.edge_index[1]]  # Extract relevant weights from W
        
        # print("weights_1d", weights_1d.shape)
        # print("selfbatch_size", self.batch_size)
        # # Expand edge weights for each graph

        batched_weights = weights_1d
        # num_features = 1
        # batched_weights = weights_1d.repeat(self.batch_size).
        batched_weights = weights_1d.repeat(self.batch_size).view(-1, 1)
                        
        # batched_weights = weights_1d.expand(self.batch_size, -1)

        # print("pred()")
        # print("x: ", x.shape)
        # print("edge_index: ", self.edge_index.shape)

        # print("batched_edge_index: ", self.batched_edge_index.shape)
        # print("batched_weights: ", batched_weights.shape)
        
        # self.pred_mu_MP(x, self.edge_index, batched_weights) batched_edge_index
        return self.pred_mu_MP(x, self.batched_edge_index, batched_weights) 

    def grad_x(self, x, e, w):

        # Gather 1D weights corresponding to connected edges
        weights_1d = w[self.edge_index[0], self.edge_index[1]]  # Extract relevant weights from W

        # # Expand edge weights for each graph
        batched_weights = weights_1d.repeat(self.batch_size).view(-1, 1)
        # batched_weights = weights_1d.expand(self.batch_size, -1)

        # dEdx = self.grad_x_MP(x, self.edge_index, e, batched_weights)
        dEdx = self.grad_x_MP(x, self.batched_edge_index, e, batched_weights)
    
        return dEdx



class PCgraph(torch.nn.Module): 

    def __init__(self, graph_type, task, activation, device, num_vertices, num_internal, adj, edge_index,
                    batch_size, learning_rates, T, incremental_learning, use_input_error,
                    update_rules, weight_init, 
                    early_stop=None, edge_type=None, use_learning_optimizer=None, use_grokfast=None, clamping=None,
                    wandb_logging=None, debug=False, **kwargs):
        super().__init__()

        self.device = device

        self.num_vertices = num_vertices
        self.num_internal = num_internal
        self.adj = torch.tensor(adj).to(self.device)
        # import torch_geometric 
        # self.edge_index = torch_geometric.utils.dense_to_sparse(adj)[0]

        self.edge_index = edge_index.to(self.device)  # PYG edge_index
        self.edge_index_single_graph = edge_index

        self.lr_x, self.lr_w = learning_rates 
        self.lr_values, self.lr_weights = learning_rates
        
        self.T_train, self.T_test = T 
        self.incremental = incremental_learning 
        # self.early_stop = early_stop

        self.epoch = 0 
        self.batch_size = batch_size  # Number of graphs in the batch

        self.f, self.dfdx = set_activation(activation)            
        # self.f, self.dfdx = get_derivative(activation)

        self.use_input_error = use_input_error
        self.trace = False 

        # self.w = nn.Parameter(torch.empty(num_vertices, num_vertices, device=self.device))
        # self.b = nn.Parameter(torch.empty(num_vertices, device=self.device))
        self.device = device 
        
        # assert torch.all(self.edge_index == edge_index)

        self.adj = torch.tensor(adj).to(DEVICE)
        self.mask = self.adj

        self.update_rules = update_rules 
        self.weight_init  = weight_init  
        self.wandb_logging = wandb_logging
        

        if self.update_rules in ["vectorized", "vanZwol_AMB"]:
            print("--------------Using vanZwol_AMB------------")
            self.reshape = True
            self.updates = vanZwol_AMB(update_type=self.update_rules, batch_size=self.batch_size, f=self.f, dfdx=self.dfdx)
        elif self.update_rules in ["MP", "MP_AMB"]:
            print("-------Using MP_AMB-------------")
            self.reshape = False

            # print("self.edge_index", self.edge_index.shape)
            self.batched_edge_index = torch.cat(
                [self.edge_index + i * self.num_vertices for i in range(self.batch_size)], dim=1
            )       
            # print("self.batched_edge_index", self.batched_edge_index.shape)
            # print("batch size", self.batch_size)
    
            self.updates = MP_AMB(update_type=self.update_rules, batch_size=self.batch_size, 
                                  edge_index=self.edge_index, 
                                  batched_edge_index=self.batched_edge_index,
                                  f=self.f, dfdx=self.dfdx)
        else:
            raise ValueError(f"Invalid update rule: {self.update_rules}")


        self.mode = "train"
        self.use_bias = False

        self._reset_grad()
        self._reset_params()

        self.use_learning_optimizer = use_learning_optimizer
        self.use_grokfast = use_grokfast    
        self.grads = None
        self.grokfast_type = "ema"  # "ema" or "ma"
        self.avg_grad = None


        self.optimizer_weights = torch.optim.Adam([self.w], lr=self.lr_w, betas=(0.9, 0.999), eps=1e-7, weight_decay=0)
        # self.optimizer_values = torch.optim.Adam([self.values_dummy], lr=self.lr_x, betas=(0.9, 0.999), eps=1e-7, weight_decay=0)
        self.optimizer_values = torch.optim.SGD([self.values_dummy], lr=self.lr_x)

        if self.use_learning_optimizer:

            # self.optimizer_weights = torch.optim.Adam([self.w], lr=self.lr_w, betas=(0.9, 0.999), eps=1e-7, weight_decay=0)
            # self.optimizer_values = torch.optim.Adam([self.values_dummy], lr=self.lr_x, betas=(0.9, 0.999), eps=1e-7, weight_decay=0)

            # self.optimizer_weights = torch.optim.AdamW([self.weights], lr=self.lr_weights, weight_decay=weight_decay)      
            # # self.optimizer_weights = torch.optim.Adam([self.weights], lr=self.lr_weights, weight_decay=weight_decay)      
            # self.optimizer_values = torch.optim.SGD([self.values_dummy], lr=self.lr_values, momentum=0, weight_decay=weight_decay, nesterov=False) # nestrov only for momentum > 0

            self.w.grad = torch.zeros_like(self.w)
            self.values_dummy.grad = torch.zeros_like(self.values_dummy)

            # self.lr_scheduler = False        
            # if self.lr_scheduler:
            #     self.scheduler_weights = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_weights, mode='min', patience=10, factor=0.1)


        # self.MP = PredictiveCodingLayer(f=self.structure.f, 
        #                                 f_prime=self.structure.dfdx)

        # update_rule = "vanZwol_AMB"
        # update_rule = "MP_AMB"
     

        self.test_supervised = self.test_iterative

    # @property
    # def hparams(self):
    #     return {"lr_x": self.lr_x, "T_train": self.T_train, "T_test": self.T_test, "incremental": self.incremental,
    #              "min_delta": self.min_delta,"early_stop": self.early_stop, "use_input_error": self.use_input_error, "node_init_std": self.node_init_std}

    # @property
    # def params(self):
    #     return {"w": self.w, "b": self.b, "use_bias": self.use_bias}
    
    # @property
    # def grads(self):
    #     return {"w": self.dw, "b": self.db}

    # def w_to_dense(self, w_sparse):
    #     # Convert 1D sparse weights back to dense (N, N) matrix
    #     adj_dense = torch.zeros((self.num_vertices, self.num_vertices), device=DEVICE)
    #     adj_dense[self.edge_index[0], self.edge_index[1]] = w_sparse
    #     return adj_dense

   
    # def w_to_sparse(self, w_sparse):
    #     # Convert sparse weights to dense (on GPU) before finding non-zero indices
    #     w_dense = w_sparse.to_dense().to(DEVICE)

    #     # Find non-zero indices
    #     # edge_index_sparse = torch.nonzero(w_dense, as_tuple=False).t()

    #     # Retrieve the corresponding weights
    #     edge_weights_sparse = w_dense[self.edge_index[0], self.edge_index[1]]

    #     return edge_weights_sparse.to(DEVICE)


        
    def _reset_params(self):

        #### WEIGHT INITIALIZATION ####
        self.w = torch.nn.Parameter(torch.empty(self.num_vertices, self.num_vertices, device=DEVICE))
        # self.weights = torch.nn.Parameter(torch.zeros(self.edge_index_single_graph.size(1), device=self.device, requires_grad=True))

        # Weight initialization
        init_type, m, std_val = self.weight_init.split()

        # Convert parsed values to appropriate types
        m = float(m)
        std_val = float(std_val)


        # init_type = "type" "mean" "std"
        # normal 0.001 0.02
        # fixed  0.001 0   
        # fixed  0.001 0.005
        # if not params, set to default values (mean = 0.0, std = 0.02)

        if init_type == "normal":
            # mean = float(params[0]) if params else 0.0
            # std_val = 0.02 # 2% 
            # std = max(0.01, abs(mean) * std_val)  # Set std to std_val% of mean or 0.01 minimum
            # nn.init.normal_(self.weights, mean=mean, std=std)

            # Because of: Predictive Coding Networks and Inference Learning: Tutorial and Survey

            nn.init.normal_(self.w, mean=m, std=std_val)   
            
        elif init_type == "uniform" or init_type == "nn.linear":
            import math
            k = 1 / math.sqrt(self.num_vertices)
            nn.init.uniform_(self.w, -k, k)
        elif init_type == "fixed":
            value = float(m) if m else 0.1
            self.w.data.fill_(value)


            # noise_std = 0.005  # Standard deviation of the noise
            # noise_std = 0  # Standard deviation of the noise

            # Fill with fixed value
            # self.weights.data.fill_(value)

            # Add small random noise
            noise = torch.randn_like(self.w) * std_val
            self.w.data.add_(noise)

        

        # Perform the operation and reassign self.w as a Parameter
        with torch.no_grad():
            self.w.copy_(self.adj * self.w)

        ######## VALUE OPTIMIZATION ########
        if self.reshape:
            self.values_dummy = torch.nn.Parameter(torch.zeros(self.batch_size, self.num_vertices, device=self.device), requires_grad=True) # requires_grad=False)                
        else:
            self.values_dummy = torch.nn.Parameter(torch.zeros(self.batch_size * self.num_vertices, 1, device=self.device), requires_grad=True)

        ## -------------------------------------------------

        # import scipy sparse coo
        from scipy.sparse import coo_matrix

        # matrix of N, N
        # self.w_sparse = coo_matrix((self.num_vertices, self.num_vertices))
        # self.weight_init(self.w_sparse)

        # Initialize edge weights as a torch.nn.Parameter
        # edge_weights = torch.nn.Parameter(torch.empty(edge_index.size(1)))
        # num_edges = self.edge_index.size(1)
        # weights_1d = torch.empty(num_edges, device=DEVICE)
        # nn.init.normal_(weights_1d, mean=0.01, std=0.05)

        # N = self.num_vertices
                
        # # Create the sparse matrix using PyTorch sparse functionality
        # self.w = torch.sparse_coo_tensor(self.edge_index, weights_1d, size=(N, N), device=DEVICE)
        # # Convert to a dense (N, N) matrix if needed
        # dense_matrix = self.w.to_dense()

        # print("Initialized 1D Weights:\n", weights_1d.shape)
        # print("\nDense (N, N) Matrix:\n", dense_matrix.shape)

        # # Extract weights corresponding to the original edge index
        # reconstructed_weights_1d = dense_matrix[self.edge_index[0], self.edge_index[1]]

        # print("Reconstructed 1D Weights:", reconstructed_weights_1d.shape)

        # # import matplotlib.pyplot as plt

        # # save png the weights 
        # plt.imshow(dense_matrix.cpu().numpy())
        # # save
        # plt.savefig("weights.png")

        # # also save the adj_dense
        # plt.imshow(self.adj.cpu().numpy())
        # plt.savefig("adj.png")

        # # close the plots 
        # plt.close()

        ## -------------------------------------------------
        # self.b = torch.empty( self.num_vertices, device=DEVICE)
        # if self.structure.use_bias:
        #     self.bias_init(self.b)

    def get_dense_weight(self):
        w = torch.tensor(self.w_sparse.toarray(), device=DEVICE)
        # w = self.w_sparse.toarray()
        assert w.shape == (self.num_vertices, self.num_vertices)
        return w

    def _reset_grad(self):
        self.dw, self.db = None, None

    def reset_nodes(self, batch_size=1):
        if self.reshape:
            self.errors = torch.empty(batch_size, self.num_vertices, device=DEVICE)
            self.values = torch.zeros(batch_size, self.num_vertices, device=DEVICE)
        else:
            num_features = 1
            self.errors = torch.empty(batch_size * self.num_vertices, num_features, device=DEVICE)
            self.values = torch.zeros(batch_size * self.num_vertices, num_features, device=DEVICE)

    # def clamp_input(self, inp):
    #     di = self.structure.shape[0]
    #     self.x[:,:di] = inp.clone()

    # def clamp_target(self, target):
    #     do = self.structure.shape[2]
    #     self.x[:,-do:] = target.clone()
        
    # def init_hidden_random(self):
    #     di = self.structure.shape[0]
    #     do = self.structure.shape[2]
    #     self.x[:,di:-do] = torch.normal(0.5, self.node_init_std,size=(self.structure.shape[1],), device=DEVICE)

    # def init_hidden_feedforward(self):
    #     self.forward(self.num_verticesum_layers-1)

    # def init_output(self):
    #     do = self.structure.shape[2]
    #     self.x[:,-do:] = torch.normal(0.5, self.node_init_std, size=(do,), device=DEVICE)

    # def forward(self, no_layers):
    #     temp = self.x.clone()
    #     for l in range(no_layers):
    #         lower = sum(self.structure.layers[:l+1])
    #         upper = sum(self.structure.layers[:l+2])
    #         temp[:,lower:upper] = self.structure.pred(x=temp, w=self.w, b=self.b )[:,lower:upper]
    #     self.x = temp

    def get_dw(self):
        
        self.errors = self.errors.view(self.batch_size, self.num_vertices)

        # self.x = self.values.view(self.batch_size, self.num_vertices)
        self.x = self.values_dummy.data.clone().view(self.batch_size, self.num_vertices)
        # self.x = 

        out = -torch.matmul(self.errors.T, self.f(self.x))

        # x_dum = self.values_dummy.view(self.batch_size, self.num_vertices)


        # assert allclose self.x and x_dum
        # assert torch.allclose(self.x, x_dum, atol=1e-6)

        # assert allclose self.errors and errors_dum
        # assert torch.allclose(self.errors, self.errors_dum, atol=1e-6)

        # out_dum = -torch.matmul(self.errors_dum.T, self.f(x_dum))

        if self.mask is not None:
            out *= self.mask


        # grad clipping 
        # delta_w = torch.clamp(delta_w, min=-1, max=1)

        # self.dw = out 
        self.dw = out
        # self.dw = out_dum.float()


        # if self.structure.use_bias:
        #     self.db = self.structure.grad_b(x=self.x, e=self.e, w=self.w, b=self.b)
            
    def set_optimizer(self, optimizer):

        # self.optimizer = optimizer

        # self.optimizer_x = torch.optim.Adam(params, lr=lr_w, betas=(0.9, 0.999), eps=1e-7, weight_decay=weight_decay)
        pass


    def train_(self, epoch=0):
        self.mode = "train"
        self.dw = None 
        self.trace = False
        self.epoch = epoch
        self.do_gen = True 

        # self.update_mask = self.update_mask_train

    def test_(self):
        self.mode = "test"
        # self.trace = True 

        print(self.mode)

        # self.update_mask = self.update_mask_test

    def init_modes(self, graph_type, graph):
        
        self.graph_type = graph_type
        
        # take first item 
        # batch_example = batch_example[0]
        # sensory_indices_single_graph, internal_indices_single_graph, supervised_labels_single_graph = batch_example.sensory_indices, batch_example.internal_indices,  batch_example.supervision_indices
        sensory_indices_single_graph, internal_indices_single_graph, supervised_labels_single_graph = graph.sensory_indices, graph.internal_indices,  graph.supervision_indices

        self.base_sensory_indices = list(sensory_indices_single_graph)
        self.base_internal_indices = list(internal_indices_single_graph)
        self.base_supervised_labels = list(supervised_labels_single_graph) if supervised_labels_single_graph else []

        # Correcting the initialization of batched indices

        # Ensuring the base indices are flattened lists of integers
        base_sensory_indices = [int(idx) for sublist in self.base_sensory_indices for idx in sublist] if isinstance(self.base_sensory_indices[0], list) else self.base_sensory_indices
        base_internal_indices = [int(idx) for sublist in self.base_internal_indices for idx in sublist] if isinstance(self.base_internal_indices[0], list) else self.base_internal_indices
        base_supervised_labels = [int(idx) for sublist in self.base_supervised_labels for idx in sublist] if isinstance(self.base_supervised_labels[0], list) else self.base_supervised_labels

        # Create batched indices by iterating over batch size and offsetting by graph index
        self.sensory_indices_batch = [
            index + i * self.num_vertices for i in range(self.batch_size) for index in base_sensory_indices
        ]
        self.internal_indices_batch = [
            index + i * self.num_vertices for i in range(self.batch_size) for index in base_internal_indices
        ]
        self.supervised_labels_batch = [
            index + i * self.num_vertices for i in range(self.batch_size) for index in base_supervised_labels
        ] if base_supervised_labels else []

        # Convert to tensors for masking purposes during updates
        # self.sensory_indices_batch = torch.tensor(sorted(sensory_indices_batch), device=self.device)
        # self.internal_indices_batch = torch.tensor(sorted(internal_indices_batch), device=self.device)
        # self.supervised_labels_batch = torch.tensor(sorted(supervised_labels_batch), device=self.device) if supervised_labels_batch else None

        print("Sensory indices batch:", len(self.sensory_indices_batch), 784, self.batch_size)
        print("Internal indices batch:", len(self.internal_indices_batch), self.num_internal, self.batch_size)
        if self.supervised_labels_batch is not None:
            print("Supervised labels batch:", len(self.supervised_labels_batch), 10, self.batch_size)

        # Update only internal nodes during training
        # self.internal_mask_train = torch.tensor(internal_indices_batch, device=self.device)
        self.internal_mask_train = torch.tensor(self.internal_indices_batch, device=self.device)

        # update_mask = self.internal_mask_train if train else self.update_mask_test
        self.update_mask_test = torch.tensor(sorted(self.internal_indices_batch + self.supervised_labels_batch), device=self.device)

        ########################################## 
        batch_size = self.batch_size
        # num_nodes = self.structure.N  # Nodes per graph
        num_nodes = self.num_vertices  # Nodes per graph

        # Offset edge_index for batched graphs
        self.batched_edge_index = torch.cat(
            [self.edge_index + i * num_nodes for i in range(batch_size)], dim=1
        )  # Concatenate and offset indices


    def set_task(self, task):
        if type(task) == list:
            task = task[0]
        self.task = task 

        if task == "classification":
            # Update both the internal and supervised nodes during classification
            self.update_mask_test = torch.tensor(sorted(self.internal_indices_batch + self.supervised_labels_batch), device=self.device)
            # print("Classification task")
            # print("self.update_mask_test", self.update_mask_test.shape)

        elif task in ["generation", "reconstruction", "denoising"]:
            # Update both the internal and sensory nodes during these tasks
            self.update_mask_test = torch.tensor(sorted(self.internal_indices_batch + self.sensory_indices_batch), device=self.device)
            # print("Generation task")
            # print("self.update_mask_test", self.update_mask_test.shape)

        elif task == "occlusion":
            # Only update internal nodes + occluded sensory nodes
            occluded_indices = list(range(392, 784))  # For example
            self.update_mask_test = torch.tensor(
                sorted(self.internal_indices_batch + [
                    index + i * self.num_vertices for i in range(self.batch_size)
                    for index in occluded_indices
                ]), device=self.device
            )

        else:
            raise ValueError(f"Invalid task: {task}")
        
        # # Update internal + supervision nodes during testing
        # # self.update_mask_test = torch.tensor(sorted(internal_indices_batch + supervised_labels_batch), device=self.device)
        # self.update_mask_test = torch.tensor(sorted(self.internal_indices_batch + self.supervised_labels_batch), device=self.device)





    # def get_filtered_batched_w_and_edge_index(self):
        
    #     # update_mask = self.internal_mask_train if train else self.update_mask_test  # Select appropriate mask

    #     self.w = self.adj * self.w

    #     batch_size = self.x.size(0)
    #     num_nodes = self.num_vertices

    #     # Offset edge_index for batched graphs
    #     batched_edge_index = torch.cat(
    #         [self.edge_index + i * num_nodes for i in range(batch_size)], dim=1
    #     )

    #     weights_1d = self.w[self.edge_index[0], self.edge_index[1]]
    #     batched_weights = weights_1d.repeat(batch_size)

    #     # **Filter edge index using internal_indices_batch**
    #     internal_nodes_tensor = self.internal_indices_batch  # Already defined in init_modes()

    #     # Mask: Keep only edges where both source and target are in `internal_indices_batch`
    #     mask = torch.isin(batched_edge_index[0], internal_nodes_tensor) & torch.isin(batched_edge_index[1], internal_nodes_tensor)

    #     # Apply mask to edge index and weights
    #     filtered_edge_index = batched_edge_index[:, mask]
    #     filtered_weights = batched_weights[mask]
    #     pass

    # def ensure_grad(self, param):
    #     if param.grad is None:
    #         param.grad = torch.zeros_like(param)
    #     else:
    #         param.grad.zero_()

    # Then use it:





    def gradient_descent_update_w(self, grad_type, parameter, delta, learning_rate, nodes_or_edge2_update, 
                                optimizer=None, use_optimizer=False):
        
        # self.use_grokfast = False 
        # self.delta_w_selection = "all"
        self.delta_w_selection = "internal_only"

        if self.use_grokfast:
            param_type = "weights"
            
            # assert 
                # Collect parameters explicitly
            params = {"weights": self.weights}

            if self.grokfast_type == 'ema':
                
                # python main_mnist.py --label test --alpha 0.8 --lamb 0.1 --weight_decay 2.0
                final_grad, self.avg_grad = gradfilter_ema_adjust(delta, self.avg_grad, alpha=0.8, lamb=0.1)
                delta = final_grad  # Replace gradient with modified version

                # # Call the gradfilter_ema function
                # self.grads = gradfilter_ema(grads=self.grads, params=params, alpha=0.8, lamb=0.1)


                # self.grads[grad_type] = gradfilter_ema(self, grads=self.grads[grad_type], alpha=0.8, lamb=0.1)
            elif self.grokfast_type == 'ma':
                self.grads[grad_type] = gradfilter_ma(self, grads=self.grads[grad_type], window_size=100, lamb=5.0, filter_type='mean')

        # ---------------------------------------------------------------------------- 

        optimizer.zero_grad()

        parameter.grad = delta
        optimizer.step()

        return True 
    
        if use_optimizer and optimizer:
            # Clear 
            optimizer.zero_grad()
            if parameter.grad is None:
                parameter.grad = torch.zeros_like(parameter)
            else:
                parameter.grad.zero_()  # Reset the gradients to zero


            # Apply the delta to the parameter
            if self.delta_w_selection == "all":
                parameter.grad = -delta.detach().clone()  # Apply full delta to the parameter
            else:
                
                if self.update_rules == "vanZwol_AMB":

                    # delta[784:-10, 784:-10] = 0
                    # parameter.grad.data[784:-10, 784:-10] = 0
                    # delta = delta * self.adj
                    # parameter.grad = -delta  # Update only specific nodes
                    # parameter.grad = delta  # Update only specific nodes

                    # self.w.grad = self.dw
                    parameter.grad = delta  # Update only specific nodes

                    # self.optimizer_weights.step()

                else:
                    parameter.grad[nodes_or_edge2_update] = -delta[nodes_or_edge2_update].detach().clone()  # Update only specific nodes
        
            # perform optimizer weight update step
            optimizer.step()

     

        else:
            # Manually update the parameter using gradient descent
            if self.delta_w_selection == "all":
            # if nodes_or_edge2_update == "all": ### BUG ### 
                parameter.data += learning_rate * delta
            else:
                
                # print shapes
                if grad_type == "weights" and self.update_rules == "Van_Zwol":

                    parameter.data += learning_rate * delta
                else:
                    
                    # parameter.data[nodes_or_edge2_update] += learning_rate * delta[nodes_or_edge2_update]
                    parameter.data[nodes_or_edge2_update] -= learning_rate * delta[nodes_or_edge2_update]
                
    def get_trace(self, trace=False):
           
        if self.trace or trace:
            if self.reshape:
                # OLD:
                # x_batch = self.values[:, 0:784].cpu().detach().numpy()
                
                # ✅ NEW: Use actual current batch size
                current_batch_size = self.values.shape[0]

                x_batch = self.values[:, 0:784].cpu().detach().numpy()

                # ✅ Reshape to [current_batch_size, 28, 28]
                x_batch = x_batch.reshape(current_batch_size, 28, 28)

                self.trace_data.append(x_batch)
                
            else:
                # Same for non-reshape cases
                current_batch_size = self.values.shape[0] // self.num_vertices
                # print("current_batch_size", current_batch_size)
                # print("self.num_vertices", self.num_vertices)
                # print("self.values", self.values.shape)

                x_batch = self.values.view(current_batch_size, self.num_vertices)
                x_batch = x_batch[:, 0:784].cpu().detach().numpy()

                x_batch = x_batch.reshape(current_batch_size, 28, 28)

                self.trace_data.append(x_batch)



    def gradient_descent_update_values(self, grad_type, parameter, delta, learning_rate, nodes_or_edge2_update,
                                       optimizer, use_optimizer, train):

     
        dEdx = delta 
        
        # if self.use_learning_optimizer:
                
        #     self.ensure_grad(self.values_dummy)

        #     self.optimizer_values.zero_grad()

        #     if self.values_dummy.grad is None:
        #         self.values_dummy.grad = torch.zeros_like(self.values_dummy)
        #     else:
        #         self.values_dummy.grad.zero_()

        #     if self.reshape:
        #         self.values_dummy.grad[:, di:upper] = delta[:, di:upper]
        #     else:
        #         self.values_dummy.grad[nodes_or_edge2_update] = delta[nodes_or_edge2_update]

        #     self.optimizer_values.step()

        # else:

        if self.reshape:
            # if self.task == "classification":
            #     di = 784
            #     upper = -10 if train else self.num_vertices
            # if self.task in ["generation", "reconstruction", "denoising", "occlusion"]:
            # if train:
            #     di = 784
            #     upper = -10
            # else:
            #     di = 0
            #     upper = 784


            dEdx = delta.flatten()[nodes_or_edge2_update]
            # print("dEdx", dEdx.shape)

            # dEdx = self.structure.grad_x(self.x, self.e, self.w, self.b,train=train) # only hidden nodes
            # self.values[:,di:upper] -= self.lr_x*dEdx 
            tmp = self.values_dummy.data.flatten()
            tmp[nodes_or_edge2_update] -= self.lr_x*dEdx
            self.values_dummy.data = tmp.reshape(self.values_dummy.shape) 
        
            
            # if train or self.train:
            #     di = 784
            #     upper = -10 
            # else:
            #     if self.task == "classification":
            #         di = 784
            #         upper = -10
                
            #     if self.task in ["generation", "reconstruction", "denoising", "occlusion"]:
            #         di = 0
            #         upper = 784

            # dEdx = delta[:, di:upper]
            # # print("dEdx", dEdx.shape)

            # # dEdx = self.structure.grad_x(self.x, self.e, self.w, self.b,train=train) # only hidden nodes
            # # self.values[:,di:upper] -= self.lr_x*dEdx 
            # self.values_dummy.data[:,di:upper] -= self.lr_x*dEdx 
    
        else:
            dEdx = dEdx[nodes_or_edge2_update]

            # clipped_dEdx = torch.clamp(dEdx, -1, 1)
            clipped_dEdx = dEdx
            # self.values[nodes_or_edge2_update] -= self.lr_x * clipped_dEdx
            self.values_dummy.data[nodes_or_edge2_update] -= self.lr_x * clipped_dEdx
    


        
        # if self.task == "classification":
        #     di = 784
        #     upper = -10 if train else self.num_vertices
        # if self.task in ["generation", "reconstruction", "denoising", "occlusion"]:
        #     if train:
        #         di = 784
        #         upper = -10
        #     else:
        #         di = 0
        #         upper = 784
        
        # update_mask = self.internal_mask_train if train else self.update_mask_test

        # # self.use_learning_optimizer = True  
        # self.use_learning_optimizer = False 
        # # delta = -delta
        
        # if self.use_learning_optimizer and optimizer:

        #     self.optimizer_values.zero_grad()

        #     # Ensure parameter has grad attribute (always true for nn.Parameter)
        #     if self.values_dummy.grad is None:
        #         self.values_dummy.grad = torch.zeros_like(self.values_dummy)

        #     self.values_dummy.grad.zero_()  # Clear previous grad values

        #     if self.reshape:
        #         # Update only internal nodes: [di:upper]
        #         self.values_dummy.grad[:, di:upper] = delta[:, di:upper]
        #     else:
        #         # Update nodes or edges specified by nodes_or_edge2_update
        #         self.values_dummy.grad[update_mask] = delta[update_mask]

        #     self.optimizer_values.step()

        # else:
        #     # Manually update the parameter using gradient descent
        #     if self.reshape:
        #         # Internal node updates only
        #         dEdx_internal = delta[:, di:upper]

        #         # self.values[:, di:upper] -= self.lr_x * dEdx_internal
        #         self.values_dummy.data[:, di:upper] -= self.lr_x * dEdx_internal  # force sync
        #     else:
        #         dEdx_internal = delta[self.internal_mask_train]

        #         # self.values[self.internal_mask_train] -= self.lr_x * dEdx_internal
        #         self.values_dummy.data[self.internal_mask_train] -= self.lr_x * dEdx_internal



    def update_xs(self, train=True, trace=False):

        # if self.early_stop:
        #     early_stopper = EarlyStopper(patience=0, min_delta=self.min_delta)

        T = self.T_train if train else self.T_test

        update_mask = self.internal_mask_train if train else self.update_mask_test

        # di = self.structure.shape[0] # 784
        di = 784
        upper = -10 if train else self.num_vertices
        
        self.lst_sensor_error = []
        self.lst_internal_error = []

        # At the start of update_xs()
        self.values_dummy.data = self.values.data.clone().to(self.device).contiguous()
        # print("self.values_dummy", self.values_dummy.shape)

        for t in range(T): 
         
            self.t = t  

            self.get_trace(trace=trace)

            # self.w = self.adj * self.w 
            # Perform the operation and reassign self.w as a Parameter
            with torch.no_grad():
                self.w.copy_(self.adj * self.w)

            # self.mu = self.updates.pred(self.values.to(self.device), self.w.to(self.device))
            self.mu = self.updates.pred(self.values_dummy, self.w.to(self.device))

            # self.errors = self.values - self.mu
            # self.errors_dum = self.values_dummy - self.mu_dum
            self.errors = self.values_dummy - self.mu
        

            if self.reshape:
               
                # BEFORE using use_input_error
                total_sensor_error = torch.mean(self.errors[:,:784]**2).item()
                total_internal_error = torch.mean(self.errors[:, 784:-10]**2).item()    

                self.lst_sensor_error.append(total_sensor_error)
                self.lst_internal_error.append(total_internal_error)
    
                if not self.use_input_error:
                    if self.task == "classification":
                        self.errors[:,:784] = 0 
                    elif self.task in ["generation", "reconstruction", "denoising", "occlusion"]:
                        self.errors[:,-10:] = 0
                        # self.errors[:,di:upper] = 0

            else:
                
                total_sensor_error   = (self.errors[self.sensory_indices_batch]**2).mean().cpu().numpy()
                total_internal_error = (self.errors[self.internal_indices_batch]**2).mean().cpu().numpy()
                # self.history.append(total_internal_error)

                self.lst_sensor_error.append(total_sensor_error)
                self.lst_internal_error.append(total_internal_error)

                if not self.use_input_error:
                    if self.task == "classification":
                        self.errors[self.sensory_indices_batch] = 0
                    elif self.task in ["generation", "reconstruction", "denoising", "occlusion"]:
                        self.errors[self.supervised_labels_batch] = 0
                        # self.errors[self.sensory_indices_batch] = 0

             
            if self.wandb_logging:
                if train:
                    wandb.log({
                                "epoch": self.epoch,
                                "Training/internal_energy_mean": total_internal_error,
                                "Training/sensory_energy_mean": total_sensor_error,
                                "step": self.t
                                })
                else:
                    wandb.log({
                                "epoch": self.epoch,
                                "Validation/internal_energy_mean": total_internal_error,
                                "Validation/sensory_energy_mean": total_sensor_error,
                                "step": self.t

                                })
                    
                         
            # x = self.x.T.contiguous().view(-1, 1).to(DEVICE)  # Shape: [num_nodes, batch_size]
            # error = self.e.T.contiguous().view(-1, 1).to(DEVICE)  # Shape: [num_nodes, batch_size]
            
            # print("-------------------")
            # print("self.values", self.values.shape)
            # print("self.errors", self.errors.shape)

            # dEdx = self.updates.grad_x(self.values, self.errors, self.w)
            dEdx = self.updates.grad_x(self.values_dummy, self.errors, self.w)
         
            # dEdx = self.updates.grad_x(self.values_dummy, self.errors_dum, self.w)
            # dEdx_dum = self.updates.grad_x(self.values_dummy, self.errors_dum, self.w)
            # print("dEdx_dum", dEdx_dum.shape)

            self.gradient_descent_update_values(
                grad_type="values",
                parameter=self.values_dummy,  
                delta=dEdx,
                learning_rate=self.lr_values,

                # nodes_or_edge2_update=self.nodes_2_update,  # Mandatory
                nodes_or_edge2_update=update_mask,  # Mandatory

                optimizer=self.optimizer_values,
                # optimizer=self.optimizer_values if self.use_learning_optimizer else None,
                use_optimizer=self.use_learning_optimizer,
                train=train)

            ## Copy updated dummy values back into self.values; also needed for testing()
            if self.reshape:
                self.values[:, :] = self.values_dummy.data
            else:
                self.values[:] = self.values_dummy.data

            # self.get_dw()

            if train and self.incremental and self.dw is not None:
            # if train and self.incremental:

                self.update_w()
                
                # self.optimizer.step(self.params, self.grads, batch_size=self.batch_size)
            # if self.early_stop:
            #     if early_stopper.early_stop( self.get_energy() ):
            #         break            

           



    def log_node_connectivity_distribution_to_wandb(self, direction="both"):
        """
        Logs a histogram of node degree counts to Weights & Biases (wandb).
        For each degree (how many connections), shows how many nodes have it.
        
        direction: 'in', 'out', or 'both' degrees (default is total degree).
        """
        
        # Get edge indices
        src, dst = self.edge_index
        num_nodes = int(torch.max(self.edge_index)) + 1

        # Calculate degrees
        in_degree = torch.bincount(dst, minlength=num_nodes)
        out_degree = torch.bincount(src, minlength=num_nodes)

        # Total degree (in + out)
        if direction == "in":
            degree_per_node = in_degree
            label = "In-degree"
        elif direction == "out":
            degree_per_node = out_degree
            label = "Out-degree"
        elif direction == "both":
            degree_per_node = in_degree + out_degree
            label = "Total-degree"
        else:
            raise ValueError("Direction must be 'in', 'out', or 'both'")

        degree_distribution = torch.bincount(degree_per_node.cpu())

        # Convert to WandB table (swap Degree and Num_Nodes in the table to switch axes)
        data = []
        for degree, count in enumerate(degree_distribution):
            data.append([count.item(), degree])  # Swapped here!

        # Create a wandb Table for horizontal bar chart
        table = wandb.Table(data=data, columns=["Num_Nodes", "Degree"])

        # Log horizontal bar chart (x = num_nodes, y = degree)
        wandb.log({
            f"Weights/{label}_Distribution_Horizontal": wandb.plot.bar(
                table,
                "Num_Nodes",  # X-axis: how many nodes
                "Degree",     # Y-axis: degree level
                title=f"{label} Distribution (Nodes per Degree)"
            )
        })
        
        # Optional print
        print(f"Logged {label} Distribution to wandb!")
        self.plot_and_log_degree_distribution()




    def plot_and_log_degree_distribution(self, save_path="trained_models/degree_distribution.png", wandb_folder="Graph/Structure"):
        """
        Generates degree distribution plots:
            - Regular degree distribution
            - Log-log scale degree distribution

        Args:
            edge_index (torch.Tensor): shape [2, num_edges], PyG style.
            save_path (str): where to save the plot.
            wandb_logging (bool): whether to log to wandb.
            wandb_folder (str): wandb folder/tag name.
        """

        # 1. Compute node degrees
        src, dst = self.edge_index
        num_nodes = int(torch.max(self.edge_index)) + 1

        in_degree = torch.bincount(dst, minlength=num_nodes)
        out_degree = torch.bincount(src, minlength=num_nodes)

        total_degree = (in_degree + out_degree).cpu().numpy()

        # 2. Compute histogram: counts of how many nodes have degree = k
        degree_values, node_counts = np.unique(total_degree, return_counts=True)

        # Avoid log(0) issues
        degree_values_safe = np.where(degree_values == 0, 1, degree_values)
        node_counts_safe = np.where(node_counts == 0, 1, node_counts)

        # 3. Plotting
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # ----- Left: Regular plot -----
        axes[0].plot(degree_values, node_counts, marker='o')
        axes[0].set_xlabel("Degree (j)")
        axes[0].set_ylabel("Number of nodes (m)")
        axes[0].set_title("Degree Distribution")

        # ----- Right: Log-Log plot -----
        axes[1].plot(np.log10(degree_values_safe), np.log10(node_counts_safe), marker='o')
        axes[1].set_xlabel("Degree (j) [log scale]")
        axes[1].set_ylabel("Number of nodes (m) [log scale]")
        axes[1].set_title("Log-Log Degree Distribution")

        plt.tight_layout()

        # 4. Save the plot locally
        plt.savefig(save_path)
        print(f"Degree distribution plot saved to: {save_path}")

        # 5. Log to wandb (optional)
        wandb.log({
            f"Weights/Degree_Distribution": wandb.Image(fig, caption="Degree Distribution + Log-Log")
        })
        print(f"Logged degree distribution to wandb: Degree_Distribution")

        plt.close(fig)






    def log_edge_weight_distribution_to_wandb(self, step=None):
        """
        Logs a histogram of edge weights to wandb.
        """
        # Get weights of existing edges only
        edge_weights = self.w[self.edge_index[0], self.edge_index[1]].detach().cpu().numpy()

        wandb.log({
            f"Weights/Edge_Weight_Distribution": wandb.Histogram(edge_weights)
        }, step=step)

        print(f"Edge Weights - Min: {edge_weights.min()}, Max: {edge_weights.max()}, Mean: {edge_weights.mean():.4f}")


    def unpack_features(self, batch, reshape=False):
        """Unpack values, errors, and predictions from the batched graph."""
        # values, errors, predictions = batch[:, 0, :].to(self.device), batch[:, 1, :].to(self.device),  None
        # # print("unpacked featreus")
        # # print(values.shape)

        values = batch 
        if reshape or self.reshape:
            values = values.view(self.batch_size, self.num_vertices)
            # error  = 
            # preds    =

        return values, None, None

    def update_w(self):

        # print("mean w 1 ", self.w.mean())
        # print("update_w")
        
        # self.optimizer_weights.zero_grad()
        self.w.grad = self.dw
        
        self.optimizer_weights.step()

        # self.edges_2_update = None 
        # self.gradient_descent_update_w(
        #     grad_type="weights",
        #     parameter=self.w,
        #     delta=self.dw,
        #     learning_rate=self.lr_weights,
      
        #     # error= self.errors if self.optimizer_values else None,
        #     nodes_or_edge2_update=self.edges_2_update,
        #     # optimizer=self.optimizer_weights if self.use_learning_optimizer else None,
        #     optimizer=self.optimizer_weights,
        #     use_optimizer=self.use_learning_optimizer, 
        # )

        # print("mean w 2", self.w.mean())



    def train_supervised(self, data):
        # edge_index = data.edge_index.to(self.device)
        self.history = []

        # self.optimizer_weights.zero_grad()
        # self.data_ptr = data.ptr
        # self.batch_size = data.x.shape[0] // self.num_vertices
        self.batch_size = data.shape[0]


        self.reset_nodes(batch_size=data.shape[0])        
                
        # Directly set graph_data into self.x
        if self.reshape:
            self.values[:, :] = data.clone()
        else:
            self.values = data.view(self.batch_size * self.num_vertices, 1)

        # self.values, _ , _ = self.unpack_features(data.x, reshape=False)
        
        self.update_xs(train=True)
        self.get_dw()

        if not self.incremental:
            self.update_w()

          
        # print("w mean", self.w.mean())

        return self.history


    def test_classifications(self, data, remove_label=True):
        # self.reset_nodes(batch_size=data.shape[0])

        # Set the graph data (flattened image + internal zeros + one-hot label)

        if self.reshape:
            self.values[:, :] = data.clone()

            # Zero out the one-hot vector (last output_size elements)
            self.values[:, -10:] = 0
        else:
            data = data.view(self.batch_size * self.num_vertices, 1)
            self.values = data
            self.values[self.supervised_labels_batch] = 0

        self.update_xs(train=False)
            
        logits = self.values.view(self.batch_size, self.num_vertices)   # batch,10
        # print("logits ", logits)
        logits = logits[:, -10:]

        y_pred = torch.argmax(logits, axis=1).squeeze()
        # print("logits ", logits.shape)
        # print("y_pred ", y_pred.shape)
        return y_pred
    


    # def test_generative(self, data, labels, remove_label=True, save_imgs=False, wandb_logging=False):
        
    #     print("self.reshape", self.reshape)
    #     # remove one_hot
    #     # if remove_label:
    #     #     for i in range(len(data)):
    #     #         sub_graph = data[i]  # Access the subgraph

    #     #         # set sensory indices to zero / random noise
    #     #         sub_graph.x[sub_graph.sensory_indices, 0] = torch.zeros_like(sub_graph.x[sub_graph.sensory_indices, 0])  # Check all feature dimensions
    #     #         # random noise
    #     #         # sub_graph.x[sub_graph.sensory_indices, 0] = torch.randn_like(sub_graph.x[sub_graph.sensory_indices, 0])  # Check all feature dimensions
    #     #         # sub_graph.x[sub_graph.sensory_indices, 0] = torch.clamp(torch.randn_like(sub_graph.x[sub_graph.sensory_indices, 0]), min=0, max=1)

    #     # control_img = data[0].x[0:784].cpu().detach().numpy().reshape(28, 28)
    #     # label_true = data[0].y[0].item()
    #     # label_ = data[0].x[-10:].cpu().detach().numpy()
    #     # assert label_true == np.argmax(label_)
    #     # plt.imshow(control_img)
    #     # plt.savefig(f"trained_models/{self.task}/control_img_{label_true}.png")
    #     # plt.close()

    #     # ------------------------

    #     # Set the graph data (flattened image + internal zeros + one-hot label)

    #     self.reset_nodes(batch_size=data.shape[0])        

       
    #     if self.reshape:
    #         self.values[:, :] = data.clone()

    #         # Zero out the imgae vector 
    #         # self.values[:, 0:784] = 0
    #         self.values[:, 0:784] = torch.randn_like(self.values[:, 0:784])  # Check all feature dimensions
    #     else:
    #         tmp = data.clone().view(self.batch_size, self.num_vertices)
    #         tmp[:, 0:784] = torch.randn_like(tmp[:, 0:784])  # Check all feature dimensions

    #         data = tmp.view(self.batch_size * self.num_vertices, 1)
    #         self.values = data
    #         self.values[self.sensory_indices_batch] = 0


    #     self.update_xs(train=False)

    #     # generated_imgs = self.values[:, :784]   # batch,10
    #     # generated_imgs = generated_imgs.view(self.batch_size, 28, 28)   # batch,10

    #     # # generated_imgs = self.values.view(self.batch_size, self.num_vertices)   # batch,10
    #     # # generated_imgs = generated_imgs[self.batch_size, :784] # batch,10
    #     # # generated_imgs = generated_imgs.view(self.batch_size, 28, 28)   # batch,10

    #     # # save img inside 1 big plt imshow plot; take first 10 images
    #     # import matplotlib.pyplot as plt

    #     # # random_offset between 0 and batch_size
    #     # random_offset = np.random.randint(0, self.batch_size-10)

    #     # fig, axs = plt.subplots(1, 10, figsize=(20, 2))
    #     # for i in range(10):

    #     #     axs[i].imshow(generated_imgs[i+random_offset].cpu().detach().numpy())
    #     #     axs[i].axis("off")
    #     #     # use label from data.y
    #     #     axs[i].set_title(labels[i+random_offset].item())

    #     # # plt.show()
    #     # # save 
    #     # plt.savefig(f"trained_models/{self.task}/generated_imgs_{self.epoch}.png")
    #     # plt.close()
       

    #     generated_imgs = self.values.view(self.batch_size, self.num_vertices)   # batch,10
    #     print("generated_imgs", generated_imgs.shape)
    #     generated_imgs = generated_imgs[:, :784] # batch,10
    #     generated_imgs = generated_imgs.view(self.batch_size, 28, 28)   # batch,10

    #     generated_imgs = generated_imgs - generated_imgs.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]
    #     generated_imgs = generated_imgs / generated_imgs.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]

    #     # Ensure no NaNs appear (in case min=max for some images)
    #     generated_imgs = torch.nan_to_num(generated_imgs, nan=0.0)

    #     # save img inside 1 big plt imshow plot; take first 10 images
    #     if save_imgs or wandb_logging:

    #         fig, axs = plt.subplots(1, 10, figsize=(20, 2))

    #         # random_offset between 0 and batch_size
    #         random_offset = np.random.randint(0, self.batch_size-10)

    #         ims = []
    #         for i in range(10):
    #             im = axs[i].imshow(generated_imgs[i + random_offset].cpu().detach().numpy(), cmap='viridis')
    #             ims.append(im)  # Store the imshow object for colorbar
    #             axs[i].axis("off")
    #             axs[i].set_title(labels[i + random_offset].item())

    #         # Add colorbar (one for all images, linked to the first one)
    #         cbar = fig.colorbar(ims[0], ax=axs, orientation='vertical', fraction=0.02, pad=0.04)

    #         if save_imgs:
    #             plt.savefig(f"trained_models/{self.graph_type}/generated_imgs_{self.epoch}.png")
    #             # plt.savefig(f"trained_models/{self.graph_type}/{self.task}/generated_imgs_{self.epoch}.png")
    #             plt.close()

    #             print("len of trace_data", len(self.trace_data))
    #             # plot self.trace_data
    #             if self.trace:
    #                 fig, axs = plt.subplots(1, self.T_test, figsize=(20, 2))
    #                 for i in range(len(self.trace_data)):
    #                     axs[i].imshow(self.trace_data[i])
    #                     axs[i].axis("off")
    #                     axs[i].set_title(labels[self.epoch].item())

    #                 # plt.show()
    #                 # save 
    #                 plt.savefig(f"trained_models/{self.task}/trace_data_{self.epoch}.png")
    #                 plt.close()

    #         if wandb_logging:
    #             wandb.log({f"Generation/Generated_Images_{self.epoch}": [wandb.Image(generated_imgs, caption="Generated Images")]})
       
    #     return 0
        
    # def test_generative(self, data, labels, remove_label=False, save_imgs=False, wandb_logging=False):
    #     """
    #     Runs a generative test where the model reconstructs digits over T_test time steps.
    #     Creates a grid matrix: digits (rows) x time steps (columns).

    #     Args:
    #         data (torch.Tensor): Batch of graph data.
    #         labels (torch.Tensor): Corresponding labels (digits 0-9).
    #         remove_label (bool): Whether to zero out the label during generation.
    #         save_imgs (bool): Whether to save the resulting grid image locally.
    #         wandb_logging (bool): Whether to log the grid to Weights & Biases.
    #     """
    #     import os
    #     import matplotlib.pyplot as plt
    #     import numpy as np
    #     import wandb

    #     DEVICE = self.device
    #     T = self.T_test
    #     input_size = 784
    #     output_size = 10
    #     num_digits_in_batch = data.shape[0]

    #     print("Running test_generative()...")
    #     print("self.reshape:", self.reshape)

    #     # --- Select up to 10 distinct digits, or all available samples if less ---
    #     selected_indices = []
    #     selected_digits = set()

    #     for idx, label in enumerate(labels):
    #         digit = label.item()
    #         if digit not in selected_digits:
    #             selected_digits.add(digit)
    #             selected_indices.append(idx)
    #         if len(selected_digits) >= min(10, num_digits_in_batch):
    #             break

    #     # Get selected data and labels
    #     selected_data = data[selected_indices]  # Shape: [N, total_size]
    #     selected_labels = labels[selected_indices]

    #     num_selected = selected_data.shape[0]
    #     print(f"Selected {num_selected} digits for generation grid.")

    #     # --- Reset nodes for batch of selected digits ---
    #     self.reset_nodes(batch_size=num_selected)

    #     # --- Initialize values with random noise on sensory inputs ---
    #     if self.reshape:
    #         self.values[:, :] = selected_data.clone()

    #         # Zero or randomize the sensory nodes (first 784)
    #         # self.values[:, 0:input_size] = torch.randn_like(self.values[:, 0:input_size])
    #         self.values[:, 0:input_size] = torch.zeros_like(self.values[:, 0:input_size])

    #         # Optionally remove label (zero out one-hot)
    #         if remove_label:
    #             self.values[:, -output_size:] = 0
    #     else:
    #         tmp = selected_data.clone().view(num_selected, self.num_vertices)
    #         tmp[:, 0:input_size] = torch.randn_like(tmp[:, 0:input_size])
    #         tmp[:, 0:input_size] = torch.zeros_like(tmp[:, 0:input_size])

    #         if remove_label:
    #             tmp[:, -output_size:] = 0

    #         selected_data = tmp.view(num_selected * self.num_vertices, 1)
    #         self.values = selected_data
    #         self.values[self.sensory_indices_batch] = 0

    #     # --- Prepare grid storage: [num_selected_digits, T_test, 28, 28] ---
    #     grid_images = torch.zeros(num_selected, T, 28, 28)

    #     # --- Iterative inference over T_test steps ---
    #     for t in range(T):
    #         self.update_xs(train=False, trace=True)

    #         reconstructed_imgs = self.values.view(num_selected, self.num_vertices)
    #         reconstructed_imgs = reconstructed_imgs[:, :input_size].view(num_selected, 28, 28)

    #         # Normalize per image
    #         imgs = reconstructed_imgs - reconstructed_imgs.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]
    #         imgs = imgs / (imgs.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0] + 1e-8)

    #         imgs = torch.nan_to_num(imgs, nan=0.0)

    #         # Store images for current time step
    #         grid_images[:, t] = imgs.detach().cpu()

    #     # --- Create a single 2D matrix for the entire grid ---
    #     # grid_images shape: [num_selected, T, 28, 28]
    #     rows = []
    #     for row in range(num_selected):
    #         # Concatenate images horizontally for each time step
    #         img_row = torch.cat([grid_images[row, t] for t in range(T)], dim=1)  # [28, T*28]
    #         rows.append(img_row)

    #     # Stack all rows vertically
    #     full_grid = torch.cat(rows, dim=0)  # [num_selected*28, T*28]

    #     # Convert to numpy for saving
    #     full_grid_np = full_grid.numpy()

    #     # --- Plot the full grid ---
    #     fig, ax = plt.subplots(figsize=(T * 2, num_selected * 2))
    #     ax.imshow(full_grid_np, cmap='gray')
    #     ax.axis('off')

    #     # Add labels on Y-axis (digit labels)
    #     tick_positions = [(i * 28) + 14 for i in range(num_selected)]
    #     tick_labels = [f'Digit {label.item()}' for label in selected_labels]

    #     ax.set_yticks(tick_positions)
    #     ax.set_yticklabels(tick_labels, fontsize=12)
    #     ax.set_xticks([])

    #     plt.tight_layout()

    #     # --- Save the figure locally ---
    #     if save_imgs:
    #         save_dir = f"trained_models/{self.graph_type}/generative_grid/"
    #         os.makedirs(save_dir, exist_ok=True)
    #         filename = f"{save_dir}/generative_grid_epoch_{self.epoch}.png"
    #         plt.savefig(filename)
    #         print(f"Saved generative grid image to: {filename}")

    #     # --- Log the figure to WandB ---
    #     if wandb_logging:
    #         wandb.log({f"Generation/Trace_Grid_Epoch_{self.epoch}": wandb.Image(fig, caption=f"Trace Grid Epoch {self.epoch}")})
    #         print(f"Logged grid image to Weights & Biases at epoch {self.epoch}")

    #     plt.close(fig)

    #     print("test_generative() complete!\n")
    #     return grid_images


    def test_generative(self, data, labels, remove_label=False, save_imgs=True, wandb_logging=False):
        """
        Runs a generative test on a batch, processes the batch, and plots:
            - 10 random generated images
            - Trace over time + sensory/internal energy (in a mosaic layout)
        """
        import os
        import matplotlib.pyplot as plt
        import numpy as np
        import wandb

        # Directory for saving outputs
        folder = self.task
        save_dir = f"trained_models/gen_test"
        # os.makedirs(save_dir, exist_ok=True)

        DEVICE = self.device
        input_size = 784
        output_size = 10

        self.batch_size = data.shape[0]
        self.reset_nodes(batch_size=self.batch_size)

        data = data.clone().view(self.batch_size, self.num_vertices)

        # ===== PREPARE GENERATION INPUT =====
        if self.task == "generation":
            if self.reshape:
                self.values[:, :] = data
                self.values[:, 0:input_size] = torch.randn_like(self.values[:, 0:input_size])

                assert (self.values[:, -output_size:].argmax(dim=1) == labels).all()
            else:
                data[:, 0:input_size] = torch.randn_like(data[:, 0:input_size])
                self.values = data.view(self.batch_size * self.num_vertices, 1)

        # ===== INFERENCE =====
        self.trace_data = []
        self.trace = True
        self.update_xs(train=False, trace=True)

        logits = self.values.view(self.batch_size, self.num_vertices)
        generated_imgs = logits[:, :784].view(self.batch_size, 28, 28)

        # ====== 1. PLOT 10 RANDOM GENERATED IMAGES ======

        n_images = min(10, len(generated_imgs))  # Don't plot more images than exist

        # Random offset: between 0 and (len - n_images), ensuring we stay within bounds
        if len(generated_imgs) > n_images:
            random_offset = np.random.randint(0, len(generated_imgs) - n_images + 1)
        else:
            random_offset = 0

        print(f"Plotting {n_images} images starting from index {random_offset}")

        fig_gen, axs = plt.subplots(1, n_images, figsize=(n_images * 2, 2))

        # If there's only 1 image, axs is not iterable, so force it into a list
        if n_images == 1:
            axs = [axs]

        # Iterate over the selected image indices
        for idx in range(n_images):
            img_idx = random_offset + idx

            axs[idx].imshow(generated_imgs[img_idx].cpu().detach().numpy(), cmap='gray')
            axs[idx].axis("off")

            # If labels available, use label index; fallback to image index
            if labels is not None and len(labels) > img_idx:
                axs[idx].set_title(f"{labels[img_idx].item()}")
            else:
                axs[idx].set_title(f"Img {img_idx}")

        plt.tight_layout()

        save_imgs = True  ## overrride save_imgs
        if save_imgs:
            img_path = os.path.join(save_dir, f"generated_imgs_epoch_{self.epoch}.png")
            plt.savefig(img_path)
            print(f"Saved generated image grid to: {img_path}")

        if wandb_logging:
            wandb.log({
                f"{folder}/Images": wandb.Image(fig_gen, caption=f"Generated Images at Epoch {self.epoch}")
            })

        plt.close(fig_gen)

        # ====== 2. TRACE + ENERGY MOSAIC ======
        if self.trace_data and len(self.trace_data) > 0:
            trace_steps = len(self.trace_data)
            # normalize_trace_images = True
            normalize_trace_images = False 

            # ====== SNAPSHOT INDICES ======
            snapshot_indices = []
            early_steps = list(range(min(trace_steps, 5)))
            snapshot_indices.extend(early_steps)

            middle_start = early_steps[-1] + 1 if early_steps else 0
            middle_end = int(0.8 * trace_steps)
            middle_steps = list(range(middle_start, middle_end, max((middle_end - middle_start) // 4, 1)))
            snapshot_indices.extend(middle_steps)

            late_steps = list(range(max(middle_end, trace_steps - 5), trace_steps))
            snapshot_indices.extend(late_steps)

            snapshot_indices = sorted(set(snapshot_indices))
            num_snapshots = len(snapshot_indices)

            # ====== CREATE MOSAIC ======
            mosaic_rows = [
                [str(i) for i in range(num_snapshots)],  # Top: Trace snapshots
                ["E"] * num_snapshots,                  # Bottom: Energy plot spanning all columns
            ]

            fig_mosaic, ax = plt.subplot_mosaic(mosaic_rows, figsize=(num_snapshots * 3, 6))
            label = labels[0].item() if labels is not None else "digit Unknown"

            fig_mosaic.suptitle(f"Trace & Energy (Digit: {label}) - Epoch {self.epoch}", fontsize=16, fontweight='bold')

            # ====== Normalize Function ======
            def normalize_image(img):
                img_min = img.min()
                img_max = img.max()
                if img_max - img_min < 1e-8:
                    return img * 0
                return (img - img_min) / (img_max - img_min)

            # ====== TOP ROW: TRACE SNAPSHOTS ======
            for idx, t_idx in enumerate(snapshot_indices):
                img = self.trace_data[t_idx][0]
                if normalize_trace_images:
                    img = normalize_image(img)

                ax[str(idx)].imshow(img, cmap='gray')
                ax[str(idx)].set_title(f"T = {t_idx}")
                ax[str(idx)].axis('off')

            # ====== BOTTOM ROW: ENERGY LINE PLOT ======
            ax_energy = ax["E"]

            steps = list(range(trace_steps))

            ax_energy.plot(steps, self.lst_internal_error, label="Internal Energy", color="tab:red")
            ax_energy.plot(steps, self.lst_sensor_error, label="Sensory Energy", color="tab:blue")

            ax_energy.set_xlabel("Time Step")
            ax_energy.set_ylabel("Energy")
            ax_energy.set_title("Energy over Time")
            ax_energy.legend()

            plt.tight_layout()

            # ====== SAVE MOSAIC LOCALLY ======
            if save_imgs:
                trace_path = os.path.join(save_dir, f"trace_and_energy_epoch_{self.epoch}.png")
                fig_mosaic.savefig(trace_path)
                print(f"Saved trace & energy mosaic to: {trace_path}")

            # ====== LOG TO WANDB ======
            if wandb_logging:
                wandb.log({
                    f"{folder}/Trace_Energy_Mosaic": wandb.Image(fig_mosaic, caption=f"Trace & Energy Mosaic at Epoch {self.epoch}")
                })
                print(f"Logged trace & energy mosaic to WandB under {folder}/Trace_Energy_Mosaic")

            plt.close(fig_mosaic)

        print("test_generative() complete!\n")
        return True


    def test_iterative(self, data, eval_types=None, remove_label=True):
        # edge_index = data.edge_index

        graph, label = data
        self.batch_size = graph.shape[0] 

        # eval_type = ["classification", "generative", "..."]

        if "classification" in eval_types:
            self.set_task("classification")   # not update the sensory nodes, only supervised nodes

            return self.test_classifications(graph.clone().to(self.device), 
                                      remove_label=remove_label)
                                      
        if "generation" in eval_types:
            self.set_task("generation")       # not update the supervised nodes, only sensory nodes

            if self.do_gen:
                self.trace_data = []
                self.trace = True 

                self.test_generative(graph.clone().to(self.device), 
                                    label.clone().to(self.device),
                                    remove_label=False, save_imgs=True, wandb_logging=True)
                self.do_gen = False 

            return 0 # Placeholder ""
        
        if "occlusion" in eval_types:
            self.set_task("occlusion")

            self.trace_data = []
            self.trace = True 
            
            occluded_graph = graph.clone().to(self.device)   # batch_size, num_vertices

            occluded_graph[:, 392:784] = 0.0   # or add noise instead of zero

            self.test_generative(occluded_graph, 
                                label.clone().to(self.device),
                                remove_label=False, save_imgs=True, wandb_logging=True)

            return 0  # Or the reconstructed images

        else:
            raise ValueError("Unknown evaluation type")
    


    def get_energy(self):
        return torch.sum(self.errors**2).item()

    def get_errors(self):
        return self.e.clone()


    def test_feedforward(self, X_batch):
        pass
        # X_batch = to_vector(X_batch)     # makes e.g. 28*28 -> 784

        # self.reset_nodes(batch_size=X_batch.shape[0])
        # self.clamp_input(X_batch)
        # self.forward(self.num_verticesum_layers)

        # return self.x[:,-self.structure.shape[2]:] 

    # def test_iterative(self, X_batch, diagnostics=None, early_stop=False):
    # def test_iterative(self, X_batch):
    #     # X_batch = to_vector(X_batch)     # makes e.g. 28*28 -> 784

    #     # self.reset_nodes(batch_size=X_batch.shape[0])
    #     # self.clamp_input(X_batch)
    #     # self.init_hidden()
    #     # self.init_output()

    #     self.update_xs(train=False)
    #     # self.update_xs(train=False, diagnostics=diagnostics, early_stop=early_stop)
    #     return self.x[:,-self.structure.shape[2]:] 

    def get_weights(self):
        return self.w.clone()

    def get_energy(self):
        return torch.sum(self.e**2).item()

    def get_errors(self):
        return self.e.clone()    


##############################