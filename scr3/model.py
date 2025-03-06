


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
relu = torch.nn.ReLU()
tanh = torch.nn.Tanh()
sigmoid = torch.nn.Sigmoid()
silu = torch.nn.SiLU()
linear = torch.nn.Identity()
leaky_relu = torch.nn.LeakyReLU()

@torch.jit.script
def sigmoid_derivative(x):
    return torch.exp(-x)/((1.+torch.exp(-x))**2)

@torch.jit.script
def relu_derivative(x):
    return torch.heaviside(x, torch.tensor(0.))

@torch.jit.script
def tanh_derivative(x):
    return 1-tanh(x)**2

@torch.jit.script
def silu_derivative(x):
    return silu(x) + torch.sigmoid(x)*(1.0-silu(x))

@torch.jit.script
def leaky_relu_derivative(x):
    return torch.where(x > 0, torch.tensor(1.), torch.tensor(0.01))

def get_derivative(f):


    print("f", f)

    if f == "sigmoid":
        f, f_prime = sigmoid, sigmoid_derivative
    elif f == "relu":
        f, f_prime = relu, relu_derivative
    elif f == "tanh":
        f, f_prime = tanh, tanh_derivative
    elif f == "silu":
        f, f_prime = silu, silu_derivative
    elif f == "linear":
        f, f_prime = 1, 1
    elif f == "leaky_relu":
        f, f_prime = leaky_relu, leaky_relu_derivative
    else:
        raise NotImplementedError(f"Derivative of {f} not implemented")

    return f, f_prime

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

    def __init__(self, update_type, batch_size, f, dfdx, edge_index):
        super().__init__(update_type, batch_size, f, dfdx)
        """ 
        Message Passing for Predictive Coding:
        self.x | values  have shape [batch_size * num_vertices]
        
        """
        self.pred_mu_MP = PredictionMessagePassing(self.f)
        self.grad_x_MP = GradientMessagePassing(self.dfdx)
        self.edge_index = edge_index


    def pred(self, x, w):
        # Gather 1D weights corresponding to connected edges
        weights_1d = w[self.edge_index[0], self.edge_index[1]]  # Extract relevant weights from W

        # # Expand edge weights for each graph
        batched_weights = weights_1d.repeat(self.batch_size)
        # batched_weights = weights_1d.expand(self.batch_size, -1)

        print("pred()")
        print("x: ", x.shape)
        print("edge_index: ", self.edge_index.shape)
        print("batched_weights: ", batched_weights.shape)
        
        self.pred_mu_MP(x, self.edge_index, batched_weights)

    def grad_x(self, x, e, w):

        # Gather 1D weights corresponding to connected edges
        weights_1d = w[self.edge_index[0], self.edge_index[1]]  # Extract relevant weights from W

        # # Expand edge weights for each graph
        batched_weights = weights_1d.repeat(self.batch_size)
        # batched_weights = weights_1d.expand(self.batch_size, -1)

        dEdx = self.grad_x_MP(x, self.edge_index, e, batched_weights)
    
        return dEdx

class PCgraph(torch.nn.Module): 

    def __init__(self, f, device, num_vertices, num_internal, adj, edge_index, batch_size, learning_rates, T_train, T_test, incremental, use_input_error, node_init_std=None, min_delta=None, early_stop=None):
        super().__init__()

        self.device = device

        self.num_vertices = num_vertices
        self.num_internal = num_internal
        self.adj = torch.tensor(adj).to(self.device)
        # import torch_geometric 
        # self.edge_index = torch_geometric.utils.dense_to_sparse(adj)[0]

        self.edge_index = edge_index.to(self.device)  # PYG edge_index

        self.lr_x, self.lr_w = learning_rates 
        self.T_train = T_train
        self.T_test = T_test
        self.node_init_std = node_init_std
        self.incremental = incremental 
        self.min_delta = min_delta
        self.early_stop = early_stop

        self.epoch = 0 
        self.batch_size = batch_size  # Number of graphs in the batch

        self.f, self.dfdx = get_derivative(f)
        self.use_input_error = use_input_error
        self.trace = False 

        # self.w = nn.Parameter(torch.empty(num_vertices, num_vertices, device=self.device))
        # self.b = nn.Parameter(torch.empty(num_vertices, device=self.device))
        self.device = device 

        
        # print("edge_index: ", edge_index)
        # print("self.edge_index: ", self.edge_index)
        # assert self.edge_index == edge_index        
        
        # assert torch.all(self.edge_index == edge_index)

        self.adj = torch.tensor(adj).to(DEVICE)
        self.mask = self.adj
        
        self._reset_grad()
        self._reset_params()

        self.optimizer_w = torch.optim.Adam([self.w], lr=self.lr_w, betas=(0.9, 0.999), eps=1e-7, weight_decay=0)


        # self.MP = PredictiveCodingLayer(f=self.structure.f, 
        #                                 f_prime=self.structure.dfdx)

        update_rule = "vanZwol_AMB"
        # update_rule = "MP_AMB"


        if update_rule in ["vectorized", "vanZwol_AMB"]:
            print("--------------Using vanZwol_AMB------------")
            self.reshape = True
            self.updates = vanZwol_AMB(update_type=update_rule, batch_size=self.batch_size, f=self.f, dfdx=self.dfdx)
        elif update_rule in ["MP", "MP_AMB"]:
            print("-------Using MP_AMB-------------")
            self.reshape = False

            self.batched_edge_index = torch.cat(
                [self.edge_index + i * self.num_vertices for i in range(self.batch_size)], dim=1
            )       
    
            self.updates = MP_AMB(update_type=update_rule, batch_size=self.batch_size, 
                                #   edge_index=self.edge_index, 
                                  edge_index=self.batched_edge_index,
                                  f=self.f, dfdx=self.dfdx)
            
        else:
            raise ValueError(f"Invalid update rule: {update_rule}")


        self.mode = "train"
        self.use_bias = False

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

        self.w = torch.nn.Parameter(torch.empty(self.num_vertices, self.num_vertices, device=DEVICE))
        # self.w = torch.empty( self.num_vertices, self.num_vertices, device=DEVICE)
       
        # best for Classification
        # nn.init.normal_(self.w, mean=0, std=0.05)  
        # # 

        # trying for generation

        # # # BEST FOR GENERATION
        # self.w.data.fill_(0.001)
        self.w.data.fill_(0.0001)
        # Add small random noise
        noise = torch.randn_like(self.w) * 0.0001
        self.w.data.add_(noise)
        

        # Perform the operation and reassign self.w as a Parameter
        with torch.no_grad():
            self.w.copy_(self.adj * self.w)

        # self.w = self.adj * self.w 


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
        self.b = torch.empty( self.num_vertices, device=DEVICE)
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
            self.errors = torch.empty(batch_size * self.num_vertices, device=DEVICE)
            self.values = torch.zeros(batch_size * self.num_vertices, device=DEVICE)

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

    def update_w(self):
        
        self.errors = self.errors.view(self.batch_size, self.num_vertices)
        self.x = self.values.view(self.batch_size, self.num_vertices)

        out = -torch.matmul(self.errors.T, self.f(self.x))
        # out = -torch.sparse.mm(self.errors.T, self.f(self.x))

        if self.mask is not None:
            out *= self.mask
        self.dw = out 
        
        # self.dw = self.structure.grad_w(x=self.x, e=self.e, w=self.w, b=self.b)
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

        # self.update_mask = self.update_mask_train

    def test_(self):
        self.mode = "test"
        # self.trace = True 

        print(self.mode)

        # self.update_mask = self.update_mask_test

    def init_modes(self, graph):
        
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

        elif task in ["generation", "reconstruction", "denoising", "occlusion"]:
            # Update both the internal and sensory nodes during these tasks
            self.update_mask_test = torch.tensor(sorted(self.internal_indices_batch + self.sensory_indices_batch), device=self.device)
            # print("Generation task")
            # print("self.update_mask_test", self.update_mask_test.shape)
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
        

    def update_xs(self, train=True, trace=False):

        # if self.early_stop:
        #     early_stopper = EarlyStopper(patience=0, min_delta=self.min_delta)

        T = self.T_train if train else self.T_test

        update_mask = self.internal_mask_train if train else self.update_mask_test

        # di = self.structure.shape[0] # 784
        di = 784
        upper = -10 if train else self.num_vertices
        
        for t in range(T): 

            # self.w = self.adj * self.w 
            # Perform the operation and reassign self.w as a Parameter
            with torch.no_grad():
                self.w.copy_(self.adj * self.w)

                # make weights[0:784, -10:] /= 2
                # self.w[0:784, -10:] /= 2 
                # self.w[-10:, 0:784] /= 2 
                
            self.mu = self.updates.pred(self.values.to(self.device), self.w.to(self.device))

            # predicted_mpU = self.pred_mu_MP.forward(self.values.view(-1,1).to(DEVICE),
            #                         self.batched_edge_index.to(DEVICE), 
            #                         batched_weights.to(DEVICE))
            # self.e = self.values - predicted_mpU

            # self.errors = self.errors.view(self.batch_size, self.num_vertices)
            # self.x = self.values.view(self.batch_size, self.num_vertices)
            
           
            # print(torch.allclose(mu_mp, mu, atol=1))  # Should be True
            # print(torch.allclose(predicted_mpU, mu, atol=1))  # Should be True
            
            # self.e = self.x - mu_mp 
            # print("predicted_mpU shape", predicted_mpU.shape)
            # print("values shape", self.values.shape)

            self.errors = self.values - self.mu
        
            # print("e1", torch.mean(self.e))
            # TODO

           
            # total_mean_error = self.errors.mean()
            # total_mean_error = torch.sum(self.errors**2).mean()

            # total_internal_error = self.errors[self.internal_indices_batch].mean()
            # self.history.append(total_mean_error.cpu().numpy())
            

            if self.reshape:
                if not self.use_input_error:
                    if self.task == "classification":
                        self.errors[:,:di] = 0 
                    # elif self.task in ["generation", "reconstruction", "denoising", "occlusion"]:
                    #     self.errors[:,di:upper] = 0
                        # self.errors[:,di:upper] = 0

                # print("self.errors", self.errors.shape)

                self.history.append(torch.sum(self.errors**2).item())
                # self.history.append(self.errors.cpu().mean().numpy()**2)
            else:

                if not self.use_input_error:
                    if self.task == "classification":
                        self.errors[self.sensory_indices_batch] = 0
                    elif self.task in ["generation", "reconstruction", "denoising", "occlusion"]:
                        self.errors[self.supervised_labels_batch] = 0
                        # self.errors[self.sensory_indices_batch] = 0

                total_internal_error = (self.errors[self.internal_indices_batch]**2).mean()
                self.history.append(total_internal_error.cpu().numpy())

            
            # x = self.x.T.contiguous().view(-1, 1).to(DEVICE)  # Shape: [num_nodes, batch_size]
            # error = self.e.T.contiguous().view(-1, 1).to(DEVICE)  # Shape: [num_nodes, batch_size]
            
            dEdx = self.updates.grad_x(self.values, self.errors, self.w)
            clipped_dEdx = torch.clamp(dEdx, -1, 1)

            if self.reshape:
                
                dEdx = dEdx[:, di:upper]
                # print("dEdx", dEdx.shape)

                # dEdx = self.structure.grad_x(self.x, self.e, self.w, self.b,train=train) # only hidden nodes
                self.values[:,di:upper] -= self.lr_x*dEdx 
            else:
                dEdx = dEdx[update_mask]

                # clipped_dEdx = torch.clamp(dEdx, -1, 1)
                clipped_dEdx = dEdx
                self.values[update_mask] -= self.lr_x * clipped_dEdx

            
            if train and self.incremental and self.dw is not None:
            # if train and self.incremental:

                # self.update_w()
                # print("optimizer step")
        
                # if self.w.is_sparse:
                #     self.w = self.w.to_dense()
                # if self.dw.is_sparse:
                #     self.dw = self.dw.to_dense()

                # # Convert m and gradients to dense if necessary
                # if self.dw.is_sparse:
                #     self.dw = self.dw.to_dense()

                # if self.optimizer.m_w.is_sparse:
                #     self.optimizer.m_w = self.optimizer.m_w.to_dense()
                
                # self.dw = torch.clamp(self.dw, -1, 1)

                self.w.grad = self.dw
                self.optimizer_w.step()
                # self.optimizer.step(self.params, self.grads, batch_size=self.batch_size)
            # if self.early_stop:
            #     if early_stopper.early_stop( self.get_energy() ):
            #         break            

            if self.trace:
                
                # print(self.x.shape)
                x_slice = self.values[0:1, 0:784].cpu().detach().numpy()
                # print("x_slice shape", x_slice.shape)

                if not isinstance(x_slice, torch.Tensor):
                    x_slice = torch.tensor(x_slice, device=self.device)

                if x_slice.numel() == 0:
                    print("Warning: x_slice is empty")
                    return

                x_slice = x_slice.contiguous().cpu().numpy()

                if not isinstance(x_slice, np.ndarray):
                    print("Error: Converted x_slice is not a NumPy array")
                    return
                
                self.trace_data.append(x_slice.reshape(28, 28))
            


    # def train_supervised(self, X_batch, y_batch): 

    #     # X_batch = to_vector(X_batch)                  # makes e.g. 28*28 -> 784
    #     # y_batch = onehot(y_batch, N=self.structure.shape[2])    # makes e.g. 3 -> [0,0,0,1,0,0,0,0,0,0]

    #     # self.reset_nodes(batch_size=X_batch.shape[0])        
    #     # self.clamp_input(X_batch)
    #     # # self.init_hidden()
    #     # # print("ommit init hidden")
    #     # self.clamp_target(y_batch)

    #     self.update_xs(train=True)
    #     self.update_w()

    #     if not self.incremental:
    #         # self.update_w()
    #         print("optimizer step end ")
    #         self.optimizer.step(self.params, self.grads, batch_size=X_batch.shape[0])

    
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


    def train_supervised(self, data):
        # edge_index = data.edge_index.to(self.device)
        self.history = []

        self.optimizer_w.zero_grad()
        # self.data_ptr = data.ptr
        # self.batch_size = data.x.shape[0] // self.num_vertices
        self.batch_size = data.shape[0]


        self.reset_nodes(batch_size=data.shape[0])        
                
        # Directly set graph_data into self.x
        self.values[:, :] = data.clone()
      
        # self.values, _ , _ = self.unpack_features(data.x, reshape=False)
        
        self.update_xs(train=True)
        self.update_w()

        if not self.incremental:
            self.w.grad = self.dw
            self.optimizer_w.step()
            # self.optimizer.step(self.params, self.grads, batch_size=self.batch_size)

        # print("w mean", self.w.mean())

        return self.history


    def test_classifications(self, data, remove_label=True):
        # self.reset_nodes(batch_size=data.shape[0])

        # Set the graph data (flattened image + internal zeros + one-hot label)
        self.values[:, :] = data.clone()

        # Zero out the one-hot vector (last output_size elements)
        self.values[:, -10:] = 0

        self.update_xs(train=False)
            
        logits = self.values.view(self.batch_size, self.num_vertices)   # batch,10
        # print("logits ", logits)
        logits = logits[:, -10:]

        y_pred = torch.argmax(logits, axis=1).squeeze()
        # print("logits ", logits.shape)
        # print("y_pred ", y_pred.shape)
        return y_pred
    


    def test_generative(self, data, labels, remove_label=True, save_imgs=False):
        
        print("self.reshape", self.reshape)
        # remove one_hot
        # if remove_label:
        #     for i in range(len(data)):
        #         sub_graph = data[i]  # Access the subgraph

        #         # set sensory indices to zero / random noise
        #         sub_graph.x[sub_graph.sensory_indices, 0] = torch.zeros_like(sub_graph.x[sub_graph.sensory_indices, 0])  # Check all feature dimensions
        #         # random noise
        #         # sub_graph.x[sub_graph.sensory_indices, 0] = torch.randn_like(sub_graph.x[sub_graph.sensory_indices, 0])  # Check all feature dimensions
        #         # sub_graph.x[sub_graph.sensory_indices, 0] = torch.clamp(torch.randn_like(sub_graph.x[sub_graph.sensory_indices, 0]), min=0, max=1)

        # control_img = data[0].x[0:784].cpu().detach().numpy().reshape(28, 28)
        # label_true = data[0].y[0].item()
        # label_ = data[0].x[-10:].cpu().detach().numpy()
        # assert label_true == np.argmax(label_)
        # plt.imshow(control_img)
        # plt.savefig(f"trained_models/{self.task}/control_img_{label_true}.png")
        # plt.close()

        # ------------------------

        # Set the graph data (flattened image + internal zeros + one-hot label)

        self.reset_nodes(batch_size=data.shape[0])        

        self.values[:, :] = data.clone()

        # Zero out the imgae vector 
        # self.values[:, 0:784] = 0
        self.values[:, 0:784] = torch.randn_like(self.values[:, 0:784])  # Check all feature dimensions

        self.update_xs(train=False)

        # generated_imgs = self.values[:, :784]   # batch,10
        # generated_imgs = generated_imgs.view(self.batch_size, 28, 28)   # batch,10

        # # generated_imgs = self.values.view(self.batch_size, self.num_vertices)   # batch,10
        # # generated_imgs = generated_imgs[self.batch_size, :784] # batch,10
        # # generated_imgs = generated_imgs.view(self.batch_size, 28, 28)   # batch,10

        # # save img inside 1 big plt imshow plot; take first 10 images
        # import matplotlib.pyplot as plt

        # # random_offset between 0 and batch_size
        # random_offset = np.random.randint(0, self.batch_size-10)

        # fig, axs = plt.subplots(1, 10, figsize=(20, 2))
        # for i in range(10):

        #     axs[i].imshow(generated_imgs[i+random_offset].cpu().detach().numpy())
        #     axs[i].axis("off")
        #     # use label from data.y
        #     axs[i].set_title(labels[i+random_offset].item())

        # # plt.show()
        # # save 
        # plt.savefig(f"trained_models/{self.task}/generated_imgs_{self.epoch}.png")
        # plt.close()
       

        generated_imgs = self.values.view(self.batch_size, self.num_vertices)   # batch,10
        print("generated_imgs", generated_imgs.shape)
        generated_imgs = generated_imgs[:, :784] # batch,10
        generated_imgs = generated_imgs.view(self.batch_size, 28, 28)   # batch,10

        generated_imgs = generated_imgs - generated_imgs.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]
        generated_imgs = generated_imgs / generated_imgs.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]

        # Ensure no NaNs appear (in case min=max for some images)
        generated_imgs = torch.nan_to_num(generated_imgs, nan=0.0)

        # save img inside 1 big plt imshow plot; take first 10 images
        if save_imgs:

            fig, axs = plt.subplots(1, 10, figsize=(20, 2))

            # random_offset between 0 and batch_size
            random_offset = np.random.randint(0, self.batch_size-10)

            ims = []
            for i in range(10):
                im = axs[i].imshow(generated_imgs[i + random_offset].cpu().detach().numpy(), cmap='viridis')
                ims.append(im)  # Store the imshow object for colorbar
                axs[i].axis("off")
                axs[i].set_title(labels[i + random_offset].item())

            # Add colorbar (one for all images, linked to the first one)
            cbar = fig.colorbar(ims[0], ax=axs, orientation='vertical', fraction=0.02, pad=0.04)

            plt.savefig(f"trained_models/{self.task}/generated_imgs_{self.epoch}.png")
            plt.close()

            # plot self.trace_data
            if self.trace:
                fig, axs = plt.subplots(1, self.T_test, figsize=(20, 2))
                # for i in range(len(self.trace_data)):
                #     axs[i].imshow(self.trace_data[i])
                for img in (self.trace_data):
                    axs[i].imshow(img)
                    axs[i].axis("off")
                    axs[i].set_title(labels[0].item())

                # plt.show()
                # save 
                plt.savefig(f"trained_models/{self.task}/trace_data_{self.epoch}.png")
                plt.close()
            
       
        return 0
    


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

            self.trace_data = []
            self.trace = True 

            self.test_generative(graph.clone().to(self.device), 
                                 label.clone().to(self.device),
                                 remove_label=remove_label, save_imgs=True)
            
            return 0 # Placeholder ""
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