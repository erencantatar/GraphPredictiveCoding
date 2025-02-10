
from re import T
import torch
import torch.nn as nn  # Ensure that `torch.nn` is only imported once

import torch.nn.init as init
import numpy as np
import math
from torch_geometric.utils import to_dense_adj, degree
from models.MessagePassing import PredictionMessagePassing, ValueMessagePassing
from helper.activation_func import set_activation
from helper.grokfast import gradfilter_ema, gradfilter_ma, gradfilter_ema_adjust
import os 
import wandb
import math 
import io
from PIL import Image

import helper.vanZwol_optim as optim_VanZwol
from helper.vanZwol_optim import get_derivative 

import torch.optim.lr_scheduler

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree
import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree
from torch_scatter import scatter


import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter
from torch_geometric.utils import degree

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter
from torch_geometric.utils import degree


import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter

""" 
STARTING WITH VANZWOL where PYG data object is shaped as batch_size,num_vertices. 
Now we introduce MEssagePassing instead of matmul operations, to be more efficient. 
"""


class PredictionMessagePassing(MessagePassing):
    def __init__(self, aggr='add', f=torch.tanh):
        super(PredictionMessagePassing, self).__init__()  # Aggregate messages using sum
        self.f = f

    def forward(self, x, edge_index, edge_weight):
        # Start propagating messages
        return self.propagate(edge_index, x=x, edge_weight=edge_weight)

    def message(self, x_j, edge_weight):
        # Compute messages as f(x_j) * w_ij
        
        return self.f(x_j) * edge_weight
        # return self.f(x_j) * edge_weight

    def update(self, aggr_out):
        # Return the aggregated prediction for each node
        return aggr_out

# e[:,lower:upper] - self.dfdx(x[:,lower:upper]) * torch.matmul(e, w.T[lower:upper,:].T)


class DeltaXUpdate(MessagePassing):
    def __init__(self, aggr='add', dfdx=None):
        super(DeltaXUpdate, self).__init__(aggr=aggr)
        self.dfdx = dfdx  # Activation derivative

    def forward(self, x, errors, edge_index, edge_weight, use_w_ji=True):
        # Store errors to access in the message method

        # Perform message passing to compute the sum term
        sum_term = self.propagate(edge_index, x=errors, edge_weight=edge_weight)
        
        # Compute the delta_x update
        delta_x = (-errors + self.dfdx(x) * sum_term)
        
        return delta_x

    def message(self, x_j, edge_weight):
        # x_j: Source node errors (tensor of shape [num_edges, 1])
        # edge_weight: Edge weights (tensor of shape [num_edges, 1])
        
        # Multiply errors by edge weights
        return edge_weight * x_j


class PC_graph_zwol_PYG(torch.nn.Module): 

    def __init__(self, f, device, num_vertices, num_internal, adj, edge_index, batch_size, lr_x, T_train, T_test, incremental, use_input_error, node_init_std=None, min_delta=None, early_stop=None):
        super().__init__()

        self.device = device

        self.num_vertices = num_vertices
        self.num_internal = num_internal
        self.adj = torch.tensor(adj).to(self.device)
        
        self.edge_index = edge_index  # PYG edge_index
        print("------VERSION WITH MESSAGE PASSING-------")
        self.lr_x = lr_x 
        self.T_train = T_train
        self.T_test = T_test
        self.node_init_std = node_init_std
        self.incremental = incremental 
        self.min_delta = min_delta
        self.early_stop = early_stop

        self.batch_size = batch_size  # Number of graphs in the batch

        self.f = f
        self.dfdx = get_derivative(f)
        self.use_input_error = use_input_error

        # self.w = nn.Parameter(torch.empty(num_vertices, num_vertices, device=self.device))
        # self.b = nn.Parameter(torch.empty(num_vertices, device=self.device))
        self.device = device 

        self._reset_grad()
        self._reset_params()
        print("self.dw init", self.dw)

        self.mode = "train"

        self.use_bias = False

        self.prediction_mp = PredictionMessagePassing(f=self.f)
        self.delta_x_MP    = DeltaXUpdate(dfdx=self.dfdx)
   
    @property
    def params(self):
        return {"w": self.w, "b": self.b, "use_bias": self.use_bias}
        
    @property
    def grads(self):
        return {"w": self.dw, "b": self.db}

    def _reset_params(self):
        print("settings params")
        
        # values = torch.randn(self.edge_index.size(1), device=self.device) * 0.05
        # self.w = torch.sparse_coo_tensor(self.edge_index, values, (self.num_vertices, self.num_vertices), device=self.device)

        self.w = torch.empty( self.num_vertices, self.num_vertices, device=self.device)
        self.b = torch.empty( self.num_vertices, device=self.device)

        self.error_heatmap = torch.empty( self.num_vertices, self.num_vertices,  device=self.device)
        self.dw_heatmap = torch.empty( self.num_vertices, self.num_vertices,  device=self.device)

        nn.init.normal_(self.w, mean=0, std=0.05)

        # add fixed value to self.w 

        # noise = torch.randn_like(self.w) * 0.001
        # self.w.data.add_(noise)

        self.w = self.w * self.adj

    
    def _reset_grad(self):
        self.dw, self.db = None, None


    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def reset_nodes(self, batch_size=1):
        # self.e = torch.empty(batch_size, sum(self.structure.shape), device=DEVICE)
        # self.x = torch.zeros(batch_size, sum(self.structure.shape), device=DEVICE)
        pass

    def train(self):
        self.mode = "train"
        self.dw = None 
        # self.update_mask = self.update_mask_train

    def test(self):
        self.mode = "test"

        print(self.mode)
        # self.update_mask = self.update_mask_test


    def init_modes(self, batch_example):
        
        # take first item 
        batch_example = batch_example[0]
        sensory_indices_single_graph, internal_indices_single_graph, supervised_labels_single_graph = batch_example.sensory_indices, batch_example.internal_indices,  batch_example.supervision_indices
        
        self.base_sensory_indices = list(sensory_indices_single_graph)
        self.base_internal_indices = list(internal_indices_single_graph)
        self.base_supervised_labels = list(supervised_labels_single_graph) if supervised_labels_single_graph else []

        # Correcting the initialization of batched indices

        # Ensuring the base indices are flattened lists of integers
        base_sensory_indices = [int(idx) for sublist in self.base_sensory_indices for idx in sublist] if isinstance(self.base_sensory_indices[0], list) else self.base_sensory_indices
        base_internal_indices = [int(idx) for sublist in self.base_internal_indices for idx in sublist] if isinstance(self.base_internal_indices[0], list) else self.base_internal_indices
        base_supervised_labels = [int(idx) for sublist in self.base_supervised_labels for idx in sublist] if isinstance(self.base_supervised_labels[0], list) else self.base_supervised_labels

        # Create batched indices by iterating over batch size and offsetting by graph index
        sensory_indices_batch = [
            index + i * self.num_vertices for i in range(self.batch_size) for index in base_sensory_indices
        ]
        internal_indices_batch = [
            index + i * self.num_vertices for i in range(self.batch_size) for index in base_internal_indices
        ]
        supervised_labels_batch = [
            index + i * self.num_vertices for i in range(self.batch_size) for index in base_supervised_labels
        ] if base_supervised_labels else []

        # Convert to tensors for masking purposes during updates
        self.sensory_indices_batch = torch.tensor(sorted(sensory_indices_batch), device=self.device)
        self.internal_indices_batch = torch.tensor(sorted(internal_indices_batch), device=self.device)
        self.supervised_labels_batch = torch.tensor(sorted(supervised_labels_batch), device=self.device) if supervised_labels_batch else None

        print("Sensory indices batch:", self.sensory_indices_batch.shape, 784, self.batch_size)
        print("Internal indices batch:", self.internal_indices_batch.shape, self.num_internal, self.batch_size)
        if self.supervised_labels_batch is not None:
            print("Supervised labels batch:", self.supervised_labels_batch.shape, 10, self.batch_size)

     
        # Update only internal nodes during training
        self.internal_mask_train = torch.tensor(internal_indices_batch, device=self.device)

        # Update internal + supervision nodes during testing
        self.update_mask_test = torch.tensor(sorted(internal_indices_batch + supervised_labels_batch), device=self.device)

        # update_mask = self.internal_mask_train if train else self.update_mask_test


        

    def unpack_features(self, batch, reshape=True):
        """Unpack values, errors, and predictions from the batched graph."""
        values, errors, predictions = batch[:, 0, :].to(self.device), batch[:, 1, :].to(self.device),  None
        # print("unpacked featreus")
        # print(values.shape)

        # reshape to (batch_size, num_vertices)
        if reshape:
            values      = values.view(self.batch_size * self.num_vertices, 1)
            errors      = errors.view(self.batch_size * self.num_vertices, 1)
            # predictions = predictions.view(self.batch_size, self.num_vertices)

        return values, errors, predictions

    def get_sparse_weight(self, w_ij):
        # Extract edge_index for a single graph
        single_graph_edge_index = self.edge_index[:, :self.edge_index.size(1) // self.batch_size]


        # w_ij = True 
        # w_ij = False
        # Get sparse weights for a single graph
        if w_ij:
          sparse_weights = self.w[single_graph_edge_index[0], single_graph_edge_index[1]]
        else:
          w_c = self.w.clone()
          sparse_weights = w_c.T[single_graph_edge_index[0], single_graph_edge_index[1]]

        # Expand weights correctly by repeating for batch size
        num_features = 1
        out = sparse_weights.repeat(self.batch_size).view(-1, num_features)
        return out


    def get_prediction(self):
        # Reshape node features
        self.values = self.values.view(self.batch_size * self.num_vertices, -1)

        # Get batched edge weights (now correctly sized)
        edge_weights_batched = self.get_sparse_weight(w_ij=True)

        # print("---get_prediction---")
        # print("values", self.values.shape)
        # print("edge_weights_batched", edge_weights_batched.shape)
        # print("self.edge_index", self.edge_index.shape)

        # Perform message passing
        prediction = self.prediction_mp(self.values, self.edge_index, edge_weights_batched)
        return prediction


    def grad_x(self):
        # Use message passing to compute gradient-based updates
        # gradx = self.errors[:,self.update_mask] - self.dfdx(self.values[:,self.update_mask]) * torch.matmul(self.errors, self.w.T[self.update_mask,:].T)

        # Get batched edge weights (now correctly sized)
        # edge_weights_batched = self.get_sparse_weight()
        edge_weights_batched = self.get_sparse_weight(w_ij=False)

        
        hidden_nodes = self.internal_indices_batch
        if self.mode == "test":
            combined_nodes = torch.cat([hidden_nodes, self.supervised_labels_batch])
        
        if self.mode == "train":
            combined_nodes = hidden_nodes

        # Step 2: Create a mask where both source and target nodes belong to the combined set
        mask = (torch.isin(self.edge_index[0], combined_nodes) & torch.isin(self.edge_index[1], combined_nodes))

        # Step 3: Subset the edge_index and edge_weight
        subgraph_edge_index = self.edge_index[:, mask]
    
        # # or all
        # subgraph_edge_index = self.edge_index
        # edge_weight=edge_weights_batched


        delta_x = self.delta_x_MP(
            x=self.values,
            errors=self.errors,  # Already precomputed
            edge_index=subgraph_edge_index ,
            # edge_weight=edge_weights_batched
            edge_weight=edge_weights_batched[mask]
        )

        # # old 
        # delta_x = self.delta_x_MP(
        #     x=self.values,
        #     errors=self.errors,  # Already precomputed
        #     edge_index=self.edge_index ,
        #     edge_weight=edge_weights_batched
        # )

        return delta_x


    # def precompute_masks(self):
    #     # Compute masks once for training and testing using batched indices
    #     internal_mask_train = self._create_batched_mask(self.internal_indices_single)
    #     update_mask_test = self._create_batched_mask(
    #         torch.cat([self.internal_indices_single, self.supervision_indices_single])
    #     )
    #     return internal_mask_train, update_mask_test

    # def _create_batched_mask(self, node_indices):
    #     # Expand node indices for the batch
    #     return torch.cat([
    #         node_indices + i * self.num_vertices for i in range(self.batch_size)
    #     ])


    def update_xs(self, train=True):
        print("------update_xs------")
        T = self.T_train if train else self.T_test

        update_mask = self.internal_mask_train if train else self.update_mask_test

        # Move all relevant tensors to self.device
       
        # if not train:
        #     print("1", self.values[:, -10:])
        # take the first 

        # print("update_mask shape", self.update_mask.shape)
        self.energies = []
        for t in range(T):
            self.prediction = self.get_prediction().to(self.device)
            
            # torch empty 
            # import gc
            # torch.cuda.empty_cache()
            # gc.collect()
            
            # print("get_prediction", self.prediction.shape)

            self.errors = self.values - self.prediction  # Update errors
            self.e = self.errors 
            self.energies.append(self.errors.mean().cpu().detach())

            if not self.use_input_error:
                tmp = self.errors.clone().view(self.batch_size, self.num_vertices)
                tmp[:, 0:784] = 0 
                tmp = tmp.view_as(self.errors)

                self.errors = tmp

            tmp = self.errors.view(self.batch_size, self.num_vertices).mean(0)
            self.error_heatmap += tmp
            # print("errors mean", self.errors.mean())

            # Determine which nodes to update
            dEdx = self.grad_x()
            # dEdx shape torch.Size([1684, 1])

            # print("dEdx shape", dEdx.shape)
            # print("values shape", values.shape)
            # self.values[:, self.update_mask] -= self.lr_x * dEdx

           # Efficiently update only the masked nodes
            # print("update_mask.shape", update_mask.shape)

            tmp = dEdx.clone().view(self.batch_size, self.num_vertices)
            self.zz = tmp 

            avg_dEdx_per_batch = tmp.mean(0)
            assert max(avg_dEdx_per_batch) > 0

            self.values = self.values.view(self.batch_size, self.num_vertices)

            if self.mode == "test":
              print(tmp[:, 784:].shape)
              print(self.values[:, 784:].shape)

              self.values[:, 784:] -= self.lr_x * tmp[:, 784:]
              print("dedX on onehot", self.values[0, -10:])

            else:
              self.values[:, 784:-10] -= self.lr_x * tmp[:, 784:-10]

            self.values = self.values.view_as(self.errors)


            # self.values[update_mask] -= self.lr_x * dEdx[update_mask]

            # print("type dw", type(self.dw))
            if self.incremental and self.dw is not None and train:
                # print("dw", self.dw.shape)
                # print(self.params["w"].shape)

                # Ensure params and grads are both on the same device
                self.params["w"] = self.params["w"].to(self.device)
                self.grads["w"] = self.grads["w"].to(self.device)
                # print("self.device", self.device)

                # print(self.params["w"].device)
                # print(self.grads["w"].device)

                self.optimizer.step(self.params, self.grads, batch_size=self.batch_size)
                # print("w mean", self.params["w"].mean())
                # print("dw mean:", self.dw.mean().item(), "dw max:", self.dw.max().item(), "dw min:", self.dw.min().item())
          
        

    def update_w(self):


        print("----update_w-----")

        # Select activations and errors based on edge_index
        # source_activations = self.f(self.values[self.edge_index[0]])  # [51520, 1]
        # target_errors = self.errors[self.edge_index[1]]  # [51520, 1]

        # Select activations and errors based on edge_index
        # source_activations = self.f(self.values).view(self.batch_size, self.num_vertices).to(self.device)  # [1684]
        # target_errors = self.errors.view(self.batch_size, self.num_vertices).to(self.device)  # [1684]

        # # Compute per-edge weight contributions
        # self.dw = -torch.matmul(target_errors.T, source_activations)  # [51520, 1]
        
        # print("source_activations.shape, target_errors.shape", source_activations.shape, target_errors.shape)
        source_activations = self.f(self.values).view(self.batch_size, self.num_vertices).to(self.device)  # (batch, N)
        target_errors = self.errors.view(self.batch_size, self.num_vertices).to(self.device)  # (batch, N)

        self.dw = -torch.matmul(target_errors.T, source_activations)  # [51520, 1]

        # batch_mode = False
        # if batch_mode:
        #     # Batch-wise weight update (Resulting shape: (batch, N, N))
        #     self.dw = -torch.einsum('bi,bj->bij', target_errors, source_activations)  # Shape: (batch, N, N)
        #     print("Batch-wise weight update shape:", self.dw.shape)
        # else:
        #     # Summed weight update across the batch (Resulting shape: (N, N))
        #     self.dw = -torch.einsum('bi,bj->ij', target_errors, source_activations)  # Shape: (N, N)
        #     print("Summed weight update shape:", self.dw.shape)

        # self.dw_heatmap += self.dw.clone()

        # # Aggregate updates for each edge (optional: sum or keep per-edge contributions)
        # # If you have a sparse weight matrix, you can directly assign these to the corresponding positions
        # # self.dw = self.mean()

        if self.adj is not None:
          # print("adj", (self.adj.shape))
          # print("dw", (self.dw.shape))
          # print("self.w", self.w.shape)
          self.out = self.dw
          self.dw *= self.adj

        # self.dw = self.dw.to(self.device)
        # self.dw = None

        # print("self.dw shape", out.shape)

    def train_supervised(self, data):
        self.edge_index = data.edge_index.to(self.device)

        self.data_ptr = data.ptr
        self.batch_size = data.x.shape[0] // self.num_vertices

        self.values, self.errors, self.predictions = self.unpack_features(data.x, reshape=True)
        
        # print("self.values 1 shape", self.values.shape)
        # print("edge_index 1 shape", self.edge_index.shape)
        print("-----------------------------")
        self.update_xs(train=True)
        # self.update_w()

        if not self.incremental and self.dw is not None:
            self.optimizer.step(self.params, self.grads, batch_size=self.batch_size)

        history = self.energies
        print("history", history)
        print("reduce error if positive", abs(history[0]) - abs(history[1]))
        return history


    def test_iterative(self, data, remove_label=True):
        edge_index = data.edge_index

        # # remove one_hot
        # if remove_label:
        #     for i in range(len(data)):
        #         sub_graph = data[i]  # Access the subgraph
        #         sub_graph.x[sub_graph.supervision_indices, 0] = torch.zeros_like(sub_graph.x[sub_graph.supervision_indices, 0])  # Check all feature dimensions

        self.values, self.errors, self.predictions = self.unpack_features(data.x, reshape=True)
        tmp = self.values.view(self.batch_size,-1)[0, -10:]
        # print("one hot 1st batch", tmp)
        # print("one hot 1st batch", tmp)
        
        self.update_xs(train=False)
        # logits = self.values[:, data.supervision_indices[0]]
        
        logits = self.values.view(self.batch_size,-1)[:, -10:]

        adjustment = tmp - logits
        assert adjustment != torch.zeros_like(adjustment)
        # print("adjustment", adjustment)
        # print("logits shape", logits.shape)
        # print("log sum", logits.sum())
        # print(logits)

        y_pred = torch.argmax(logits, axis=1).squeeze()
        return y_pred.cpu().detach()
    
    def get_energy(self):
        return torch.sum(self.errors**2).item()

    def get_errors(self):
        return self.e.clone()
