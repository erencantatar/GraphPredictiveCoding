
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

class PredictionMessagePassing(MessagePassing):
    def __init__(self, aggr="add", f=torch.tanh):
        super().__init__(aggr=aggr)  # Aggregation method (e.g., 'add', 'mean')
        self.f = f

    def forward(self, x, edge_index, w):
        """ Message passing for predictions. """
        return self.propagate(edge_index, x=x, w=w)

    def message(self, x_j, w):
        return w * self.f(x_j)

    def aggregate(self, inputs, index):
        return scatter(inputs, index, dim=0, reduce=self.aggr)

class GradXMessagePassing(MessagePassing):
    def __init__(self, aggr="add", dfdx=torch.sigmoid):
        super().__init__(aggr=aggr)
        self.dfdx = dfdx  # Activation derivative

    def forward(self, e, edge_index, w):
        """ Message passing for computing dEdx. """
        return self.propagate(edge_index, x=e, w=w)

    def message(self, x_j, w):
        return w * self.dfdx(x_j)

    def aggregate(self, inputs, index):
        return scatter(inputs, index, dim=0, reduce=self.aggr)


class PC_graph_zwol_PYG(torch.nn.Module): 

    def __init__(self, f, device, num_vertices, edge_index, lr_x, T_train, T_test, incremental, use_input_error, node_init_std=None, min_delta=None, early_stop=None):
        super().__init__()
        self.num_vertices = num_vertices
        self.edge_index = edge_index  # PYG edge_index

        self.lr_x = lr_x 
        self.T_train = T_train
        self.T_test = T_test
        self.node_init_std = node_init_std
        self.incremental = incremental 
        self.min_delta = min_delta
        self.early_stop = early_stop

        self.f = f
        self.dfdx = get_derivative(f)
        self.use_input_error = use_input_error
        self.device = device

        # self.w = nn.Parameter(torch.empty(num_vertices, num_vertices, device=self.device))
        # self.b = nn.Parameter(torch.empty(num_vertices, device=self.device))


        self.device = device 

        self._reset_grad()
        self._reset_params()

        self.use_bias = False

        # print(type(self.mask))
        # print(type(self.w))
        # self.w = self.mask * self.w 

        # Message-passing layers
        self.prediction_mp = PredictionMessagePassing(aggr="add", f=f)
        self.gradx_mp = GradXMessagePassing(aggr="add", dfdx=self.dfdx)
        # self.reset_nodes()
    
    @property
    def params(self):
        return {"w": self.w, "b": self.b, "use_bias": self.use_bias}
    
    @property
    def grads(self):
        return {"w": self.dw, "b": self.db}

    def _reset_params(self):
        self.w = torch.empty( self.num_vertices, self.num_vertices, device=self.device)
        self.b = torch.empty( self.num_vertices, device=self.device)

        nn.init.normal_(self.w, mean=0, std=0.05)   

    def _reset_grad(self):
        self.dw, self.db = None, None


    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    # def reset_nodes(self, batch_size=1):
    #     self.e = torch.empty(batch_size, sum(self.structure.shape), device=DEVICE)
    #     self.x = torch.zeros(batch_size, sum(self.structure.shape), device=DEVICE)

    # def reset_nodes(self, total_nodes=1):
        
    #     self.x = torch.zeros(total_nodes, self.num_vertices, device=self.device)
    #     self.e = torch.zeros_like(self.x)
    
    def unpack_features(self, batch):
        """Unpack values, errors, and predictions from the batched graph."""
        values, errors, predictions = batch[:, 0, :], batch[:, 1, :], batch[:, 2, :]
        print("unpacked featreus")
        print(values.shape)
        return values, errors, predictions

    def get_prediction(self, values, edge_index):
        """Use message passing to get predictions from neighbors."""
        w_expanded = self.w[edge_index[0], edge_index[1]].unsqueeze(1)
        prediction = self.prediction_mp(values, edge_index, w_expanded)

        print("prediction shape", prediction.shape)
        return prediction

    def grad_x(self, errors, edge_index):
        """Use message passing to compute dEdx."""
        w_expanded = self.w[edge_index[0], edge_index[1]].unsqueeze(1)
        print("errors shape", errors.shape)
        gradx = errors - self.gradx_mp(errors, edge_index, w_expanded)
        return gradx

    def update_xs(self, edge_index, values, errors, sensory_indices, supervision_indices, internal_indices, train=True):
        T = self.T_train if train else self.T_test

        for t in range(T):
            prediction = self.get_prediction(values, edge_index)
            print("values shape,", values.shaoe)
            print("pred shape", prediction.shape)

            errors[:] = values - prediction  # Update errors
            print("errors", errors)

            # if not self.use_input_error:
            #     errors[sensory_indices] = 0

            # Determine which nodes to update
            update_mask = internal_indices if train else torch.cat([internal_indices, supervision_indices])

            dEdx = self.grad_x(errors, edge_index)
            values[update_mask] -= self.lr_x * dEdx[update_mask]

    def update_w(self, values, errors):
        self.dw = -torch.matmul(errors.T, self.f(values))
        self.w.data.add_(self.dw)

    def train_supervised(self, data):
        edge_index = data.edge_index
        values, errors, predictions = self.unpack_features(data.x)

        self.update_xs(edge_index, values, errors, data.sensory_indices, data.supervision_indices, data.internal_indices, train=True)
        self.update_w(values, errors)

    def test_iterative(self, data):
        edge_index = data.edge_index
        values, errors, predictions = self.unpack_features(data.x)

        self.update_xs(edge_index, values, errors, data.sensory_indices, data.supervision_indices, data.internal_indices, train=False)
        return values[data.supervision_indices]

    def get_energy(self):
        return torch.sum(self.e**2).item()

    def get_errors(self):
        return self.e.clone()
