
from re import T
import torch
import torch.nn as nn  # Ensure that `torch.nn` is only imported once

import torch.nn.init as init
import numpy as np
import math
from torch_geometric.utils import to_dense_adj, degree
# from models.MessagePassing import PredictionMessagePassing, ValueMessagePassing
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
