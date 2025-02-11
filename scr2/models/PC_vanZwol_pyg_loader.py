
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



class PC_graph_zwol_PYG(torch.nn.Module): 

    def __init__(self, f, device, num_vertices, num_internal, adj, edge_index, batch_size, lr_x, T_train, T_test, incremental, use_input_error, node_init_std=None, min_delta=None, early_stop=None):
        super().__init__()

        self.device = device

        self.num_vertices = num_vertices
        self.num_internal = num_internal
        self.adj = torch.tensor(adj).to(self.device)
        
        self.edge_index = edge_index  # PYG edge_index

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
        self.mode == "train"
        self.update_mask = self.update_mask_train

    def test(self):
        self.mode == "test"
        self.update_mask = self.update_mask_test


    def init_modes(self, batch_example):
        
        sensory_indices, supervision_indices, internal_indices = batch_example.sensory_indices, batch_example.supervision_indices, batch_example.internal_indices, 
        
        self.sensory_indices = torch.tensor(sensory_indices, device=self.device)[0].flatten()

        internal_indices = torch.tensor(internal_indices, device=self.device)[0].flatten()
        supervision_indices = torch.tensor(supervision_indices, device=self.device)[0].flatten()
  
        self.update_mask_train = internal_indices 
        self.update_mask_train = torch.tensor(self.update_mask_train)
        self.update_mask_train = self.update_mask_train.clone().detach()

        self.update_mask_test = torch.cat([internal_indices, supervision_indices])
        self.update_mask_test = torch.tensor(self.update_mask_test)
        self.update_mask_test = self.update_mask_test.clone().detach()

        

    def unpack_features(self, batch, reshape=True):
        """Unpack values, errors, and predictions from the batched graph."""
        values, errors, predictions = batch[:, 0, :].to(self.device), batch[:, 1, :].to(self.device),  None
        # print("unpacked featreus")
        # print(values.shape)

        # reshape to (batch_size, num_vertices)
        if reshape:
            values      = values.view(self.batch_size, self.num_vertices)
            errors      = errors.view(self.batch_size, self.num_vertices)
            # predictions = predictions.view(self.batch_size, self.num_vertices)

        return values, errors, predictions

    def get_prediction(self, values):
        
        bias = 0
        prediction = torch.matmul(self.f(values), self.w.T) + bias

        return prediction


    def grad_x(self):
        """Use message passing to compute dEdx."""
        # lower = 784
        # # upper = -self.shape[2] if train else sum(self.num_vertices)
        # upper = -10 if train else self.num_vertices

        gradx = self.errors[:,self.update_mask] - self.dfdx(self.values[:,self.update_mask]) * torch.matmul(self.errors, self.w.T[self.update_mask,:].T)
        
        # print("gradx shape:", gradx.shape)
        return gradx


    def update_xs(self, train=True):
        T = self.T_train if train else self.T_test

        # Move all relevant tensors to self.device
       
        # if not train:
        #     print("1", self.values[:, -10:])
        # take the first 

        # print("update_mask shape", self.update_mask.shape)

        for t in range(T):
            self.prediction = self.get_prediction(self.values).to(self.device)
      
            self.errors = self.values - self.prediction  # Update errors
            self.e = self.errors 

            if not self.use_input_error:
                self.errors[:, self.sensory_indices] = 0

            # print("errors mean", self.errors.mean())

            # Determine which nodes to update
            dEdx = self.grad_x()

            # print("dEdx shape", dEdx.shape)
            # print("values shape", values.shape)
            self.values[:, self.update_mask] -= self.lr_x * dEdx

            # if not train:
                # print("2", self.values[:, -10:])

            if self.incremental and self.dw is not None:
                # print("dw", self.dw.shape)
                # print(self.params["w"].shape)
                self.optimizer.step(self.params, self.grads, batch_size=self.batch_size)
                # print("w mean", self.params["w"].mean())
                # print("dw mean:", self.dw.mean().item(), "dw max:", self.dw.max().item(), "dw min:", self.dw.min().item())


    def update_w(self):
        # self.dw = -torch.matmul(errors.T, self.f(values))
        # self.dw = self.structure.grad_w(x=self.x, e=self.e, w=self.w, b=self.b)
        # self.errors = self.errors.view(self.batch_size, self.num_vertices)
        # self.values = self.values.view(self.batch_size, self.num_vertices)

        # print("errors mean", self.errors.mean())
        # print("values mean", self.values.mean())

        out = -torch.matmul(self.errors.T, self.f(self.values)).to(self.device)
        
        # print("out shape", out.shape)
        # print("w", self.w.shape)
        if self.adj is not None:
            out *= self.adj
        self.dw = out
        # print("dw mean 2:", self.dw.mean().item(), "dw max:", self.dw.max().item(), "dw min:", self.dw.min().item())

        # print("self.dw shape", out.shape)

    def train_supervised(self, data):
        # edge_index = data.edge_index.to(self.device)

        # self.data_ptr = data.ptr
        self.batch_size = data.x.shape[0] // self.num_vertices

        self.values, self.errors, self.predictions = self.unpack_features(data.x, reshape=True)
        
        self.update_xs(train=True)
        self.update_w()

        if not self.incremental:
            self.optimizer.step(self.params, self.grads, batch_size=self.batch_size)

        # print("w mean", self.w.mean())

    def test_iterative(self, data, remove_label=True):
        # edge_index = data.edge_index

        # remove one_hot
        if remove_label:
            for i in range(len(data)):
                sub_graph = data[i]  # Access the subgraph
                sub_graph.x[sub_graph.supervision_indices, 0] = torch.zeros_like(sub_graph.x[sub_graph.supervision_indices, 0])  # Check all feature dimensions

        self.values, self.errors, self.predictions = self.unpack_features(data.x, reshape=True)
        
        # print("0", self.values[:, -10:].shape, self.values[:, -10:])

        self.update_xs(train=False)
        # logits = self.values[:, data.supervision_indices[0]]
        logits = self.values[:, -10:]

        y_pred = torch.argmax(logits, axis=1).squeeze()
        return y_pred.cpu().detach()
    
    def get_energy(self):
        return torch.sum(self.errors**2).item()

    def get_errors(self):
        return self.e.clone()
