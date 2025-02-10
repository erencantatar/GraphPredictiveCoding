
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


class PC_graph_zwol(torch.nn.Module): 

    def __init__(self, f, device, num_vertices, mask,
                 lr_x, T_train, T_test, 
                 incremental, use_input_error,
                 node_init_std=None,
                 min_delta=None,
                 early_stop=None,
                 ):
        super().__init__()  # 'add' aggregation
        self.num_vertices = num_vertices
        

        self.lr_x = lr_x 
        self.T_train=T_train
        self.T_test=T_test
        self.node_init_std = node_init_std
        self.incremental=incremental 
        self.min_delta=min_delta
        self.early_stop=early_stop
        self.mask = mask

        self.f = f
        self.dfdx = get_derivative(f)
        self.use_input_error = use_input_error 
       
        # self.lr_values , self.lr_weights = learning_rate  

        # self.T = T[0]  # Number of iterations for gradient descent (T_train, T_test)
        # self.T_train, self.T_test = T  # Number of iterations for gradient descent (T_train, T_test) 

        # iF incremental for now ony use PCG_AMB update rules pred (mu) = wf(x) + b  
        self.incremental_learning = incremental  # False ]/True"standard" or "incremental" PC 

        # self.edge_index_single_graph = graph_structure  # a geometric graph structure

        # self.task = None
        # self.device = device
        # self.weight_init = weight_init
        # self.internal_clock = 0
        # self.clamping = clamping
        # self.wandb_logger = wandb_logger


        # self.adj = to_dense_adj(self.edge_index_single_graph).squeeze(0).to(self.device)
        # # transpose the adj matrix; to match 
        # self.mask = self.adj.t()


        # if self.debug:
        #     ic.enable()
        # else: 
        #     ic.disable()
        # assert num_vertices == len(sensory_indices) + len(internal_indices), "Number of vertices must match the sum of sensory and internal indices"

        # self.batchsize = batch_size
        # TODO use torch.nn.Parameter instead of torch.zeros
        
        

        # Initialize grads for weight updates only
        # self.grads = {"weights": None}

        # for grokfast optionally 
        # self.grads = {
        #     # "values" : None,
        #     "weights": None, 
        # }

        self.device = device 

        self._reset_grad()
        self._reset_params()

        self.use_bias = False

        print(type(self.mask))
        print(type(self.w))
        self.w = self.mask * self.w 


        """ ------------------------------------------- INFO -------------------------------------------
        - VanZwol:        values and errors are (batch_size, num_vertices) and weights are (num_vertices, num_vertices)
        - Salvatori:      using MessagePassing where values and errors are both shape (num_vertices * batch_size, 1) as given by the PYG dataloader
        and by using the edge_index (extented to batch_size) to get the correct values,errors for each subgraph in the batch.       
        weights are (num_edges)
        - vectorized:             
        """

    
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


    def get_prediction(self):
        # bias = b if self.use_bias else 0
        bias = 0
        mu = torch.matmul(self.f(self.x), self.w.T) + bias
        return mu

    def print_debug(self, out, debug=False):
        if self.debug or debug:
            print(out)

    def grad_x(self, train):
        lower = 784
        # upper = -self.shape[2] if train else sum(self.num_vertices)
        upper = -10 if train else self.num_vertices
        # self.print_debug("from", lower, upper)
        gradx = self.e[:,lower:upper] - self.dfdx(self.x[:,lower:upper]) * torch.matmul(self.e, self.w.T[lower:upper,:].T)
        # self.print_debug("shape gradx", gradx.shape)
        return gradx
    
    def update_xs(self, train=True):
        # if self.early_stop:
        #     early_stopper = EarlyStopper(patience=0, min_delta=self.min_delta)

        di = 784
        upper = -10 if train else self.num_vertices
        # self.print_debug("di, upper", di, upper)
        T = self.T_train if train else self.T_test
        
        self.di = di
        self.upper = upper
        # print("di, upper", di, upper)

        for t in range(T): 

            self.e = self.x - self.get_prediction()
            if not self.use_input_error:
                self.e[:,:di] = 0 
            
            dEdx = self.grad_x(train) # only hidden nodes
            # print("dEdx shape", dEdx.shape)
            # print("self.x sjape", self.x[:,di:upper].shape)
            self.x[:,di:upper] -= self.lr_x*dEdx 

            if self.incremental and self.dw is not None:
                self.optimizer.step(self.params, self.grads, batch_size=self.x.shape[0])
                
            # if self.early_stop:
            #     if early_stopper.early_stop( self.get_energy() ):
            #         break            

    

    def onehot(self, y_batch, N):
        """
        y_batch: tensor of shape (batch_size, 1)
        N: number of classes
        """
        return torch.eye(N, device=self.device)[y_batch.squeeze().long()].float()

    def to_vector(self, batch):
        batch_size = batch.size(0)
        return batch.reshape(batch_size, -1).squeeze()


    def reset_nodes(self, batch_size=1):
        # self.e = torch.empty(batch_size, sum(self.structure.shape), device=self.device)
        # self.x = torch.zeros(batch_size, sum(self.structure.shape), device=self.device)
        self.e = torch.empty(batch_size, (self.num_vertices), device=self.device)
        self.x = torch.zeros(batch_size, (self.num_vertices), device=self.device)

    def clamp_input(self, inp):
        di = 784
        self.x[:,:di] = inp.clone()

    def clamp_target(self, target):
        do = 10
        self.x[:,-do:] = target.clone()
        

    def train_supervised(self, X_batch, y_batch): 
        self.debug = False
        X_batch = self.to_vector(X_batch)                  # makes e.g. 28*28 -> 784
        y_batch = self.onehot(y_batch, N=10)    # makes e.g. 3 -> [0,0,0,1,0,0,0,0,0,0]

        self.reset_nodes(batch_size=X_batch.shape[0])        
        self.clamp_input(X_batch)
        # self.init_hidden()
        # print("ommit init hidden")
        self.clamp_target(y_batch)

        self.update_xs(train=True)
        self.update_w()

        if not self.incremental:
            self.optimizer.step(self.params, self.grads, batch_size=X_batch.shape[0])


    # def test_iterative(self, X_batch, diagnostics=None, early_stop=False):
    def test_iterative(self, X_batch):
        X_batch = self.to_vector(X_batch)     # makes e.g. 28*28 -> 784

        self.reset_nodes(batch_size=X_batch.shape[0])
        self.clamp_input(X_batch)
        # self.init_hidden()
        # self.init_output()

        self.update_xs(train=False)
        # self.update_xs(train=False, diagnostics=diagnostics, early_stop=early_stop)
        # return self.x[:,-self.structure.shape[2]:] 
        return self.x[:,-10:] 
    

    def update_w(self):


        # self.dw = self.structure.grad_w(x=self.x, e=self.e, w=self.w, b=self.b)
        out = -torch.matmul(self.e.T, self.f(self.x))
        if self.mask is not None:
            out *= self.mask
        self.dw = out
    

        # def test_iterative(self, X_batch, diagnostics=None, early_stop=False):
   
    def get_weights(self):
        return self.w.clone()

    def get_energy(self):
        return torch.sum(self.e**2).item()

    def get_errors(self):
        return self.e.clone()    