
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

import torch.optim.lr_scheduler

class PCGraphConv(torch.nn.Module): 
    def __init__(self, num_vertices, sensory_indices, internal_indices, 
                 learning_rate, T, graph_structure,
                 batch_size, edge_type, incremental_learning, update_rules, delta_w_selection, use_bias=False, use_learning_optimizer=False, 
                 use_grokfast=False,
                 weight_init="normal", clamping=None,  
                 supervised_learning=False, normalize_msg=False, debug=False, activation=None, 
                 log_tensorboard=True, wandb_logger=None, device="cpu"):
        super(PCGraphConv, self).__init__()  # 'add' aggregation
        self.num_vertices = num_vertices
        self.edge_type = edge_type
        
        # these are fixed and inmutable, such that we can copy 
        self.sensory_indices_single_graph = sensory_indices
        self.internal_indices_single_graph = internal_indices
        self.supervised_labels_single_graph = supervised_learning 
        
        # init, but these are going to updated depending on batchsize
        self.sensory_indices = self.sensory_indices_single_graph
        self.internal_indices = self.internal_indices_single_graph
        self.supervised_labels = self.supervised_labels_single_graph 
        
        self.lr_values , self.lr_weights = learning_rate  

        self.T = T[0]  # Number of iterations for gradient descent (T_train, T_test)
        self.T_train, self.T_test = T  # Number of iterations for gradient descent (T_train, T_test) 

        # iF incremental for now ony use PCG_AMB update rules pred (mu) = wf(x) + b  
        self.incremental_learning = incremental_learning  # False ]/True"standard" or "incremental" PC 

   
   
        self.debug = debug
        self.edge_index_single_graph = graph_structure  # a geometric graph structure
        self.mode = ""

        self.delta_w_selection = delta_w_selection
        print("--------delta_w_selection-------------", self.delta_w_selection)

        self.task = None
        self.device = device
        self.weight_init = weight_init
        self.internal_clock = 0
        self.clamping = clamping
        self.wandb_logger = wandb_logger

        self.trace_activity_values, self.trace_activity_errors, self.trace_activity_preds = False, False, False  
        self.trace = {
            "values": [], 
            "errors": [],
            "preds" : [],
        }


        self.energy_vals = {
            # training
            "internal_energy": [],
            "sensory_energy": [],
            "supervised_energy": [], 

            "mean_internal_energy_sign": [],
            "mean_sensory_energy_sign": [],
            "mean_supervised_energy_sign":[],
     
            'energy_drop': [],
            'weight_update_gain': [],

            # testing 
            "internal_energy_testing": [],
            "sensory_energy_testing": [],
            "supervised_energy_testing": [],
    
            "internal_energy_batch": [],
            "sensory_energy_batch": [],

            "energy_t0": [],
            "energy_tT": [],
        }

        self.w_log = []

        # Metrics for energy drop and weight update gain
        self.energy_metrics = {
            'internal_energy_t0': None,
            'internal_energy_tT': None,
            'internal_energy_tT_plus_1': None,
            'energy_drop': None,
            'weight_update_gain': None,
        }

        self.gradients_minus_1 = 1 # or -1 
        # self.gradients_minus_1 = -1 # or -1 #NEVER 

        print("------------------------------------")
        print(f"gradients_minus_1: x and w.grad += {self.gradients_minus_1} * grad")
        print("------------------------------------")

        self.data = None 
        self.values_at_t = []
        
        self.TODO = """ 
                    - think about removing self loops
                    - requires_grad=False for values and errors
                    - understand aggr_out_i, aggr_out is the message passed to the update function
                    """
        self.log = {"Zero_pred" : [], }   # Zero prediction error encountered for node {vertex}

        # if self.debug:
        #     ic.enable()
        # else: 
        #     ic.disable()
        # assert num_vertices == len(sensory_indices) + len(internal_indices), "Number of vertices must match the sum of sensory and internal indices"

        self.batchsize = batch_size
        # TODO use torch.nn.Parameter instead of torch.zeros
        
        self.use_optimizers = use_learning_optimizer
        

        # Initialize grads for weight updates only
        # self.grads = {"weights": None}

        # for grokfast optionally 
        # self.grads = {
        #     # "values" : None,
        #     "weights": None, 
        # }

        self.grads = None

        self.grokfast_type = "ema"  # "ema" or "ma"
        self.avg_grad = None

        self.use_grokfast = use_grokfast
        print(f"----- using grokfast: {self.use_grokfast}")

        self.use_bias = use_bias 
        print(f"----- using use_bias: {self.use_bias}")

        if self.wandb_logger:
            self.wandb_logger.log({"use_bias": self.use_bias})
            self.wandb_logger.log({"use_grokfast": self.use_grokfast})

        
        self.grad_accum_method = "mean" 
        assert self.grad_accum_method in ["sum", "mean"]
        
        # Initialize weights with uniform distribution in the range (-k, k)

        # ------------- init weights -------------------------- 

        # https://pytorch.org/docs/stable/generated/torch.nn.Linear.html

        # self.update_rules = "Van_Zwol"  # "Van_Zwol" or "salvatori "
        # self.update_rules = "Salvatori"  # "Van_Zwol" or "salvatori " salvatori tommaso
        # self.update_rules = "vectorized"  # "Van_Zwol" or "salvatori " salvatori tommaso

        self.update_rules = update_rules
        if self.wandb_logger:
            self.wandb_logger.log({"update_rules": self.update_rules})
        
        """ ------------------------------------------- INFO -------------------------------------------
        - VanZwol:        values and errors are (batch_size, num_vertices) and weights are (num_vertices, num_vertices)
        - Salvatori:      using MessagePassing where values and errors are both shape (num_vertices * batch_size, 1) as given by the PYG dataloader
        and by using the edge_index (extented to batch_size) to get the correct values,errors for each subgraph in the batch.       
        weights are (num_edges)
        - vectorized:             
        """


        if self.update_rules == "Van_Zwol" or self.update_rules == "vectorized":

            if self.update_rules == "vectorized":
                print("using vectorized update rules")
            else:
                print("Using Van Zwol update rules with mu = f(wx+b) (optional bias) without MessagePassing " )
            
            print("weights shape", self.num_vertices, self.num_vertices)
            # Initialize weights as a parameter with correct shape and device
          
            # self.weights = torch.nn.Parameter(
            #     torch.zeros(self.num_vertices, self.num_vertices, device=self.device, requires_grad=True)
            # )
            self.weights = torch.empty(self.num_vertices, self.num_vertices, device=self.device)
            self.delta_w = None

            nn.init.normal_(self.weights, mean=0, std=0.05)   

            self.lr_w = 0.00001              # learning rate
            weight_decay = 0             
            grad_clip = 1
            batch_scale = False
            self.dw, self.db = None, None

            self.b = torch.empty( self.num_vertices, device=self.device)

            self.params = {"w": self.weights, "b": self.b, "use_bias": False}

            self.van_zwol_optimizer_weights = optim_VanZwol.Adam(
                self.params,
                learning_rate=self.lr_w,  # lr_w  / lr_weights
                grad_clip=grad_clip,
                batch_scale=batch_scale,
                weight_decay=weight_decay,
            )
            
            #     def weight_init(self, param):
            #     nn.init.normal_(param, mean=0, std=0.05)   

            # def bias_init(self, param):
            #     nn.init.normal_(param, mean=0, std=0) 

            self.biases = torch.nn.Parameter(torch.zeros(self.batchsize * self.num_vertices, device=self.device), requires_grad=False) # requires_grad=False)                
            nn.init.normal_(self.biases, mean=0, std=0) 
            
        elif self.update_rules == "Salvatori":
            print("Using Salvatori update rules with mu = (wf(x)+b) (optional bias) with Pytorch geometric MessagePassing")

            self.prediction_mp  = PredictionMessagePassing(activation=activation)
            self.values_mp      = ValueMessagePassing(activation=activation)

            # USING BATCH SIZE, we want the same edge weights at each subgraph of the batch
            self.weights = torch.nn.Parameter(torch.zeros(self.edge_index_single_graph.size(1), device=self.device, requires_grad=True))
            # init.uniform_(self.weights.data, -k, k)
            
            # if self.use_bias:
            #     self.biases = torch.nn.Parameter(torch.zeros(self.batchsize * self.num_vertices, device=self.device), requires_grad=False) # requires_grad=False)                
            #     # TODO 
            # print("INIT BIAS WITH THE MEAN OF THE DATASET???")
            # self.biases..data.fill_(0.01) 
            # self.biases.data = torch.full_like(self.biases.data, 0.01)




        # Weight initialization
        init_type, m, std_val = weight_init.split()

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

            nn.init.normal_(self.weights, mean=m, std=std_val)   
            
        elif init_type == "uniform" or init_type == "nn.linear":
            k = 1 / math.sqrt(num_vertices)
            nn.init.uniform_(self.weights, -k, k)
        elif init_type == "fixed":
            value = float(m) if m else 0.1
            self.weights.data.fill_(value)


            # noise_std = 0.005  # Standard deviation of the noise
            # noise_std = 0  # Standard deviation of the noise

            # Fill with fixed value
            # self.weights.data.fill_(value)

            # Add small random noise
            noise = torch.randn_like(self.weights) * std_val
            # TODO
            # noise = torch.ones_Like(self.weights) * std_val
            self.weights.data.add_(noise)

        ### Apply mask to weights
        if self.update_rules == "Van_Zwol":
        
            # Convert edge index to dense adjacency matrix and apply to weights
            self.adj = to_dense_adj(self.edge_index_single_graph).squeeze(0).to(self.device)

            # transpose the adj matrix; to match 
            self.adj = self.adj.t()
            # self.adj = self.adj
            
            with torch.no_grad():
                self.weights.data *= self.adj

        
        if self.wandb_logger:
            # log the weights init and params
            self.wandb_logger.log({"Weights/weights_init": weight_init})
            self.wandb_logger.log({"Weights/weight_m": m})
            self.wandb_logger.log({"Weights/weight_std": std_val})


        # Bias initialization (if use_bias is enabled)
        if self.use_bias:

            nn.init.normal_(self.biases, mean=0, std=0) 

            # dataset_mean = 0.1307  # Replace with actual dataset mean (0.1307 MNIST)
            # self.biases.data.fill_(dataset_mean)

            # raise NotImplementedError 
            # bias_type, *bias_params = bias_init.split()
            # if bias_type == "normal":
            #     mean = float(bias_params[0]) if bias_params else 0.0
            #     std = max(0.01, abs(mean) * 0.1)
            #     nn.init.normal_(self.biases, mean=mean, std=std)
            # elif bias_type == "uniform":
            #     k = 1 / math.sqrt(num_vertices)
            #     nn.init.uniform_(self.biases, -k, k)
            # elif bias_type == "fixed":
            #     value = float(bias_params[0]) if bias_params else 0.0
            #     self.biases.data.fill_(value)

        # self.w_t_min_1 = self.weights.clone()

        # https://chatgpt.com/c/0f9c0802-c81b-40df-8870-3cea4d2fc9b7

        # graph object is stored as one big graph with one edge_index vector with disconnected sub graphs as batch items 
        self.values_dummy = torch.nn.Parameter(torch.zeros(self.batchsize, self.num_vertices, device=self.device), requires_grad=True) # requires_grad=False)                
        self.values = None
        self.errors = None
        self.predictions = None  

        self.optimizer_values, self.optimizer_weights = None, None 
        
        

        self.scheduler_weights = None

        # self.use_optimizers = False

        print("check ______________--", self.use_optimizers)
        # true if use_optimizers is a list with a float or int as the first element; and if so take the first element as the weight_decay

        # Check if self.use_optimizers is a list and its first element is a valid float or int (not False or similar invalid values)
        if isinstance(self.use_optimizers, list) and len(self.use_optimizers) > 0 and isinstance(self.use_optimizers[0], (float, int)) and self.use_optimizers[0] is not False:
            weight_decay = self.use_optimizers[0]
            
            # weight_decay = self.use_optimizers[0]

            print(f"------------Using optimizers with weight_decay {weight_decay} ------------")
            
            # paper: 
            # SGD configured as plain gradient descent

            # if overfitting is a risk or bigger more complex model, use AdamW
            self.optimizer_weights = torch.optim.AdamW([self.weights], lr=self.lr_weights, weight_decay=weight_decay)      
            # self.optimizer_weights = torch.optim.Adam([self.weights], lr=self.lr_weights, weight_decay=weight_decay)      
            self.optimizer_values = torch.optim.SGD([self.values_dummy], lr=self.lr_values, momentum=0, weight_decay=weight_decay, nesterov=False) # nestrov only for momentum > 0

            self.weights.grad = torch.zeros_like(self.weights)
            self.values_dummy.grad = torch.zeros_like(self.values_dummy)

            self.lr_scheduler = False        
            if self.lr_scheduler:
                self.scheduler_weights = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_weights, mode='min', patience=10, factor=0.1)

            # log wandb that using optim in delta_w/
            if self.wandb_logger:
                self.wandb_logger.log({"Using optimizer for delta_w": True})
        else:
            self.use_optimizers = False
            print("------------Not using optimizers ------------")

        self.effective_learning = {}
        self.effective_learning["w_mean"] = []
        self.effective_learning["w_max"] = []
        self.effective_learning["w_min"] = []


        self.effective_learning["v_mean"] = []
        self.effective_learning["v_max"] = []
        self.effective_learning["v_min"] = []

        self.global_step = 0 # Initialize global step counter for logging

        self.use_convergence_monitor = False 
        if self.use_convergence_monitor:
            print(f"----- using use_convergence_monitor: {self.use_convergence_monitor}")
            from helper.converge_monitor import CombinedConvergence, AdaptiveEnergyConvergence, GradientEnergyConvergence
            self.convergence_tracker = CombinedConvergence(energy_threshold=0.05, gradient_threshold=1, patience=10)
            self.gradients_log = torch.zeros_like(self.weights) # or and None

        # used for when training with T until convergence 
        self.T_MAX = 5 * self.T
        
        # Create a vector for edge-type-specific learning rates
        self.adjust_delta_w = False 
        print("Using self.adjust_delta_w, ", self.adjust_delta_w)
        if self.adjust_delta_w:
            scaling_factors = torch.zeros_like(self.weights)
            scaling_factors[self.edge_type == 0] = 0.1 / 0.656775   # Sens2Sens
            scaling_factors[self.edge_type == 1] = 0.1 / 0.009169   # Sens2Inter
            scaling_factors[self.edge_type == 2] = 0.1 / 9.061984   # Sens2Sup
            scaling_factors[self.edge_type == 3] = 0.1 / 0.0006583   # Inter2Sens
            scaling_factors[self.edge_type == 4] = 0.1 / 0.00007534  # Inter2Inter
            scaling_factors[self.edge_type == 5] = 0.1 / 0.0074454   # Inter2Sup
            scaling_factors[self.edge_type == 6] = 0.1 / 7.248817   # Sup2Sens
            scaling_factors[self.edge_type == 7] = 0.1 / 0.100325   # Sup2Inter
            scaling_factors[self.edge_type == 8] = 0.1 / 0.095290   # Sup2Sup
            self.lr_by_edtype = scaling_factors

        # if self.wandb_logger:
            
            # self.wandb_logger.watch(self, log="all", log_freq=100)  # Log all gradients and parameters

            # watch the parameters weights and 
            # self.wandb_logger.watch(self.weights, log="all", log_freq=40)
            
        # 2. during training set batch.e

        # k = 1.0 / num_vertices ** 0.5
        # init.uniform_(self.weights, -k, k)
            
        # using graph_structure to initialize mask Data(x=x, edge_index=edge_index, y=label)
        # self.mask = self.initialize_mask(graph_structure)
        
        print("normalize_msg", normalize_msg)
        if normalize_msg:

            edge_index_batch = torch.cat([self.edge_index_single_graph for _ in range(batch_size)], dim=1)

            # Compute normalization for the entire batch edge_index, num_nodes, device):
            self.norm = self.compute_normalization(edge_index_batch, self.num_vertices * batch_size, device)

            print(self.edge_index_single_graph.shape)
            print(self.norm.shape)
            
            assert self.norm.shape[0] == self.edge_index_single_graph.shape[1], "Norm shape must match the number of edges"
            # self.norm_single_batch = self.compute_normalization(self.edge_index_single_graph, self.num_vertices, self.device)
            print("-----compute_normalization-----")
        else:
            self.norm_single_batch = torch.ones(self.edge_index_single_graph.size(1))
            # self.norm = self.norm_single_batch.repeat(1, self.batchsize).to(self.device)
            
            
            self.norm = torch.tensor(1)

        self.norm = self.norm.cpu()

        # Apply mask to weights
        # self.weights.data *= self.mask

    

        self.set_phase('initialize') # or 'weight_update'
         
        if activation:
            self.f, self.f_prime = set_activation(activation)
            set_activation(activation)
            tmp = f'Activation func set to {activation}'
            self.set_phase(tmp) # or 'weight_update'
            # for now MUST SET Activation function before calling self.prediction_msg_passing
        assert activation, "Activation function not set"

        self.t = ""

        # Find edges between sensory nodes (sensory-to-sensory)
        # self.s2s_mask = (self.edge_index_single_graph[0].isin(self.sensory_indices)) & (self.edge_index_single_graph[1].isin(self.sensory_indices))

        # sensory_indices_set = set(self.sensory_indices_single_graph)
        # s2s_mask = [(src in sensory_indices_set and tgt in sensory_indices_set) 
        #             for src, tgt in zip(self.edge_index_single_graph[0], self.edge_index_single_graph[1])]

        # # Convert mask to tensor (if needed)
        # self.s2s_mask = torch.tensor(s2s_mask, dtype=torch.bool, device=self.device)

        # if activation == "tanh":
        #     print("Since using tanh, using xavier_normal_")
        #     init.xavier_normal_(self.weights)   # or xavier_normal_ / xavier_uniform_
        
        # elif activation == "relu":
        #     # kaiming_uniform_ or kaiming_normal_
        #     print("Since using relu, using kaiming_normal_")
        #     init.kaiming_normal_(self.weights)   # or xavier_normal_ / xavier_uniform_
        # else:
        #     print("Std random weight initialization --> xavier_uniform_")
        #     init.xavier_uniform_(self.weights)   # or xavier_normal_ / xavier_uniform_


    # def initialize_mask(self, edge_index):
    #     """Initialize the mask using the graph_structure's edge_index."""
    #     self.set_phase("Initialize the mask using the graph_structure's edge_index")

    #     # mask = torch.zeros(self.num_vertices, self.num_vertices, dtype=torch.float32, device=self.weights.device)
    #     self.edge_index = edge_index
    #     # print("inint mask", edge_index.shape)

    #     # # Set mask to 1 for edges that exist according to edge_index
    #     # mask[edge_index[0], edge_index[1]] = 1

    #     from torch_geometric.utils import to_dense_adj

    #     # get structure
    #     mask = to_dense_adj(edge_index).squeeze(0) 
        
    #     if self.include_self_connections:
    #         print("Including self connections")
    #         mask = mask.fill_diagonal_(1)
    #     else:
    #         # Set the diagonal to zero to avoid self-connections
    #         mask = mask.fill_diagonal_(0)

    #     mask = mask.to(self.weights.device)
    #     return mask
    
        from torch_geometric.utils import degree

        self.gpu_cntr = 0 
        self.print_GPU = False 

    def log_activations(self, activations, edge_types):
        """
        Log activations for each edge type in a batch using wandb.
        Parameters:
        - activations: Tensor of activations corresponding to edges.
        - edge_types: Tensor of edge types for each edge in the batch.
        """
        if self.wandb_logger is None:
            return  # Skip logging if no wandb logger is provided

        # Check if the sizes of activations and edge_types match
        if activations.size(0) != edge_types.size(0):
            raise ValueError(
                f"Size mismatch: activations has size {activations.size(0)}, "
                f"but edge_types has size {edge_types.size(0)}"
            )
    
        edge_type_map = {
            0: "Sens2Sens", 1: "Sens2Inter", 2: "Sens2Sup", 
            3: "Inter2Sens", 4: "Inter2Inter", 5: "Inter2Sup", 
            6: "Sup2Sens", 7: "Sup2Inter", 8: "Sup2Sup"
        }

        # Prepare histograms for activations by edge type
        histograms = {}
        for etype, etype_name in edge_type_map.items():
            mask = edge_types == etype
            activations_etype = activations[mask]
            
            if activations_etype.numel() > 0:
                # Store histogram for logging
                histograms[f"activations/{etype_name}"] = wandb.Histogram(activations_etype.cpu().numpy())
        
        # Log all histograms in a single step
        self.wandb_logger.log(histograms)


    def log_gradients(self, log_histograms=False, log_every_n_steps=100):
        """
        Log the gradients of each layer after backward pass.
        Args:
            log_histograms (bool): If True, log gradient histograms (which is more time-consuming).
            log_every_n_steps (int): Log histograms every N steps to reduce overhead.
        """
        print("Logging gradients")
        if self.global_step != 0 and (self.global_step % log_every_n_steps == 0):
            grad_data = {"gradients": {}}
            for name, param in self.named_parameters():
                if param.grad is not None:
                    grad_data["gradients"][name] = {
                        "_grad_magnitude": param.grad.norm().item()
                    }

                    if log_histograms:
                        grad_hist = param.grad.cpu().numpy()
                        grad_data["gradients"][name]["_grad_distribution"] = wandb.Histogram(grad_hist)

            if self.wandb_logger:
                self.wandb_logger.log(grad_data)

        # Increment step counter
        self.global_step += 1

    def log_weights(self):
        print("Logging weights")
        """Log the weight matrix norms and distributions."""
        for name, param in self.named_parameters():
            if 'weight' in name:
                weight_norm = param.norm().item()
                weight_hist = param.cpu().detach().numpy()

                if self.wandb_logger:
                    # Log weight matrix norm
                    self.wandb_logger.log({f"{name}_weight_norm": weight_norm})
                    # Log weight matrix distribution
                    self.wandb_logger.log({f"{name}_weight_distribution": wandb.Histogram(weight_hist)})

     # Method to calculate energy drop and weight update gain
    def calculate_energy_metrics(self):
        """
        Calculate energy drop (from t=0 to t=T) and weight update gain (from t=T to t=T+1).
        """
        self.energy_vals['energy_drop'].append(self.energy_metrics['internal_energy_t0'] - self.energy_metrics['internal_energy_tT'])
        self.energy_vals['weight_update_gain'].append(self.energy_metrics['internal_energy_tT'] - self.energy_metrics['internal_energy_tT_plus_1'])
        
        print(f"Energy drop (t=0 to t=T): {self.energy_metrics['energy_drop']}")
        print(f"Weight update gain (t=T to t=T+1): {self.energy_metrics['weight_update_gain']}")

        print("test1")
        self.energy_vals["internal_energy_batch"] = np.mean(self.energy_vals["internal_energy"])
        self.energy_vals["sensory_energy_batch"] = np.mean(self.energy_vals["sensory_energy"])
        print("test1")
        

    def helper_GPU(self,on=False):
        if on:
            current_memory_allocated = torch.cuda.memory_allocated()
            current_memory_reserved = torch.cuda.memory_reserved()
            print(f"{self.gpu_cntr} current_memory_allocated", current_memory_allocated)
            print(f"{self.gpu_cntr} current_memory_reserved", current_memory_reserved)
            
            self.gpu_cntr += 1 

    def compute_normalization(self, edge_index, num_nodes, device):
        # Calculate degree for normalization
        row, col = edge_index
        deg = degree(col, num_nodes, dtype=torch.int16)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Using sparse matrix for normalization to reduce memory consumption
        # norm = torch.sparse_coo_tensor(edge_index, deg_inv_sqrt[row] * deg_inv_sqrt[col], size=(num_nodes, num_nodes)).to(device)

        return norm
    def initialize_weights(self, init_method):
        if init_method == "uniform":
            init.uniform_(self.weights, -0.1, 0.1)
        elif init_method == "normal":
            init.normal_(self.weights, mean=0, std=0.1)
        elif init_method == "xavier_uniform":
            init.xavier_uniform_(self.weights)
        elif init_method == "xavier_normal":
            init.xavier_normal_(self.weights)
        elif init_method == "kaiming_uniform":
            init.kaiming_uniform_(self.weights, nonlinearity='relu')
        elif init_method == "kaiming_normal":
            init.kaiming_normal_(self.weights, nonlinearity='relu')
        else:
            raise ValueError(f"Unknown init_method: {init_method}")
    
    
    def copy_node_values_to_dummy(self, node_values):
        """
        Copy node values from the graph to the dummy parameter.
        Args:
            node_values (torch.Tensor): The node values from the graph's node features.
        """

        # self.values_dummy.data = node_values.view(-1).detach().clone()  # Copy node values into dummy parameter
        self.values_dummy.data = node_values.view(-1).clone()  # Copy node values into dummy parameter
        
    def copy_dummy_to_node_values(self):
        """
        Copy updated dummy parameter values back to the graph's node features.
        Args:
            node_values (torch.Tensor): The node values to be updated in the graph's node features.
        """
        # self.data.x[self.nodes_2_update, 0] = self.values_dummy.data[self.nodes_2_update].unsqueeze(-1).detach()  # Detach to avoid retaining the computation graph
        self.data.x[self.nodes_2_update, 0] = self.values_dummy.data[self.nodes_2_update].unsqueeze(-1)  # Detach to avoid retaining the computation graph
        # node_values.data = self.values_dummy.view(node_values.shape).detach()  

    def gradient_descent_update(self, grad_type, parameter, delta, learning_rate, nodes_or_edge2_update, 
                                error=None, optimizer=None, use_optimizer=False):
        """
        Perform a gradient descent update on a parameter (weights or values).

        Args:
            type (str): either 'values' or 'weights' if we want specific dynamics or either parameter update. 
            parameter (torch.nn.Parameter): The parameter to be updated (weights or values).
            delta (torch.Tensor): The computed delta (change) for the parameter.
            learning_rate (float): The learning rate to be applied to the delta.
            optimizer (torch.optim.Optimizer, optional): Optimizer to be used for updates (if specified).
            use_optimizer (bool): Whether to use the optimizer for updating the parameter.
            nodes_2_update (torch.Tensor, optional): Indices of the nodes to update (for partial updates).
        """

        # self.use_grokfast = False 

        # MAYBE SHOULD NOT DO GROKFAST FOR BOTH VALUE NODE UPDATES AND WEIGHTS UPDATE (ONLY this one) 
        self.grokfast_type = "ema"
        """ 
            Grokfast-EMA (with alpha = 0.8, lamb = 0.1) achieved 22x faster generalization compared to the baseline.
            Grokfast-MA (with window size = 100, lamb = 5.0) achieved slower generalization improvement compared to EMA, but was still faster than the baseline. 
        """

        if not parameter.requires_grad:
            parameter.requires_grad = True
        parameter.requires_grad_(True)

        # ------------------- Grokfast ------------------- 
        """  Optionally adjust gradients based on grokfast 
        In Grokfast, the goal is to amplify the slow gradient component stored in self.grads[grad_type].
        After calculating the current gradient delta (based on the error or loss function), Grokfast adds a weighted version of this slow gradient component to delta:
        CAN be used for both weights and values
        CAN be used with optimizer or without optimizer
        """
        if self.use_grokfast:
            if grad_type == "weights" and self.use_grokfast:
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





        if use_optimizer and optimizer:
            # Clear 
            optimizer.zero_grad()
            if parameter.grad is None:
                parameter.grad = torch.zeros_like(parameter)
            else:
                parameter.grad.zero_()  # Reset the gradients to zero



            # print("use_optimizer and optimizer")
            # print("error", error)
            # print("parameter", parameter)

            # print("error grad", error.grad)
            # print("parameter grad", parameter.grad)

            # Set gradients for relevant indices
            ## --------- New ----------- 
            # if self.delta_w_selection == "all":

            #     error = error.mean() if self.grad_accum_method == "mean" else error.mean()
            #     error.backward(retain_graph=False)
            # else:
            #     # Manually set gradients only for specified nodes
            #     # take grad E w.r.t. parameter (weights or values)
            #     temp_grad = torch.autograd.grad(error, parameter, retain_graph=True, allow_unused=True)[0]

            #     if temp_grad is None:
            #         raise RuntimeError(f"{grad_type} parameter has no gradient.")
            #     # parameter.grad[nodes_or_edge2_update] = temp_grad[nodes_or_edge2_update].detach().clone()
            #     parameter.grad[nodes_or_edge2_update] = temp_grad[nodes_or_edge2_update]
                
            ### ---------- OLD ---------- set the gradients

            if delta.shape != parameter.shape:
                print("delta.shape", delta.shape)
                print("parameter.shape", parameter.shape)
                # delta = delta.view_as(parameter)

            if self.delta_w_selection == "all":
                parameter.grad = -delta.detach().clone()  # Apply full delta to the parameter
            else:
                
                

                # if self.graph_type == "single_hidden_layer" and self.update_rules == "Van_Zwol":
                
                # if grad_type == "weights" and self.update_rules == "Van_Zwol":

                #     # only update hidden nodes; 
                #     delta = delta * self.adj
                #     delta[784:-10, 784:-10] = 0

                #     parameter.grad = -delta.detach().clone()  # Apply full delta to the parameter

                #     # delta[784:-10, 784:-10] = 0
                #     # parameter.grad.data[784:-10, 784:-10] = 0
                #     # delta = delta * self.adj
                #     # parameter.grad = -delta  # Update only specific nodes
                #     # parameter.grad = delta  # Update only specific nodes

                # else:
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

                    # print("nodes_or_edge2_update", nodes_or_edge2_update)
                    # print("nodes_or_edge2_update shape", nodes_or_edge2_update.shape)
                    # print("delta.shape", delta.shape)
                    # print("parameter.data.shape", parameter.data.shape)
                    

                    # parameter.data.view(-1)[nodes_or_edge2_update] += learning_rate * delta.view(-1)[nodes_or_edge2_update]
                    # parameter.data = parameter.data.view_as(self.weights)

                    # transpose delta
                    # delta = delta.T
                    # delta[784:-10, 784:-10] = 0
                    # delta = delta * self.adj

                    # parameter.data *= self.adj

                    # parameter.data -= learning_rate * delta
                    parameter.data += learning_rate * delta
                else:
                    
                    parameter.data[nodes_or_edge2_update] += learning_rate * delta[nodes_or_edge2_update]
                


    # def update(self, aggr_out, x):
    def update_values(self):

        # Only iterate over internal indices for updating values
        # for i in self.internal_indices:

        """ 
        Query by initialization: Again, every value node is randomly initialized, but the value nodes of
        specific nodes are initialized (for t = 0 only), but not fixed (for all t), to some desired value. This
        differs from the previous query, as here every value node is unconstrained, and hence free to change
        during inference. The sensory vertices will then converge to the minimum found by gradient descent,
        when provided with that specific initialization. 
        """ 

        # self.optimizer_values.zero_grad()  # Reset value node gradients

        self.get_graph()

    
        if self.update_rules == "Salvatori":

            weights_batched_graph = self.weights.repeat(1, self.batchsize).to(self.device)


            weights_batched_graph_T = self.weights.T.repeat(1, self.batchsize).to(self.device)

            # self.predictions = self.prediction_mp(self.data.x.to(self.device), self.data.edge_index.to(self.device), weights_batched_graph, norm=self.norm.to(self.device))
            # self.predictions = self.prediction_mp(self.data.x.to(self.device), self.data.edge_index.to(self.device), self.weights)
            
            # ------------- TODO NEW ----------------------
            # print("self.data.x[:, 1].shape", self.data.x[:, 1].shape)
            # print("self.predictions.shape", self.predictions.shape)

            # self.errors = self.data.x[:, 1].to(self.device) - self.predictions
            # # write error back 
            # self.data.x[:, 2] = self.errors

            with torch.no_grad():
                
                # calculate dE/dx 
                # def forward(self, values, errors, edge_index, weight_matrix, norm=None):
                print(self.values.view(-1, 1).shape)

                delta_x = self.values_mp(self.values.view(-1, 1), self.errors.view(-1, 1),
                                         self.data.edge_index.to(self.device), weights_batched_graph, norm=self.norm.to(self.device)).squeeze()
                # delta_x = self.values_mp(self.data.x.to(self.device), self.data.edge_index.to(self.device), weights_batched_graph, norm=self.norm.to(self.device)).squeeze()
                # delta_x = self.values_mp(self.data.x.to(self.device), self.data.edge_index.to(self.device), weights_batched_graph_T, norm=self.norm.to(self.device)).squeeze()
                # delta_x = delta_x.detach()
                print("delta_x", delta_x.shape)
                delta_x = delta_x.view(self.batchsize, self.num_vertices)

        if self.update_rules == "Van_Zwol":
            # Ensure correct shapes for operations
            # Ensure correct dimensions
            # self.errors = self.errors.view(self.batchsize, self.num_vertices, 1)  # [8, 1094, 1]
            # self.values = self.values.view(self.batchsize, self.num_vertices, 1)  # [8, 1094, 1]
            
            # # Expand weights across batches
            # batch_weights = self.weights.repeat(self.batchsize, 1, 1)  # [8, 1094, 1094]
            
            # # Compute weighted input for each batch
            # weighted_input = torch.bmm(batch_weights, self.values)  # [8, 1094, 1]
            # f_prime_weighted = self.f_prime(weighted_input)  # Apply activation derivative
            
            # # Compute delta_x for each batch
            # delta_x_batch = self.errors - torch.bmm(batch_weights.transpose(1, 2), self.errors * f_prime_weighted)
            
            # # Flatten back to full batched graph
            # delta_x = delta_x_batch.view(-1, 1)  # [8752, 1]

            ### ------------------------------------ NEW ------------------------------------ ###
            # batchsize 1
            e = self.errors.view(self.batchsize, self.num_vertices)  # [200]
            # use input_error 
            # e[:, :784] = 0 # Sensory nodes are fixed
            
            a = self.values.view(self.batchsize, self.num_vertices)  # [200]
            W = self.weights # N, N
            # fp_wa = self.f_prime(W @ a)

            # dEdx = e - torch.matmul(W.T, e) * (fp_wa)

            # fp_wa = self.f_prime(torch.matmul(a, W.T))

            #    = [200] - [200] * [200] @ [200, 200]
            low = 784
            high = -10 if self.mode == "training" else self.num_vertices

            # training: update graph (values) from 784:-10
            # testing: update graph (values) from 784:num_vertices

            # dEdx = e - self.f_prime(a) * torch.matmul(e, W)  
            dEdx = e[:,low:high] - self.f_prime(a[:,low:high]) * torch.matmul(e, W.T[low:high,:].T)
            # dEdx = e[:,low:high] - self.f_prime(a[:,low:high]) * torch.matmul(e, W)

            print("dEdx", dEdx.shape)
            print("self.values", self.values.shape)

            # dEdx_selected = dEdx[self.nodes_2_update].unsqueeze(-1)  # Shape [200, 1]

            # only update the values of the hidden nodes
            # if self.mode == "training":
            #     self.values[:, ] -= self.lr_values * dEdx_selected
            # else:
            #     self.values[:, low:high] -= self.lr_values * dEdx_selected
            
            # nodes_2_update (train) = internal_indices
            # nodes_2_update (test) =  internal_indices + (either sensory or supervision indices)
            
            # print("self.nodes_2_update", self.nodes_2_update)
            print("self.nodes_2_update", self.nodes_2_update.shape)
            print("self.values", self.values.shape)
            print("a", a.shape)

            # Edx torch.Size([4, 200])
            # self.values torch.Size([3976, 1])
            # self.nodes_2_update torch.Size([200])
            # self.values torch.Size([3976, 1])
            # a torch.Size([4, 994])
            a[:, low:high] -= (self.lr_values * dEdx)
            a = a.view(-1, 1)

            self.values = a 

            # delta_values = self.errors - batch_weights.T @ (self.errors * self.f_prime(batch_weights @ self.values))

        if self.update_rules == "vectorized":

            feedback = torch.matmul(self.errors, self.weights.T)
            delta_x = -self.errors + self.f_prime(self.values) * feedback

        # #### TODO NEW 
        # if self.delta_w_selection == "internal_only":
        #     delta_x[self.sensory_indices_batch] = 0  # Sensory nodes are fixed

        # self.copy_node_values_to_dummy(self.values)
        # self.copy_node_values_to_dummy(self.data.x[:, 0])
                
        if self.update_rules != "Van_Zwol":

            # Use the gradient descent update for updating values
            self.gradient_descent_update(
                grad_type="values",
                parameter=self.values_dummy,  # Assuming values are in self.data.x[:, 0]
                delta=delta_x,
                learning_rate=self.lr_values,
                nodes_or_edge2_update=self.nodes_2_update,  # Mandatory, only updating hidden nodes (not sensory)

                error=self.errors if self.optimizer_values else None,
                optimizer=self.optimizer_values if self.use_optimizers else None,
                use_optimizer=self.use_optimizers
            )


        # self.copy_dummy_to_node_values()
        # self.data.x[self.nodes_2_update, 0] = self.values_dummy.data[self.nodes_2_update].unsqueeze(-1).detach()  # Detach to avoid retaining the computation graph


        _, _, pred = self.get_graph()

        if self.trace_activity_preds:
            preds = self.prediction.view(self.batchsize, self.num_vertices)
            preds_first = preds[0].detach()
            self.trace["preds"].append(preds_first)
            # self.trace["preds"].append((self.data.x[:, 2].detach()))
        if self.trace_activity_values:
            vals_first = self.values.view(self.batchsize, self.num_vertices)[0].detach()
            # self.trace["values"].append((self.data.x[:, 0].detach()))
            self.trace["values"].append(vals_first)

        # TODO
        # if self.trace_activity_preds:
        #     # tracing errors (preds are weird)
        #     self.trace["preds"].append(self.data.x[:, 1].cpu().detach())
        #     # self.trace["preds"].append(pred.cpu().detach())
        # if self.trace_activity_values:
        #     self.trace["values"].append(self.data.x[:, 0].cpu().detach())
            # self.trace["values"].append(torch.zeros_like(self.values))


        # if self.trace_activity_preds:
        #     self.trace["preds"].append(torch.randn_like(self.data.x[:, 2].detach()))
        # if self.trace_activity_values:
        #     self.trace["values"].append(torch.randn_like(self.data.x[:, 0].detach()))

        # delta_x = self.values_mp(self.data.x.to(self.device), self.data.edge_index.to(self.device), self.weights)

        # self.values.data[self.nodes_2_update, :] += delta_x[self.nodes_2_update, :]
        # data.x[self.nodes_2_update, 0] += delta_x[self.nodes_2_update, :]

    

        # https://chatgpt.com/share/54c649d0-e7de-48be-9c00-442bef5a24b8
        # This confirms that the optimizer internally performs the subtraction of the gradient (grad), which is why you should assign theta.grad = grad rather than theta.grad = -grad. If you set theta.grad = -grad, it would result in adding the gradient to the weights, which would maximize the loss instead of minimizing it.

        # if self.use_optimizers:
        #     self.optimizer_values.zero_grad()
        #     if self.values_dummy.grad is None:
        #         self.values_dummy.grad = torch.zeros_like(self.values_dummy)
        #     else:
        #         self.values_dummy.grad.zero_()  # Reset the gradients to zero
            
        #     # print("ai ai ")
        #     self.values_dummy.grad[self.nodes_2_update] = delta_x[self.nodes_2_update]
        #     self.optimizer_values.step()

        #     # print(self.data.x[self.nodes_2_update, 0].shape)
        #     # print(self.values_dummy.data[self.nodes_2_update].shape)
        #     self.data.x[self.nodes_2_update, 0] = self.values_dummy.data[self.nodes_2_update].unsqueeze(-1).detach()  # Detach to avoid retaining the computation graph
        #     # self.values[self.nodes_2_update] = self.values_dummy.data
    
        # else:
        #     # self.values_dummy.data[self.nodes_2_update] += self.gamma * delta_x[self.nodes_2_update].detach() 
        #     self.data.x[self.nodes_2_update, 0] += self.gradients_minus_1 * self.lr_values * delta_x[self.nodes_2_update].unsqueeze(-1).detach()  # Detach to avoid retaining the computation graph
        

        # Calculate the effective learning rate
        # effective_lr = self.lr_values * delta_x
        # self.effective_learning["v_mean"].append(effective_lr.mean().item())
        # self.effective_learning["v_max"].append(effective_lr.max().item())
        # self.effective_learning["v_min"].append(effective_lr.min().item())

    def get_predictions(self, data):
        self.get_graph()

        # with a single batch of n items the weights are shared/the same (self.weights.to(self.device))
        weights_batched_graph = self.weights.repeat(1, self.batchsize).to(self.device)

        if self.update_rules == "Salvatori":
            

            # self.predictions = self.prediction_mp(self.data.x.to(self.device), self.data.edge_index.to(self.device), weights_batched_graph, norm=self.norm.to(self.device))
            self.predictions = self.prediction_mp(self.values.view(-1, 1), 
                                                    self.data.edge_index.to(self.device), weights_batched_graph, norm=self.norm.to(self.device))
            # self.predictions = self.prediction_mp(self.data.x.to(self.device), self.data.edge_index.to(self.device), self.weights)
            self.predictions = self.predictions.view(self.batchsize, self.num_vertices)

        if self.update_rules == "Van_Zwol":


            # make weights a NxN matrix using the edge_index
            # self.weights_NxN = 
            # self.weights_NxN = self.weights.repeat(1, self.batchsize).to(self.device)

            # a = self.data.x[:, 0].view(self.batchsize, self.num_vertices)  # Shape: (N,)
            a = self.values.view(self.batchsize, self.num_vertices)  # Shape: (N,)

            # pred = f(wa) where w (NxN) and a (N,)
            W = self.weights   # CHANGED TO W.T
            # self.prediction = self.f(W @ a).to(self.device)
            # self.prediction = torch.matmul(self.f(a), W)  # + bias 
            self.predictions = torch.matmul(self.f(a), W.T)  # + bias 

            print("predictions shape 3", self.predictions.shape)
            print("Van_Zwol:       mean self.prediction", self.predictions.mean())

        if self.update_rules == "vectorized":
            
            # self.errors = self.errors.view(self.batchsize, self.num_vertices)
            self.values = self.values.view(self.batchsize, self.num_vertices)
            # delta_w = torch.einsum('bi,bj->ij', self.errors, self.f(self.values))

            # self.predictions = np.dot(self.values.to(self.device), self.weights.T)  # (batch_size, N) @ (N, N) -> (batch_size, N)
            # self.predictions = torch.matmul(self.values.to(self.device), self.weights.T.to(self.device)) # (batch_size, N) @ (N, N) -> (batch_size, N)
            self.predictions = torch.matmul(self.f(self.values), self.weights)  # Shape: (batch, N)

            print("predictions shape 4", self.predictions.shape)

        # self.predictions = self.predictions.detach()

        if self.use_bias:
            # print(self.predictions.shape)
            # print(self.biases.shape)
            self.predictions += self.biases.unsqueeze(-1)

        # if self.trace_activity_preds:
        #     self.trace["preds"].append(self.predictions.detach())

        return self.predictions


    def get_graph(self):
        """ 
        Don't need to reset preds/errors/values because we already set them in the dataloader to be zero's
        """
        self.values, self.errors, self.predictions = self.data.x[:, 0], self.data.x[:, 1], self.data.x[:, 2]
        # print("get graph")
        # print("values shape", self.values.shape)
        # print("errors shape", self.errors.shape)
        # print("predictions shape", self.predictions.shape)

        if self.update_rules == "Van_Zwol":
            # reshape into batch_size x num_vertices
            self.values = self.values.view(self.batchsize, self.num_vertices)
            self.errors = self.errors.view(self.batchsize, self.num_vertices)
            self.predictions = self.predictions.view(self.batchsize, self.num_vertices)


        # self.weights_NxN = 
        self.weights_NxN = self.weights.repeat(1, self.batchsize).to(self.device)

        return self.values, self.errors, self.predictions

    def energy(self):
        """
        Compute the total energy of the network, defined as:
        E_t = 1/2 * ∑_i (ε_i,t)**2,
        where ε_i,t is the error at vertex i at time t.

        For batching         
        """
        self.get_graph()

        # self.helper_GPU(self.print_GPU)


        self.predictions = self.get_predictions(self.data)
        # self.data.x[:, 2] = self.predictions.detach()

        # view the predictions as batch_size x num_vertices
        # self.predictions = self.predictions.view(self.batchsize, self.num_vertices)

        # Ensure predictions are correctly shaped

        # Assign to the data tensor
        # if self.update_rules == "Salvatori":
        #     # self.predictions = self.predictions.view(-1, 1)  # Flatten batch and add singleton dimension if necessary

        #     self.data.x[:, 2] = self.predictions  # Remove the extra dimension if needed
        
  
        # else:
        #     self.data.x[:, 2] = self.predictions  # Remove the extra dimension if needed
        
        # print("predictions shape", self.predictions.shape)

        # print("errors shape  --", self.errors.shape)
        # print("values shape --", self.values.shape)
        # print("predictions shape --", self.predictions.shape)
        
        # self.errors = (self.values.to(self.device) - self.predictions.to(self.device)).squeeze(-1) 
        # self.errors = (self.values.to(self.device) - self.predictions.to(self.device)).detach()  # Detach to avoid retaining the computation graph
        # self.errors = (self.values.to(self.device) - self.predictions.to(self.device))  # USED FOR loss.backward()
        
        print("self.errors shape", self.errors.shape)
        print("self.values shape", self.values.shape)
        print("self.predictions shape", self.predictions.shape)

        self.values = self.values.view(self.batchsize, self.num_vertices)
        self.errors = self.errors.view(self.batchsize, self.num_vertices)
        self.predictions = self.predictions.view(self.batchsize, self.num_vertices)

        self.errors.data = (self.values.to(self.device) - self.predictions.to(self.device))  # USED FOR loss.backward()


        self.use_input_error = None
        # TODO if using Van_Zwol and either generative or discriminative 
        # self.use_input_error = ["sensory", "internal", "supervised"]
        # self.use_input_error = ["sensory"]
        # self.use_input_error = ["supervised"]
    
        if self.mode == "training" and self.use_input_error:
            # use_input_error from VanZwol 
            """ 
            Note that to get the same updates as a PCN, use_input_error = False is required.
            if not self.use_sensory_error:
            self.errors[self.sensory_indices] = 0
            """
      
            if "sensory" in self.use_input_error:
                self.errors[self.sensory_indices] = 0
            if "internal" in self.use_input_error:
                self.errors[self.internal_indices] = 0
            if "supervised" in self.use_input_error:
                self.errors[self.supervised_labels] = 0

        
        # TODO IMPORTANT TESTING
        # self.errors = (self.errors ** 2) / 2

        # manually add error
        # print(self.errors.shape)
        # self.errors[self.sensory_indices] += 1
        # self.errors[self.supervised_labels] += 1

        # self.errors[self.internal_indices] += 0.1
        # print("!!!!!!!!!!!!!!!!!!!!!!!! IMPORTANT")        
        
        # self.errors = self.errors.view(self.batchsize, self.num_vertices)

        # self.predictions = self.predictions.view(-1, 1)  # Flatten batch and add singleton dimension if necessary
        
        # if self.update_rules == "Salvatori":
        #     self.errors = self.errors.squeeze()
        #     self.data.x[:, 1] = self.errors.view(-1, 1)


        # if self.update_rules == "Van_Zwol":
        #     # write prediction to the data tensor
        #     self.data.x[:, 2] = self.predictions.view(-1, 1)

        #     self.errors = self.errors.squeeze()
        #     self.data.x[:, 1] = self.errors.view(-1, 1)


            
        

        # print("errors shape", self.errors.shape)
        # print("values shape", self.values.shape)
        # print("predictions shape", self.predictions.shape)
        # print("weights shape", self.weights.shape)
        # print("self.data.x[:, 1] shape", self.data.x[:, 1].shape)

        # print("Shape before assignment:", self.errors.shape)

        # TODO
        # ASSIGN self.data.x to self.errors, self.values, self.predictions
        # print(self.data.x[:, 1])




        # self.data.x[:, 1] = self.errors.view(self.batchsize * self.num_vertices, 1)
        
        # self.data.x[:, 1] = self.errors.flatten()  # Flatten the errors and assign to the data tensor
        # data.x[self.nodes_2_update, 1] = errors[self.nodes_2_update, :]

       
        energy = {
                "internal_energy": [],
                "supervised_energy": [],
                "sensory_energy":  [],
        }

        if self.update_rules == "Salvatori":
            
            # print("len internal energ", len(self.internal_indices))
            # print(self.errors[self.internal_indices[0]])
            energy['internal_energy'] = 0.5 * (self.errors[self.internal_indices] ** 2).sum().item()
            energy['sensory_energy']  = 0.5 * (self.errors[self.sensory_indices] ** 2).sum().item()
            energy['supervised_energy']  = 0.5 * (self.errors[self.supervised_labels] ** 2).sum().item()
            energy['energy_total']  = 0.5 * (self.errors** 2).sum().item()
            
            self.energy_vals['mean_internal_energy_sign'].append(self.errors[self.internal_indices].mean().item())
            self.energy_vals['mean_sensory_energy_sign'].append(self.errors[self.sensory_indices].mean().item())
            self.energy_vals['mean_supervised_energy_sign'].append(self.errors[self.supervised_labels].mean().item())

            if self.mode == "training":

                self.energy_vals["internal_energy"].append(energy["internal_energy"])
                self.energy_vals["sensory_energy"].append(energy["sensory_energy"])
                self.energy_vals["supervised_energy"].append(energy["supervised_energy"])

                if self.wandb_logger:
                    self.wandb_logger.log({"Training/energy_total": energy["energy_total"]})
                    self.wandb_logger.log({"Training/energy_internal": energy["internal_energy"]})
                    self.wandb_logger.log({"Training/energy_sensory": energy["sensory_energy"]})

                    self.wandb_logger.log({"Training/mean_internal_energy_sign": self.errors[self.internal_indices].mean().item()})
                    self.wandb_logger.log({"Training/mean_sensory_energy_sign": self.errors[self.sensory_indices].mean().item()})
                    self.wandb_logger.log({"Training/mean_supervised_energy_sign": self.errors[self.supervised_labels].mean().item()})

            if self.mode == "testing":

                if self.wandb_logger:
                    self.wandb_logger.log({"Testing/energy_internal_testing": energy["internal_energy"]})
                    self.wandb_logger.log({"Testing/energy_sensory_testing": energy["sensory_energy"]})
        
                self.energy_vals["internal_energy_testing"].append(energy["internal_energy"])
                self.energy_vals["sensory_energy_testing"].append(energy["sensory_energy"])
                self.energy_vals["supervised_energy_testing"].append(energy["supervised_energy"])    

        return energy

    
    def test_class(self, data):
        
        # reset nodes
        # self.restart_activity()

        # fix IMG to sensory nodes
        # already done in the dataloader where 

        self.data = data
        
        # init values of supervision nodes in all batches to 0 
        self.data.x[:, 0][self.supervised_labels_batch] = 0
        
        tmp = self.data.x[:, 0].clone()
        tmp = tmp.view(self.batchsize, self.num_vertices)

        # tmp[:, -10:] = torch.normal(mean=0.5, std=1.0, size=tmp[:, -10:].shape, device=self.device)
        tmp[:, -10:] = 0.1
        self.values = tmp
        
        self.data.x[:, 0] = tmp.view(-1, 1)

        # update values
        self.inference()

        # return predictions on supervised nodes
        x_hat = self.values.view(self.batchsize, self.num_vertices)
        logits = x_hat[:, -10:]                       # batch_size x 10
        y_pred = torch.argmax(logits, axis=1)         # batch_size

        return y_pred



    def set_sensory_nodes(self):
        """ 
        When presented with a training point s̄ taken from a training set, the value nodes of
        the sensory vertices are fixed to be equal to the entries of s̄ for the whole duration of the training
        process, i.e., for every t. 
        """

        print("------------Setting sensory nodes--------------")
        # disable gradient computation, since we are not updating the values of the sensory nodes
        print("all information already in the graph object")
        pass 
        # x: (batch X nodes) X features (pixel values)
        # assert x.shape == [self.num_vertices, 1], f"Shape of x is {x.shape} and num_vertices is {self.num_vertices}"

    def restart_activity(self):

        self.set_phase("Restarting activity (pred/errors/values)")

        # Initialize tensors to zeros without creating new variables where not needed
        with torch.no_grad():
            self.values = torch.zeros(self.data.size(0), device=self.device) if self.values is None else self.values.zero_()
            self.errors = torch.zeros(self.data.size(0), device=self.device) if self.errors is None else self.errors.zero_()
            self.predictions = torch.zeros(self.data.size(0), device=self.device) if self.predictions is None else self.predictions.zero_()

            # Use in-place operation for values_dummy to reduce memory overhead
            self.values_dummy.data.zero_()  # Zero out values_dummy without creating a new tensor

            # 
            if len(self.trace["values"]) > 2 and len(self.trace["values"]) < 5:
                self.trace["values"] = []
                self.trace["preds"]  = []

        # Reset optimizer gradients if needed
        if self.use_optimizers:
            self.optimizer_values.zero_grad()
            self.optimizer_weights.zero_grad()

    def inference(self, restart_activity=False):

        """      During the inference phase, the weights are fixed, and the value nodes are continuously updated via gradient descent for T iterations, where T is
        a hyperparameter of the model. The update rule is the following (inference) (Eq. 3)
        
        First, get the aggregated messages for each node
        This could involve a separate function or mechanism to perform the message aggregation
        For simplicity, let's assume you have a function get_aggregated_messages that does this

        Update values as per Eq. (3)
        self.set_sensory_nodes(data.x)

        if restart_activity:
            self.restart_activity()
        """

        assert self.mode in ['training', 'testing', 'classification'], "Mode not set, (training or testing / classification )"
        
        energy = 0

        # restart trace 
        self.trace = {
            "values": [], 
            "errors": [],
            "preds" : [],
         }

        # add random noise
        # self.data.x[:, 0][self.sensory_indices] = torch.rand(self.data.x[:, 0][self.sensory_indices].shape).to(self.device)
        # self.data.x[:, 0][self.internal_indices] = torch.rand(self.data.x[:, 0][self.internal_indices].shape).to(self.device)

        # self.edge_weights = self.extract_edge_weights(edge_index=self.edge_index, weights=self.weights, mask=self.mask)
        # self.values, _pred_ , self.errors, = data.x[:, 0], data.x[:, 1], data.x[:, 2]

        # self.helper_GPU(self.print_GPU)

        self.get_graph()

        # self.helper_GPU(self.print_GPU)

        # Energy at t=0
        # Recalculate energy
        if self.update_rules == "Salvatori":
            energy = self.energy()
            
            self.energy_metrics['internal_energy_t0'] = energy['internal_energy']

            self.energy_vals["energy_t0"].append(energy["energy_total"])
                
            print(f"Initial internal energy (t=0): {self.energy_metrics['internal_energy_t0']}")

        from tqdm import tqdm

        t_bar = tqdm(range(self.T), leave=False)

        # t_bar.set_description(f"Total energy at time 0 {energy} Per avg. vertex {energy['internal_energy'] / len(self.internal_indices)}")
        t_bar.set_description(f"Total energy at time 0 {energy}")
        # print(f"Total energy at time 0", energy, "Per avg. vertex", energy["internal_energy"] / len(self.internal_indices))

        # for t in t_bar:
            
        #     # aggr_out = self.forward(data)
        #     self.t = t 

        #     self.update_values()
            
        #     energy = self.energy()
        #     t_bar.set_description(f"Total energy at time {t+1} / {self.T} {energy},")

        #     if self.use_convergence_monitor:
        #         if self.convergence_tracker.update(energy['internal_energy'], self.gradients_log):
        #             print("Both energy and gradients have converged, stopping at iteration:", t)
        #             break

        # Initialize progress bar with unknown total length if convergence monitor is used
        total_iterations = self.T if not self.use_convergence_monitor else 5000
        t = 0
        with tqdm(total=total_iterations, leave=False) as t_bar:
            t_bar.set_description(f"Initial energy: {energy}")
            self.t = t 

            while True:
                
                # 1. predictions 
                self.predictions = self.get_predictions(self.data)

                # 2. errors
                self.errors = self.values - self.predictions

                # self.data.x[:, 1] = self.errors.view(-1, 1)
                # self.data.x[:, 2] = self.predictions.view(-1, 1)
                
                # 3. Update values 
                self.update_values()

                if self.incremental_learning and self.mode == "training": #and self.delta_w is not None:
                    self.set_phase('weight_update')
                    self.weight_update()
                    self.set_phase('weight_update done')
                
                if self.delta_w is not None:
                    print("delta_w", self.delta_w.shape)
                else:
                    print("-----------delta_w-----------", self.delta_w)

                # Recalculate energy
                if self.update_rules == "Salvatori":
                    energy = self.energy()
                
                    # Break conditions
                    if self.use_convergence_monitor:
                        # Check for convergence
                        if self.convergence_tracker.update(energy['internal_energy'], None) or t >= self.T_MAX:
                        # if self.convergence_tracker.update(energy['internal_energy'], self.gradients_log) or t >= 300:
                            print(f"Convergence reached at iteration {t}")
                            break
                
                # Stop after T iterations if not monitoring convergence
                if t >= self.T - 1:
                    break

                # Update progress bar
                t_bar.set_description(f"Iteration {t+1}, Energy: {energy}")
                t_bar.update(1)

                t += 1

        if self.update_rules == "Salvatori":
            
            # Energy at t=T
            self.energy_metrics['internal_energy_tT'] = energy['internal_energy']
            print(f"Final internal energy (t=T): {self.energy_metrics['internal_energy_tT']}")

            self.energy_vals["energy_tT"].append(energy["energy_total"])
            
        print(self.mode)

        # # if in 'testing' mode, return the predictions for the sensory nodes/supervised nodes instead of clearing the graph values 
        # if self.mode == "training" and not self.incremental_learning:
        #     self.restart_activity()


        return True

    def set_mode(self, mode, task=None):
        """
        Setting which nodes to update their values based on the mode (training or testing) and task (classification, generation, reconstruction, etc.)
        """

        self.mode = mode
        self.task = task

        if self.task is None:
            assert self.mode == "training", "Task must be set for testing mode"
        assert self.mode in ['training', 'testing', 'classification'], "Mode not set, (training or testing / classification )"

        # Base indices (for a single graph instance)
        base_sensory_indices = list(self.sensory_indices_single_graph)
        base_internal_indices = list(self.internal_indices_single_graph)
        base_supervised_labels = list(self.supervised_labels_single_graph) if self.supervised_labels else []

        # Extended indices to account for the batch size
        self.sensory_indices_batch = []
        self.internal_indices_batch = []
        self.supervised_labels_batch = []

        for i in range(self.batchsize):
            self.sensory_indices_batch.extend([index + i * self.num_vertices for index in base_sensory_indices])
            self.internal_indices_batch.extend([index + i * self.num_vertices for index in base_internal_indices])
            if base_supervised_labels:
                self.supervised_labels_batch.extend([index + i * self.num_vertices for index in base_supervised_labels])

        # print("vertix", self.num_vertices)
        # print("before after", len(base_sensory_indices), len(self.sensory_indices_batch))
        # print("before after", len(base_internal_indices), len(self.internal_indices_batch))
        # print("before after", len(base_supervised_labels), len(self.supervised_labels_batch))

        self.sensory_indices   = self.sensory_indices_batch
        self.internal_indices  = self.internal_indices_batch
        self.supervised_labels = self.supervised_labels_batch

        if self.mode == "training":
            self.T = self.T_train

            # Update only the internal nodes during training
            self.nodes_2_update_base = self.internal_indices_batch


        elif self.mode == "testing":
            self.T = self.T_test
            # during testing the batch size is set to 1 for now
            # assert self.task in ["classification", "generation", "occlusion", "reconstruction", "denoising", "Associative_Memories"], \
            #     "Task not set, (generation, reconstruction, denoising, Associative_Memories)"

            print("----------using T of -----", self.T)

            if self.task == "classification":
                # Update both the internal and supervised nodes during classification
                self.nodes_2_update_base = self.internal_indices_batch + self.supervised_labels_batch

            # elif self.task in ["generation", "reconstruction", "denoising", "occlusion"]:
            else:
                # Update both the internal and sensory nodes during these tasks
                self.nodes_2_update_base = self.internal_indices_batch + self.sensory_indices_batch

        # Ensure nodes_2_update are expanded to include all batch items
        self.nodes_2_update = self.nodes_2_update_base
        # make torch array 
        

        # ---------------------------------------------
        if self.delta_w_selection == "all":
        
            # self.edges_2_update = torch.ones(self.edge_index.size(1), dtype=torch.bool, device=self.device)

            #         # Convert edge_type to tensor (use integers or map to string if preferred)
            # edge_type_mapsensory_indices_batch = {"Sens2Sens": 0, "Sens2Inter": 1, "Sens2Sup": 2, "Inter2Sens": 3, "Inter2Inter": 4, "Inter2Sup": 5, "Sup2Sens": 6, "Sup2Inter": 7, "Sup2Sup": 8}
            # self.edge_type = torch.tensor([edge_type_map[etype] for etype in self.edge_type], dtype=torch.long)

            # use size of edge_type
            self.edges_2_update = torch.ones(self.edge_type.size(0), dtype=torch.bool, device=self.device)
        elif self.delta_w_selection == "internal_only":
            # we want to have the option to only update the weights from and to the internal nodes
            # Define edge types involving internal nodes
            # self.edge_type_map = {"Sens2Sens": 0, "Sens2Inter": 1, "Sens2Sup": 2, "Inter2Sens": 3, "Inter2Inter": 4, "Inter2Sup": 5, "Sup2Sens": 6, "Sup2Inter": 7, "Sup2Sup": 8}
            # "Sens2Inter": 1, "Inter2Sens": 3, "Inter2Inter": 4, "Inter2Sup": 5, "Sup2Inter": 7,

            edge_types_involving_internal = torch.tensor([1, 3, 4, 5, 7], device=self.device)

            # Create a mask for edges to update
            # self.edges_2_update = torch.isin(self.edge_type, edge_types_involving_internal).to(self.device)

            # print("------------self.edge_type", self.edge_type)
            self.edges_2_update = torch.isin(self.edge_type, edge_types_involving_internal.to(self.edge_type.device))

            # For VanZwol use the dense (2d weight matrix)
            if self.graph_type == "single_hidden_layer" and self.update_rules == "Van_Zwol":
                pass
                # # Create a 2D mask based on the edge index (for dense weights)
                # weight_mask = torch.zeros_like(self.weights, dtype=torch.bool)

                # # Use the edge index to set the mask for corresponding weight entries
                # weight_mask[self.edge_index[0], self.edge_index[1]] = self.edges_2_update

                # Apply gradients selectively using this mask
                # weight_mask = torch.zeros_like(self.weights, dtype=torch.bool)
                # # template[784:-10, 784:-10] = 1

                # self.edges_2_update = weight_mask

                # print("previous self.edges_2_update.shapoooe", self.edges_2_update.shape)
                # # make a template for the weights mask with bool
                # template = torch.zeros_like(self.weights)  # 2d matrix 

                # # nodes_or_edge2_update = torch.zeros_like(self.weights, dtype=torch.bool)

                # template[784:-10, 784:-10] = 1
                # print("template.shape", template.shape)
                # print("weights.shape", self.weights.shape)

                # # flatten the 2d matrix into a 1d vector
                # self.edges_2_update = template.view(-1)

                # # make sure the type is bool
                # self.edges_2_update = self.edges_2_update.bool()

                # print("new self.edges_2_update.shapoooe", self.edges_2_update.shape)
                # print("new type self.edges_2_update.shapoooe", (self.edges_2_update.dtype))


        # if self.delta_w_selection == "nodes_2_update":
        else:
            self.edges_2_update = self.nodes_2_update


        # assert self.nodes_2_update.numel() > 0, "No nodes selected for updating"
        # assert self.edges_2_update.any(), "No edges selected for updating"
        # assert self.nodes_2_update, "No nodes selected for updating"

        # wandb. log in delta_w  the shape of edges_2_update and the shape of self.nodes_2_update
        if self.wandb_logger:
            # make sure both are not list but tensors or np arrays 
            copy_edges_2_update = self.edges_2_update.cpu().detach().numpy()
            
            # if type list 
            if isinstance(self.nodes_2_update, list):
                copy_nodes_2_update = np.array(self.nodes_2_update)
            else:
                copy_nodes_2_update = self.nodes_2_update.cpu().detach().numpy

            self.wandb_logger.log({"delta_w/edges_2_update_shape": copy_edges_2_update.shape, 
                                   "delta_w/nodes_2_update_shape": copy_nodes_2_update.shape,
                                   "delta_w/edge_type": self.edge_type.shape,
                                   })
            
        print(f"-------------mode {self.mode}--------------")
        print(f"-------------task {self.task}--------------")

        self.nodes_2_update = torch.tensor(self.nodes_2_update, device=self.device)

        
    def set_phase(self, phase):
        self.phase = phase
        print(f"-------------{self.phase}--------------")


    def init_hidden_feedforward(self):

        pass 
        # if self.graph_type == "single_hidden_layer":
            
        #     values_clone = self.data.x[:, 0].clone()

        #     for l in range(self.num_layers):
        #         lower = 
        #         upper = 

        #         values_clone[]


        #     delta_x = values_clone
        
    def learning(self, data, restart_activity=False):

        
        # metric zeros 
        self.energy_vals["internal_energy_batch"] = []
        self.energy_vals["sensory_energy_batch"] = []

        self.data = data    

        # random inint value of internal nodes
        # data.x[:, 0][self.internal_indices] = torch.rand(data.x[:, 0][self.internal_indices].shape).to(self.device)

        # # not need but improves a bit the performance see PRECO notebook (VanZwol)
        # if self.graph_type == "single_hidden_layer":
        #     # init the hidden layer (internal nodes values) with predictions layer for layer 
        #     self.init_hidden_feedforward()


        # self.copy_node_values_to_dummy(self.data.x[:, 0])

        # random inint errors of internal nodes
        # data.x[:, 1][self.internal_indices] = torch.rand(data.x[:, 1][self.internal_indices].shape).to(self.device)

        # self.restart_activity()

        # need for the wieght update
        _, self.edge_index = self.data.x, self.data.edge_index
        
        # self.helper_GPU(self.print_GPU)

        ## INFERENCE: This process of iteratively updating the value nodes distributes the output error throughout the PC graph. 
        self.set_phase('inference')
        self.inference()
        self.set_phase('inference done')
        
        if not self.incremental_learning:
            ## WEIGHT UPDATE 
            self.set_phase('weight_update')
            self.weight_update()
            self.set_phase('weight_update done')

        # Energy at t=T+1 after weight update
        if self.update_rules == "Salvatori":
                
            energy = self.energy()
            self.energy_metrics['internal_energy_tT_plus_1'] = energy['internal_energy']
            print(f"Internal energy after weight update (t=T+1): {self.energy_metrics['internal_energy_tT_plus_1']}")

            # Calculate energy drop and weight update gain
            self.calculate_energy_metrics()


    
        # if self.wandb_logger:
        #     self.wandb_logger.log({"Training/lr_values": current_lr_values, "lr_weights": current_lr_weights})

        
        # self.helper_GPU(self.print_GPU)

    # def log_delta_w(self, delta_w):
         
    #     # self.w_log.append(delta_w.detach().cpu())

    #     # Extract the first batch from delta_w and edge_index
    #     first_batch_delta_w = delta_w[:self.edge_index_single_graph.size(1)]  # Assuming edge_index_single_graph represents a single graph's edges

    #     # Find sensory-to-sensory connections in the first batch
    #     sensory_indices_set = set(self.sensory_indices_single_graph)
    #     s2s_mask = [(src in sensory_indices_set and tgt in sensory_indices_set) 
    #                 for src, tgt in zip(self.edge_index_single_graph[0], self.edge_index_single_graph[1])]

    #     # Convert mask to tensor (if needed)
    #     s2s_mask = torch.tensor(s2s_mask, dtype=torch.bool, device=self.device)

    #     # Find the rest (edges that are not sensory-to-sensory) in the first batch
    #     rest_mask = ~s2s_mask

    #     # Apply the mask to delta_w for sensory-to-sensory and rest
    #     delta_w_s2s = first_batch_delta_w[s2s_mask]
    #     delta_w_rest = first_batch_delta_w[rest_mask]

    #     # Check if delta_w_s2s is non-empty before calculating max and mean
    #     delta_w_s2s_mean = delta_w_s2s.mean().item() if delta_w_s2s.numel() > 0 else 0
    #     delta_w_s2s_max = delta_w_s2s.max().item() if delta_w_s2s.numel() > 0 else 0

    #     # Check if delta_w_rest is non-empty before calculating max and mean
    #     delta_w_rest_mean = delta_w_rest.mean().item() if delta_w_rest.numel() > 0 else 0
    #     delta_w_rest_max = delta_w_rest.max().item() if delta_w_rest.numel() > 0 else 0

    #     # Log the delta_w values for the first batch
    #     self.wandb_logger.log({
    #         "delta_w_s2s_mean_first_batch": delta_w_s2s_mean,
    #         "delta_w_s2s_max_first_batch": delta_w_s2s_max,
    #         "delta_w_rest_mean_first_batch": delta_w_rest_mean,
    #         "delta_w_rest_max_first_batch": delta_w_rest_max
    #     })


    def log_delta_w(self, print_log=False):
        """
        Log delta_w values separately for each edge connection category type defined by self.edge_type.
        Create a combined histogram plot of all edge types using matplotlib and log it to WandB.
        Parameters:
        - print_log: Whether to print log information for debugging.
        """
        import matplotlib.pyplot as plt

        delta_w = self.gradients_log  # Assuming `gradients_log` holds the delta weights
        edge_type = self.edge_type  # Using self.edge_type for the connection types

        if self.update_rules == "Van_Zwol":
            # Ensure delta_w is flattened for size comparison
            if delta_w.dim() > 1:
                delta_w = delta_w.view(-1)

            # Use edge indices to gather the corresponding delta_w elements
            if delta_w.dim() == 2:  # Ensure 2D weights
                delta_w_edges = delta_w[self.edge_index[0], self.edge_index[1]]
            else:
                delta_w_edges = delta_w

        # Check if the sizes of delta_w and edge_type match
        if delta_w.size(0) != edge_type.size(0):
            raise ValueError(
                f"Size mismatch: delta_w has size {delta_w.size(0)}, "
                f"but edge_type has size {edge_type.size(0)}"
            )


      

        # Ensure sizes match
        if delta_w_edges.size(0) != edge_type.size(0):
            raise ValueError(
                f"Size mismatch: delta_w_edges has size {delta_w_edges.size(0)}, "
                f"but edge_type has size {edge_type.size(0)}"
            )


        # Mapping of edge types
        edge_type_map = {
            "Sens2Sens": 0, "Sens2Inter": 1, "Sens2Sup": 2,
            "Inter2Sens": 3, "Inter2Inter": 4, "Inter2Sup": 5,
            "Sup2Sens": 6, "Sup2Inter": 7, "Sup2Sup": 8
        }
        reverse_edge_type_map = {v: k for k, v in edge_type_map.items()}

        # Prepare a dictionary to store histograms for WandB
        histograms = {}

        # Initialize a figure for the combined histogram plot
        plt.figure(figsize=(10, 7))

        num_bins = 25
        # Iterate through each connection category type in edge_type_map
        for etype, etype_name in reverse_edge_type_map.items():
            # Mask to select delta_w values corresponding to the current edge type
            mask = (edge_type == etype)
            
            # Select delta_w values for this edge type
            delta_w_etype = delta_w[mask]
            
            if delta_w_etype.numel() > 0:  # Plot and log only if there are elements
                # Compute statistics
                # delta_mean = delta_w_etype.mean().item()
                # delta_max = delta_w_etype.max().item()

                # Log histograms and stats to WandB
                # if self.wandb_logger:
                #     histograms[f"delta_w/{etype_name}_mean"] = delta_mean
                #     histograms[f"delta_w/{etype_name}_max"] = delta_max

                # Plot histogram for this edge type
                plt.hist(
                    delta_w_etype.cpu().numpy(), 
                    # bins='auto',  # Automatic bin size based on data distribution
                    bins=num_bins,  # Automatic bin size based on data distribution
                    alpha=0.5,    # Transparency to overlay histograms
                    label=etype_name
                )

                # Print log for debugging if required
                # if print_log:
                #     print(f"delta_w_{etype_name}: mean={delta_mean}, max={delta_max}")
            else:
                # Log NaN if no values for this edge type
                if self.wandb_logger:
                    histograms[f"delta_w/{etype_name}_mean"] = float('nan')
                    histograms[f"delta_w/{etype_name}_max"] = float('nan')

        # Compute the 5th and 95th percentiles of delta_w values
        delta_min = np.percentile(delta_w.cpu().numpy(), 1)
        delta_max = np.percentile(delta_w.cpu().numpy(), 99)


        # Extend the range by 20% on both sides for broader visualization
        range_extension = 0.7 * (delta_max - delta_min)
        x_min = delta_min - range_extension
        x_max = delta_max + range_extension

        # Cap the x-axis with the broader range
        plt.xlim(x_min, x_max)

        # Add title, labels, and legend to the combined histogram plot
        plt.title("Delta Weight Distribution by Edge Type")
        plt.xlabel("Delta Weight")
        plt.ylabel("Frequency")
        plt.legend()
        
        # Save the figure
        # combined_hist_path = "combined_delta_w_histogram.png"
        # plt.savefig(combined_hist_path)

        # Save the figure to a BytesIO buffer instead of saving locally
        # Save the figure to a BytesIO buffer
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)

        # Convert the BytesIO object to a PIL Image for WandB
        image = Image.open(buffer)

        # Log the histogram image to WandB
        if self.wandb_logger:
            self.wandb_logger.log({"delta_w/combined_histogram_plot": wandb.Image(image)})

        # Close the plot to free resources
        plt.close()
        buffer.close()
        # 
        # Log all histograms at once to WandB
        # if self.wandb_logger:
        #     self.wandb_logger.log(histograms)

        # Clear the plot to free memory
        plt.clf()
        
        print("delta_w distributions for each edge type and combined histogram plot logged.")


    def weight_update(self):
        
        # self.optimizer_weights.zero_grad()  # Reset weight gradients

        self.get_graph()                
        # self.values, self.errors, self.predictions, = self.data.x[:, 0], self.data.x[:, 1], self.data.x[:, 2]
        
        # errors = self.errors.squeeze().detach() 
        # f_x    = self.f(self.values).squeeze().detach()  #* self.mask  # * self.mask

        errors = self.errors.squeeze() 
        f_x    = self.f(self.values).squeeze()  #* self.mask  # * self.mask

  
        # print("Errors / max, mean", errors.max(), errors.mean())
        # print("f_x / max, mean", f_x.max(), f_x.mean())



        if self.update_rules == "Salvatori":


            # Gather the indices of the source and target nodes for each edge
            source_nodes = self.edge_index[0]   # all i's 
            target_nodes = self.edge_index[1]   # all j's 

            #### TODO IMPORTANT SWITCHED

            ## GOOD 
            fx   = f_x[source_nodes]       # f(x_i)
            errors = errors[target_nodes]  # (x_j - mu_j)

            # WRONG 
            # fx = f_x[target_nodes]       # f(xj,T)
            # errors = errors[source_nodes]  # εi,T

            delta_w_batch = errors * fx  # α · εi,T f(xj,T)

            delta_w = delta_w_batch.view(self.batchsize, -1)

            if self.grad_accum_method == "sum":
                # delta_w = delta_w.sum(0).detach()
                delta_w = delta_w.sum(0)
            if self.grad_accum_method == "mean":
                # delta_w = delta_w.mean(0).detach()
                delta_w = delta_w.mean(0)


        if self.update_rules == "Van_Zwol":

            # --------------------  old code --------------------
            # bias = self.biases if self.use_bias else 0

            # # Ensure correct dimensions
            # batch_size = self.batchsize
            # num_nodes = self.num_vertices
            # x = self.data.x[:, 0].view(batch_size, num_nodes, 1)  # [8, 944, 1]
            # self.errors = self.errors.view(batch_size, num_nodes, 1)  # [8, 944, 1]
            
            # # Reshape weights for batched computation
            # batch_weights = self.weights.view(batch_size, num_nodes, num_nodes)  # [8, 944, 944]
            # bias = self.biases.view(batch_size, num_nodes, 1) if self.use_bias else 0  # [8, 944, 1]

            # # Compute weighted input and `temp`
            # weighted_input = torch.bmm(batch_weights, x)  # [8, 944, 1]
            # temp = self.errors * self.f_prime(weighted_input + bias)  # [8, 944, 1]

            # # Compute delta weights for each batch
            # delta_w_batch = -torch.bmm(temp, x.transpose(1, 2))  # [8, 944, 944]

            # # Aggregate across batches (mean or sum)
            # delta_w = delta_w_batch.mean(dim=0)  # [944, 944]

            # # Correct shape of delta_w after computation
            # if delta_w.dim() == 1 or delta_w.shape != self.weights.shape:
            #     delta_w = delta_w.view(self.weights.shape)

            # # TODO ???
            # delta_w = delta_w_batch.view(self.weights.shape)  # Correct the shape before the update

            # --------------- New code --------------------
            # a = self.data.x[:, 0].view(self.num_vertices)   # a_hat 
            # e = self.data.x[:, 1].view(self.num_vertices)
            # # W = self.weights 

            # e = e.view(-1, 1)  # Shape [994, 1]
            # f_a = self.f(a).view(1, -1)  # Shape [1, 994]

            # # Outer product: [994, 1] @ [1, 994] = [994, 994]
            # delta_w = -torch.matmul(e, f_a)


            # a = self.data.x[:, 0].view(self.batchsize, self.num_vertices)   # a_hat 
            # e = self.data.x[:, 1].view(self.batchsize, self.num_vertices)
            a = self.values.view(self.batchsize, self.num_vertices)   # a_hat 
            e = self.errors.view(self.batchsize, self.num_vertices)

            # Outer product: [994, 1] @ [1, 994] = [994, 994]
            self.delta_w = -torch.matmul(e.T, self.f(a))

            # print("delta_w", delta_w.shape)  # Should be [

            print("delta_w", self.delta_w.shape)  # Should be [994, 994]
            print("a", a.shape)  # [994]
            print("e", e.shape)  # [994, 1]
            
            # delta_w.data *= self.adj
            assert self.delta_w.shape == self.weights.shape
            
            self.params = {"w": self.weights, 
                           "b": self.b, 
                           "use_bias": False}

            self.grads = {"w": self.delta_w, "b": None}

            print("before mean of the gradients", self.delta_w.mean())
            # print mean of the new weights
            print("before mean of the new weights", self.weights.mean())
            

            self.van_zwol_optimizer_weights.step(self.params, self.grads,  batch_size=self.batchsize)

            # print mean of the gradients
            print("mean of the gradients", self.delta_w.mean())
            # print mean of the new weights
            print("mean of the new weights", self.weights.mean())
            
            return True

            # self.weights.data -= delta_w
            # self.weights.data *= self.adj

            # f_prime = self.f_prime(W @ a)  # f'(W * a_hat)
            # print("f_prime", f_prime.shape)
            # print("e", e.shape)
            # print("a", a.shape)
            # print("W", W.shape)
            # print("e.view(-1, 1)", e.view(-1, 1).shape)
            # print("f_prime.view(-1, 1)", f_prime.view(-1, 1).shape)
            # print("a.view(1, -1)", a.view(1, -1).shape)

            # delta_w = (e.view(-1, 1) * f_prime.view(-1, 1)) @ a.view(1, -1)
            # delta_w = -delta_w
            # print("delta_w", delta_w.shape)




            # bias = b if self.use_bias else 0
            # temp = e*self.dfdx( torch.matmul(x, w.T) + bias )
            # out = -torch.matmul( temp.T,  x ) # matmul takes care of batch sum
            # out *= self.mask if self.mask is not None else 1

          

        if self.update_rules == "vectorized":
            
            # self.errors = self.errors.view(self.batchsize, self.num_vertices)
          
            delta_w = np.einsum('bi,bj->ij', self.errors, self.f(self.values))  # Outer product, (N, N)



        # grad clipping 
        delta_w = torch.clamp(delta_w, min=-1, max=1)


     
        # print("self.delta_w shape", delta_w.shape)

        # self.log_gradients(log_histograms=True, log_every_n_steps=20 * self.T)

        # Adjust delta_w by element-wise multiplication with lr_by_edtype
        if self.adjust_delta_w:
            adjusted_delta_w = delta_w * self.lr_by_edtype

        # self.gradients_log = delta_w.detach()
        self.gradients_log = delta_w

        print(self.edges_2_update, delta_w.shape)

        if self.update_rules != "Van_Zwol":
            self.gradient_descent_update(
                grad_type="weights",
                parameter=self.weights,
                delta=adjusted_delta_w if self.adjust_delta_w else delta_w,
                learning_rate=self.lr_weights,
        
                error= self.errors if self.optimizer_values else None,
                nodes_or_edge2_update=self.edges_2_update,
                optimizer=self.optimizer_weights if self.use_optimizers else None,
                use_optimizer=self.use_optimizers, 
            )

        # log delta_w 
        # # self.log_delta_w(delta_w)
        # if self.edge_type.numel() > 0:
        #     self.log_delta_w(adjusted_delta_w if self.adjust_delta_w else delta_w, self.edge_type, log=False)
            
        if self.use_bias:
            # print((self.lr_weights * self.errors[self.internal_indices].detach()).shape)
            # print(self.biases.data[self.internal_indices].shape)

            self.biases.data[self.internal_indices] += (self.lr_weights * self.errors[self.internal_indices].detach()).squeeze()


        # if self.use_optimzers:
            
        #     # self.optimizer_weights.zero_grad()             self.optimizer_values.grad = torch.zeros_like(self.values.grad)  # .zero_grad()
        #     # self.weights.grad = torch.zeros_like(self.weights.grad)  # .zero_grad()
        #     self.optimizer_weights.zero_grad()
        #     self.weights.grad = delta_w
        #     self.optimizer_weights.step()

        #     # self.optimizer_weights.zero_grad()
        #     # # Accumulate gradients
        #     # self.weights.backward(delta_w)
        #     # # Optional: Gradient clipping
        #     # torch.nn.utils.clip_grad_norm_(self.weights, max_norm=1.0)
        #     # # Update weights
        #     # self.optimizer_weights.step()

        # else:
        #     print(self.lr_weights, delta_w.shape)
        #     print(self.weights.data.shape)

        #     self.weights.data += self.gradients_minus_1 * (self.lr_weights * delta_w)
        #     # self.weights.data += self.lr_weights * self.delta_w
        
        #     # self.weights.data = (1 - self.damping_factor) * self.w_t_min_1 + self.damping_factor * (self.weights.data)
        
        #     self.w_t_min_1 = self.weights.data.clone()
        
        # # Calculate the effective learning rate
        # effective_lr = self.lr_weights * delta_w


        # self.effective_learning["w_mean"].append(effective_lr.mean().item())
        # self.effective_learning["w_max"].append(effective_lr.max().item())
        # self.effective_learning["w_min"].append(effective_lr.min().item())

        # print("leeen", len(self.effective_learning["w_mean"]))

        # ## clamp weights to be above zero
        # print("--------------------CLAMP THE WEIGHTS TO BE ABOVE ZERO-----------")
        # self.weights.data = torch.clamp(self.weights.data, min=0)

        print("----------------NO CLAMPING----------------")

    


class PCGNN(nn.Module):
    def __init__(self, num_vertices, sensory_indices, internal_indices, 
                 lr_params, T, graph_structure, 
                 batch_size, edge_type, incremental_learning, update_rules,
                 delta_w_selection, use_bias, 
                 use_learning_optimizer=False, 
                 use_grokfast=False,
                 weight_init="xavier", clamping=None, supervised_learning=False, 
                 normalize_msg=False, 
                 debug=False, activation=None, log_tensorboard=True, wandb_logger=None, device='cpu'):
        super(PCGNN, self).__init__()  # Ensure the correct super call
        
        """ TODO: in_channels, hidden_channels, out_channels, """
        # INSIDE LAYERS CAN HAVE PREDCODING - intra-layer 
        self.pc_conv1 = PCGraphConv(num_vertices, sensory_indices, internal_indices, 
                                    lr_params, T, graph_structure, 
                                    batch_size, edge_type, incremental_learning, update_rules, delta_w_selection, use_bias, use_learning_optimizer, 
                                    use_grokfast, weight_init, clamping, supervised_learning, 
                                    normalize_msg, 
                                    debug, activation, log_tensorboard, wandb_logger, device)

        self.use_optimizers = use_learning_optimizer
        self.original_weights = None  # Placeholder for storing the original weights
        self.update_rules = update_rules

    def log():
        pass
    
    def learning(self, batch):       
        

        self.pc_conv1.mode = "training"
        self.pc_conv1.learning(batch)
        
        if self.use_optimizers:

            if self.pc_conv1.scheduler_weights is not None and self.pc_conv1.lr_scheduler:
                self.pc_conv1.scheduler_weights.step(self.pc_conv1.energy_vals["internal_energy_batch"])
                
            # self.pc_conv1.scheduler_values.step(self.pc_conv1.energy_vals["internal_energy_batch"])

            # current_lr_values = self.pc_conv1.optimizer_values.param_groups[0]['lr']
            # current_lr_weights = self.pc_conv1.optimizer_weights.param_groups[0]['lr']
            # print(f"Current LR for values: {current_lr_values}")
            # print(f"Current LR for weights: {current_lr_weights}")
        
        if self.update_rules == "Salvatori":
            history = {
                "internal_energy_mean": self.pc_conv1.energy_vals["internal_energy_batch"],
                "sensory_energy_mean": self.pc_conv1.energy_vals["sensory_energy_batch"],
            
                "internal_energy_last": self.pc_conv1.energy_vals["internal_energy"][-1],
                "sensory_energy_last": self.pc_conv1.energy_vals["sensory_energy"][-1],
            }
        
            return history
        return None
    
    def trace(self, values=False, errors=False):
        
        # self.pc_conv1.trace = {
        #     "values": [], 
        #     "errors": [],
        #  }
    
        if values:
            self.pc_conv1.trace_activity_values = True 
            
        if errors:
            self.pc_conv1.trace_activity_errors = True  


    def Disable_connection(self, from_indices, to_indices):
        """
        Temporarily disable connections between specified nodes by setting their weights to zero.

        Parameters:
        - from_indices: list of node indices from which connections originate.
        - to_indices: list of node indices to which connections lead.
        """
        if self.original_weights is None:
            # Make a copy of the original weights the first time a connection is disabled
            self.original_weights = self.pc_conv1.weights.clone()

        masks = []
        for from_idx in from_indices:
            for to_idx in to_indices:
                # Find the corresponding edge in the graph
                edge_mask = (self.pc_conv1.edge_index_single_graph[0] == from_idx) & \
                            (self.pc_conv1.edge_index_single_graph[1] == to_idx)
                # Temporarily set the weights of these edges to zero
                masks.append(edge_mask)
                self.pc_conv1.weights.data[edge_mask] = 0
        return masks

    def enable_all_connections(self):
        """
        Restore the original weights for all connections that were disabled.
        """
        if self.original_weights is not None:
            self.pc_conv1.weights.data = self.original_weights
            self.original_weights = None  # Clear the backup after restoration

    def retrieve_connection_strength(self, from_group, to_group):
        """
        Retrieve the connection strengths (weights) between two specific groups of nodes.

        Parameters:
        - from_group: list of node indices from which connections originate (e.g., sensory_indices).
        - to_group: list of node indices to which connections lead (e.g., supervision_label_indices).

        Returns:
        - connection_strengths: A dictionary with keys as tuples (from_idx, to_idx) and values as the corresponding weights.
        """
        connection_strengths = {}
        
        for from_idx in from_group:
            for to_idx in to_group:
                # Find the corresponding edge in the graph
                edge_mask = (self.pc_conv1.edge_index_single_graph[0] == from_idx) & \
                            (self.pc_conv1.edge_index_single_graph[1] == to_idx)
                # Retrieve the connection weight for this edge
                connection_weights = self.pc_conv1.weights[edge_mask]
                
                if connection_weights.numel() > 0:  # If there is a connection
                    connection_strengths[(from_idx, to_idx)] = connection_weights.item()

        return connection_strengths


    def load_weights(self, path, data_eg):

        print("Settng weights of self.pc_conv1")
        self.pc_conv1.weights = torch.load(f"{path}/weights.pt")
        print("loaded weights matrix W")

        self.pc_conv1.edge_index = torch.load(f"{path}/graph.pt")
        print("loaded edge index for the graphs")

        if self.pc_conv1.use_bias:
            self.pc_conv1.bias = torch.load(f"{path}/bias.pt")


        # self.pc_conv1.values = torch.zeros(self..num_vertices,self.batch_size,device=self.device) # requires_grad=False)    
        self.pc_conv1.data = data_eg
        self.pc_conv1.restart_activity()
        print("Done")

    def save_weights(self, path, overwrite=False):
        # Ensure the directory exists
        os.makedirs(path, exist_ok=True)
        import time 
        # Define base file paths
        weights_path = os.path.join(path, "weights.pt")
        graph_path = os.path.join(path, "graph.pt")
        bias_path = os.path.join(path, "bias.pt") if self.pc_conv1.use_bias else None

        # Check for existing files and adjust names if needed
        if os.path.exists(weights_path) or os.path.exists(graph_path) or (bias_path and os.path.exists(bias_path)):
            if not overwrite:
                # Add timestamp to avoid overwriting
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                weights_path = os.path.join(path, f"weights_{timestamp}.pt")
                graph_path = os.path.join(path, f"graph_{timestamp}.pt")
                if bias_path:
                    bias_path = os.path.join(path, f"bias_{timestamp}.pt")
                print(f"Files already exist. Saving to new files: {weights_path}, {graph_path}" +
                    (f", {bias_path}" if bias_path else ""))
            else:
                print("----Overwriting existing files----")

        # Save weights, graph, and bias (if applicable)
        torch.save(self.pc_conv1.weights, weights_path)
        torch.save(self.pc_conv1.edge_index, graph_path)
        print(f"Saved weights and edge_index to {path}")

        if self.pc_conv1.use_bias:
            torch.save(self.pc_conv1.biases, bias_path)
            print(f"Saved bias weights to {path}")


    def query(self, method, random_internal=True, data=None):
        
        print("Random init values of all internal nodes")

        if random_internal:
            data.x[:, 0][self.pc_conv1.internal_indices] = torch.rand(data.x[:, 0][self.pc_conv1.internal_indices].shape).to(self.pc_conv1.device)


        self.pc_conv1.energy_vals["internal_energy_testing"] = []
        self.pc_conv1.energy_vals["sensory_energy_testing"] = []
        self.pc_conv1.energy_vals["supervised_energy_testing"] = []

        assert self.pc_conv1.mode == "testing"
        assert data is not None, "We must get the labels (conditioning) or the image data (initialization)"

        self.pc_conv1.set_phase('---reconstruction---')
        print("TASK IS", self.pc_conv1.task)

        if method == "query_by_initialization":
            if self.pc_conv1.task == "denoising":
                self.pc_conv1.set_sensory_nodes(data, generaton=False, mode='initialization')
                self.pc_conv1.inference()
            elif self.pc_conv1.task == "reconstruction":
                self.pc_conv1.set_sensory_nodes(data, generaton=False, mode='initialization')
                self.pc_conv1.inference()
            elif self.pc_conv1.task == "occlusion":
                
                # x.view(-1)[i]
                self.pc_conv1.set_sensory_nodes(data, generaton=False, mode='initialization')
                self.pc_conv1.inference()
            
                print("TODO")
            else:
                raise Exception(f"Unknown task given {method}")
        
        elif method == "query_by_conditioning":
            assert len(self.pc_conv1.supervised_labels) > 0, "Can't do conditioning on labels without labels"

            '''            
            While each value node is randomly re-initialized, the value nodes of
            specific vertices are fixed to some desired value, and hence not allowed to change during the energy
            minimization process. The unconstrained sensory vertices will then converge to the minimum of the
            energy given the fixed vertices, thus computing the conditional expectation of the latent vertices given
            the observed stimulus.
            '''

            if self.pc_conv1.task == "classification":
                # classification, where internal nodes are fixed to the pixels of an image, and the sensory nodes are
                # ... fixed to a 1-hot vector with the labels

                self.pc_conv1.set_sensory_nodes(data, generaton=False, mode='conditioning')

            else:
                # conditioning model on the label
                
                if self.pc_conv1.task == "generation":
                    # here the single value node encoding the class information is fixed, 
                    # .. and the value nodes of the sensory nodes converge to an image of that clas
                    print()
                elif self.pc_conv1.task == "reconstruction":
                    
                    self.pc_conv1.set_sensory_nodes(data, generaton=False, mode='conditioning')

                    # such as image completion, where a fraction of the sensory nodes are fixed to # the available pixels of an image, 
                else:
                    raise Exception(f"unkown task given {method}")
        elif method == "pass":
            print("Pass")
        else:
            raise Exception(f"unkown method: {method}")

        if self.pc_conv1.trace_activity_preds:
            self.pc_conv1.trace["preds"].append(data.x[:, 2][0:784].detach())
        if self.pc_conv1.trace_activity_values:
            self.pc_conv1.trace["values"].append(data.x[:, 0][0:784].detach())

        self.inference(data)
        
        print("QUery by condition or query by init")

        return self.pc_conv1.get_graph()
        
   
    
    def inference(self, data, random_internal=False):

        # print("------------------ experimental ===================")
        # TODO
        if random_internal:
            data.x[:, 0][self.pc_conv1.internal_indices] = torch.rand(data.x[:, 0][self.pc_conv1.internal_indices].shape).to(self.pc_conv1.device)

        self.pc_conv1.data = data

        # self.pc_conv1.restart_activity()
        
        if self.pc_conv1.trace_activity_preds:
            self.pc_conv1.trace["preds"].append(data.x[:, 2].cpu().detach())

        if self.pc_conv1.trace_activity_values:
            self.pc_conv1.trace["values"].append(data.x[:, 0].cpu().detach())

        self.pc_conv1.inference()
        print("Inference completed.")
        return True
 