import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import numpy as np
import math
from torch_geometric.utils import to_dense_adj
from models.MessagePassing import PredictionMessagePassing, ValueMessagePassing
from helper.activation_func import set_activation
import os

# import base PC-graph arch.
from models.PC import PCGraphConv


class IPCGraphConv(PCGraphConv): 
    def __init__(self, num_vertices, sensory_indices, internal_indices, 
                 learning_rate, T, graph_structure,
                 batch_size, use_learning_optimizer, weight_init, clamping,
                 supervised_learning=False, debug=False, activation=None, 
                 log_tensorboard=True, wandb_logger=None, device="cpu"):
        super(IPCGraphConv, self).__init__(num_vertices, sensory_indices, internal_indices,
                                           learning_rate, T, graph_structure,
                                           batch_size, use_learning_optimizer, weight_init, clamping,
                                           supervised_learning, debug, activation,
                                           log_tensorboard, wandb_logger, device)  # 'add' aggregation
        
        self.wandb_logger = wandb_logger  # Make sure the logger is passed


    # Overwrite the learning function
    def learning(self, data):

            self.data = data
            x, self.edge_index = self.data.x, self.data.edge_index
            
            # edge_index: has shape [2, E * batch] where E is the number of edges, but in each batch the edges are the same

            # 1. fix value nodes of sensory vertices to be 
            # self.restart_activity()
            # self.set_sensory_nodes()

            ## 2. Then, the total energy of Eq. (2) is minimized in two phases: inference and weight update. 
            ## INFERENCE: This process of iteratively updating the value nodes distributes the output error throughout the PC graph. 
            self.set_phase('inference')
            self.inference(self.data)
            self.set_phase('inference done')

            ## WEIGHT UPDATE inside of inference

            energy = self.energy(self.data)
            
            

    # Overwrite the inference function
    def inference(self, data, restart_activity=True):

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


        # self.edge_weights = self.extract_edge_weights(edge_index=self.edge_index, weights=self.weights, mask=self.mask)
        self.data = data
        # self.values, _pred_ , self.errors, = data.x[:, 0], data.x[:, 1], data.x[:, 2]

        print("Aaaa", self.data.x.shape)
        self.get_graph()

        energy = self.energy(self.data)

        from tqdm import tqdm

        t_bar = tqdm(range(self.T), leave=False)

        # t_bar.set_description(f"Total energy at time 0 {energy} Per avg. vertex {energy['internal_energy'] / len(self.internal_indices)}")
        t_bar.set_description(f"Total energy at time 0 {energy}")
        # print(f"Total energy at time 0", energy, "Per avg. vertex", energy["internal_energy"] / len(self.internal_indices))
        
        if self.mode == "training":

            for t in t_bar:
                
                # aggr_out = self.forward(data)
                self.t = t 

                self.update_values(self.data)
                
                self.set_phase('weight_update')
                self.weight_update(self.data)
                self.set_phase('weight_update done')

                energy = self.energy(self.data)
                t_bar.set_description(f"Total energy at time {t+1} / {self.T} {energy},")

        else:
            for t in t_bar:
        
                # aggr_out = self.forward(data)
                self.t = t 

                self.update_values(self.data)
                
                energy = self.energy(self.data)
                t_bar.set_description(f"Total energy at time {t+1} / {self.T} {energy},")
                
                             
        return True



class IPCGNN(torch.nn.Module):
    def __init__(self, num_vertices, sensory_indices, internal_indices, 
                 lr_params, T, graph_structure, 
                 batch_size, 
                 use_learning_optimizer=False, weight_init="xavier", clamping=None, supervised_learning=False, 
                 debug=False, activation=None, log_tensorboard=True, wandb_logger=None, device='cpu'):
        super(IPCGNN, self).__init__()
        
        """ TODO: in_channels, hidden_channels, out_channels, """
        # INSIDE LAYERS CAN HAVE PREDCODING - intra-layer 
        self.pc_conv1 = IPCGraphConv(num_vertices, sensory_indices, internal_indices, 
                                    lr_params, T, graph_structure, 
                                    batch_size, use_learning_optimizer, weight_init, clamping, supervised_learning, 
                                    debug, activation, log_tensorboard, wandb_logger, device)

        self.original_weights = None  # Placeholder for storing the original weights


    def log():
        pass
    
    def learning(self, batch):       
        
        self.pc_conv1.mode = "training"
        self.pc_conv1.learning(batch)
        
        history = {
            "internal_energy_mean": np.mean(self.pc_conv1.energy_vals["internal_energy"]),
            "sensory_energy_mean": np.mean(self.pc_conv1.energy_vals["sensory_energy"]),
            }
        
        return history
    

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

        for from_idx in from_indices:
            for to_idx in to_indices:
                # Find the corresponding edge in the graph
                edge_mask = (self.pc_conv1.edge_index_single_graph[0] == from_idx) & \
                            (self.pc_conv1.edge_index_single_graph[1] == to_idx)
                # Temporarily set the weights of these edges to zero
                self.pc_conv1.weights[edge_mask] = 0

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


    def load_weights(self, W, graph, b=None):

        print("Settng weights of self.pc_conv1")
        self.pc_conv1.weights = W 
        self.pc_conv1.edge_index = graph 

        if self.pc_conv1.use_bias:
            self.pc_conv1.bias = b  

        self.pc_conv1.values = torch.zeros(self.num_vertices,self.batch_size,device=self.device) # requires_grad=False)                

    
    def save_weights(self, path):
        

        # make dir if not exist
        if not os.path.exists(path):
            os.makedirs(path)

        W = self.pc_conv1.weights
        b = self.pc_conv1.biases 
        graph = self.pc_conv1.edge_index

        # save to '"trained_models/weights.pt"' 
        torch.save(W, f"{path}/weights.pt")
        torch.save(graph, f"{path}/graph.pt")

        if self.pc_conv1.use_bias:
            torch.save(b, f"{path}/bias.pt")


    def query(self, method, data=None):
        
        print("Random init values of all internal nodes")
        data.x[:, 0][self.pc_conv1.internal_indices] = torch.rand(data.x[:, 0][self.pc_conv1.internal_indices].shape).to(self.pc_conv1.device)


        self.pc_conv1.energy_vals["internal_energy_testing"] = []
        self.pc_conv1.energy_vals["sensory_energy_testing"] = []

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
                one_hot = torch.zeros(10)
                one_hot[data.y] = 1
                # one_hot = one_hot.view(-1, 1)
                one_hot = one_hot.to(self.device)
                self.pc_conv1.values.data[self.pc_conv1.supervised_labels] = one_hot
                
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

        
        self.inference(data)
        
        print("QUery by condition or query by init")

        return self.pc_conv1.get_graph()
        
   
    
    def inference(self, data):
        self.pc_conv1.inference(data)
        print("Inference completed.")
        return True
 