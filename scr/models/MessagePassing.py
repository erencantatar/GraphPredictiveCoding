import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree


""" 
MessagePassing Classes for nodepredictions and node value updates :

---------------PredictionMessagePassing---------------
prediction u = ∑




---------------ValueMessagePassing---------------

Δx = .... + ∑

"""

from helper.activation_func import set_activation 


class PredictionMessagePassing(MessagePassing):
    def __init__(self, activation):
        super(PredictionMessagePassing, self).__init__(aggr="add",flow="source_to_target")
        # Initialize the activation function and its derivative
        self.f, self.activation_derivative = set_activation(activation)
        # GET THIS with init. the 
        # self.edge_index = edge_index

    def forward(self, x, edge_index, weight_matrix, norm):
        # x: Node features (values, errors, predictions)
        # edge_index: Graph connectivity (2, num_edges)
        # edge_weight: Edge weights (num_edges,)

        # Perform message passing
        # μi,t = ∑_j θj,i * f(xj,t)

        # need to have  num_nodes,3 (features) not N,3,1 
        if x.dim() == 3 and x.size(2) == 1:
            x = x.squeeze(2)  # Now x should have shape (num_nodes, 3)

        
        #  Step 3: Compute normalization.
        # if self.normalize_msg: 
        #     row, col = edge_index
        #     deg = degree(col, x.size(0), dtype=x.dtype)
        #     deg_inv_sqrt = deg.pow(-0.5)
        #     deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        #     norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        # else:
        #     # make norm the identity
        #     norm = torch.ones(edge_index.size(1), device=x.device)

        return self.propagate(edge_index, x=x, weight_matrix=weight_matrix, norm=norm)
        
            

    def message(self, x_j, weight_matrix, norm):
        # x_j: Node features of the neighboring nodes
        # edge_weight: Weights of the edges

        # Compute the message for each edge, which is the weighted activated value of the neighboring node

        # COMPUTES θj,i * f(xj,t) for each edge
        
        # return thetaJI * self.activation(x_j[:, 0]).view(-1, 1)

        return weight_matrix.view(-1, 1) * self.f(x_j[:, 0]).view(-1, 1) 
        # return weight_matrix.view(-1, 1) * self.f(x_j[:, 0]).view(-1, 1) * norm 
        # return norm.view(-1, 1) * edge_weight.view(-1, 1) * self.activation(x_j[:, 0]).view(-1, 1)

    def update(self, aggr_out, x):
        # aggr_out: Aggregated messages for each node
        # x: Node features (values, errors, predictions)

        # Extract current values, errors, and predictions
        # values, errors, predictions = x[:, 0], x[:, 1], x[:, 2]

        # Update predictions based on aggregated messages
        # μi,t is updated with the aggregated message
        predictions = aggr_out

        return predictions.view(-1,1)
       


import torch
from torch_geometric.nn import MessagePassing

class ValueMessagePassing(MessagePassing):
    def __init__(self, activation):
        super(ValueMessagePassing, self).__init__(aggr="add", flow="source_to_target")
        # Initialize the activation function and its derivative
        self.activation, self.f_prime = set_activation(activation)
        

    def forward(self, x, edge_index, weight_matrix, norm):
        # x: Node features (values, errors, predictions)
        # edge_index: Graph connectivity (2, num_edges)
        # edge_weight: Edge weights (num_edges,)

        # need to have  num_nodes,3 (features) not N,3,1 
        if x.dim() == 3 and x.size(2) == 1:
            x = x.squeeze(2)  # Now x should have shape (num_nodes, 3)

        # #  Step 3: Compute normalization.
        """ COmputed ones since the graph are the same"""
        # if self.normalize_msg: 
        #     row, col = edge_index
        #     deg = degree(col, x.size(0), dtype=x.dtype)
        #     deg_inv_sqrt = deg.pow(-0.5)
        #     deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        #     norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        # else:
        #     # make norm the identity
        #     norm = torch.ones(edge_index.size(1), device=x.device)

        return self.propagate(edge_index, x=x, weight_matrix=weight_matrix, norm=norm)
        # return self.propagate(edge_index, x=x, weight_matrix=weight_matrix)

    def message(self, x_j, weight_matrix, norm):
        # x_j: Node features of the neighboring nodes (source nodes in edge_index)
        # x_i: Node features of the destination nodes in edge_index
        # edge_weight: Weights of the edges

        # Compute the message for each edge, which is the error times the edge weight
        # ε_{k,t} θ_{k,i}
        # errors_j = x_j[:, 1].view(-1, 1)  # Errors of the source nodes
        # return edge_weight.view(-1, 1) * errors_j

        errors_j = x_j[:, 1].view(-1, 1)
        return weight_matrix.view(-1, 1) * errors_j
        # return weight_matrix.view(-1, 1) * errors_j * norm 


    def update(self, aggr_out, x):
        # aggr_out: Aggregated messages for each node
        # x: Node features (values, errors, predictions)

        # Extract current values and errors
        values, errors = x[:, 0], x[:, 1]

        # Compute the derivative of the activation function applied to the values
        f_prime_x_i = self.f_prime(values).view(-1, 1)

        # Compute the change in values using the provided equation
        # Δx_{i,t} = γ (-ε_{i,t} + f'(x_{i,t}) ∑_k ε_{k,t} θ_{k,i})
        delta_x = (-errors.view(-1, 1) + f_prime_x_i * aggr_out)

        return delta_x.view(-1,1)