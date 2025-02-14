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
        super(PredictionMessagePassing, self).__init__(aggr="add", flow="source_to_target")
        # super(PredictionMessagePassing, self).__init__(aggr="add", flow="target_to_source")

        self.f, self.activation_derivative = set_activation(activation)

    def forward(self, values, edge_index, weight_matrix, norm=None):
        # Ensure default normalization
        if norm is None:
            norm = torch.ones(edge_index.size(1), device=values.device)

        # Perform prediction message passing (aggregating predictions)
        return self.propagate(edge_index, values=values, weight_matrix=weight_matrix, norm=norm)

    def message(self, values_j, weight_matrix, norm):
        # θj,i * f(xj,t)
        return weight_matrix.view(-1, 1) * self.f(values_j).view(-1, 1) * norm.view(-1, 1)

    def update(self, aggr_out):
        # Update predictions as aggregated messages
        return aggr_out.view(-1, 1)


import torch
from torch_geometric.nn import MessagePassing



class ValueMessagePassing(MessagePassing):
    def __init__(self, activation):
        super(ValueMessagePassing, self).__init__(aggr="add", flow="source_to_target")
        self.activation, self.f_prime = set_activation(activation)

    def forward(self, values, errors, edge_index, weight_matrix, norm=None):
        # Ensure default normalization
        if norm is None:
            norm = torch.ones(edge_index.size(1), device=values.device)

        # Perform value update message passing (aggregating errors and updating values)
        return self.propagate(edge_index, values=values, errors=errors, weight_matrix=weight_matrix, norm=norm)

    def message(self, errors_j, weight_matrix, norm):
        # ε_{k,t} θ_{k,i}
        return weight_matrix.view(-1, 1) * errors_j.view(-1, 1) * norm.view(-1, 1)

    def update(self, aggr_out, values, errors):
        # Compute the derivative of the activation function applied to the values
        f_prime_values = self.f_prime(values).view(-1, 1)

        # Compute change in values using Δx_{i,t} = γ (-ε_{i,t} + f'(x_{i,t}) ∑_k ε_{k,t} θ_{k,i})
        # delta_x = -values.view(-1, 1) + f_prime_values * aggr_out
        delta_x = -errors.view(-1, 1) + f_prime_values * aggr_out

        return delta_x.view(-1, 1)