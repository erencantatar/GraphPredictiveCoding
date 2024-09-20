
activation_functions = [
    "tanh",
    "relu",
    "leaky relu",
    "linear",
    "sigmoid",
    "hard_tanh",
    "swish"]


import torch
import random 
from models.MessagePassing import PredictionMessagePassing, ValueMessagePassing
from helper.activation_func import set_activation


import inspect

def compare_class_args(class1, class2):
    # Get the signature (i.e., the argument details) of both class initializers
    sig1 = inspect.signature(class1.__init__)
    sig2 = inspect.signature(class2.__init__)
    
    # Extract the parameter names, ignoring 'self'
    params1 = [param.name for param in sig1.parameters.values() if param.name != 'self' and param.kind != param.VAR_POSITIONAL and param.kind != param.VAR_KEYWORD]
    params2 = [param.name for param in sig2.parameters.values() if param.name != 'self' and param.kind != param.VAR_POSITIONAL and param.kind != param.VAR_KEYWORD]
    
    # Compare parameter sets
    diff_class1 = set(params1) - set(params2)
    diff_class2 = set(params2) - set(params1)

    print(f"Arguments in {class1.__name__} but not in {class2.__name__}: {diff_class1}")
    print(f"Arguments in {class2.__name__} but not in {class1.__name__}: {diff_class2}")



def verify_prediction_message_passing(activation_functions):

    act = random.choice(list(activation_functions))
    print("Using activation function: ", act)

    # Node features: (values, errors, predictions)
    node_values = torch.tensor([[1.0], [0.5], [0.3]])
    node_errors = torch.tensor([[0.1], [0.2], [0.4]])
    node_preds = torch.tensor([[0.2], [0.3], [0.5]])
    
    # Combine them into a single tensor
    x = torch.cat((node_values, node_errors, node_preds), dim=1)
  
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])  # Edge connections
    
    # Define edge weights corresponding to the edges in edge_index
    # edge_weights = torch.tensor([1.0, 0.8, 0.5])  # Edge weights for edges 0->1, 1->2, 2->0
    # random edge weights 
    edge_weights = torch.randn(3)

    # Instantiate the PredictionMessagePassing with ReLU activation
    mp = PredictionMessagePassing(activation=act)
    output = mp(x, edge_index, edge_weights)

    # Manually calculate expected outputs based on the update rule from the paper
    f, f_prime = set_activation(act)

    expected_output = torch.zeros(x.size(0), 1)
    
    for i in range(x.size(0)):
        incoming_edges = (edge_index[1] == i).nonzero(as_tuple=True)[0]
        messages = []
        for edge in incoming_edges:
            src_node = edge_index[0, edge].item()
            weight = edge_weights[edge].item()
            messages.append(weight * f(x[src_node, 0]).item())
        expected_output[i, 0] = sum(messages)

    # Check if output matches the expected values based on the paper's formula
    print("Prediction Message Passing Output:\n", output)
    print("Expected Output:\n", expected_output)

    assert torch.allclose(output, expected_output, atol=1e-5), "PredictionMessagePassing output does not match the expected output."



import torch

def verify_value_message_passing(activation_functions):

    act = random.choice(list(activation_functions))
    print("Using activation function", act)
    # Define node features: (values, errors, predictions)
    node_values = torch.tensor([[1.0], [0.5], [0.3]])
    node_errors = torch.tensor([[0.1], [0.2], [0.4]])
    node_preds = torch.tensor([[0.2], [0.3], [0.5]])
    
    # Combine them into a single tensor
    x = torch.cat((node_values, node_errors, node_preds), dim=1)
    
    # Define the edge index and edge weights
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])  # Edge connections
    edge_weights = torch.tensor([1.0, 0.8, 0.5])  # Edge weights for edges 0->1, 1->2, 2->0

    # Instantiate the ValueMessagePassing with ReLU activation
    mp = ValueMessagePassing(activation=act)
    output = mp(x, edge_index, edge_weights)

    # Manually calculate expected updates based on the update rule from the paper
    f, f_prime = set_activation(act)

    expected_output = torch.zeros(node_values.size(0), 1)
    
    for i in range(node_values.size(0)):
        incoming_edges = (edge_index[1] == i).nonzero(as_tuple=True)[0]
        aggr_message = 0.0
        for edge in incoming_edges:
            src_node = edge_index[0, edge].item()
            weight = edge_weights[edge].item()
            aggr_message += weight * node_errors[src_node, 0].item()
        
        # Calculate the update for the node value
        f_prime_x_i = f_prime(node_values[i, 0])
        expected_output[i, 0] = -node_errors[i, 0] + f_prime_x_i * aggr_message

    # Check if output matches the expected updates based on the paper's formula
    print("Value Message Passing Output:\n", output)
    print("Expected Output:\n", expected_output)

    assert torch.allclose(output, expected_output, atol=1e-5), "ValueMessagePassing output does not match the expected output."




def validate_messagePassing():

    activation_functions = [
        "tanh",
        "relu",
        "leaky relu",
        "linear",
        "sigmoid",
        "hard_tanh",
        "swish"]

    # Run verification tests
    verify_prediction_message_passing(activation_functions)

    # Run verification tests
    verify_value_message_passing(activation_functions)