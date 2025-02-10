from typing import Any

# ------------------------------------------

import torch
from typing import Any

activation_functions = {
        "tanh": torch.tanh,
        "relu": torch.relu,
        "leaky relu": torch.nn.functional.leaky_relu,  # Note: requires passing an additional argument for the negative slope, handled below
        "linear": lambda x: x,
        "sigmoid": torch.sigmoid,
        "hard_tanh": torch.nn.Hardtanh(min_val=-1.0, max_val=1.0, inplace=False, min_value=None, max_value=None),
        "swish": None,
    }

class Swish:
    def forward(self, x):
        return x * torch.sigmoid(x)

    def f_prime(self, x):
        # Derivative of Swish: sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
        sigmoid_x = torch.sigmoid(x)
        return sigmoid_x + x * sigmoid_x * (1 - sigmoid_x)



def set_activation(activation):
    
    swish = Swish()

    activation_functions = {
        "tanh": torch.tanh,
        "relu": torch.relu,
        "leaky relu": torch.nn.functional.leaky_relu,  # Note: requires passing an additional argument for the negative slope, handled below
        "linear": lambda x: x,
        "sigmoid": torch.sigmoid,
        "hard_tanh": torch.nn.Hardtanh(min_val=-1.0, max_val=1.0, inplace=False, min_value=None, max_value=None),
        "swish": swish.forward,
    }

    derivative_functions = {
        "tanh": lambda x: 1 - torch.tanh(x)**2,
        "relu": lambda x: torch.where(x > 0, torch.ones_like(x), torch.zeros_like(x)),
        "leaky relu": lambda x: torch.where(x > 0, torch.ones_like(x), 0.1 * torch.ones_like(x)),
        "linear": lambda x: torch.ones_like(x),
        "sigmoid": lambda x: torch.sigmoid(x) * (1 - torch.sigmoid(x)),
        "hard_tanh": lambda x: torch.where((x >= -1.0) & (x <= 1.0), torch.ones_like(x), torch.zeros_like(x)), 
        "swish": swish.f_prime
    }
    if activation not in activation_functions:
        raise NotImplementedError(f"Invalid activation function: {activation}. Supported activations are: {', '.join(activation_functions.keys())}")

    # Special-handling for "leaky relu" to pass the negative_slope parameter
    if activation == "leaky relu":
        f = lambda x: activation_functions[activation](x, negative_slope=0.1)
    else:
        f = activation_functions[activation]

    f_prime = derivative_functions[activation]

    return f, f_prime
