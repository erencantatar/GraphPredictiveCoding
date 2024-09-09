
import torch 
import numpy as np 

def calculate_mse(tensor1, tensor2):
    """
    Calculate the Mean Squared Error between two PyTorch tensors.

    Parameters:
    tensor1 (torch.Tensor): The first tensor.
    tensor2 (torch.Tensor): The second tensor.

    Returns:
    float: The Mean Squared Error (MSE) between the two tensors.
    """
    
    # Ensure the tensors have the same shape
    if tensor1.shape != tensor2.shape:
        raise ValueError("Tensors must have the same dimensions.")
       # Convert numpy arrays to tensors if necessary
    if isinstance(tensor1, np.ndarray):
        tensor1 = torch.tensor(tensor1)
    if isinstance(tensor2, np.ndarray):
        tensor2 = torch.tensor(tensor2)
    
    # Ensure tensors are detached
    if tensor1.requires_grad:
        tensor1 = tensor1.detach()
    if tensor2.requires_grad:
        tensor2 = tensor2.detach()
    
    # Calculate the MSE
    mse = torch.mean((tensor1 - tensor2) ** 2).item()
    return mse
