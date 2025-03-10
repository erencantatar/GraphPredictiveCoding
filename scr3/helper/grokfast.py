from collections import deque
from typing import Dict, Optional, Literal
import torch
import torch.nn as nn

### Imports
from collections import deque
from typing import Dict, Optional, Literal
import torch
from typing import Dict, Optional

import torch
import torch.nn as nn
import wandb


def gradfilter_ema_adjust(
    new_grad: torch.Tensor,  # Current gradient 
    old_grad: Optional[torch.Tensor] = None,  # Previous averaged gradient
    alpha: float = 0.99,  # EMA decay factor
    lamb: float = 5.0,  # Amplification factor
) -> torch.Tensor:
    """
    Combines current gradient with historical gradient using exponential moving average.
    
    Args:
        new_grad: Current gradient tensor
        old_grad: Previous averaged gradient tensor (None for first iteration)
        alpha: EMA decay factor (higher means more weight on history)
        lamb: Amplification factor for the averaged gradient
        
    Returns:
        final_grad: Modified gradient combining current and historical information
    """
    # Initialize historical gradient if None
    if old_grad is None:
        old_grad = new_grad.detach().clone()
        
    # Compute EMA of gradients
    avg_grad = old_grad * alpha + new_grad.detach() * (1 - alpha)
    
    # Combine current gradient with amplified historical average
    final_grad = new_grad + avg_grad * lamb
    
    return final_grad, avg_grad  # Return both for next iteration



### Grokfast-EMA adjusted
def gradfilter_ema(
    grads: Optional[Dict[str, torch.Tensor]] = None,
    params: Optional[Dict[str, torch.nn.Parameter]] = None,
    alpha: float = 0.99,
    lamb: float = 5.0,
) -> Dict[str, torch.Tensor]:
    """
    Exponential Moving Average (EMA) for gradients.

    Args:
        grads: Dictionary of existing gradients.
        params: Dictionary of named parameters from the model.
        alpha: EMA decay factor (0 < alpha < 1).
        lamb: Scaling factor for the EMA.

    Returns:
        Updated `grads` dictionary with EMA-applied gradients.
    """
    # Initialize grads if not provided
    if grads is None:
        grads = {n: p.grad.data.detach() for n, p in params.items() if p.requires_grad and p.grad is not None}

    for n, p in params.items():
        if p.requires_grad and p.grad is not None:  # Ensure the gradient exists
            # Update EMA of gradients

            grads[n] = grads[n] * alpha + p.grad.data.detach() * (1 - alpha)
            p.grad.data = p.grad.data + grads[n] * lamb

            # Log to wandb that the gradient exists
            wandb.log({f"has grad for {n}": 1})
            print(f"has grad for {n}")
        else:
            # Log to wandb if no gradient exists
            print(f"no grad for {n}")
            wandb.log({f"no grad for {n}": 1})

    return grads

### Grokfast-MA
def gradfilter_ma(
    m: nn.Module,
    grads: Optional[Dict[str, deque]] = None,
    window_size: int = 128,
    lamb: float = 5.0,
    filter_type: Literal['mean', 'sum'] = 'mean',
    warmup: bool = True,
    trigger: bool = False,
) -> Dict[str, deque]:
    if grads is None:
        grads = {n: deque(maxlen=window_size) for n, p in m.named_parameters() if p.requires_grad and p.grad is not None}

    for n, p in m.named_parameters():
        if p.requires_grad and p.grad is not None:  # Ensure the gradient exists
            grads[n].append(p.grad.data.detach())

            if not warmup or (len(grads[n]) == window_size and not trigger):
                if filter_type == "mean":
                    avg = sum(grads[n]) / len(grads[n])
                elif filter_type == "sum":
                    avg = sum(grads[n])
                else:
                    raise ValueError(f"Unrecognized filter_type {filter_type}")
                p.grad.data = p.grad.data + avg * lamb

    return grads
