from collections import deque
from typing import Dict, Optional, Literal
import torch
import torch.nn as nn


def gradfilter_ma(
    m: nn.Module,
    grads: Optional[Dict[str, deque]] = None,
    window_size: int = 100,
    lamb: float = 5.0,
    filter_type: Literal['mean', 'sum'] = 'mean',
    warmup: bool = True,
    trigger: bool = False, # For ablation study.
    param_type: str = "weights"  # Add param_type to control which params to filter
) -> Dict[str, deque]:
    if grads is None:
        grads = {n: deque(maxlen=window_size) for n, p in m.named_parameters() if p.requires_grad and p.grad is not None}

    for n, p in m.named_parameters():
        if p.requires_grad and p.grad is not None:
            grads[n].append(p.grad.data.detach()) # .cpu())

            # Modify the gradients.
            if not warmup or len(grads[n]) == window_size and not trigger:
                if filter_type == "mean":
                    avg = sum(grads[n]) / len(grads[n])
                elif filter_type == "sum":
                    avg = sum(grads[n])
                else:
                    raise ValueError(f"Unrecognized filter_type {filter_type}")
                p.grad.data = p.grad.data + avg * lamb

    return grads


def gradfilter_ema(
    m: nn.Module,
    grads: Optional[Dict[str, torch.Tensor]] = None,
    alpha: float = 0.98,
    lamb: float = 2.0,
    param_type: str = "weights"  # Add param_type to control which params to filter
) -> Dict[str, torch.Tensor]:
    

    if grads is None:
        if param_type == 'values_dummy':
            grads = {'values_dummy': m.values_dummy.grad.detach()}

        elif param_type == 'weights':
            grads = {'weights': m.weights.grad.detach()}
        else:
            grads = {}

    # if param_type == 'values_dummy':
    #     grads['values_dummy'] = grads['values_dummy'] * alpha + m.values_dummy.grad.detach() * (1 - alpha)
    #     m.values_dummy.grad = m.values_dummy.grad + grads['values_dummy'] * lamb
    if param_type == 'weights':
        grads['weights'] = grads['weights'] * alpha + m.weights.grad.detach() * (1 - alpha)
        m.weights.grad = m.weights.grad + grads['weights'] * lamb

    return grads