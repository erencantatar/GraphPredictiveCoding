import numpy as np
import torch
# from PRECO.utils import *


import numpy as np
import torch
from datetime import datetime


# Check if CUDA (GPU) is available
if torch.cuda.is_available():
    # Set the default device to the first available GPU (index 0)
    print("CUDA available, using GPU")
    torch.cuda.set_device(0)
    DEVICE = torch.device('cuda:0')
else:
    print("WARNING: CUDA not available, using CPU")
    DEVICE = torch.device('cpu')


#########################################################
# ACTIVATION FUNCTIONS
#########################################################

relu = torch.nn.ReLU()
tanh = torch.nn.Tanh()
sigmoid = torch.nn.Sigmoid()
silu = torch.nn.SiLU()
linear = torch.nn.Identity()
leaky_relu = torch.nn.LeakyReLU()

@torch.jit.script
def sigmoid_derivative(x):
    return torch.exp(-x)/((1.+torch.exp(-x))**2)

@torch.jit.script
def relu_derivative(x):
    return torch.heaviside(x, torch.tensor(0.))

@torch.jit.script
def tanh_derivative(x):
    return 1-tanh(x)**2

@torch.jit.script
def silu_derivative(x):
    return silu(x) + torch.sigmoid(x)*(1.0-silu(x))

@torch.jit.script
def leaky_relu_derivative(x):
    return torch.where(x > 0, torch.tensor(1.), torch.tensor(0.01))

def get_derivative(f):
    if f == sigmoid:
        return sigmoid_derivative
    elif f == relu:
        return relu_derivative
    elif f == tanh:
        return tanh_derivative
    elif f == silu:
        return silu_derivative
    elif f == linear:
        return 1
    elif f == leaky_relu:
        return leaky_relu_derivative
    else:
        raise NotImplementedError(f"Derivative of {f} not implemented")


#########################################################
# GENERAL METHODS
#########################################################

def onehot(y_batch, N):
    """
    y_batch: tensor of shape (batch_size, 1)
    N: number of classes
    """
    return torch.eye(N, device=DEVICE)[y_batch.squeeze().long()].float()

def to_vector(batch):
    batch_size = batch.size(0)
    return batch.reshape(batch_size, -1).squeeze()

def preprocess_batch(batch):
    batch[0] = set_tensor(batch[0])
    batch[1] = set_tensor(batch[1])
    return (batch[0], batch[1])

def preprocess(dataloader):
    return list(map(preprocess_batch, dataloader))

def set_tensor(tensor):
    return tensor.to(DEVICE)

def seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def count_parameters(layers, use_bias):
    """
    Counts the number of parameters in a hierarchical network with given layer arrangement.
    """
    n_params = 0
    for i in range(len(layers)-1):
        n_params += layers[i]*layers[i+1]
        if use_bias:
            n_params += layers[i+1]
    return n_params

#########################################################
# LITTLE DATA METHODS
#########################################################

def train_subset_indices(train_set, n_classes, no_per_class):
    """
    Selects indices of a subset of the training set, with no_per_class samples per
    class. Useful for training with less data.
    """
    if no_per_class ==0:  # return all indices
        return np.arange(len(train_set))
    else:
        train_indices = []
        for i in range(n_classes):
            train_targets = torch.tensor([train_set.dataset.targets[i] for i in train_set.indices]) # SLOW but fine for now
            indices = np.where(train_targets == i)[0]
            indices = np.random.choice(indices, size=no_per_class, replace=False)
            train_indices += list(indices)
        return train_indices

def print_class_counts(loader):
    """
    Prints the number of samples per class in the given dataloader.
    """
    n_classes = loader.dataset.dataset.targets.unique().shape[0]
    counts = torch.zeros(n_classes)
    for _, (_, y_batch) in enumerate(loader):
        for i in range(n_classes):
            temp = (torch.argmax(y_batch,dim=1) == i)
            counts[i] += torch.count_nonzero(temp) 
    print(f"Class counts: {counts}")


#########################################################
# PCG METHODS
#########################################################

def get_mask_hierarchical(layers, symmetric=False):
    """
    Generates a hierarchical mask for the given layer arrangement.
    Returns:
        torch.Tensor: A binary mask matrix of shape (N, N) where N is the total number of nodes.
    """
    rows, cols = get_mask_indices_hierarchical(layers)
    N = np.sum(layers)
    M = torch.zeros((N, N), device=DEVICE)
    M[rows, cols] = torch.ones(len(rows), device=DEVICE)
    M = M.T
    if symmetric:
        # Make the matrix symmetric
        M = torch.tril(M) + torch.tril(M).T
    return M

def get_nodes_partition(layers):
    """
    Partitions nodes into layers based on the layer arrangement.
    Returns:
        list[list[int]]: A list of lists, where each sublist contains the nodes of a layer.
    """
    nodes = np.arange(sum(layers))  # Number the nodes 0, ..., N-1

    nodes_partition = []
    for i in range(len(layers)):
        a = np.sum(layers[:i]).astype(int)
        b = np.sum(layers[:i+1]).astype(int)
        nodes_partition.append(nodes[a:b])
    return nodes_partition 

def get_mask_indices_hierarchical(layers):
    """
    Finds the matrix indices of nonzero weights for a hierarchical mask.
    Returns:
        tuple[list[int], list[int]]: Two lists representing the row and column indices of the nonzero weights.
    """
    # Partition nodes into layers
    nodes_partition = get_nodes_partition(layers)

    # Helper function to combine rows and columns
    def combine(x, y):
        z = np.array([(x_i, y_j) for x_i in x for y_j in y])
        rows, cols = z.T
        return rows.tolist(), cols.tolist()  # Returns rows, cols of matrix indices

    # Find matrix indices of nonzero weights
    all_rows, all_cols = [], []
    for i in range(len(layers)-1):
        rows, cols = combine(nodes_partition[i], nodes_partition[i+1])
        all_rows += rows
        all_cols += cols

    return all_rows, all_cols



class Optimizer(object):
    def __init__(self, params, optim_type, learning_rate, batch_scale=False, grad_clip=None, weight_decay=None):
        self._optim_type = optim_type
        self._params = params
        self.n_params = len(params)
        self.learning_rate = learning_rate
        self.grad_clip = grad_clip
        self.batch_scale = batch_scale
        self.weight_decay = weight_decay
        if isinstance(params["w"], list):  
            self.is_list = True  # PCN
        else:  
            self.is_list = False # PCG

        self._hparams = f"{optim_type}_lr={self.learning_rate}_gclip={self.grad_clip}_bscale={self.batch_scale}_wd={self.weight_decay}"

    @property
    def hparams(self):
        return self._hparams 
    
    @property
    def hparams_dict(self):
        return {"lr": self.learning_rate, "gradclip": self.grad_clip, "batchscale": self.batch_scale, "wd": self.weight_decay}

    def clip_grads(self, grad):
        if self.grad_clip is not None:
            grad = torch.clamp(grad, -self.grad_clip, self.grad_clip)

    def scale_batch(self, grad, batch_size):
        if self.batch_scale:
            grad /= batch_size

    # def decay_weights(self, param):
    #     if self.weight_decay is not None:
    #         param.grad["weights"] = param.grad["weights"] - self.weight_decay * param.weights

    def step(self, *args, **kwargs):
        raise NotImplementedError
    

class SGD(Optimizer):
    def __init__(self, params, learning_rate, batch_scale=False, grad_clip=None, weight_decay=None, model=None):
        super().__init__(params, optim_type="SGD", learning_rate=learning_rate, batch_scale=batch_scale, grad_clip=grad_clip,
                         weight_decay=weight_decay)

    def step(self, params, grads, batch_size):
        if self.is_list:
            for i in range(len(params["w"])):
                self._update_single_param(params["w"], grads["w"], i, batch_size, self.learning_rate)
                if params["use_bias"]:
                    self._update_single_param(params["b"], grads["b"], i, batch_size, self.learning_rate)
        else:
            self._update_single_param(params["w"], grads["w"], None, batch_size, self.learning_rate)
            if params["use_bias"]:
                self._update_single_param(params["b"], grads["b"], None, batch_size, self.learning_rate)

    def _update_single_param(self, param_group, grad_group, i, batch_size, learning_rate):
        if i is not None:
            param = param_group[i]
            grad = grad_group[i]
        else:
            param = param_group
            grad = grad_group

        self.scale_batch(grad, batch_size)
        self.clip_grads(grad)
        param -= learning_rate * grad

        if i is not None:
            param_group[i] = param
        else:
            param_group.copy_(param)


class Adam(Optimizer):
    def __init__(self, params, learning_rate, batch_scale=False, grad_clip=None, 
                 beta1=0.9, beta2=0.999, epsilon=1e-7, weight_decay=0, AdamW=False):
        super().__init__(params, optim_type="Adam", learning_rate=learning_rate, batch_scale=batch_scale,
                         grad_clip=grad_clip, weight_decay=weight_decay)

        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0  # Time step
        self.weight_decay = weight_decay
        self.AdamW = AdamW

        if self.is_list: # PCN
            self.m_w = [torch.zeros_like(param, device=DEVICE) for param in params["w"]]
            self.v_w = [torch.zeros_like(param, device=DEVICE) for param in params["w"]]
            self.m_b = [torch.zeros_like(param, device=DEVICE) for param in params["b"]]
            self.v_b = [torch.zeros_like(param, device=DEVICE) for param in params["b"]]
        else: # PCG
            self.m_w = torch.zeros_like(params["w"], device=DEVICE)
            self.v_w = torch.zeros_like(params["w"], device=DEVICE)
            self.m_b = torch.zeros_like(params["b"], device=DEVICE)
            self.v_b = torch.zeros_like(params["b"], device=DEVICE)

    def step(self, params, grads, batch_size):
        self.t += 1
        lr_t = self.learning_rate * np.sqrt(1. - self.beta2 ** self.t) / (1. - self.beta1 ** self.t)

        if self.is_list:
            for i in range(len(params["w"])):
                self._update_single_param(params["w"], grads["w"], self.m_w, self.v_w, i, batch_size, lr_t, self.AdamW)
                if params["use_bias"]:
                    self._update_single_param(params["b"], grads["b"], self.m_b, self.v_b, i, batch_size, lr_t, self.AdamW)
        else:
            self._update_single_param(params["w"], grads["w"], self.m_w, self.v_w, None, batch_size, lr_t, self.AdamW)
            if params["use_bias"]:
                self._update_single_param(params["b"], grads["b"], self.m_b, self.v_b, None, batch_size, lr_t, self.AdamW)

    def _update_single_param(self, param_group, grad_group, m_group, v_group, i, batch_size, lr_t, AdamW):
        if i is not None:
            param = param_group[i]
            grad = grad_group[i]
            m = m_group[i]
            v = v_group[i]
        else:
            param = param_group
            grad = grad_group
            m = m_group
            v = v_group

        # print("param_group, grad_group", param_group, grad_group)

        self.scale_batch(grad, batch_size)
        self.clip_grads(grad)

        grad += self.weight_decay * param
        m = self.beta1 * m + (1 - self.beta1) * grad
        v = self.beta2 * v + (1 - self.beta2) * (grad ** 2)
        step = lr_t * m / (torch.sqrt(v) + self.epsilon)

        if AdamW:
            param *= (1. - self.weight_decay * self.learning_rate)
        param -= step

        if i is not None:
            param_group[i] = param
            m_group[i] = m
            v_group[i] = v
        else:
            param_group.copy_(param)
            m_group.copy_(m)
            v_group.copy_(v)



class EarlyStopper:
    def __init__(self, patience=1, min_delta=0, verbose=True, relative=False):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_obj = float('inf')
        self.verbose = verbose
        self.relative = relative

    def early_stop(self, validation_obj):
        if np.isnan(validation_obj):
            print("Validation objective is NaN. Stopping early.")
            return True
        difference = validation_obj - self.min_validation_obj
        if self.relative:
            difference /= self.min_validation_obj
        if validation_obj < self.min_validation_obj:
            if self.verbose:
                print(f"Validation objective decreased ({self.min_validation_obj:.6f} --> {validation_obj:.6f}).")
            self.min_validation_obj = validation_obj
            self.counter = 0
        elif difference >= self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"Validation objective increased ({self.min_validation_obj:.6f} --> {validation_obj:.6f}).")
                print(f"Early stopping counter: {self.counter} out of {self.patience}\n")
            if self.counter >= self.patience:
                return True
        return False