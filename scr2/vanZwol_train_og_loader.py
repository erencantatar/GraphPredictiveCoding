
import sys
# sys.path.append('/Users/6884407/PRECO')
sys.path.append('../')

# from PRECO.utils import *
# import PRECO.optim as optim
# from PRECO.PCG import *

import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, SubsetRandomSampler, random_split
from tqdm import tqdm
import torch

torch.set_printoptions(threshold=10000)


#################################### IMPORTS ####################################
import numpy as np
import torch
from datetime import datetime
# import logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

start_time = datetime.now()
dt_string = start_time.strftime("%Y%m%d-%H.%M")

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

# ---------------------------------------
import numpy as np
import torch

import torch
from torch_geometric.nn import MessagePassing
import torch.nn.functional as F

class PredictiveCodingLayer(MessagePassing):
    def __init__(self, f, f_prime):
        # super().__init__(aggr='add')  # Sum aggregation
        super().__init__(aggr='add', flow="target_to_source")  # Sum aggregation

        self.f, self.f_prime = f, f_prime
        
    def message_mu(self, x_j, weight):
        """
        Forward pass to compute predictions μ.
        - x_j: Neighboring node activations (scalar values)
        - W: Weights associated with edges
        """
        # return self.f(x_j) * weight
        # return self.f(x_j) * weight.unsqueeze(-1)   
        return self.f(x_j) * weight  
        # return weight * self.f(x_j) 
    

    def message_delta_x(self, epsilon_j, x_j, weight):
        """
        Backward pass for updating node activations using W^T.
        - epsilon_j: Error from neighboring nodes
        - x_j: Neighboring node activations
        """
        return self.f_prime(x_j) * epsilon_j * weight

    def forward(self, x, edge_index, weight):
        """
        Compute prediction μ, error ε, and update Δx.
        Inputs:
        - x: Node activations (shape [num_nodes, 1])
        - edge_index: Edge index tensor (shape [2, num_edges])
        """

        # reversed_edge_index = edge_index[[1, 0]]  # Transpose the edges for W^T
        reversed_edge_index = edge_index.flip(0)  # Transpose the edges for W^T

        # Step 1: Compute predictions μ using W (forward pass)
        # mu = self.propagate(reversed_edge_index, x=x, weight=weight, message=self.message_mu)
        mu = self.propagate(edge_index, x=x, weight=weight, message=self.message_mu)
        
        # Step 2: Compute prediction error ε = x - μ
        epsilon = x - mu

        # Step 3: Compute Δx using W^T explicitly

        # delta_x = -self.gamma * self.propagate(reversed_edge_index, x=x, epsilon=epsilon, message=self.message_delta_x)
        delta_x = self.propagate(reversed_edge_index, x=x, epsilon=epsilon, weight=weight, message=self.message_delta_x)
        
        return epsilon, mu, delta_x




# Message passing layer
class PredictionMessagePassing(MessagePassing):
    def __init__(self, f):
        super().__init__(aggr='add', flow="target_to_source")  # Sum aggregation
        # super().__init__(aggr='add', flow="source_to_target")  # Sum aggregation
        # self.f = torch.tanh
        self.f = f

    def forward(self, x, edge_index, weight):
        # Start message passing
        return self.propagate(edge_index, x=x, weight=weight)
    
    def message(self, x_j, weight):
        # Apply activation to the source node's feature
        # return self.f(x_j) * weight.unsqueeze(-1) weight[:, None]
        return self.f(x_j) * weight[:, None]
        # return self.f(x_j) 
    
    def update(self, aggr_out):
        # No bias or additional transformation; return the aggregated messages directly
        return aggr_out
    


class GradientMessagePassing(MessagePassing):
    def __init__(self, dfdx):
        # super().__init__(aggr='add', flow="target_to_source")  # Reverse flow for backward pass
        super().__init__(aggr='add', flow="source_to_target")  # Reverse flow for backward pass

        self.dfdx = dfdx  # Activation derivative

    def forward(self, x, edge_index, error, weight):
        # Propagate messages backward using edge_index and node data
        return self.propagate(edge_index, x=x, error=error, weight=weight)

    def message(self, error_j, x_i, weight):
        # Compute the propagated error using the activation derivative and weight

        # error_j = error_j.view(-1, 1)  # Ensure shape consistency
        # x_i = x_i.view(-1, 1)  # Ensure shape consistency
        # weight = weight.view(-1, 1)  # Ensure shape consistency

        return self.dfdx(x_i) * error_j * weight

    def update(self, aggr_out, error):
        # Align with: e - dfdx(x) * torch.matmul(e, w.T)
        return error - aggr_out


    # def grad_x(self, x, e, w, b, train):
    #     lower = self.shape[0]
    #     upper = -self.shape[2] if train else sum(self.shape)
    #     return e[:,lower:upper] - self.dfdx(x[:,lower:upper]) * torch.matmul(e, w.T[lower:upper,:].T)



# --------------------------------------------------

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
            self.m_w = torch.zeros_like(params["w"].to_dense(), device=DEVICE)
            self.v_w = torch.zeros_like(params["w"].to_dense(), device=DEVICE)
            self.m_b = torch.zeros_like(params["b"].to_dense(), device=DEVICE)
            self.v_b = torch.zeros_like(params["b"].to_dense(), device=DEVICE)

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

        # Ensure m is dense
        if m.is_sparse:
            m = m.to_dense()

        # Ensure grad is dense (just in case)
        if grad.is_sparse:
            grad = grad.to_dense()

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


# --------------------------------------------------

# from PRECO.optim import *
# from PRECO.utils import *
# from PRECO.structure import *
from torch import nn
from scipy.ndimage import label, find_objects
import torch

class PCStructure:
    """
    Abstract class for PC structure.

    Args:
        f (torch.Tensor -> torch.Tensor): Activation function.
        use_bias (bool): Whether to use bias.
    """
    def __init__(self, f, use_bias):
        self.f = f
        self.dfdx = get_derivative(f)
        self.use_bias = use_bias

class PCmodel:
    """
    Abstract class for PC model.
    
    Args:
        structure (PCStructure): Structure of the model.
        lr_x (float): Learning rate for the input.
        T_train (int): Number of training iterations.
        incremental (bool): Whether to use incremental EM.
        min_delta (float): Minimum change in energy for early stopping.
        early_stop (bool): Whether to use early stopping.
    """
    def __init__(self, structure: PCStructure, lr_x: float, T_train: int, 
                 incremental: bool, min_delta: float, early_stop: bool):
        self.structure = structure
        self.lr_x = torch.tensor(lr_x, dtype=torch.float, device=DEVICE)
        self.T_train = T_train
        self.incremental = incremental
        self.min_delta = min_delta
        self.early_stop = early_stop

    def weight_init(self, param):
        nn.init.normal_(param, mean=0, std=0.05)   

    def bias_init(self, param):
        nn.init.normal_(param, mean=0, std=0) 


class PCGStructure(PCStructure):
    """
    Abstract class for PCG structure.

    Args:
        shape (tuple): Number of input, hidden, and output nodes.
        f (torch.Tensor -> torch.Tensor): Activation function.
        use_bias (bool): Whether to use bias.
        mask (torch.Tensor): Mask for the weight matrix.
    """
    def __init__(self, shape, f, use_bias, mask):
        super().__init__(f, use_bias)
        self.shape = shape
        self.mask = mask

        if self.mask is not None:
            # Move the mask to the CPU before using NumPy
            if np.all(np.triu(self.mask.cpu().numpy(), k=1) == 0):
                labeled_matrix, self.num_layers = label(self.mask.cpu().numpy(), structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]))
                blocks = np.array([(slice_obj[1].stop - slice_obj[1].start,
                                slice_obj[0].stop - slice_obj[0].start)
                                for slice_obj in find_objects(labeled_matrix)])
                self.layers = blocks[:, 0].tolist() + [blocks[-1, -1]]
                # logging.info(f"Hierarchical mask, layers: {self.num_layers}, using feedforward initialization and testing.")
            else:
                self.num_layers = None
                # logging.info("Non-hierarchical mask, using random initialization and iterative testing.")

        self.N = sum(self.shape)

    @property
    def hparams(self):
        return {"shape": self.shape, "f": self.f, "use_bias": self.use_bias, "mask": self.mask}

    def pred(self, x, w, b):
        raise NotImplementedError

    def grad_x(self, x, e, w, b, train):
        raise NotImplementedError

    def grad_w(self, x, e, w, b):
        raise NotImplementedError

    def grad_b(self, x, e, w, b):
        raise NotImplementedError
    

class PCG_AMB(PCGStructure):
    """
    PCGStructure class with convention: mu = wf(x)+b.
    """
    def pred(self, x, w, b):
        bias = b if self.use_bias else 0
        return torch.matmul(self.f(x), w.T) + bias

    def grad_x(self, x, e, w, b, train):
        lower = self.shape[0]
        upper = -self.shape[2] if train else sum(self.shape)
        return e[:,lower:upper] - self.dfdx(x[:,lower:upper]) * torch.matmul(e, w.T[lower:upper,:].T)

    def grad_w(self, x, e, w, b):
        out = -torch.matmul(e.T, self.f(x))
        if self.mask is not None:
            out *= self.mask
        return out

    def grad_b(self, x, e, w, b,):
        return -torch.sum(e, axis=0)


class PCG_MBA(PCGStructure):
    """
    PCGStructure class with convention: mu = f(xw+b).
    """
    def pred(self, x, w, b):
        bias = b if self.use_bias else 0
        return self.f(torch.matmul(x, w.T) + bias)
    
    def grad_x(self, x, e, w, b, train):
        lower = self.shape[0]
        upper = -self.shape[2] if train else sum(self.shape)
        bias = b[lower:upper] if self.use_bias else 0
        temp = self.dfdx( torch.matmul(x, w.T)[:,lower:upper] + bias)
        return e[:,lower:upper] - temp*torch.matmul(e, w.T[lower:upper,:].T)
    
    def grad_w(self, x, e, w, b):
        bias = b if self.use_bias else 0
        temp = e*self.dfdx( torch.matmul(x, w.T) + bias )
        out = -torch.matmul( temp.T,  x ) # matmul takes care of batch sum
        out *= self.mask if self.mask is not None else 1
        return out

    def grad_b(self, x, e, w, b):
        bias = b if self.use_bias else 0
        temp = self.dfdx( torch.matmul(x, w.T) + bias )
        return torch.sum(-e*temp, axis=0) # batch sum



class PCNStructure(PCStructure):
    """
    Abstract class for PCN structure.

    Args:
        layers (list): Number of nodes in each layer.
        f (torch.Tensor -> torch.Tensor): Activation function.
        use_bias (bool): Whether to use bias.
        upward (bool): Whether the structure is upward (discriminative) or downward (generative).
        fL (torch.Tensor -> torch.Tensor): Activation function for the last layer.
    """
    def __init__(self, layers, f, use_bias, upward, fL=None):
        super().__init__(f, use_bias)
        self.layers = layers
        self.upward = upward
        if fL is None:
            self.fL = f
            self.dfLdx = self.dfdx
        else:
            self.fL = fL
            self.dfLdx = get_derivative(fL)
        self.L = len(layers) - 1

    @property
    def hparams(self):
        return {"layers": self.layers, "f": self.f, "use_bias": self.use_bias, "upward": self.upward}

    # NOTE: this implementation is somewhat inefficient; costs around 1s/epoch for MNIST
    def fl(self, x, l):
        if l == self.L:
            return self.fL(x)
        else:
            return self.f(x)
        
    def dfldx(self, x, l):
        if l == self.L:
            return self.dfLdx(x)
        else:
            return self.dfdx(x)

    def pred(self, l, x, w, b):
        raise NotImplementedError

    def grad_x(self, l, x, e, w, b, train):
        raise NotImplementedError

    def grad_w(self, l, x, e, w, b):
        raise NotImplementedError

    def grad_b(self, l, x, e, w, b):
        raise NotImplementedError


class PCN_AMB(PCNStructure):
    """
    PCGNtructure class with convention mu = wf(x)+b.
    """
    def pred(self, l, x, w, b):
        k = l - 1 if self.upward else l + 1
        bias = b[k] if self.use_bias else 0
        out = torch.matmul(self.fl(x[k], l), w[k])
        return out + bias

    def grad_x(self, l, x, e, w, b, train):
        k = l + 1 if self.upward else l - 1

        if l != self.L:
            grad = e[l] - self.dfldx(x[l], k) * (torch.matmul(e[k], w[l].T))
        else:
            if train:
                grad = 0
            else:
                if self.upward:
                    grad = e[l]
                else:
                    grad = -self.dfldx(x[l], k) * (torch.matmul(e[k], w[l].T))
        return grad

    def grad_w(self, l, x, e, w, b):
        k = l + 1 if self.upward else l - 1
        return -torch.matmul(self.fl(x[l].T, k), e[k])

    def grad_b(self, l, x, e, w, b):
        k = l + 1 if self.upward else l - 1
        return -e[k]


class PCN_MBA(PCNStructure):
    """
    PCNStructure class with convention mu = f(xw+b).
    """
    def pred(self, l, x, w, b):
        k = l - 1 if self.upward else l + 1
        bias = b[k] if self.use_bias else 0
        out = torch.matmul(x[k], w[k])
        return self.fl(out + bias, l)

    def grad_x(self, l, x, e, w, b, train):
        k = l + 1 if self.upward else l + 1
        bias = b[l] if self.use_bias else 0

        if l != self.L:
            temp = torch.matmul(x[l], w[l]) + bias
            grad = e[l] - torch.matmul(e[k] * self.dfldx(temp, k), w[l].T)
        else:
            if train:
                grad = 0
            else:
                if self.upward:
                    grad = e[l]
                else:
                    temp = torch.matmul(x[l], w[l]) + bias
                    grad = -torch.matmul(e[k] * self.dfldx(temp, k), w[l].T)
        return grad

    def grad_w(self, l, x, e, w, b):
        k = l + 1 if self.upward else l - 1
        bias = b[l] if self.use_bias else 0
        temp = e[k] * self.dfldx(torch.matmul(x[l], w[l]) + bias, k)
        return -torch.matmul(x[l].T, temp)

    def grad_b(self, l, x, e, w, b):
        k = l + 1 if self.upward else l - 1
        bias = b[l] if self.use_bias else 0
        return -e[k] * self.dfldx(torch.matmul(x[l], w[l]) + bias, k)  # same calc as grad_w so could make more efficient

# -------------------------------------------------------- 
# from PRECO.optim import *
# from PRECO.structure import *



class PCgraph(torch.nn.Module): 

    def __init__(self, f, device, num_vertices, num_internal, adj, edge_index, batch_size, lr_x, T_train, T_test, incremental, use_input_error, node_init_std=None, min_delta=None, early_stop=None):
        super().__init__()

        self.device = device

        self.num_vertices = num_vertices
        self.num_internal = num_internal
        self.adj = torch.tensor(adj).to(self.device)
        
        self.edge_index = edge_index  # PYG edge_index

        self.lr_x = lr_x 
        self.T_train = T_train
        self.T_test = T_test
        self.node_init_std = node_init_std
        self.incremental = incremental 
        self.min_delta = min_delta
        self.early_stop = early_stop

        self.epoch = 0 
        self.batch_size = batch_size  # Number of graphs in the batch

        self.f = f
        self.dfdx = get_derivative(f)
        self.use_input_error = use_input_error
        self.trace = False 

        # self.w = nn.Parameter(torch.empty(num_vertices, num_vertices, device=self.device))
        # self.b = nn.Parameter(torch.empty(num_vertices, device=self.device))
        self.device = device 

        import torch_geometric 
        self.edge_index = torch_geometric.utils.dense_to_sparse(adj)[0]
        # print("edge_index: ", edge_index)
        # print("self.edge_index: ", self.edge_index)
        # assert self.edge_index == edge_index        
        
        # assert torch.all(self.edge_index == edge_index)

        self.adj = torch.tensor(adj).to(DEVICE)
        self.mask = self.adj
        
        self._reset_grad()
        self._reset_params()

        self.optimizer_w = torch.optim.Adam([self.w], lr=lr_w, betas=(0.9, 0.999), eps=1e-7, weight_decay=0)


        # self.MP = PredictiveCodingLayer(f=self.structure.f, 
        #                                 f_prime=self.structure.dfdx)

        self.pred_mu_MP = PredictionMessagePassing(self.f)
        self.grad_x_MP = GradientMessagePassing(self.dfdx)

        self.mode = "train"
        self.use_bias = False

        self.test_supervised = self.test_iterative

    # @property
    # def hparams(self):
    #     return {"lr_x": self.lr_x, "T_train": self.T_train, "T_test": self.T_test, "incremental": self.incremental,
    #              "min_delta": self.min_delta,"early_stop": self.early_stop, "use_input_error": self.use_input_error, "node_init_std": self.node_init_std}

    # @property
    # def params(self):
    #     return {"w": self.w, "b": self.b, "use_bias": self.use_bias}
    
    # @property
    # def grads(self):
    #     return {"w": self.dw, "b": self.db}

    # def w_to_dense(self, w_sparse):
    #     # Convert 1D sparse weights back to dense (N, N) matrix
    #     adj_dense = torch.zeros((self.num_vertices, self.num_vertices), device=DEVICE)
    #     adj_dense[self.edge_index[0], self.edge_index[1]] = w_sparse
    #     return adj_dense

   
    # def w_to_sparse(self, w_sparse):
    #     # Convert sparse weights to dense (on GPU) before finding non-zero indices
    #     w_dense = w_sparse.to_dense().to(DEVICE)

    #     # Find non-zero indices
    #     # edge_index_sparse = torch.nonzero(w_dense, as_tuple=False).t()

    #     # Retrieve the corresponding weights
    #     edge_weights_sparse = w_dense[self.edge_index[0], self.edge_index[1]]

    #     return edge_weights_sparse.to(DEVICE)


        
    def _reset_params(self):

        self.w = torch.nn.Parameter(torch.empty(self.num_vertices, self.num_vertices, device=DEVICE))
        # self.w = torch.empty( self.num_vertices, self.num_vertices, device=DEVICE)
       
        # best for Classification
        # nn.init.normal_(self.w, mean=0, std=0.05)  
        # # 

        # # trying for generation
        # nn.init.normal_(self.w, mean=0.1, std=0.05)  


        # # # # # BEST FOR GENERATION
        self.w.data.fill_(0.001)
        # self.w.data.fill_(0.0001)
        # Add small random noise
        noise = torch.randn_like(self.w) * 0.0001
        self.w.data.add_(noise)
        

        # Perform the operation and reassign self.w as a Parameter
        with torch.no_grad():
            self.w.copy_(self.adj * self.w)

        # self.w = self.adj * self.w 


        ## -------------------------------------------------

        # import scipy sparse coo
        from scipy.sparse import coo_matrix

        # matrix of N, N
        # self.w_sparse = coo_matrix((self.num_vertices, self.num_vertices))
        # self.weight_init(self.w_sparse)

        # Initialize edge weights as a torch.nn.Parameter
        # edge_weights = torch.nn.Parameter(torch.empty(edge_index.size(1)))
        # num_edges = self.edge_index.size(1)
        # weights_1d = torch.empty(num_edges, device=DEVICE)
        # nn.init.normal_(weights_1d, mean=0.01, std=0.05)

        # N = self.num_vertices
                
        # # Create the sparse matrix using PyTorch sparse functionality
        # self.w = torch.sparse_coo_tensor(self.edge_index, weights_1d, size=(N, N), device=DEVICE)
        # # Convert to a dense (N, N) matrix if needed
        # dense_matrix = self.w.to_dense()

        # print("Initialized 1D Weights:\n", weights_1d.shape)
        # print("\nDense (N, N) Matrix:\n", dense_matrix.shape)

        # # Extract weights corresponding to the original edge index
        # reconstructed_weights_1d = dense_matrix[self.edge_index[0], self.edge_index[1]]

        # print("Reconstructed 1D Weights:", reconstructed_weights_1d.shape)

        # # import matplotlib.pyplot as plt

        # # save png the weights 
        # plt.imshow(dense_matrix.cpu().numpy())
        # # save
        # plt.savefig("weights.png")

        # # also save the adj_dense
        # plt.imshow(self.adj.cpu().numpy())
        # plt.savefig("adj.png")

        # # close the plots 
        # plt.close()

        ## -------------------------------------------------
        self.b = torch.empty( self.num_vertices, device=DEVICE)
        # if self.structure.use_bias:
        #     self.bias_init(self.b)

    def get_dense_weight(self):
        w = torch.tensor(self.w_sparse.toarray(), device=DEVICE)
        # w = self.w_sparse.toarray()
        assert w.shape == (self.num_vertices, self.num_vertices)
        return w

    def _reset_grad(self):
        self.dw, self.db = None, None

    def reset_nodes(self, batch_size=1):
        self.e = torch.empty(batch_size, sum(self.structure.shape), device=DEVICE)
        self.x = torch.zeros(batch_size, sum(self.structure.shape), device=DEVICE)

    def clamp_input(self, inp):
        di = self.structure.shape[0]
        self.x[:,:di] = inp.clone()

    def clamp_target(self, target):
        do = self.structure.shape[2]
        self.x[:,-do:] = target.clone()
        
    def init_hidden_random(self):
        di = self.structure.shape[0]
        do = self.structure.shape[2]
        self.x[:,di:-do] = torch.normal(0.5, self.node_init_std,size=(self.structure.shape[1],), device=DEVICE)

    def init_hidden_feedforward(self):
        self.forward(self.num_verticesum_layers-1)

    def init_output(self):
        do = self.structure.shape[2]
        self.x[:,-do:] = torch.normal(0.5, self.node_init_std, size=(do,), device=DEVICE)

    def forward(self, no_layers):
        temp = self.x.clone()
        for l in range(no_layers):
            lower = sum(self.structure.layers[:l+1])
            upper = sum(self.structure.layers[:l+2])
            temp[:,lower:upper] = self.structure.pred(x=temp, w=self.w, b=self.b )[:,lower:upper]
        self.x = temp

    def update_w(self):
        
        self.errors = self.errors.view(self.batch_size, self.num_vertices)
        self.x = self.values.view(self.batch_size, self.num_vertices)

        out = -torch.matmul(self.errors.T, self.f(self.x))
        # out = -torch.sparse.mm(self.errors.T, self.f(self.x))

        if self.mask is not None:
            out *= self.mask
        self.dw = out 
        
        # self.dw = self.structure.grad_w(x=self.x, e=self.e, w=self.w, b=self.b)
        # if self.structure.use_bias:
        #     self.db = self.structure.grad_b(x=self.x, e=self.e, w=self.w, b=self.b)
            
    def set_optimizer(self, optimizer):

        # self.optimizer = optimizer

        # self.optimizer_x = torch.optim.Adam(params, lr=lr_w, betas=(0.9, 0.999), eps=1e-7, weight_decay=weight_decay)
        pass


    def train(self):
        self.mode = "train"
        self.dw = None 
        # self.update_mask = self.update_mask_train

    def test(self):
        self.mode = "test"
        print(self.mode)

        # self.update_mask = self.update_mask_test

    def init_modes(self, batch_example):
        
        # take first item 
        batch_example = batch_example[0]
        sensory_indices_single_graph, internal_indices_single_graph, supervised_labels_single_graph = batch_example.sensory_indices, batch_example.internal_indices,  batch_example.supervision_indices
        
        self.base_sensory_indices = list(sensory_indices_single_graph)
        self.base_internal_indices = list(internal_indices_single_graph)
        self.base_supervised_labels = list(supervised_labels_single_graph) if supervised_labels_single_graph else []

        # Correcting the initialization of batched indices

        # Ensuring the base indices are flattened lists of integers
        base_sensory_indices = [int(idx) for sublist in self.base_sensory_indices for idx in sublist] if isinstance(self.base_sensory_indices[0], list) else self.base_sensory_indices
        base_internal_indices = [int(idx) for sublist in self.base_internal_indices for idx in sublist] if isinstance(self.base_internal_indices[0], list) else self.base_internal_indices
        base_supervised_labels = [int(idx) for sublist in self.base_supervised_labels for idx in sublist] if isinstance(self.base_supervised_labels[0], list) else self.base_supervised_labels

        # Create batched indices by iterating over batch size and offsetting by graph index
        self.sensory_indices_batch = [
            index + i * self.num_vertices for i in range(self.batch_size) for index in base_sensory_indices
        ]
        self.internal_indices_batch = [
            index + i * self.num_vertices for i in range(self.batch_size) for index in base_internal_indices
        ]
        self.supervised_labels_batch = [
            index + i * self.num_vertices for i in range(self.batch_size) for index in base_supervised_labels
        ] if base_supervised_labels else []

        # Convert to tensors for masking purposes during updates
        # self.sensory_indices_batch = torch.tensor(sorted(sensory_indices_batch), device=self.device)
        # self.internal_indices_batch = torch.tensor(sorted(internal_indices_batch), device=self.device)
        # self.supervised_labels_batch = torch.tensor(sorted(supervised_labels_batch), device=self.device) if supervised_labels_batch else None

        print("Sensory indices batch:", len(self.sensory_indices_batch), 784, self.batch_size)
        print("Internal indices batch:", len(self.internal_indices_batch), self.num_internal, self.batch_size)
        if self.supervised_labels_batch is not None:
            print("Supervised labels batch:", len(self.supervised_labels_batch), 10, self.batch_size)

        # Update only internal nodes during training
        # self.internal_mask_train = torch.tensor(internal_indices_batch, device=self.device)
        self.internal_mask_train = torch.tensor(self.internal_indices_batch, device=self.device)

        # update_mask = self.internal_mask_train if train else self.update_mask_test
        self.update_mask_test = torch.tensor(sorted(self.internal_indices_batch + self.supervised_labels_batch), device=self.device)

        ########################################## 
        batch_size = self.batch_size
        # num_nodes = self.structure.N  # Nodes per graph
        num_nodes = self.num_vertices  # Nodes per graph

        # Offset edge_index for batched graphs
        self.batched_edge_index = torch.cat(
            [self.edge_index + i * num_nodes for i in range(batch_size)], dim=1
        )  # Concatenate and offset indices


    def set_task(self, task):
        self.task = task 
        if task == "classification":
            # Update both the internal and supervised nodes during classification
            self.update_mask_test = torch.tensor(sorted(self.internal_indices_batch + self.supervised_labels_batch), device=self.device)
            print("Classification task")
            print("self.update_mask_test", self.update_mask_test.shape)

        elif task in ["generation", "reconstruction", "denoising", "occlusion"]:
            # Update both the internal and sensory nodes during these tasks
            self.update_mask_test = torch.tensor(sorted(self.internal_indices_batch + self.sensory_indices_batch), device=self.device)
            print("Generation task")
            print("self.update_mask_test", self.update_mask_test.shape)
        else:
            raise ValueError(f"Invalid task: {task}")
        
        # # Update internal + supervision nodes during testing
        # # self.update_mask_test = torch.tensor(sorted(internal_indices_batch + supervised_labels_batch), device=self.device)
        # self.update_mask_test = torch.tensor(sorted(self.internal_indices_batch + self.supervised_labels_batch), device=self.device)





    # def get_filtered_batched_w_and_edge_index(self):
        
    #     # update_mask = self.internal_mask_train if train else self.update_mask_test  # Select appropriate mask

    #     self.w = self.adj * self.w

    #     batch_size = self.x.size(0)
    #     num_nodes = self.num_vertices

    #     # Offset edge_index for batched graphs
    #     batched_edge_index = torch.cat(
    #         [self.edge_index + i * num_nodes for i in range(batch_size)], dim=1
    #     )

    #     weights_1d = self.w[self.edge_index[0], self.edge_index[1]]
    #     batched_weights = weights_1d.repeat(batch_size)

    #     # **Filter edge index using internal_indices_batch**
    #     internal_nodes_tensor = self.internal_indices_batch  # Already defined in init_modes()

    #     # Mask: Keep only edges where both source and target are in `internal_indices_batch`
    #     mask = torch.isin(batched_edge_index[0], internal_nodes_tensor) & torch.isin(batched_edge_index[1], internal_nodes_tensor)

    #     # Apply mask to edge index and weights
    #     filtered_edge_index = batched_edge_index[:, mask]
    #     filtered_weights = batched_weights[mask]
    #     pass
        

    def update_xs(self, train=True, trace=False):

        if self.early_stop:
            early_stopper = EarlyStopper(patience=0, min_delta=self.min_delta)

        T = self.T_train if train else self.T_test

        update_mask = self.internal_mask_train if train else self.update_mask_test

        for t in range(T): 

            # self.w = self.adj * self.w 
            # Perform the operation and reassign self.w as a Parameter
            with torch.no_grad():
                self.w.copy_(self.adj * self.w)

                # make weights[0:784, -10:] /= 2
                # self.w[0:784, -10:] /= 2 
                # self.w[-10:, 0:784] /= 2 
                

            # self.weights_1d = self.w_to_sparse(self.w)
            # Gather 1D weights corresponding to connected edges
            weights_1d = self.w[self.edge_index[0], self.edge_index[1]]  # Extract relevant weights from W
            # # weights_1d = self.w.T[self.edge_index[0], self.edge_index[1]]  # Extract relevant weights from W

            # # Expand edge weights for each graph
            batched_weights = weights_1d.repeat(self.batch_size)
            # predicted_mpU = self.pred_mu_MP.forward(self.values.view(-1,1).to(DEVICE),
            #                         self.batched_edge_index.to(DEVICE), 
                                    # batched_weights.to(DEVICE))
            # self.e = self.values - predicted_mpU

            # self.errors = self.errors.view(self.batch_size, self.num_vertices)
            self.x = self.values.view(self.batch_size, self.num_vertices)
            if self.trace:
                
                # print(self.x.shape)
                x_slice = self.x[0:1, 0:784]
                # print("x_slice shape", x_slice.shape)

                if not isinstance(x_slice, torch.Tensor):
                    x_slice = torch.tensor(x_slice, device=self.device)

                if x_slice.numel() == 0:
                    print("Warning: x_slice is empty")
                    return

                x_slice = x_slice.contiguous().cpu().numpy()

                if not isinstance(x_slice, np.ndarray):
                    print("Error: Converted x_slice is not a NumPy array")
                    return
                
                self.trace_data.append(x_slice.reshape(28, 28))
            
            mu = torch.matmul(f(self.x), self.w.T)
            mu = mu.view(-1, 1)
            # print(torch.allclose(mu_mp, mu, atol=1))  # Should be True
            # print(torch.allclose(predicted_mpU, mu, atol=1))  # Should be True
            
            # self.e = self.x - mu_mp 
            # print("predicted_mpU shape", predicted_mpU.shape)
            # print("values shape", self.values.shape)

            self.errors = self.values - mu
        
            # print("e1", torch.mean(self.e))
            # TODO

            if not self.use_input_error:
                if self.task == "classification":
                    self.errors[self.sensory_indices_batch] = 0
                elif self.task in ["generation", "reconstruction", "denoising", "occlusion"]:
                    self.errors[self.supervised_labels_batch] = 0
                    # self.errors[self.sensory_indices_batch] = 0

            # total_mean_error = self.errors.mean()
            # total_mean_error = torch.sum(self.errors**2).mean()

            # total_internal_error = self.errors[self.internal_indices_batch].mean()
            # self.history.append(total_mean_error.cpu().numpy())
            
            total_internal_error = (self.errors[self.internal_indices_batch]**2).mean()
            self.history.append(total_internal_error.cpu().numpy())
                # self.e[:,:di] = 0 

            # print("mean error", torch.mean(self.errors))

            # print("dx----0")

            # AMB convention 
            # lower = self.shape[0]
            # upper = -self.shape[2] if train else sum(self.shape)
            # return e[:,lower:upper] - self.dfdx(x[:,lower:upper]) * torch.matmul(e, w.T[lower:upper,:].T)
            # dEdx = self.structure.grad_x(self.x.to(DEVICE), self.e.to(DEVICE), self.w.to(DEVICE), self.b.to(DEVICE),train=train) # only hidden nodes

            torch.cuda.empty_cache()

            # print(self.x.view(-1,1).shape)
            # print(batched_edge_index.shape)
            # print(self.e.view(-1,1).shape)
            # print(batched_weights.shape)

            # x = self.x.T.contiguous().view(-1, 1).to(DEVICE)  # Shape: [num_nodes, batch_size]
            # error = self.e.T.contiguous().view(-1, 1).to(DEVICE)  # Shape: [num_nodes, batch_size]

            dEdx_ = self.grad_x_MP.forward( 
                    x=self.values.view(-1,1),
                    edge_index=self.batched_edge_index.to(DEVICE),  
                    error=self.errors.view(-1,1), 
                    weight=batched_weights.view(-1,1).to(DEVICE),
            )

            dEdx = dEdx_[update_mask]
        
        
            # clipped_dEdx = torch.clamp(dEdx, -1, 1)
            clipped_dEdx = dEdx

                    
            self.values[update_mask] -= self.lr_x * clipped_dEdx

            # dEdx_ = dEdx_.view(self.batch_size, self.num_vertices)
            # dEdx_ = dEdx_[:,di:upper]
            
            # self.update_w()
            
            # if train and self.incremental and self.dw is not None:
            if train and self.incremental:

                self.update_w()
                # print("optimizer step")
        
                # if self.w.is_sparse:
                #     self.w = self.w.to_dense()
                # if self.dw.is_sparse:
                #     self.dw = self.dw.to_dense()

                # # Convert m and gradients to dense if necessary
                # if self.dw.is_sparse:
                #     self.dw = self.dw.to_dense()

                # if self.optimizer.m_w.is_sparse:
                #     self.optimizer.m_w = self.optimizer.m_w.to_dense()
                
                # self.dw = torch.clamp(self.dw, -1, 1)

                self.w.grad = self.dw
                self.optimizer_w.step()
                # self.optimizer.step(self.params, self.grads, batch_size=self.batch_size)
            if self.early_stop:
                if early_stopper.early_stop( self.get_energy() ):
                    break            

    # def train_supervised(self, X_batch, y_batch): 

    #     # X_batch = to_vector(X_batch)                  # makes e.g. 28*28 -> 784
    #     # y_batch = onehot(y_batch, N=self.structure.shape[2])    # makes e.g. 3 -> [0,0,0,1,0,0,0,0,0,0]

    #     # self.reset_nodes(batch_size=X_batch.shape[0])        
    #     # self.clamp_input(X_batch)
    #     # # self.init_hidden()
    #     # # print("ommit init hidden")
    #     # self.clamp_target(y_batch)

    #     self.update_xs(train=True)
    #     self.update_w()

    #     if not self.incremental:
    #         # self.update_w()
    #         print("optimizer step end ")
    #         self.optimizer.step(self.params, self.grads, batch_size=X_batch.shape[0])

    
    def unpack_features(self, batch, reshape=False):
        """Unpack values, errors, and predictions from the batched graph."""
        # values, errors, predictions = batch[:, 0, :].to(self.device), batch[:, 1, :].to(self.device),  None
        # # print("unpacked featreus")
        # # print(values.shape)

        # # reshape to (batch_size, num_vertices)
        # if reshape:
        #     values      = values.view(self.batch_size, self.num_vertices)
        #     errors      = errors.view(self.batch_size, self.num_vertices)
        #     # predictions = predictions.view(self.batch_size, self.num_vertices)

        # return values, errors, predictions
        values = batch

        return values, None, None


    def train_supervised(self, data):
        # edge_index = data.edge_index.to(self.device)
        self.history = []

        self.optimizer_w.zero_grad()
        # self.data_ptr = data.ptr
        self.batch_size = data.x.shape[0] // self.num_vertices

        self.values, _ , _ = self.unpack_features(data.x, reshape=False)
        
        self.update_xs(train=True)
        self.update_w()

        if not self.incremental:
            self.w.grad = self.dw
            self.optimizer_w.step()
            # self.optimizer.step(self.params, self.grads, batch_size=self.batch_size)

        # print("w mean", self.w.mean())

        return self.history

    def test_classifications(self, data, remove_label=True):
            
        # remove one_hot
        if remove_label:
            for i in range(len(data)):
                sub_graph = data[i]  # Access the subgraph
                sub_graph.x[sub_graph.supervision_indices, 0] = torch.zeros_like(sub_graph.x[sub_graph.supervision_indices, 0])  # Check all feature dimensions

        self.values, _ , _ = self.unpack_features(data.x, reshape=False)
        
        # print("0", self.values[:, -10:].shape, self.values[:, -10:])

        self.update_xs(train=False)
        # logits = self.values[:, data.supervision_indices[0]]
        # logits = self.values[:, -10:]
        # print("supervised_labels_batch ", self.supervised_labels_batch)
      
        # logits = self.values[self.supervised_labels_batch]
        # logits = logits.view(self.batch_size, len(self.base_supervised_labels))   # batch,10
        # OR 
        logits = self.values.view(self.batch_size, self.num_vertices)   # batch,10
        # print("logits ", logits)
        logits = logits[:, -10:]

        y_pred = torch.argmax(logits, axis=1).squeeze()
        # print("logits ", logits.shape)
        # print("y_pred ", y_pred.shape)
        return y_pred.cpu().detach()
    


    def test_generative(self, data, remove_label=True, save_imgs=False):
              
        # remove one_hot
        if remove_label:
            for i in range(len(data)):
                sub_graph = data[i]  # Access the subgraph

                # set sensory indices to zero / random noise
                sub_graph.x[sub_graph.sensory_indices, 0] = torch.zeros_like(sub_graph.x[sub_graph.sensory_indices, 0])  # Check all feature dimensions
                # random noise
                # sub_graph.x[sub_graph.sensory_indices, 0] = torch.randn_like(sub_graph.x[sub_graph.sensory_indices, 0])  # Check all feature dimensions
                # sub_graph.x[sub_graph.sensory_indices, 0] = torch.clamp(torch.randn_like(sub_graph.x[sub_graph.sensory_indices, 0]), min=0, max=1)

        self.values, _ , _ = self.unpack_features(data.x, reshape=False)
        
        # print("0", self.values[:, -10:].shape, self.values[:, -10:])

        self.update_xs(train=False)
        # logits = self.values[:, data.supervision_indices[0]]
        # logits = self.values[:, -10:]
        # print("supervised_labels_batch ", self.supervised_labels_batch)
      
        # logits = self.values[self.supervised_labels_batch]
        # logits = logits.view(self.batch_size, len(self.base_supervised_labels))   # batch,10
        # OR 
        generated_imgs = self.values[self.sensory_indices_batch]   # batch,10
        generated_imgs = generated_imgs.view(self.batch_size, 28, 28)   # batch,10

        # generated_imgs = self.values.view(self.batch_size, self.num_vertices)   # batch,10
        # generated_imgs = generated_imgs[self.batch_size, :784] # batch,10
        # generated_imgs = generated_imgs.view(self.batch_size, 28, 28)   # batch,10

        # save img inside 1 big plt imshow plot; take first 10 images
        if save_imgs:
            import matplotlib.pyplot as plt

            # random_offset between 0 and batch_size
            random_offset = np.random.randint(0, self.batch_size-10)

            fig, axs = plt.subplots(1, 10, figsize=(20, 2))
            for i in range(10):

                axs[i].imshow(generated_imgs[i+random_offset].cpu().detach().numpy())
                axs[i].axis("off")
                # use label from data.y
                axs[i].set_title(data.y[i+random_offset].item())

            # plt.show()
            # save 
            plt.savefig(f"trained_models/{self.task}/generated_imgs_{self.epoch}.png")
            plt.close()

            # plot self.trace_data
            if self.trace:
                fig, axs = plt.subplots(1, self.T_test, figsize=(20, 2))
                for i in range(self.T_test):
                    axs[i].imshow(self.trace_data[i])
                    axs[i].axis("off")
                    axs[i].set_title(data.y[0].item())

                # plt.show()
                # save 
                plt.savefig(f"trained_models/{self.task}/trace_data_{self.epoch}.png")
                plt.close()
            
       
        return 0
    


    def test_iterative(self, data, eval_types=None, remove_label=True):
        # edge_index = data.edge_index

        self.batch_size = data.x.shape[0] // self.num_vertices

    

        # eval_type = ["classification", "generative", "..."]

        if "classification" in eval_types:
            self.set_task("classification")   # not update the sensory nodes, only supervised nodes

            return self.test_classifications(data.clone().to(self.device), 
                                      remove_label=remove_label)
                                      
        if "generation" in eval_types:
            self.set_task("generation")       # not update the supervised nodes, only sensory nodes

            self.trace_data = []
            self.trace = True 

            self.test_generative(data.clone().to(self.device), 
                                 remove_label=remove_label, save_imgs=True)
            
            return 0 # Placeholder ""
        else:
            raise ValueError("Unknown evaluation type")
    
    def get_energy(self):
        return torch.sum(self.errors**2).item()

    def get_errors(self):
        return self.e.clone()


    def test_feedforward(self, X_batch):
        pass
        # X_batch = to_vector(X_batch)     # makes e.g. 28*28 -> 784

        # self.reset_nodes(batch_size=X_batch.shape[0])
        # self.clamp_input(X_batch)
        # self.forward(self.num_verticesum_layers)

        # return self.x[:,-self.structure.shape[2]:] 

    # def test_iterative(self, X_batch, diagnostics=None, early_stop=False):
    # def test_iterative(self, X_batch):
    #     # X_batch = to_vector(X_batch)     # makes e.g. 28*28 -> 784

    #     # self.reset_nodes(batch_size=X_batch.shape[0])
    #     # self.clamp_input(X_batch)
    #     # self.init_hidden()
    #     # self.init_output()

    #     self.update_xs(train=False)
    #     # self.update_xs(train=False, diagnostics=diagnostics, early_stop=early_stop)
    #     return self.x[:,-self.structure.shape[2]:] 

    def get_weights(self):
        return self.w.clone()

    def get_energy(self):
        return torch.sum(self.e**2).item()

    def get_errors(self):
        return self.e.clone()    


##############################


args = {
    "model_type": "IPC",
    # "ype": "stochastic_block",  # Type of graph
    "update_rules": "Van_Zwol",  # Update rules for learning

    # "graph_type": "fully_connected",  # Type of graph

    "graph_type": "single_hidden_layer",  # Type of graph
    # "discriminative_hidden_layers": [32, 16],  # Hidden layers for discriminative model
    # "generative_hidden_layers": [0],  # Hidden layers for generative model

    "discriminative_hidden_layers": [0],  # Hidden layers for discriminative model
    "generative_hidden_layers": [50,50],  # Hidden layers for generative model


    "delta_w_selection": "all",  # Selection strategy for weight updates
    "weight_init": "fixed 0.001 0.001",  # Weight initialization method
    "use_grokfast": True,  # Whether to use GrokFast
    "optimizer": 1.0,  # Optimizer setting
    "remove_sens_2_sens": True,  # Remove sensory-to-sensory connections
    "remove_sens_2_sup": True,  # Remove sensory-to-supervised connections
    "set_abs_small_w_2_zero": False,  # Set small absolute weights to zero
    "mode": "experimenting",  # Mode of operation (training/experimenting)
    "use_wandb": "offline",  # WandB logging mode
    "tags": "PC_vs_IPC",  # Tags for logging
    "use_bias": False,  # Whether to use bias
    "normalize_msg": False,  # Normalize message passing
    "dataset_transform": ["normalize_mnist_mean_std"],  # Data transformations
    "numbers_list": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],  # Classes to include
    "N": "all",  # Number of samples per class
    "supervision_label_val": 1,  # Value assigned for supervision
    "num_internal_nodes": 1000,  # Number of internal nodes in the network
    "T": 5,  # Number of inference iterations
    "lr_values": 0.01,  # Learning rate for value updates
    "lr_weights": 0.00001,  # Learning rate for weight updates
    "activation_func": "swish",  # Activation function
    "epochs": 10,  # Number of training epochs
    # "batch_size": 0,  # Batch size for training; fine for discriminative
    "batch_size": 30,  # Batch size for training
    # "batch_size": 200,  # Batch size for training
    "seed": 2,  # Random seed
}

torch.set_default_dtype(torch.float32)  # Ensuring consistent precision

# Use compiled model for speed optimization (if PyTorch 2.0+)
USE_TORCH_COMPILE = True


# Access the arguments just like you would with argparse
print(args['dataset_transform'])  # Example of accessing an argument


class Args:
    def __init__(self, **entries):
        self.__dict__.update(entries)


# Create an object from the dictionary
args = Args(**args)


# Make True of False bool
args.normalize_msg = args.normalize_msg == 'True'
args.use_bias = args.use_bias == 'True'
args.set_abs_small_w_2_zero = args.set_abs_small_w_2_zero == 'True'
args.grokfast = args.use_grokfast == 'True'

tags_list = args.tags.split(",") if args.tags else []



# Using argparse values
torch.manual_seed(args.seed)

generator_seed = torch.Generator()
generator_seed.manual_seed(args.seed)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
print(f"Seed used", args.seed)
if torch.cuda.is_available():
    print("Device name: ", torch.cuda.get_device_name(0))

print("---------------model.pc_conv1.log_delta_w() turned off-----------------")

# Make True of False bool
args.normalize_msg = args.normalize_msg == 'True'
args.use_bias = args.use_bias == 'True'
args.set_abs_small_w_2_zero = args.set_abs_small_w_2_zero == 'True'
args.grokfast = args.use_grokfast == 'True'

tags_list = args.tags.split(",") if args.tags else []

import torchvision.transforms as transforms
import numpy as np


# The ToTensor() transformation is the one responsible for scaling (MNIST) images to the range [0, 1].
transform_list = [
    transforms.ToTensor()
]

if args.dataset_transform:

    if "normalize_min1_plus1" in args.dataset_transform:
        transform_list.append(transforms.Normalize((0.5,), (0.5,)))

    if "normalize_mnist_mean_std" in args.dataset_transform:
        transform_list.append(transforms.Normalize((0.1307,), (0.3081,)))
    
    if "random_rotation" in args.dataset_transform:
        transform_list.append(transforms.RandomRotation(degrees=20))
    




# # Create the transform
print("TODO ADD COMPASE TRANSFORMS")
# transform = transforms.Compose(transform_list)

# mnist_trainset = torchvision.datasets.MNIST(root=args.data_dir, train=True, download=True, transform=transform)
# mnist_testset  = torchvision.datasets.MNIST(root=args.data_dir, train=False, download=True, transform=transform)



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')




######################################################################################################### 
####                                            Dataset                                             #####
######################################################################################################### 


## Subset of the dataset (for faster development)
# subset_size = 100  # Number of samples to use from the training set
# indices = list(range(len(mnist_trainset)))
# random.shuffle(indices)
# subset_indices = indices[:subset_size]

# mnist_train_subset = torch.utils.data.Subset(mnist_trainset, subset_indices)
# print("USSSSSING SUBSEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEET")

# CustomGraphDataset params
# dataset_params = {
#     "mnist_dataset":            mnist_trainset,
#     # "mnist_dataset":            mnist_train_subset,
#     "supervised_learning":      True,
#     "numbers_list":             args.numbers_list,
#     "same_digit":               False,
#     "add_noise":                False,
#     "noise_intensity":          0.0,
#     "N":                        args.N,     # taking the first n instances of each digit or use "all"

#     "edge_index":               None,
#     "supervision_label_val":    args.supervision_label_val,         # Strength of label signal within the graph. MNIST ~0-1, label_vector[label] = self.supervision_label_val
# } 

print("------------------Importing Graph Params ---------------- ")
from graphbuilder import graph_type_options

# Define the graph type
# Options: "fully_connected", "fully_connected_w_self", "barabasi", "stochastic_block", "fully_connected_no_sens2sup"
graph_params = {
    "internal_nodes": args.num_internal_nodes,  # Number of internal nodes
    "supervised_learning": True,  # Whether the task involves supervised learning
    "graph_type": {    
        "name": args.graph_type, # Options: "fully_connected", "fully_connected_w_self", "barabasi", "stochastic_block"
        "params": graph_type_options[args.graph_type]["params"], 
        # "params_general": {
        #     "remove_sens_2_sens": args.remove_sens_2_sens,  # Set from command line
        #     "remove_sens_2_sup": args.remove_sens_2_sup,    # Set from command line
        #     },
        },  
    "seed": args.seed,   
}

eval_generation, eval_classification, eval_denoise, eval_occlusion = True, True, 0, 0 


# add graph specific info: 
# print("zzz", args.remove_sens_2_sens, args.remove_sens_2_sup)
graph_params["graph_type"]["params"]["remove_sens_2_sens"] = args.remove_sens_2_sens  
graph_params["graph_type"]["params"]["remove_sens_2_sup"]  = args.remove_sens_2_sup 


if graph_params["graph_type"]["name"] == "stochastic_block":
    
    # override internal nodes if doing clustering
    graph_params["internal_nodes"] = (graph_params["graph_type"]["params"]["num_communities"] * graph_params["graph_type"]["params"]["community_size"])

if graph_params["graph_type"]["name"] == "stochastic_block_hierarchy":
    raise ValueError("Not implemented yet")


if graph_params["graph_type"]["name"] in ["custom_two_branch","two_branch_graph"]:
    # Configure internal nodes for two_branch_graph
    # This assumes two branches with specified configurations
    branch1_layers, branch1_clusters_per_layer, branch1_nodes_per_cluster = graph_params["graph_type"]["params"]["branch1_config"]
    branch2_layers, branch2_clusters_per_layer, branch2_nodes_per_cluster = graph_params["graph_type"]["params"]["branch2_config"]
    
    # Calculate total internal nodes for both branches
    # Branch 1
    branch1_internal_nodes = branch1_layers * branch1_clusters_per_layer * branch1_nodes_per_cluster
    # Branch 2 (Reversed order)
    branch2_internal_nodes = branch2_layers * branch2_clusters_per_layer * branch2_nodes_per_cluster
    
    # The total number of internal nodes will be the sum of both branches
    graph_params["internal_nodes"] = branch1_internal_nodes + branch2_internal_nodes



if graph_params["graph_type"]["name"] in ["single_hidden_layer"]:

    
    # # discriminative_hidden_layers = [0]  # Adjust if layers change
    # generative_hidden_layers = [50, 100, 200] # Adjust if layers change
    
    # # Calculate total number of nodes
    # discriminative_hidden_layers = [200, 100, 50]  # Adjust if layers change
    # # generative_hidden_layers = [0] # Adjust if layers change

    discriminative_hidden_layers = args.discriminative_hidden_layers or [200, 100, 50]  # Default if not provided
    generative_hidden_layers = args.generative_hidden_layers or [50, 100, 200]  # Default if not provided


    num_discriminative_nodes = sum(discriminative_hidden_layers)
    num_generative_nodes = sum(generative_hidden_layers)

    graph_params["graph_type"]["params"]["discriminative_hidden_layers"] = discriminative_hidden_layers
    graph_params["graph_type"]["params"]["generative_hidden_layers"]  = generative_hidden_layers
   
    graph_params["internal_nodes"] = num_discriminative_nodes + num_generative_nodes

    # edge_index, N = test_single_hidden_layer(discriminative_hidden_layers, generative_hidden_layers,
    #                                         no_sens2sens=True, no_sens2supervised=True)

    if sum(discriminative_hidden_layers) == 0:
        eval_classification = False
    if sum(generative_hidden_layers) == 0:
        eval_generation = False

    # TODO ; still unsure about which graph does which task
    eval_generation, eval_classification, eval_denoise, eval_occlusion = True, True, 0, 0 

# if graph_params["graph_type"]["name"] not in ["single_hidden_layer"]:
#     # Ensure these arguments are not specified for other graph types
#     if "discriminative_hidden_layers" in args:
#         assert args.discriminative_hidden_layers is None, \
#             "The argument --discriminative_hidden_layers can only be used if graph_type is 'single_hidden_layer'."
#         assert args.generative_hidden_layers is None, \
#             "The argument --generative_hidden_layers can only be used if graph_type is 'single_hidden_layer'."

# if graph_params["graph_type"]["name"] in ["custom_two_branch", "two_branch_graph"]:
#     # Configure internal nodes for two_branch_graph
#     # This assumes two branches with specified configurations
#     branch1_config = graph_params["graph_type"]["params"]["branch1_config"]
#     branch2_config = graph_params["graph_type"]["params"]["branch2_config"]
    
#     # Calculate total internal nodes for both branches
#     # Sum up the total internal nodes for Branch 1
#     branch1_internal_nodes = sum([clusters * nodes_per_cluster for clusters, nodes_per_cluster in branch1_config])
    
#     # Sum up the total internal nodes for Branch 2 (Reversed order if required)
#     branch2_internal_nodes = sum([clusters * nodes_per_cluster for clusters, nodes_per_cluster in branch2_config])
    
#     # The total number of internal nodes will be the sum of both branches
#     graph_params["internal_nodes"] = branch1_internal_nodes + branch2_internal_nodes


TASK = []
if args.graph_type == "fully_connected" or args.graph_type == "stochastic_block":
    TASK = ["classification", "generation"]

if args.graph_type == "single_hidden_layer":
    if sum(args.discriminative_hidden_layers) > 0:
        TASK.append("classification")
    else:
        TASK.append("generation")
        
import os 
# if not exist make folder trained_models/args.graph_type/
if not os.path.exists(f"trained_models/{args.graph_type}"):
    # create 
    os.makedirs(f"trained_models/{args.graph_type}")


print("graph_params 1 :", graph_params)

from graphbuilder import GraphBuilder
from helper.plot import plot_adj_matrix

print("graph_params", graph_params)
graph = GraphBuilder(**graph_params)


single_graph = graph.edge_index

adj_matrix_pyg = plot_adj_matrix(single_graph, model_dir=f"trained_models/{args.graph_type}", node_types=None)




from dataset_vanZwol import PCGraphDataset, train_subset_indices

from torch_geometric.data import Batch

def custom_collate_fn(batch):

    # batched_data = Data.from_batch(batch)  # Use the built-in PyG batching method
    batched_data = Batch.from_data_list(batch)

    sensory_indices = []
    internal_indices = []
    supervision_indices = []

    node_offset = 0  # Keeps track of node offset as we batch graphs

    for data in batch:
        # Adjust the indices by adding the current node offset
        sensory_indices.append(data.sensory_indices + node_offset)
        internal_indices.append(data.internal_indices + node_offset)
        supervision_indices.append(data.supervision_indices + node_offset)

        # Increment offset by the number of nodes in the current graph
        node_offset += data.x.size(0)

    # Concatenate indices across the batch
    batched_data.sensory_indices = torch.cat(sensory_indices)
    batched_data.internal_indices = torch.cat(internal_indices)
    batched_data.supervision_indices = torch.cat(supervision_indices)

    return batched_data


from torch_geometric.data import DataLoader as GeoDataLoader
import torchvision
from torchvision import transforms

# Dataset and DataLoader configurations

DATASET_PATH = "../data"
batch_size = args.batch_size

# # Initialize the GraphBuilder
# custom_dataset_train = PCGraphDataset(graph_params, **dataset_params)

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

train_dataset = torchvision.datasets.MNIST(root=DATASET_PATH, train=True, transform=transform, download=True)
test_set = torchvision.datasets.MNIST(root=DATASET_PATH, train=False, transform=transform, download=True)

train_set, val_set = torch.utils.data.random_split(train_dataset, [50000, 10000])
train_indices = train_subset_indices(train_set, 10, no_per_class=0)  # Set `no_per_class` as needed

# Initialize CustomGraphDataset for train, validation, and test sets
train_graph_dataset = PCGraphDataset(graph, train_set, supervised_learning=True, numbers_list=list(range(10)))
val_graph_dataset = PCGraphDataset(graph, val_set, supervised_learning=True, numbers_list=list(range(10)))
test_graph_dataset = PCGraphDataset(graph, test_set, supervised_learning=True, numbers_list=list(range(10)))

# PYG DataLoaders
train_loader = GeoDataLoader(train_graph_dataset, batch_size=batch_size, collate_fn=custom_collate_fn, shuffle=True, pin_memory=True, num_workers=4, drop_last=True)
val_loader = GeoDataLoader(val_graph_dataset, batch_size=batch_size, collate_fn=custom_collate_fn, shuffle=False,  pin_memory=True, num_workers=4, drop_last=True)
test_loader = GeoDataLoader(test_graph_dataset, batch_size=batch_size, collate_fn=custom_collate_fn, shuffle=False,  pin_memory=True, num_workers=4, drop_last=True)


#### GET EXAMPLE BATCH #### 
import matplotlib.pyplot as plt

print(len(train_loader))
testing_remove_label = True
testing_remove_data  = False

for batch in train_loader:
    
    # Set the graph-level labels to zero
    # batch.y[:] = 0
    for i in range(len(batch)):
        
        ### batch.x[batch.supervision_indices, :] = 0 
        if testing_remove_label:
            sub_graph = batch[i]  # Access the subgraph
            sub_graph.x[sub_graph.supervision_indices, 0] = torch.zeros_like(sub_graph.x[sub_graph.supervision_indices, 0])  # Check all feature dimensions

    # train_loader.dataset.zero_img_in_graph  = True
    # train_loader.dataset.zero_y_in_graph    = True 

    print(batch)
    print(batch.x.shape)

    batch_item = 0  # Select a specific graph within the batch
    
    sub_graph = batch[batch_item]  # Access the subgraph
    sensory_indices = sub_graph.sensory_indices
    image_tensor = sub_graph.x[sensory_indices, 0].view(28, 28).detach().numpy()
    
    sup_tensor = sub_graph.x[sub_graph.supervision_indices, 0]  # Check all feature dimensions
    
    print("Supervision tensor after zeroing:")
    print(sup_tensor)

    # Display the image
    # plt.imshow(image_tensor, cmap='gray')
    # plt.title(f"Sensory Node Image Representation (Label set to {sub_graph.y})")

    print("---------------------")
    print("Batched sensory indices:", batch.sensory_indices)
    print("Batched internal indices:", batch.internal_indices)
    print("Batched supervision indices:", batch.supervision_indices)

    print("-------Edge_index (single) vs (batched) ---------")

    single_graph = graph.edge_index
    print(single_graph.shape)
    print(batch.edge_index.shape)

    # plt.show()

    break  # Only process the first batch

###########################



# from models.PC_vanZwol import PC_graph_zwol 

# """ WITH MESSAGE_PASSING """
# from models.PC_vanZwolPYG import PC_graph_zwol_PYG

""" WITHOUT MESSAGE_PASSING """
# from models.PC_vanZwol_pyg_loader import PC_graph_zwol_PYG

from helper.vanZwol_optim import *

################################# discriminative model lr         ##########################################
################################# generative model lr         ##########################################

# Inference
f = tanh
# f = relu
lr_x = 0.5                  # inference rate                   # inference rate 
T_train = 5                 # inference time scale
T_test = 10              # unused for hierarchical model
incremental = True          # whether to use incremental EM or not
use_input_error = False     # whether to use errors in the input layer or not

# Learning
lr_w = 0.00001      
# Learning
# lr_w = 0.00001              # learning rate hierarchial model
# lr_w = 0.000001              # learning rate generative model



################################# fully connected model lr         ##########################################

# # GOOD FOR CLASSIFCATION
# lr_x = 0.01                  # inference rate                   # inference rate 
# lr_w = 0.00001              # learning rate hierarchial model

# # OKAY FOR GENRATION
# lr_x = 0.001                  # inference rate                   # inference rate 
lr_w = 0.001              # learning rate hierarchial model
# T_train = 15                 # inference time scale
# T_test = 20  

# lr_x = 0.01                  # inference rate                   # inference rate 
# lr_w = 0.000001              # learning rate hierarchial model


weight_decay = 0             
grad_clip = 1
batch_scale = False
 
import helper.vanZwol_optim as optim

# vertices = [784, 48, 10] # input, hidden, output
# mask = get_mask_hierarchical([784,32,16,10])

PCG = PCgraph(f,
        device=device,
        num_vertices=graph.num_vertices,
        num_internal=sum(graph.internal_indices),
        adj=adj_matrix_pyg,
        edge_index=graph.edge_index,
        batch_size=batch_size,
        # mask=mask,
        lr_x=lr_x, 
        T_train=T_train,
        T_test=T_test,
        incremental=incremental, 
        use_input_error=use_input_error,
        )

# optimizer = Adam(
#     PCG.params,
#     learning_rate=lr_w,
#     grad_clip=grad_clip,
#     batch_scale=batch_scale,
#     weight_decay=weight_decay,
# ) 
# PCG.set_optimizer(optimizer)

PCG.init_modes(batch_example=batch)

# PCG.set_task(TASK)

model = PCG
model = torch.compile(model, disable=True) 
# torch.compile(model, dynamic=True)


model.task = ""   # classification or generation, or both 

from datetime import datetime
from tqdm import tqdm
import torch

epochs = 100
start_time = datetime.now()

train_energy, train_loss, train_acc = [], [], []
val_loss, val_acc, val_acc2 = [], [], []
num_epochs = 40

# break_num = 150 
# break_num = 250 
# break_num = int(len(train_loader) -1 )
# break_num = 100

break_num = 200
break_num = 100
break_num = 100
# break_num = 30

with torch.no_grad():

    epoch_history = []

    for epoch in tqdm(range(num_epochs), desc="Epoch Progress"):
        model.train()
        model.epoch = epoch
        total_loss = 0
        energy = 0


        print("\n-----train_supervised-----")
        print(len(train_loader))

        for batch_no, batch in enumerate(tqdm(train_loader, total=min(break_num, len(train_loader)), desc=f"Epoch {epoch+1} - Training", leave=False)):
            batch = batch.to(model.device)
            history = model.train_supervised(batch)  # history is [..., ...]
            # append all items in history to epoch_history
            for item in history:
                epoch_history.append(item)

            if batch_no >= break_num:
                break
    
        #### 
        loss, acc = 0, 0
        model.test()
        cntr = 0

        # break_num_eval = 20
        # if TASK == "generation":
        #     break_num_eval = 1
        break_num_eval = 10
            
        print("\n----test_iterative-----")
        accs = []
        TASK_copy = TASK.copy()

        # for batch_no, batch in enumerate(tqdm(val_loader, total=min(len(val_loader)), desc=f"Epoch {epoch+1} - Validation", leave=False)):
        for batch_no, batch in enumerate(tqdm(val_loader, total=len(val_loader), desc=f"Epoch {epoch+1} - Validation | {TASK}", leave=False)):
            y_batch = batch.y.clone()
            batch = batch.to(model.device)
            # y_pred = PCG.test_iterative(batch, eval_types=TASK_copy, remove_label=True)

            for task in TASK_copy:
                y_pred = PCG.test_iterative(batch, eval_types=[task], remove_label=True)

            # do generation once
            if "generation" in TASK_copy:
                TASK_copy = ["classification"]

            # print("y_pred", y_pred.shape)
            # print("y_pred", y_batch.shape)
            if "classification" in TASK_copy:
                correct = torch.mean((y_pred == y_batch).float()).item()
                acc += correct
                accs.append(correct)

            cntr += 1
            if batch_no >= break_num_eval:
                break


        # save model weights plt.imshow to "trained_models/{TASK}/weights/model_{epoch}.png"
        # make folder if not exist
        import os 
        if not os.path.exists(f"trained_models/{args.graph_type}/weights/"):
            os.makedirs(f"trained_models/{args.graph_type}/weights/")
        
        # save weights
        w = PCG.w.detach().cpu().numpy()
        plt.imshow(w)
        plt.colorbar()
        plt.savefig(f"trained_models/{args.graph_type}/weights/model_{epoch}.png")
        plt.close()



        if "classification" in TASK:
            
            # Corrected validation accuracy calculations
            val_acc.append(acc / len(val_loader))
            val_acc2.append(acc / cntr)
            val_loss.append(loss)

        
            print("val_acc2", val_acc2)
            print("Last prediction:", y_pred)
            print("Last y_batch:", y_batch)
            print("accs", accs)
            print("accs", sum(accs) / len(accs))

            print(f"\nEpoch {epoch+1}/{num_epochs} Completed")
            print(f"  Validation Accuracy: {val_acc[-1]:.3f}")
            print(f"  Validation Accuracy (limited): {val_acc2[-1]:.3f}")


        # plot history of energy
        import matplotlib.pyplot as plt
        print("epoch_history", len(epoch_history))
        plt.plot(epoch_history)

        plt.savefig(f"trained_models/{args.graph_type}/energy_history.png")
        plt.close()





