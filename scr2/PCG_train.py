
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

class PCgraph(PCmodel):
    """
    Predictive Coding Graph trained with Inference Learning (IL)/Expectation Maximization (EM).

    Args:
        lr_x (float): Inference rate/learning rate for nodes (partial E-step).
        T_train (int): Number of inference iterations (partial E-steps) during training.
        T_test (int): Number of inference iterations (partial E-steps) during testing.
        incremental (bool): Whether to use incremental EM (partial M-step after each partial E-step).
        init (dict): Initialization parameters: {"weights": std, "bias": std, "x_hidden": std, "x_output": std}.
        min_delta (float): Minimum change in energy for early stopping during inference.
        use_input_error (bool): Whether to use input error in training (to obtain exact PCN updates for hierarchical mask, set to False).
    """
    def __init__(self, lr_x: float, T_train: int, T_test: int, structure: PCGStructure, 
                 incremental: bool = False, 
                 node_init_std: float = None,
                 min_delta: float = 0, early_stop: bool = False,
                 use_input_error: bool = True):
         
        super().__init__(structure=structure, lr_x=lr_x, T_train=T_train, incremental=incremental, min_delta=min_delta, early_stop=early_stop)
        self.T_test = T_test

        self.use_input_error = use_input_error
        self.node_init_std = node_init_std

        self._reset_grad()
        self._reset_params()

        if self.structure.mask is not None:
            self.w = self.structure.mask * self.w 
            self.mask_density = torch.count_nonzero(self.w)/self.structure.N**2   
            # hierarchical structure
            if self.structure.num_layers is not None:
                # self.test_supervised = self.test_feedforward
                self.test_supervised = self.test_iterative

                self.init_hidden = self.init_hidden_feedforward
                if use_input_error:

                    print("Using input error in training with hierarchical mask (no input error recommended).")
            # non-hierarchical structure
            else: 
                self.test_supervised = self.test_iterative
                self.init_hidden = self.init_hidden_random
                if not use_input_error:
                    print("Not using input error in training with non-hierarchical mask.")


    @property
    def hparams(self):
        return {"lr_x": self.lr_x, "T_train": self.T_train, "T_test": self.T_test, "incremental": self.incremental,
                 "min_delta": self.min_delta,"early_stop": self.early_stop, "use_input_error": self.use_input_error, "node_init_std": self.node_init_std}

    @property
    def params(self):
        return {"w": self.w, "b": self.b, "use_bias": self.structure.use_bias}
    
    @property
    def grads(self):
        return {"w": self.dw, "b": self.db}

    def _reset_params(self):
        self.w = torch.empty( self.structure.N, self.structure.N, device=DEVICE)
        self.b = torch.empty( self.structure.N, device=DEVICE)
        
        self.weight_init(self.w)
        if self.structure.use_bias:
            self.bias_init(self.b)

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
        self.forward(self.structure.num_layers-1)

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
        self.dw = self.structure.grad_w(x=self.x, e=self.e, w=self.w, b=self.b)
        if self.structure.use_bias:
            self.db = self.structure.grad_b(x=self.x, e=self.e, w=self.w, b=self.b)
            
    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def update_xs(self, train=True):
        if self.early_stop:
            early_stopper = EarlyStopper(patience=0, min_delta=self.min_delta)

        di = self.structure.shape[0]
        upper = -self.structure.shape[2] if train else self.structure.N

        T = self.T_train if train else self.T_test
        


        
        for t in range(T): 
            self.e = self.x - self.structure.pred( x=self.x, w=self.w, b=self.b )
            if not self.use_input_error:
                self.e[:,:di] = 0 
            
            # dEdx = self.structure.grad_x(self.x, self.e, self.w, self.b,train=train) # only hidden nodes

            # loss = self.e[:,lower:upper] - self.dfdx(x[:,lower:upper]) * torch.matmul(e, w.T[lower:upper,:].T)
            # loss = self.e - self.dfdx(x) * torch.matmul(self.e, self.w.T[lower:upper,:].T)
            loss = self.e - self.structure.dfdx(self.x) * torch.matmul(self.e, self.w)

            # lower = self.shape[0]
            # upper = -self.shape[2] if train else sum(self.shape)
            # print("loss", loss.shape)
            dEdx = loss[:, di:upper]
            # print("dEdx", dEdx.shape)

            # dEdx = self.structure.grad_x(self.x, self.e, self.w, self.b,train=train) # only hidden nodes
            self.x[:,di:upper] -= self.lr_x*dEdx 

            
            # self.update_w()

            # if train and self.incremental:
            #     self.update_w()
            if self.incremental and self.dw is not None:
                self.optimizer.step(self.params, self.grads, batch_size=self.x.shape[0])
                
            if self.early_stop:
                if early_stopper.early_stop( self.get_energy() ):
                    break            

    # def train_supervised(self, X_batch, y_batch): 
    #     X_batch = to_vector(X_batch)                  # makes e.g. 28*28 -> 784
    #     y_batch = onehot(y_batch, N=self.structure.shape[2])    # makes e.g. 3 -> [0,0,0,1,0,0,0,0,0,0]

    #     self.reset_nodes(batch_size=X_batch.shape[0])        
    #     self.clamp_input(X_batch)
    #     # self.init_hidden()
    #     # print("ommit init hidden")
    #     self.clamp_target(y_batch)

    #     self.update_xs(train=True)
    #     self.update_w()

    #     if not self.incremental:
    #         # self.update_w()
    #         self.optimizer.step(self.params, self.grads, batch_size=X_batch.shape[0])

    
    def train_supervised(self, X_batch, y_batch): 
     
        self.reset_nodes(batch_size=X_batch.shape[0])        
                
        # Directly set graph_data into self.x
        self.x[:, :] = X_batch.clone()

        self.update_xs(train=True)
        self.update_w()

        if not self.incremental:
            # self.update_w()
            self.optimizer.step(self.params, self.grads, batch_size=X_batch.shape[0])


    def test_feedforward(self, X_batch):
        pass
        # X_batch = to_vector(X_batch)     # makes e.g. 28*28 -> 784

        # self.reset_nodes(batch_size=X_batch.shape[0])
        # self.clamp_input(X_batch)
        # self.forward(self.structure.num_layers)

        # return self.x[:,-self.structure.shape[2]:] 

    # def test_iterative(self, X_batch, diagnostics=None, early_stop=False):

    def test_iterative(self, graph_data):
        self.reset_nodes(batch_size=graph_data.shape[0])

        # Set the graph data (flattened image + internal zeros + one-hot label)
        self.x[:, :] = graph_data.clone()

        # Zero out the one-hot vector (last output_size elements)
        self.x[:, -self.structure.shape[2]:] = 0

        self.update_xs(train=False)

        return self.x[:, -self.structure.shape[2]:] # Return only the output layer predictions


    # def test_iterative(self, X_batch):
    #     X_batch = to_vector(X_batch)     # makes e.g. 28*28 -> 784

    #     self.reset_nodes(batch_size=X_batch.shape[0])
    #     self.clamp_input(X_batch)
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


#################################### IMPORTS ####################################

DATASET_PATH = './data'
SAVE_PATH = f"output/PCG_{dt_string}"

seed(0)

class GraphFormattedMNIST(torch.utils.data.Dataset):
    def __init__(self, dataset, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Precompute dataset transformation (instead of transforming on-the-fly)
        self.data = []
        self.labels = []

        one_hot_lookup = torch.eye(output_size)  # One-hot encoding lookup

        for img, label in dataset:
            image_flat = img.view(-1).to(torch.float32)  # Flattened
            internal_nodes = torch.zeros(hidden_size, dtype=torch.float32)  # Internal nodes
            one_hot_label = one_hot_lookup[label]

            # Precomputed graph data
            graph_data = torch.cat([image_flat, internal_nodes, one_hot_label])

            self.data.append(graph_data)
            self.labels.append(label)

        # Convert to tensors for fast access
        self.data = torch.stack(self.data)  # Shape: [N, input_size + hidden_size + output_size]
        self.labels = torch.tensor(self.labels, dtype=torch.long)  # Shape: [N]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]  # No transformations here!



f = tanh
use_bias = False
# use_bias = False # errie set to False 
shape = [784, 48, 10] # input, hidden, output
mask = get_mask_hierarchical([784,32,16,10])

# mask = torch.ones_like(mask)

structure = PCG_MBA(f=f, 
                    use_bias=use_bias,
                    shape=shape,
                    mask=mask,
                    )


# Inference
lr_x = 0.5                  # inference rate 
T_train = 5                 # inference time scale
T_test = 10                 # unused for hierarchical model
incremental = True          # whether to use incremental EM or not
use_input_error = False     # whether to use errors in the input layer or not
# use_input_error = True     # whether to use errors in the input layer or not

# Learning
lr_w = 0.00001              # learning rate
# lr_w = 0.0001              # learning rate
batch_size = 100 
weight_decay = 0             
grad_clip = 1
batch_scale = False

PCG = PCgraph(structure=structure,
            lr_x=lr_x, 
            T_train=T_train,
            T_test=T_test,
            incremental=incremental, 
            use_input_error=use_input_error,
            )

optimizer = Adam(
    PCG.params,
    learning_rate=lr_w,
    grad_clip=grad_clip,
    batch_scale=batch_scale,
    weight_decay=weight_decay,
)

PCG.set_optimizer(optimizer)



# transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

# train_dataset = torchvision.datasets.MNIST(root=DATASET_PATH, train=True, transform=transform, download=True)
# test_set = torchvision.datasets.MNIST(root=DATASET_PATH, train=False, transform=transform, download=True)

# train_set, val_set = random_split(train_dataset, [50000, 10000])
# train_indices = train_subset_indices(train_set, 10, no_per_class=0) # if a certain number of samples per class is required, set no_per_class to that number. 0 means all samples are used.

# train_loader = preprocess( DataLoader(train_set, batch_size=batch_size, sampler=SubsetRandomSampler( train_indices ), drop_last=False) ) # subsetrandomsampler shuffles the data.
# val_loader = preprocess( DataLoader(val_set, batch_size=len(val_set), shuffle=False, drop_last=False) )
# test_loader = preprocess( DataLoader(test_set, batch_size=len(test_set), shuffle=False, drop_last=False) )


# Define dataset sizes
input_size = 784  # Flattened MNIST image
hidden_size = 32+16  # Number of internal nodes in graph
output_size = 10  # Number of classes

# Load the raw MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

base_train_dataset = torchvision.datasets.MNIST(root=DATASET_PATH, train=True, transform=transform, download=True)
base_test_dataset = torchvision.datasets.MNIST(root=DATASET_PATH, train=False, transform=transform, download=True)

# Wrap it in the GraphFormattedMNIST class
train_dataset = GraphFormattedMNIST(base_train_dataset, input_size, hidden_size, output_size)
test_dataset = GraphFormattedMNIST(base_test_dataset, input_size, hidden_size, output_size)

# Split the dataset
train_set, val_set = random_split(train_dataset, [50000, 10000])

train_indices = train_subset_indices(train_set, 10, no_per_class=0)

# Create DataLoaders
# train_loader = DataLoader(train_set, batch_size=batch_size, sampler=SubsetRandomSampler(train_indices), drop_last=False, num_workers=4, pin_memory=True)
train_loader = DataLoader(train_set, batch_size=batch_size, drop_last=False, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_set, batch_size=len(val_set), shuffle=False, drop_last=False, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False, drop_last=False, num_workers=4, pin_memory=True)




MSE = torch.nn.MSELoss()

train_energy, train_loss, train_acc = [], [], []
val_loss, val_acc = [], []

early_stopper = EarlyStopper(patience=5, min_delta=0)

epochs = 10
epochs = 15

start_time = datetime.now()

with torch.no_grad():
    for i in tqdm(range(epochs)):
        
        energy = 0
        for batch_no, (X_batch, y_batch) in enumerate(train_loader):
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)  # Move to GPU

            PCG.train_supervised(X_batch, y_batch)
            energy += PCG.get_energy()
        train_energy.append(energy/len(train_loader))

        loss, acc = 0, 0
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)  # Move to GPU

            y_batch = y_batch.to(DEVICE)
            y_pred = PCG.test_supervised(X_batch)

            loss += MSE(y_pred, onehot(y_batch, N=10) ).item()
            acc += torch.mean(( torch.argmax(y_pred, axis=1) == y_batch ).float()).item()

        val_acc.append(acc/len(val_loader))
        val_loss.append(loss)

        print(f"\nEPOCH {i+1}/{epochs} \n #####################")   
        print(f"VAL acc:   {val_acc[i]:.3f}, VAL MSE:   {val_loss[i]:.3f}, TRAIN ENERGY:   {train_energy[i]:.3f}")

        if early_stopper.early_stop(val_loss[i]):
            print(f"\nEarly stopping at epoch {i+1}")          
            break

print(f"\nTraining time: {datetime.now() - start_time}")
print(torch.argmax(y_pred, axis=1)[:20])
print(y_batch[:20])

loss, acc = 0, 0
for X_batch, y_batch in test_loader:
    X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)  # Move to GPU

    y_pred = PCG.test_supervised(X_batch) 

    loss += MSE(y_pred, onehot(y_batch,N=10) ).item()
    acc += torch.mean(( torch.argmax(y_pred, axis=1) == y_batch).float()).item() 

test_energy = energy/len(test_loader)
test_acc = acc/len(test_loader)
test_loss = loss/len(test_loader)

print(f"\nTEST acc:   {test_acc:.3f}, TEST MSE:   {test_loss:.3f}")
print("Training & testing finished in %s" % str((datetime.now() - start_time)).split('.')[0])

import matplotlib.pyplot as plt

plt.imshow(PCG.w.cpu().detach())
plt.show()