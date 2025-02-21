
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
import numpy as np
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from graphbuilder import GraphBuilder


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

class PCGraphDataset(Dataset):
    def __init__(self, graph, mnist_dataset, supervised_learning, 
                 numbers_list,  
                 add_noise=False, 
                 noise_intensity=0.1, 
                 supervision_label_val=1, 
                 indices=None):
        self.mnist_dataset = mnist_dataset
        # self.graph_structure = graph_structure
        # self.add_noise = add_noise
        self.noise_intensity = noise_intensity
        self.supervised_learning = supervised_learning
        self.supervision_label_val = supervision_label_val 

        self.zero_img_in_graph          = False 
        self.zero_y_in_graph            = False 

        if self.supervised_learning:
            print("Supervised learning")
        else:
            print("Un supervised learning")
        
        # if numbers_list:
        #     self.numbers_list = numbers_list
        # else:
        #     self.numbers_list = list(range(10))

        # # shuffle the numbers_list
        # self.numbers_list_copy = self.numbers_list.copy()
        # np.random.shuffle(self.numbers_list_copy)
        
        # # Instead of storing just the first occurrence, store all occurrences of each digit
        # self.indices = {int(digit): [] for digit in self.numbers_list_copy}

        # # Populate the indices dictionary with all occurrences of each digit
        # for idx, (image, label) in enumerate(self.mnist_dataset):
        #     if int(label) in self.numbers_list_copy:
        #         self.indices[int(label)].append(idx)

        self.edge_index = graph.edge_index
        self.edge_index_tensor = graph.edge_index
        
        # if edge_index is not None:
        #     self.edge_index = edge_index
        #     self.edge_index_tensor = self.edge_index
        # else:

        #     print("graph_params", graph_params)
        #     loader = GraphBuilder(**graph_params)
            
        #     self.edge_type = loader.edge_type
        #     self.edge_index = loader.edge_index
        #     self.edge_index_tensor = self.edge_index

        #     # self.NUM_INTERNAL_NODES_range = loader.INTERNAL
        #     # self.NUM_INTERNAL_NODES = sum(loader.num_internal_nodes)
        
        # if indices:
        #     self.num_vertices, self.sensory_indices, self.internal_indices, self.supervision_indices = indices
        # else:
        self.num_vertices = graph.num_vertices
        self.sensory_indices = graph.sensory_indices
        self.internal_indices = graph.internal_indices
        self.supervision_indices = graph.supervision_indices
        
        print("-----Done-----")
        print(self.num_vertices)
        print(self.sensory_indices)
        print(self.internal_indices)
        print(self.supervision_indices)
        print("-----Done-----")
       
    def __len__(self):
        # TODO check this --> maybe len(self.indices)
        # return len(self.mnist_dataset)
        return len(self.mnist_dataset)

    def __getitem__(self, idx):

        # Get the image and label from the dataset using the randomly selected index
        image, label = self.mnist_dataset[idx]

        # # Optionally add noise to the image
        # if self.add_noise:
        #     noise = torch.randn(image.size()) * self.noise_intensity
        #     image = image + noise
        #     # Ensure the noisy image is still within valid range
        #     image = torch.clamp(image, 0, 1)
       
        ##### (new) Initialize the values tensor with the number of vertices
        values = torch.zeros(self.num_vertices, 1)

        ## Assign values to sensory nodes based on image
        image_flattened = image.view(-1, 1)
        for sensory_idx in self.sensory_indices:
            # values[sensory_idx] = image_flattened[sensory_idx]
            values[sensory_idx] = 0 if self.zero_img_in_graph else image_flattened[sensory_idx]

        values[:784] = image.view(-1, 1)

        # Initialize internal nodes with zeros
        for internal_idx in self.internal_indices:
            values[internal_idx] = 0  # Or some other initial value

        if self.supervised_learning:
            # Create a one-hot encoded vector of the label
            label_vector = torch.zeros(10)  # Assuming 10 classes for MNIST
            label_vector[label] = self.supervision_label_val

            # Zero out the labels if zero_y_in_graph is True
            label_vector = label_vector * (0 if self.zero_y_in_graph else 1)
                                       
            # Assign values to the supervision nodes
            for i, supervision_idx in enumerate(self.supervision_indices):
                values[supervision_idx] = label_vector[i]

        # Node features: value, prediction, and error for each node
        # errors = torch.zeros_like(values)
        # predictions = torch.zeros_like(values)

        # Combine attributes into a feature matrix
        # features = torch.stack((values, errors, predictions), dim=1)
        features = values

        # Data.x [784, internal, supervision]

        edge_attr = torch.ones(self.edge_index_tensor.size(1))

        # return Data(x=features, edge_index=self.edge_index_tensor, 
        #             y=label, edge_attr=edge_attr)
        return Data(
                x=features,
                edge_index=self.edge_index_tensor,
                y=label,
                edge_attr=edge_attr,
                sensory_indices=self.sensory_indices,
                internal_indices=self.internal_indices,
                supervision_indices=self.supervision_indices,
        )
