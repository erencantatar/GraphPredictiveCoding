
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
import numpy as np
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from graphbuilder import GraphBuilder

class CustomGraphDataset(Dataset):
    def __init__(self, graph_params, mnist_dataset, supervised_learning, numbers_list,  
                 same_digit=False,
                 add_noise=False, noise_intensity=0.1, N=2,
                 edge_index=None, supervision_label_val=1, indices=None):
        self.mnist_dataset = mnist_dataset
        # self.graph_structure = graph_structure
        self.NUM_INTERNAL_NODES = graph_params["internal_nodes"]
        self.add_noise = add_noise
        self.noise_intensity = noise_intensity
        self.same_digit = same_digit
        self.supervised_learning = supervised_learning
        self.N = N.lower() if type(N)==str else N
        self.supervision_label_val = supervision_label_val 

        if self.supervised_learning:
            print("Supervised learning")
        else:
            print("Un supervised learning")
        
        print(f"Taking first n={self.N} digits from each class")

        if numbers_list:
            self.numbers_list = numbers_list
        else:
            self.numbers_list = list(range(10))

        # Instead of storing just the first occurrence, store all occurrences of each digit
        self.indices = {int(digit): [] for digit in self.numbers_list}

        # Populate the indices dictionary with all occurrences of each digit
        for idx, (image, label) in enumerate(self.mnist_dataset):
            if int(label) in self.numbers_list:
                self.indices[int(label)].append(idx)

        # ------------------- Create the graph structure -------------------
        # FOR TESTING THE METHOD of MESSAGE PASSING
        # SENSORY_NODES = 10 # 784
        # SENSORY_NODES = 784 # 784
        # NUM_INTERNAL_NODES = self.NUM_INTERNAL_NODES 
        # num_sensor_nodes    = range(SENSORY_NODES)
        # num_internal_nodes  = range(SENSORY_NODES, SENSORY_NODES + NUM_INTERNAL_NODES)

        # if self.supervised_learning:
        #     num_all_nodes = range(SENSORY_NODES + NUM_INTERNAL_NODES + 10)
        # else:
        #     num_all_nodes = range(SENSORY_NODES + NUM_INTERNAL_NODES)



        if edge_index is not None:
            self.edge_index = edge_index
            self.edge_index_tensor = self.edge_index
        else:

            from graphbuilder import graph_type_options

            loader = GraphBuilder(graph_type_options, **graph_params)
            
            self.edge_index = loader.edge_index
            self.edge_index_tensor = self.edge_index

            # self.NUM_INTERNAL_NODES_range = loader.INTERNAL
            # self.NUM_INTERNAL_NODES = sum(loader.num_internal_nodes)
        
        if indices:
            self.num_vertices, self.sensory_indices, self.internal_indices, self.supervision_indices = indices
        else:
            self.num_vertices = loader.num_vertices
            self.sensory_indices = loader.sensory_indices
            self.internal_indices = loader.internal_indices
            self.supervision_indices = loader.supervision_indices
            
        print("-----Done-----")
        print(self.num_vertices)
        print(self.sensory_indices)
        print(self.internal_indices)
        print(self.supervision_indices)
        print("-----Done-----")
            # Convert the edge list to a PyTorch tensor and transpose it to match the expected shape (2, num_edges)
            # self.edge_index_tensor = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

        # # Initialize the weights for the edges
        # self.weights = torch.ones(self.edge_index_tensor.size(1))
        
        # # Convert to sparse representation
        # self.sparse_weights = torch.sparse.FloatTensor(self.edge_index_tensor, self.weights, torch.Size([len(num_all_nodes), len(num_all_nodes)]))

    # def get_edge_weight(self, i, j):
    #     """
    #     Retrieve the weight of the edge from node i to node j.

    #     Parameters:
    #     i (int): Source node index.
    #     j (int): Target node index.

    #     Returns:
    #     float: Weight of the edge from node i to node j.
    #     """
    #     edge_indices = (self.edge_index_tensor[0] == i) & (self.edge_index_tensor[1] == j)
    #     if edge_indices.any():
    #         return self.weights[edge_indices.nonzero(as_tuple=True)[0]].item()
    #     else:
    #         return 0.0, False  # Return 0 if no such edge exists
        
    def __len__(self):
        # TODO check this --> maybe len(self.indices)
        # return len(self.mnist_dataset)
        return sum(len(indices) for indices in self.indices.values())


    def __getitem__(self, idx):
        
        # if self.same_digit:
        #     # fix idx to be the same digit
        #     selected_idx = 9

        # else:
        #     digit = np.random.choice(self.numbers_list)
        
        #     # Randomly select an index from the list of indices for the selected digit
        #     digit_indices = self.indices[digit]

        #     if self.N == "all":
        #         self.N = len(digit_indices)
        #     digit_indices = digit_indices[0:self.N]
        #     selected_idx = np.random.choice(digit_indices)

        flat_indices = [(digit, idx_in_digit) for digit, digit_indices in self.indices.items() for idx_in_digit in digit_indices]

        # Get the correct digit and its associated sample index
        digit, selected_idx = flat_indices[idx]

        if idx == 0:  # Only print for the first batch to avoid repetitive output

            print("Selected idx: ", selected_idx)

        # Get the image and label from the dataset using the randomly selected index
        image, label = self.mnist_dataset[selected_idx]
        clean_image = image.clone()  # Store the clean image

        # Optionally add noise to the image
        if self.add_noise:
            noise = torch.randn(image.size()) * self.noise_intensity
            image = image + noise
            # Ensure the noisy image is still within valid range
            image = torch.clamp(image, 0, 1)
        
        # Initialize sensory nodes with image values
        # if self.supervised_learning:
        #     # the sensory nodes are fixed to a 1-hot vector with the labels
        #     # we want outgoing edges from the label_nodes (1 hot of 10 classes) to the pixel nodes

        #     # Create a one-hot encoded vector of the label
        #     label_vector = torch.zeros(10)  # Assuming 10 classes for MNIST
        #     label_vector[label] = self.supervision_label_val

        #     # Concatenate the label vector with the image vector
        #     # concat the image, internal nodes (zeros) and label nodes

        #     values = torch.cat((image.view(-1, 1), torch.zeros(self.NUM_INTERNAL_NODES).view(-1, 1), label_vector.view(-1, 1)), dim=0)    
        #     # values = torch.cat((one_hot.view(-1, 1), torch.zeros(self.NUM_INTERNAL_NODES).view(-1, 1), label_vector.view(-1, 1)), dim=0)

        #     # x = torch.cat((image.view(-1, 1), label_vector.view(-1, 1)), dim=0)
        # else:
        #     # Flatten image to use as part of the node features
        #     # NO: concat the image, internal nodes (zeros) and label nodes
        #     # YES: concat the image
        #     values = torch.cat((image.view(-1, 1), torch.zeros(self.NUM_INTERNAL_NODES).view(-1, 1)), dim=0)
        #     # values = torch.cat((one_hot.view(-1, 1), torch.zeros(self.NUM_INTERNAL_NODES).view(-1, 1)), dim=0)
        #     # x = image.view(-1, 1)

        ##### (new) Initialize the values tensor with the number of vertices
        values = torch.zeros(self.num_vertices, 1)

        ## Assign values to sensory nodes based on image
        image_flattened = image.view(-1, 1)
        for sensory_idx in self.sensory_indices:
            values[sensory_idx] = image_flattened[sensory_idx]

        values[:784] = image.view(-1, 1)

        # Initialize internal nodes with zeros
        for internal_idx in self.internal_indices:
            values[internal_idx] = 0  # Or some other initial value

        if self.supervised_learning:
            # Create a one-hot encoded vector of the label
            label_vector = torch.zeros(10)  # Assuming 10 classes for MNIST
            label_vector[label] = self.supervision_label_val

            # Assign values to the supervision nodes
            for i, supervision_idx in enumerate(self.supervision_indices):
                values[supervision_idx] = label_vector[i]

            if idx == 0:  # Only print for the first batch to avoid repetitive output
                print("Adding label", label, label_vector)

        print("Done for idx", idx)
        # Node features: value, prediction, and error for each node
        errors = torch.zeros_like(values)
        predictions = torch.zeros_like(values)

        # Combine attributes into a feature matrix
        features = torch.stack((values, errors, predictions), dim=1)

        # assign the weights to the edges to be 1 
        # print("edge_index_tensor shape: ", self.edge_index_tensor.shape)
        edge_attr = torch.ones(self.edge_index_tensor.size(1))

        self.edge_attr = edge_attr

        return Data(x=features, edge_index=self.edge_index_tensor, y=label, edge_attr=edge_attr), clean_image.squeeze(0)
        # return Data(x=features, edge_index=self.edge_index_tensor, y=label), clean_image.squeeze(0)

