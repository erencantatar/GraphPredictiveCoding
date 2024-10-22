import os 
import json 

from torch_geometric.utils import from_networkx, to_dense_adj
import networkx as nx
# import matplotlib.pyplot as plt
# from torch_geometric.data import Data
# from torch_geometric.nn import MessagePassing
import numpy as np 
import torch 
import random 

graph_type_options = {
        "fully_connected": {
            "params": {
                # "remove_sens_2_sens": True, 
                # "remove_sens_2_sup": False, 
            }
        },
        "fully_connected_w_self": {
            "params": {
                # "remove_sens_2_sens": True, 
                # "remove_sens_2_sup": False, 
            }
        },

        "fully_connected_no_sens2sup": {
            "params": {
                # "remove_sens_2_sens": True, 
                # "remove_sens_2_sup": False, 
            }
        }, 

        
        "stochastic_block": {
            "params": {
                "num_communities": 150,      # Number of communities
                "community_size": 40,       # Size of each community
                "p_intra": 0.3,             # Probability of edges within the same community
                "p_inter": 0.1,             # Probability of edges between different communities
                "full_con_last_cluster_w_sup": True,
                # "remove_sens_2_sens": False, 
                # "remove_sens_2_sup": False, 
                }
        },
        

        "stochastic_block_w_supvision_clusters": {
            "params": {
                "num_communities": 50,      # Number of communities
                "community_size": 30,       # Size of each community
                "p_intra": 0.5,             # Probability of edges within the same community
                "p_inter": 0.1,             # Probability of edges between different communities
                # "remove_sens_2_sens": True, 
                # "remove_sens_2_sup": True
                }
        },

        "barabasi": {
            "params": {
                "num_edges_to_attach": 5  # Example parameter for Barabasi graph
            }
        },
    }



class GraphBuilder:

    def __init__(self, internal_nodes, supervised_learning, graph_type, seed):

        # Initialize the seed if provided
        self.seed = seed
        self.set_seed(self.seed)  

        self.graph_type = graph_type

        # check if graph tpye aligns with the needed graph_params 
        # TODO 

        # init base, but can change depending on the graph type.
        print("--------Init base indices for sensory, internal, supervision nodes--------")
        
        self.SENSORY_NODES = 784
        # self.num_sensor_nodes = range(self.SENSORY_NODES)
        # self.num_internal_nodes = range(self.SENSORY_NODES, self.SENSORY_NODES + self.NUM_INTERNAL_NODES)

        self.NUM_INTERNAL_NODES = internal_nodes
        self.num_sensor_nodes = list(range(self.SENSORY_NODES))
        self.num_internal_nodes = list(range(self.SENSORY_NODES, self.SENSORY_NODES + self.NUM_INTERNAL_NODES))
        self.INTERNAL = self.num_internal_nodes

        self.supervised_learning = supervised_learning
        
        # Define total number of nodes depending on supervised or unsupervised learning
        if self.supervised_learning:
            self.num_all_nodes = range(self.SENSORY_NODES + self.NUM_INTERNAL_NODES + 10)  # Adding 10 label nodes
            self.supervision_indices = range(self.SENSORY_NODES + self.NUM_INTERNAL_NODES, self.SENSORY_NODES + self.NUM_INTERNAL_NODES + 10)
        else:
            self.num_all_nodes = range(self.SENSORY_NODES + self.NUM_INTERNAL_NODES)

        # Initialize the number of vertices and indices
        print("--------Updating base indices for sensory, internal, supervision nodes--------")
        self.num_vertices = len(self.num_sensor_nodes) + len(self.num_internal_nodes)
        if self.supervised_learning:
            self.num_vertices += len(self.supervision_indices)  # Add supervision nodes to the vertex count
        self.sensory_indices = list(range(self.SENSORY_NODES))  # Assuming SENSORY_NODES is the number of sensory nodes
        self.internal_indices = list(self.num_internal_nodes)

        self.graph_params = graph_type["params"]

        ### TODO 
        self.create_new_graph = True   
        self.save_graph = True 
        self.dir = f"graphs/{self.graph_type['name']}/"
    

        print(self.graph_params)

        # Modify the path based on the graph configuration (removing sens2sens or sens2sup)
        if self.graph_params["remove_sens_2_sens"] and self.graph_params["remove_sens_2_sup"]:
            self.dir += "_no_sens2sens_no_sens2sup"
        elif self.graph_params["remove_sens_2_sens"]:
            self.dir += "_no_sens2sens"
        elif self.graph_params["remove_sens_2_sup"]:
            self.dir += "_no_sens2sup"
        else:
            self.dir += "_normal"  # If neither are removed, label the folder as 'normal'
                
        if self.create_new_graph:
            self.graph = self.create_graph()
        else:
            self.graph = self.use_old_graph()
            if not self.graph:
                self.graph = self.create_graph()


    def use_old_graph(self):
        # Check if the directory for the specific seed exists
        graph_folder = os.path.join(self.dir, str(self.seed))
        graph_file = os.path.join(graph_folder, "edge_index.pt")
        params_file = os.path.join(graph_folder, "graph_type.json")

        # Check if both files exist in the folder
        if os.path.exists(graph_file) and os.path.exists(params_file):
            print(f"Found existing graph files in {graph_folder}")

            # Load the saved graph parameters from the JSON file
            with open(params_file, "r") as f:
                saved_graph_params = json.load(f)

            # Compare saved parameters with current graph parameters
            if saved_graph_params == self.graph_type:
                print("Graph parameters match, loading the existing graph.", graph_file)
                # Load the existing edge_index tensor
                self.edge_index = torch.load(graph_file)
                return True
            else:
                print(f"Graph parameters in {graph_folder} do not match.")
        else:
            print(f"No graph found for seed {self.seed}.")
        return False

    def create_graph(self):

        self.edge_index = []

        print(f"Creating graph structure for {self.graph_type['name']}")
        if self.graph_type["name"] == "fully_connected":
            self.fully_connected(self_connection=False, 
                                 no_sens2sens=self.graph_params["remove_sens_2_sens"], 
                                 no_sens2supervised=self.graph_params["remove_sens_2_sup"])
        elif self.graph_type["name"] == "fully_connected_w_self":
            self.fully_connected(self_connection=True,
                                 no_sens2sens=self.graph_params["remove_sens_2_sens"], 
                                 no_sens2supervised=self.graph_params["remove_sens_2_sup"])
        elif self.graph_type["name"] == "barabasi":
            self.barabasi()
        elif self.graph_type["name"] == "stochastic_block":
            self.stochastic_block()

        elif self.graph_type["name"] == "stochastic_block_w_supvision_clusters":
            self.stochastic_block_w_supervision_clusters()
        else:
            raise ValueError(f"Invalid graph type: {self.graph_type['name']}")
        

        # Convert edge_index to tensor only if it's not already a tensor
        if not isinstance(self.edge_index, torch.Tensor):
            self.edge_index = torch.tensor(self.edge_index, dtype=torch.long)
        
        print("graph created")

        # Transpose if edge_index was manually created
        if self.graph_type["name"] in ["fully_connected", "fully_connected_w_self", "fully_connected_no_sens2sup", "barabasi"]:
            self.edge_index = self.edge_index.t().contiguous()
            self.edge_index_tensor = self.edge_index.t().contiguous()

        assert self.edge_index.shape[0] == 2
        print("self.edge_index", self.edge_index.shape)


        # save to self.dir 
        if self.save_graph:
            self.save_graph_to_file()

    def save_graph_to_file(self):
        print("---------save_graph_to_file--------------------")
        
        # Use the seed as the folder name
        new_folder_path = os.path.join(self.dir, str(self.seed))
        
        # Create the directory if it doesn't exist
        if not os.path.exists(new_folder_path):
            os.makedirs(new_folder_path)

        # Save edge_index as a torch file in the new folder
        torch.save(self.edge_index, os.path.join(new_folder_path, "edge_index.pt"))

        # Save graph_type["params"] as a JSON file in the new folder
        with open(os.path.join(new_folder_path, "graph_type.json"), "w") as f:
            json.dump(self.graph_type, f)

        print(f"Graph data saved in {new_folder_path}")
    
    def set_seed(self, seed):
        if seed is not None:
            print(f"Setting seed: {seed}")
            np.random.seed(seed)  # Seed for numpy
            torch.manual_seed(seed)  # Seed for torch
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)  # Seed for GPU
            random.seed(seed)  # Seed for random module (if used)
        else:
            print("No seed provided. Using default behavior.")


    def fully_connected_no_sens2sup(self):

        print("Creating fully connected graph without sensory to internal (and otherwayaround)")
        # Sensory to sensory (both ways)
        for i in self.num_sensor_nodes:
            for j in self.num_sensor_nodes:
                if i != j:
                    self.edge_index.append([i, j])

        # Sensory to internal (both ways )
        for i in self.num_sensor_nodes:
            for j in self.num_internal_nodes:
                self.edge_index.append([i, j])
                self.edge_index.append([j, i])

        # Internal to internal (both directions)
        for i in self.num_internal_nodes:
            for j in self.num_internal_nodes:
                if i != j:
                    self.edge_index.append([i, j])

        if self.supervised_learning:
            # Internal to label (both directions)
            label_nodes = range(self.SENSORY_NODES + self.NUM_INTERNAL_NODES, self.SENSORY_NODES + self.NUM_INTERNAL_NODES + 10)
            for i in self.num_internal_nodes:
                for j in label_nodes:
                    self.edge_index.append([i, j])
                    self.edge_index.append([j, i])
            

    def fully_connected(self, self_connection, no_sens2sens=False, no_sens2supervised=False):

        if (not no_sens2sens) and (not no_sens2supervised):
            if self_connection:
                print("Creating fully connected directed graph with self connections")
                self.edge_index += [[i, j] for i in self.num_all_nodes for j in self.num_all_nodes]
            else:
                print("Creating fully connected directed graph without self connections")
                self.edge_index += [[i, j] for i in self.num_all_nodes for j in self.num_all_nodes if i != j]

        else:

            print(f"Doing from scratch; Creating fully connected directed graph with self connections: {self_connection}")

            # 1. Sensory to all others (sensory, internal, supervision)
            for i in self.sensory_indices:
                # Sensory to sensory
                if not no_sens2sens:
                    for j in self.sensory_indices:
                        if i != j or self_connection:  # Allow self-connection only if specified
                            self.edge_index.append([i, j])

                # Sensory to internal (and vice versa)
                for j in self.internal_indices:
                    self.edge_index.append([i, j])  # Sensory to internal
                    self.edge_index.append([j, i])  # Internal to sensory

                # Sensory to supervision (and vice versa)
                if self.supervised_learning and not no_sens2supervised:
                    for j in self.supervision_indices:
                        self.edge_index.append([i, j])  # Sensory to supervision
                        self.edge_index.append([j, i])  # Supervision to sensory

            # 2. Internal to all others (sensory, internal, supervision)
            for i in self.internal_indices:
                # Internal to internal
                for j in self.internal_indices:
                    if i != j or self_connection:  # Internal to internal (both ways)
                        self.edge_index.append([i, j])

                # Internal to sensory (and vice versa)
                for j in self.sensory_indices:
                    self.edge_index.append([i, j])  # Internal to sensory
                    self.edge_index.append([j, i])  # Sensory to internal

                # Internal to supervision (and vice versa)
                if self.supervised_learning:
                    for j in self.supervision_indices:
                        self.edge_index.append([i, j])  # Internal to supervision
                        self.edge_index.append([j, i])  # Supervision to internal

            # 3. Supervision to all others (sensory, internal, supervision)
            if self.supervised_learning:
                for i in self.supervision_indices:
                    # Supervision to sensory (and vice versa)
                    if not no_sens2supervised:
                        for j in self.sensory_indices:
                            self.edge_index.append([i, j])  # Supervision to sensory
                            self.edge_index.append([j, i])  # Sensory to supervision

                    # Supervision to internal (and vice versa)
                    for j in self.internal_indices:
                        self.edge_index.append([i, j])  # Supervision to internal
                        self.edge_index.append([j, i])  # Internal to supervision

                    # Supervision to supervision
                    for j in self.supervision_indices:
                        if i != j or self_connection:  # Supervision to supervision (both ways)
                            self.edge_index.append([i, j])

            print("Fully connected graph creation complete.")




    def barabasi(self):
        num_nodes = len(self.num_all_nodes)
        m = self.graph_params.get("m", 5)  # Number of edges to attach from a new node to existing nodes
        G = nx.barabasi_albert_graph(num_nodes, m, seed=self.seed)
        self.edge_index = [[u, v] for u, v in G.edges()]
        print(f"Creating Barabási-Albert graph with {num_nodes} nodes and {m} edges to attach per new node")

    
    def stochastic_block(self):

        # Given code parameters; else take default 40, 50
        num_communities = self.graph_params.get("num_communities", 40)  # Example block sizes
        community_size = self.graph_params.get("community_size",   50)
        p_intra = self.graph_params.get("p_intra", 0.5)  # Probability of edges within the same community
        p_inter = self.graph_params.get("p_inter", 0.1)  # Probability of edges between different communities
        
        print(num_communities, community_size,  len(self.num_internal_nodes) )

        assert (num_communities * community_size) == self.NUM_INTERNAL_NODES, "must be equal"
        # assert (num_communities * community_size) == len(self.num_internal_nodes), "must be equal"

        # Sizes of communitieser
        sizes = [community_size for _ in  range(num_communities)]
        # SENSORY_NODES = range(0, range(self.SENSORY_NODES))
        self.INTERNAL = range(self.SENSORY_NODES, (self.SENSORY_NODES+sum(sizes)))
        self.num_internal_nodes = sum(self.INTERNAL)
        # SUPERVISED_NODES = range(self.SENSORY_NODES+sum(sizes), sum(sizes)+10)


        sizes.insert(0, self.SENSORY_NODES)
        sizes.append(10)
        community_sizes = sizes

        num_communities = len(community_sizes)
  
        # Create the stochastic block model graph
        p = np.full((num_communities, num_communities), p_inter)
        np.fill_diagonal(p, p_intra)

        # removing sensory to sensory connection
        if self.graph_params.get("remove_sens_2_sens", True):
            p[0, 0] = 0
        
        # adding connections from community to community  
        for i in range(1, num_communities):
            p[0, i] = 0.1
            p[i, 0] = 0.1

        if self.graph_params.get("remove_sens_2_sup", True):
            # remove sensory to supervised
            p[0, -1] = 0 
            p[-1, 0] = 0 

        fully_connect_last_cluster = self.graph_params.get("full_con_last_cluster_w_sup", False)
        # If requested, fully connect the last community with the supervised nodes
        if fully_connect_last_cluster and self.supervised_learning:
            last_cluster_idx = num_communities - 2  # Last internal community
            supervision_cluster_idx = num_communities - 1  # Supervised nodes
            p[last_cluster_idx, supervision_cluster_idx] = 1.0
            p[supervision_cluster_idx, last_cluster_idx] = 1.0

        print("Got the sizes", sizes)
        G = nx.stochastic_block_model(sizes, p, directed=True, seed=self.seed)
        
        print("Created the graph stochastic_block_model") 

        # Convert the graph to an adjacency matrix
        # adj_matrix = nx.adjacency_matrix(G).todense()

        # Convert the NetworkX graph to PyTorch Geometric format
        data = from_networkx(G)

        # Extract the edge_index tensor
        self.edge_index = data.edge_index

        print("done creating Stochastic Block Model graph")

    def stochastic_block_w_supervision_clusters(self):

        # Parameters
        num_communities = self.graph_params.get("num_communities", 3)  # 3 internal clusters
        community_size = self.graph_params.get("community_size", 5)  # Each internal cluster has 5 nodes
        supervision_cluster_size = 10  # 10 supervision clusters

        # Ensure that the total number of internal nodes is correct
        assert (num_communities * community_size) == len(self.num_internal_nodes), "Internal node count must match."

        # Initialize ranges
        SENSORY_NODES = self.SENSORY_NODES
        self.INTERNAL = range(self.SENSORY_NODES, self.SENSORY_NODES + num_communities * community_size)
        self.num_internal_nodes = sum(self.INTERNAL)

        # Sizes for internal clusters
        internal_community_sizes = [community_size for _ in range(num_communities)]

        # Indices for supervision clusters, 1 supervision node + 9 internal nodes per cluster
        supervision_internal_nodes_per_cluster = 9
        self.SUPERVISION = []  # Supervision nodes
        supervision_internal_indices = []  # Internal nodes from the supervision clusters

        supervision_start_idx = self.SENSORY_NODES + num_communities * community_size
        for i in range(supervision_cluster_size):
            supervision_node_idx = supervision_start_idx + i * (supervision_internal_nodes_per_cluster + 1)
            self.SUPERVISION.append(supervision_node_idx)  # First node is the supervision node

            # The remaining 9 nodes in the supervision cluster are internal nodes
            supervision_internal_indices.extend(range(supervision_node_idx + 1, supervision_node_idx + 10))

        # Combine all internal nodes (from internal clusters + supervision clusters)
        self.internal_indices = list(range(self.SENSORY_NODES, self.SENSORY_NODES + num_communities * community_size)) + supervision_internal_indices

        # Set up intra-cluster and inter-cluster probabilities
        p_intra = self.graph_params.get("p_intra", 0.5)
        p_inter = self.graph_params.get("p_inter", 0.1)

        # Total communities: internal + supervision
        total_community_sizes = internal_community_sizes + [supervision_internal_nodes_per_cluster + 1] * supervision_cluster_size

        # Create block probability matrix
        p = np.full((len(total_community_sizes), len(total_community_sizes)), p_inter)
        np.fill_diagonal(p, p_intra)

        # removing sensory to sensory connection
        if self.graph_params.get("remove_sens_2_sens", True):
            p[0, 0] = 0
        
        # Adding connections from community to community
        for i in range(1, len(total_community_sizes)):
            p[0, i] = 0.1
            p[i, 0] = 0.1

        # Remove sensory to supervised connections if specified
        if self.graph_params.get("remove_sens_2_sup", True):
            # Sensory community is at index 0, and supervision nodes are at the end
            p[0, -1] = 0
            p[-1, 0] = 0

        # Build stochastic block model
        G = nx.stochastic_block_model(total_community_sizes, p, directed=True, seed=self.seed)

        # Convert the graph to PyTorch Geometric format
        data = from_networkx(G)

        # Update edge_index tensor
        self.edge_index = data.edge_index

        # Update the self.indices to reflect the new structure
        print("--------Updating base indices for sensory, internal, supervision nodes--------")
        self.num_vertices = len(self.sensory_indices) + len(self.internal_indices) + len(self.supervision_indices)
        self.sensory_indices = list(range(SENSORY_NODES))

        self.supervision_indices = self.SUPERVISION
        self.internal_indices = list(self.INTERNAL) + supervision_internal_indices


        print("Created Stochastic Block Model graph with supervision clusters")
