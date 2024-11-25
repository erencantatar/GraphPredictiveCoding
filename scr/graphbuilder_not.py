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
        
        "stochastic_block_hierarchy": {
            "params": {
                            # "remove_sens_2_sens": True, 
                            # "remove_sens_2_sup": False, 
                        }
        },

        
        "stochastic_block": {
            "params": {
                "num_communities": 50,      # Number of communities
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

        "custom_two_branch": {
            "params": {
                "branch1_config": (3, 10, 10),  # 2 layers, 5 clusters per layer, 5 nodes per cluster for Branch 1
                "branch2_config": (3, 10, 10),  # 2 layers, 5 clusters per layer, 5 nodes per cluster for Branch 2
                # "branch1_config": (2, 5, 10),  # 2 layers, 5 clusters per layer, 5 nodes per cluster for Branch 1
                # "branch2_config": (2, 5, 10),  # 2 layers, 5 clusters per layer, 5 nodes per cluster for Branch 2
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
        print(self.graph_params)

        ### TODO 
        self.create_new_graph = True   
        self.save_graph = True 
        self.dir = f"graphs/{self.graph_type['name']}/"
    


        # Modify the path based on the graph configuration (removing sens2sens or sens2sup)
        if self.graph_params["remove_sens_2_sens"] and self.graph_params["remove_sens_2_sup"]:
            self.dir += "_no_sens2sens_no_sens2sup"
        elif self.graph_params["remove_sens_2_sens"]:
            self.dir += "_no_sens2sens"
        elif self.graph_params["remove_sens_2_sup"]:
            self.dir += "_no_sens2sup"
        else:
            self.dir += "_normal"  # If neither are removed, label the folder as 'normal'
                
        self.edge_type = []  # Store edge types

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

        elif self.graph_type["name"] == "stochastic_block_hierarchy":
            self.stoic_block_hierarchy( 
                                no_sens2sens=self.graph_params["remove_sens_2_sens"], 
                                no_sens2supervised=self.graph_params["remove_sens_2_sup"])
        elif self.graph_type["name"] == "stochastic_block_w_supvision_clusters":
            self.stochastic_block_w_supervision_clusters()
        elif self.graph_type["name"] == "custom_two_branch":
            self.custom_two_branch(self.SENSORY_NODES, 
                                   self.graph_params["branch1_config"] , 
                                   self.graph_params["branch2_config"], 
                                   10)
            
        else:
            raise ValueError(f"Invalid graph type: {self.graph_type['name']}")
        
        print("Recalculating the number of vertices after graph creation")
        self.num_vertices = len(self.sensory_indices) + len(self.internal_indices)
        if self.supervised_learning:
            self.num_vertices += len(self.supervision_indices)  # Add supervision nodes to the vertex count
            
        # Convert edge_index to tensor
        self.edge_index = torch.tensor(self.edge_index, dtype=torch.long).t().contiguous()

        # Convert edge_type to tensor (use integers or map to string if preferred)
        edge_type_map = {"Sens2Sens": 0, "Sens2Inter": 1, "Sens2Sup": 2, "Inter2Sens": 3, "Inter2Inter": 4, "Inter2Sup": 5, "Sup2Sens": 6, "Sup2Inter": 7, "Sup2Sup": 8}
        
        if self.edge_type is [] or self.edge_type is None:
            self.edge_type = torch.tensor([edge_type_map[etype] for etype in self.edge_type], dtype=torch.long)

        # Convert edge_index to tensor only if it's not already a tensor
        if not isinstance(self.edge_index, torch.Tensor):
            self.edge_index = torch.tensor(self.edge_index, dtype=torch.long)
        
        print("graph created")

        # Transpose if edge_index was manually created
        if self.graph_type["name"] in ["fully_connected", "fully_connected_w_self", "fully_connected_no_sens2sup", "barabasi"]:
            self.edge_index = self.edge_index.t().contiguous()
            self.edge_index_tensor = self.edge_index.t().contiguous()

        # Ensure edge_index is a 2-row tensor
        if self.edge_index.shape[0] != 2:
            self.edge_index = self.edge_index.t().contiguous()
        assert self.edge_index.shape[0] == 2, "Edge index must have 2 rows (source and target nodes)"

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

        # Save edge_type as a torch file in the new folder
        torch.save(self.edge_type, os.path.join(new_folder_path, "edge_type.pt"))

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

    def stoic_block_hierarchy(self, no_sens2sens=False, no_sens2supervised=False):
        sensory_grid_size = 28
        sensory_patch_size = 4  # Each patch is 4x4
        patches_per_row = sensory_grid_size // sensory_patch_size
        num_sensory_nodes = sensory_grid_size ** 2

        # Define clusters per layer
        layer_1_clusters = 49  # First layer with 4x4 clusters
        layer_2_clusters = layer_1_clusters
        final_layer_clusters = 10  # Larger clusters before the supervision layer

        # Track all node indices as lists for flexibility
        sensory_nodes = list(range(num_sensory_nodes))
        layer_1_nodes = list(range(num_sensory_nodes, num_sensory_nodes + layer_1_clusters * 16))
        layer_2_nodes = list(range(layer_1_nodes[-1] + 1, layer_1_nodes[-1] + 1 + layer_2_clusters * 16))
        final_layer_nodes = list(range(layer_2_nodes[-1] + 1, layer_2_nodes[-1] + 1 + final_layer_clusters))
        supervision_nodes = list(range(final_layer_nodes[-1] + 1, final_layer_nodes[-1] + 1 + 10))  # Assume 10 supervision nodes

        # Create a list to hold edges and edge types
        self.edge_index = []
        self.edge_type = []

        # Step 1: Sensory-to-Sensory Connections (if allowed by no_sens2sens)
        if not no_sens2sens:
            for i in sensory_nodes:
                for j in range(i + 1, num_sensory_nodes):  # Avoid double edges
                    self.edge_index.append([i, j])
                    self.edge_type.append("Sens2Sens")
                    self.edge_index.append([j, i])
                    self.edge_type.append("Sens2Sens")

        # Optional Sensory-to-Supervision Connections
        if self.supervised_learning and not no_sens2supervised:
            for sensory_node in sensory_nodes:
                for sup_node in supervision_nodes:
                    self.edge_index.append([sensory_node, sup_node])
                    self.edge_type.append("Sens2Sup")
                    self.edge_index.append([sup_node, sensory_node])
                    self.edge_type.append("Sup2Sens")

        # Step 2: Layer 1 Connections (Sensory to Layer 1 Clusters)
        for patch_row in range(patches_per_row):
            for patch_col in range(patches_per_row):
                block_id = patch_row * patches_per_row + patch_col
                start_row, start_col = patch_row * sensory_patch_size, patch_col * sensory_patch_size
                start_index = block_id * 16
                internal_nodes_in_block = layer_1_nodes[start_index:start_index + 16]

                # Connect each sensory node in the 4x4 block to all nodes in its corresponding internal 4x4 cluster
                for i in range(sensory_patch_size):
                    for j in range(sensory_patch_size):
                        sensory_node = (start_row + i) * sensory_grid_size + (start_col + j)
                        for internal_node in internal_nodes_in_block:
                            # Sensory to Internal and Internal to Sensory
                            self.edge_index.append([sensory_node, internal_node])
                            self.edge_type.append("Sens2Inter")
                            self.edge_index.append([internal_node, sensory_node])
                            self.edge_type.append("Inter2Sens")

                # Fully connect nodes within each 4x4 internal block
                for node_i in internal_nodes_in_block:
                    for node_j in internal_nodes_in_block:
                        if node_i != node_j:
                            self.edge_index.append([node_i, node_j])
                            self.edge_type.append("Inter2Inter")

        # Step 3: Layer 2 Connections (Layer 1 to Layer 2 Clusters)
        for i in range(layer_1_clusters):
            layer_1_block = layer_1_nodes[i * 16:(i + 1) * 16]
            layer_2_block = layer_2_nodes[i * 16:(i + 1) * 16]

            # Connect nodes between corresponding clusters in Layer 1 and Layer 2
            for node_1 in layer_1_block:
                for node_2 in layer_2_block:
                    self.edge_index.append([node_1, node_2])
                    self.edge_type.append("Inter2Inter")
                    self.edge_index.append([node_2, node_1])
                    self.edge_type.append("Inter2Inter")

            # Fully connect nodes within each 4x4 block in Layer 2
            for node_i in layer_2_block:
                for node_j in layer_2_block:
                    if node_i != node_j:
                        self.edge_index.append([node_i, node_j])
                        self.edge_type.append("Inter2Inter")

        # Step 4: Final Layer Connections (Layer 2 to Final Large Clusters)
        for node_2 in layer_2_nodes:
            for large_cluster in final_layer_nodes:
                self.edge_index.append([node_2, large_cluster])
                self.edge_type.append("Inter2Inter")
                self.edge_index.append([large_cluster, node_2])
                self.edge_type.append("Inter2Inter")

        # Step 5: Final Layer to Supervision Connections
        if self.supervised_learning:
            for large_cluster in final_layer_nodes:
                for sup_node in supervision_nodes:
                    self.edge_index.append([large_cluster, sup_node])
                    self.edge_type.append("Inter2Sup")
                    self.edge_index.append([sup_node, large_cluster])
                    self.edge_type.append("Sup2Inter")

    def fully_connected(self, self_connection, no_sens2sens=False, no_sens2supervised=False):

        # Define node groups for easy reference
        node_groups = {
            "Sensory": set(self.sensory_indices),
            "Internal": set(self.internal_indices),
            "Supervision": set(self.supervision_indices) if self.supervised_learning else set(),
        }

        def get_edge_type(src, dest):
            if src in node_groups["Sensory"] and dest in node_groups["Sensory"]:
                return "Sens2Sens"
            elif src in node_groups["Sensory"] and dest in node_groups["Internal"]:
                return "Sens2Inter"
            elif src in node_groups["Sensory"] and dest in node_groups["Supervision"]:
                return "Sens2Sup"
            elif src in node_groups["Internal"] and dest in node_groups["Sensory"]:
                return "Inter2Sens"
            elif src in node_groups["Internal"] and dest in node_groups["Internal"]:
                return "Inter2Inter"
            elif src in node_groups["Internal"] and dest in node_groups["Supervision"]:
                return "Inter2Sup"
            elif src in node_groups["Supervision"] and dest in node_groups["Sensory"]:
                return "Sup2Sens"
            elif src in node_groups["Supervision"] and dest in node_groups["Internal"]:
                return "Sup2Inter"
            elif src in node_groups["Supervision"] and dest in node_groups["Supervision"]:
                return "Sup2Sup"

        # Create fully connected edges with edge types
        if (not no_sens2sens) and (not no_sens2supervised):
            for i in self.num_all_nodes:
                for j in self.num_all_nodes:
                    if i != j or self_connection:
                        self.edge_index.append([i, j])
                        self.edge_type.append(get_edge_type(i, j))
        else:

            print(f"Doing from scratch; Creating fully connected directed graph with self connections: {self_connection}")

            # 1. Sensory to all others (sensory, internal, supervision)
            for i in self.sensory_indices:
                # Sensory to sensory
                if not no_sens2sens:
                    for j in self.sensory_indices:
                        if i != j or self_connection:  # Allow self-connection only if specified
                            self.edge_index.append([i, j])
                            self.edge_type.append("Sens2Sens")


                # Sensory to internal (and vice versa)
                for j in self.internal_indices:
                    self.edge_index.append([i, j])  # Sensory to internal
                    self.edge_type.append("Sens2Inter")

                    self.edge_index.append([j, i])  # Internal to sensory
                    self.edge_type.append("Inter2Sens")


                # Sensory to supervision (and vice versa)
                if self.supervised_learning and not no_sens2supervised:
                    for j in self.supervision_indices:
                        self.edge_index.append([i, j])  # Sensory to supervision
                        self.edge_type.append("Sens2Sup")

                        self.edge_index.append([j, i])  # Supervision to sensory
                        self.edge_type.append("Sup2Sens")


            # 2. Internal to all others (sensory, internal, supervision)

            # 2. Internal to all others (sensory, internal, supervision)
            for i in self.internal_indices:
                # Internal to internal
                for j in self.internal_indices:
                    if i != j or self_connection:  # Internal to internal (both ways)
                        self.edge_index.append([i, j])
                        self.edge_type.append("Inter2Inter")

                # Internal to sensory (and vice versa)
                for j in self.sensory_indices:
                    self.edge_index.append([i, j])  # Internal to sensory
                    self.edge_type.append("Inter2Sens")

                    self.edge_index.append([j, i])  # Sensory to internal
                    self.edge_type.append("Sens2Inter")

                # Internal to supervision (and vice versa)
                if self.supervised_learning:
                    for j in self.supervision_indices:
                        self.edge_index.append([i, j])  # Internal to supervision
                        self.edge_type.append("Inter2Sup")

                        self.edge_index.append([j, i])  # Supervision to internal
                        self.edge_type.append("Sup2Inter")

            # 3. Supervision to all others (sensory, internal, supervision)
            if self.supervised_learning:
                for i in self.supervision_indices:
                    # Supervision to sensory (and vice versa)
                    if not no_sens2supervised:
                        for j in self.sensory_indices:
                            self.edge_index.append([i, j])  # Supervision to sensory
                            self.edge_type.append("Sup2Sens")

                            self.edge_index.append([j, i])  # Sensory to supervision
                            self.edge_type.append("Sens2Sup")

                    # Supervision to internal (and vice versa)
                    for j in self.internal_indices:
                        self.edge_index.append([i, j])  # Supervision to internal
                        self.edge_type.append("Sup2Inter")

                        self.edge_index.append([j, i])  # Internal to supervision
                        self.edge_type.append("Inter2Sup")

                    # Supervision to supervision
                    for j in self.supervision_indices:
                        if i != j or self_connection:  # Supervision to supervision (both ways)
                            self.edge_index.append([i, j])
                            self.edge_type.append("Sup2Sup")

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



    def custom_two_branch(self, sensory_nodes, branch1_config, branch2_config, supervision_nodes):
        """
        Builds a custom graph with two branches:
        - Lower triangle branch (sensory -> branch1 -> supervision)
        - Upper triangle branch (supervision -> branch2 -> sensory)
        """
        print(f"Building custom two-branch graph...")

        # Initialize variables
        self.sensory_nodes = sensory_nodes
        self.branch1_config = branch1_config  # Config: (layers, clusters per layer, nodes per cluster)
        self.branch2_config = branch2_config
        self.supervision_nodes = supervision_nodes

        # Calculate total nodes
        self.branch1_internal_nodes = branch1_config[0] * branch1_config[1] * branch1_config[2]
        self.branch2_internal_nodes = branch2_config[0] * branch2_config[1] * branch2_config[2]
        self.total_nodes = sensory_nodes + self.branch1_internal_nodes + self.branch2_internal_nodes + supervision_nodes
        self.edge_index = []
        self.edge_type = []

        # Define node indices
        self.sensory_indices = list(range(self.sensory_nodes))
        self.branch1_internal_indices = list(range(sensory_nodes, sensory_nodes + self.branch1_internal_nodes))
        self.branch2_internal_indices = list(
            range(sensory_nodes + self.branch1_internal_nodes,
                sensory_nodes + self.branch1_internal_nodes + self.branch2_internal_nodes)
        )
        self.supervision_indices = list(
            range(sensory_nodes + self.branch1_internal_nodes + self.branch2_internal_nodes,
                sensory_nodes + self.branch1_internal_nodes + self.branch2_internal_nodes + supervision_nodes)
        )

        ## Step 1: Lower Triangle (Sensory -> Branch1 -> Supervision)
        branch1_layers = self.create_internal_layers(self.branch1_config, self.sensory_nodes, "branch1")
        print(f"Branch 1 layers: {branch1_layers}")
        self.connect_clusters(self.sensory_indices, branch1_layers[0], "Sens2Inter")  # Sensory -> Layer 1
        for i in range(len(branch1_layers) - 1):
            self.connect_clusters(branch1_layers[i], branch1_layers[i + 1], "Inter2Inter", branch_type="branch1")  # Layer i -> Layer i+1
        self.connect_clusters(branch1_layers[-1], self.supervision_indices, "Inter2Sup", branch_type="branch1")  # Last layer -> Supervision

        # Branch2: Supervision -> Branch2 -> Sensory
        branch2_start_idx = self.sensory_nodes + self.branch1_internal_nodes + len(self.supervision_indices)
        branch2_layers = self.create_internal_layers(self.branch2_config, branch2_start_idx, "branch2")
        print(f"Branch2 Layers: {branch2_layers}")
        self.connect_clusters(self.supervision_indices, branch2_layers[0], "Sup2Inter", branch_type="branch2")
        for i in range(len(branch2_layers) - 1):
            print(f"-----------------------Connecting layer {i} to layer {i + 1} in branch2...")
            self.connect_clusters(branch2_layers[i], branch2_layers[i + 1], "Inter2Inter", branch_type="branch2")
            print(f"Connections from layer {i} to layer {i + 1} completed.")
        self.connect_clusters(branch2_layers[-1], self.sensory_indices, "Inter2Sens", branch_type="branch2")

        # Fully connect supervision nodes
        self.connect_fully(self.supervision_indices, "Sup2Sup")

        # Step 3: Verify Inter2Inter Connections
        # self.verify_layer_connections(branch1_layers, "branch1")
        self.verify_layer_connections(branch2_layers, "branch2")

        print("Custom two-branch graph created successfully with Inter2Inter connections for both branches!")

    def verify_layer_connections(self, layers, branch_name):
        """
        Verifies that there are connections between consecutive layers within the given branch.
        """
        # Ensure edge_index is a tensor
        if not isinstance(self.edge_index, torch.Tensor):
            self.edge_index = torch.tensor(self.edge_index, dtype=torch.long).t().contiguous()

        for i in range(len(layers) - 1):
            layer1 = [node for cluster in layers[i] for node in cluster]
            layer2 = [node for cluster in layers[i + 1] for node in cluster]
            
            print(f"Verifying connections between layers {i} and {i + 1} in {branch_name}...")
            print(f"Layer {i} nodes: {layer1}")
            print(f"Layer {i + 1} nodes: {layer2}")

            found_connection = any(
                (edge[0] in layer1 and edge[1] in layer2) or (edge[0] in layer2 and edge[1] in layer1)
                for edge in self.edge_index.t().tolist()
            )
            
            if not found_connection:
                print(f"No connections found between layers {i} and {i + 1} in {branch_name}.")
            
            assert found_connection, f"Missing Inter2Inter connections between layers {i} and {i + 1} in {branch_name}."


    def create_internal_layers(self, config, start_idx, branch_name):
        """
        Creates layers of internal nodes organized into clusters.
        - Config: (num_layers, clusters_per_layer, nodes_per_cluster)
        """
        num_layers, clusters_per_layer, cluster_size = config
        layers = []
        current_idx = start_idx

        for layer_num in range(num_layers):
            layer = []
            for cluster_num in range(clusters_per_layer):
                cluster = list(range(current_idx, current_idx + cluster_size))
                if branch_name == "branch2":
                    print(f"Creating Cluster {cluster_num} in Layer {layer_num} of {branch_name} with nodes {cluster}")
                self.connect_fully(cluster, "Inter2Inter")  # Fully connect within cluster
                layer.append(cluster)
                current_idx += cluster_size
            layers.append(layer)

            # Extend the branch-specific indices
            if branch_name == "branch1":
                self.branch1_internal_indices.extend([node for cluster in layer for node in cluster])
            elif branch_name == "branch2":
                self.branch2_internal_indices.extend([node for cluster in layer for node in cluster])

        return layers

    def connect_fully(self, nodes, edge_type):
        """
        Fully connects all nodes in a given list. This method ensures that all nodes
        within the same cluster are bidirectionally connected (if applicable).
        """
        print(f"Fully connecting {len(nodes)} nodes with edge type {edge_type}...")
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):  # Avoid self-loops and duplicate edges
                self.edge_index.append([nodes[i], nodes[j]])
                self.edge_index.append([nodes[j], nodes[i]])  # Add bidirectional edge
                self.edge_type.append(edge_type)
                self.edge_type.append(edge_type)

        # Debug: Print connections for small clusters
        # if len(nodes) <= 10:
        #     print(f"Connected nodes: {nodes}")


    def connect_clusters(self, source_nodes, target_clusters, edge_type, branch_type=None, dense=False):
        """
        Connects clusters in dense or sparse configurations, ensuring directed connections based on the branch’s flow.
        """
        if isinstance(source_nodes[0], list):
            source_nodes = [node for cluster in source_nodes for node in cluster]

        if isinstance(target_clusters[0], int):
            target_nodes = target_clusters
        else:
            target_nodes = [node for cluster in target_clusters for node in cluster]


        for source in source_nodes:
            for target in target_nodes:
                # if branch_type:
                #     # Validate connection direction based on branch type
                #     if not self.is_valid_connection(source, target, branch_type):
                #         print(f"Invalid connection: {source} -> {target} in {branch_type}.")
                #         continue
                if dense or random.random() < 0.2:  # Dense or sparse connection
                    

                    if branch_type == "branch2":
                        # print("aa")
                        if edge_type == "Inter2Inter":
                            # print("bb")
                            self.edge_index.append([target, source])
                            self.edge_type.append(edge_type)
                    else:
                        self.edge_index.append([source, target])
                        self.edge_type.append(edge_type)

    def is_valid_connection(self, source, target, branch_type):
        """
        Validates if the connection direction is valid for the branch type.
        """
        if branch_type == "branch1":
            return source < target
        elif branch_type == "branch2":
            return source > target
        return True

 
    def get_edge_type_index(self, edge_type):
        """Maps edge type to an index."""
        edge_type_map = {
            "Sens2Sens": 0,
            "Sens2Inter": 1,
            "Sens2Sup": 2,
            "Inter2Sens": 3,
            "Inter2Inter": 4,
            "Inter2Sup": 5,
            "Sup2Sens": 6,
            "Sup2Inter": 7,
            "Sup2Sup": 8,
        }
        return edge_type_map[edge_type]


    def verify_edge_direction(self, source, target, direction):
        if direction == "Internal -> Supervision":
            return source < target
        elif direction == "Supervision -> Internal":
            return source > target
        elif direction == "Internal -> Internal":
            return source != target
        elif direction == "Internal -> Sensory":
            return source > target
        else:
            return True

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
