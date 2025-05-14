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

# import os
# os.environ['NX_CUGRAPH_AUTOCONFIG'] = 'True'


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

        "single_hidden_layer": {
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
                "num_communities": 20,      # Number of communities (50)
                "community_size": 50,       # Size of each community (40)
                "p_intra": 0.25,             # Probability of edges within the same community
                "p_inter": 0.1,             # Probability of edges between different communities
                "full_con_last_cluster_w_sup": True,
                "min_full_con_last_cluster_w_sup": 2,
                # "remove_sens_2_sens": False, 
                # "remove_sens_2_sup": False, 
                }
        },
        
        "sbm_two_branch_chain": {
            "params": {
                "branch1_config": (2, 20, 20),  # layers, clusters per layer, nodes per cluster
                "branch2_config": (2, 20, 20)
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
                "branch1_config": (3, 40, 10),  # 2 layers, 5 clusters per layer, 5 nodes per cluster for Branch 1
                "branch2_config": (3, 40, 10),  # 2 layers, 5 clusters per layer, 5 nodes per cluster for Branch 2
                # "branch1_config": (2, 5, 10),  # 2 layers, 5 clusters per layer, 5 nodes per cluster for Branch 1
                # "branch2_config": (2, 5, 10),  # 2 layers, 5 clusters per layer, 5 nodes per cluster for Branch 2
            }
        },


        # "custom_two_branch": {
        #     "params": {
        #         # Branch 1 configuration: Each tuple represents (cluster size, number_nodes_per_cluster) for each layer
        #         "branch1_config": [
        #             (100, 10),  # Layer 1: 
        #             (50, 10),   # Layer 2: 
        #             (20, 10),   # Layer 3: 
        #         ],
                
        #         # Branch 2 configuration: Same format as Branch 1
        #         "branch2_config": [
        #             (100, 10),  # Layer 1: 
        #             (50, 10),   # Layer 2: 
        #             (20, 10),   # Layer 3: 
        #         ],
        #     }
        # },

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
        self.edge_type_map = {"Sens2Sens": 0, "Sens2Inter": 1, "Sens2Sup": 2, "Inter2Sens": 3, "Inter2Inter": 4, "Inter2Sup": 5, "Sup2Sens": 6, "Sup2Inter": 7, "Sup2Sup": 8}


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
        # time to create the graph
        import time 
        start = time.time()


        print(f"Creating graph structure for {self.graph_type['name']}")
        if self.graph_type["name"] == "fully_connected":
            self.fully_connected(self_connection=False, 
                                 no_sens2sens=self.graph_params["remove_sens_2_sens"], 
                                 no_sens2supervised=self.graph_params["remove_sens_2_sup"])
        elif self.graph_type["name"] == "fully_connected_w_self":
            self.fully_connected(self_connection=True,
                                 no_sens2sens=self.graph_params["remove_sens_2_sens"], 
                                 no_sens2supervised=self.graph_params["remove_sens_2_sup"])
            
        elif self.graph_type["name"] == "single_hidden_layer":
            self.single_hidden_layer(
                                    # discriminative_hidden_layers=[200, 100, 50],
                                    # generative_hidden_layers=[50, 100, 200],
                                    discriminative_hidden_layers=self.graph_params["discriminative_hidden_layers"], 
                                    generative_hidden_layers=self.graph_params["generative_hidden_layers"], 
                                    no_sens2sens=self.graph_params["remove_sens_2_sens"], 
                                    no_sens2supervised=self.graph_params["remove_sens_2_sup"],
                                    bidirectional_hidden=False)
        
        elif self.graph_type["name"] == "barabasi":
            self.barabasi()
        elif self.graph_type["name"] == "stochastic_block":
            self.stochastic_block(no_sens2sens=self.graph_params["remove_sens_2_sens"], 
                                 no_sens2supervised=self.graph_params["remove_sens_2_sup"])
        elif self.graph_type["name"] == "sbm_two_branch_chain":
            self.sbm_two_branch_chain(
                self.graph_params["branch1_config"],
                self.graph_params["branch2_config"]
            )
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
        

        end_time = time.time()
        print(f"Graph creation time: {end_time - start:.2f} seconds")
        print("Created graph type:", self.graph_type["name"])

        print("Recalculating the number of vertices after graph creation")
        self.num_vertices = len(self.sensory_indices) + len(self.internal_indices)
        if self.supervised_learning:
            self.num_vertices += len(self.supervision_indices)  # Add supervision nodes to the vertex count
            
        # Convert edge_index to tensor
        self.edge_index = torch.tensor(self.edge_index, dtype=torch.long).t().contiguous()

        # Convert edge_type to tensor (use integers or map to string if preferred)

        self.edge_type = torch.tensor([self.edge_type_map[etype] for etype in self.edge_type], dtype=torch.long)

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



    def single_hidden_layer(self, discriminative_hidden_layers, generative_hidden_layers,
                                no_sens2sens=False, no_sens2supervised=False, bidirectional_hidden=False):
        """Creates a graph with shared internal nodes and layers for discriminative and generative paths."""
        # edge_index = []
        # edge_type = []

        # swap 
        discriminative_hidden_layers, generative_hidden_layers = generative_hidden_layers, discriminative_hidden_layers

        # Sensory nodes
        num_sensor_nodes = range(0, 784)

        # Define discriminative and generative layer ranges
        discriminative_layers = [
            range(
                784 + sum(discriminative_hidden_layers[:i]),
                784 + sum(discriminative_hidden_layers[:i + 1])
            )
            for i in range(len(discriminative_hidden_layers))
        ]

        generative_layers = [
            range(
                784 + sum(discriminative_hidden_layers) + sum(generative_hidden_layers[:i]),
                784 + sum(discriminative_hidden_layers) + sum(generative_hidden_layers[:i + 1])
            )
            for i in range(len(generative_hidden_layers))
        ]

        # Supervision nodes
        supervision_indices = range(
            784 + sum(discriminative_hidden_layers) + sum(generative_hidden_layers),
            784 + sum(discriminative_hidden_layers) + sum(generative_hidden_layers) + 10
        )

        # ------------ Generative -----------------
        ## Connect sensory -> first generative hidden layer
        for sensory_node in num_sensor_nodes:
            for hidden_node in generative_layers[0]:
                self.edge_index.append([hidden_node, sensory_node])  # Direction: hidden -> sensory
                self.edge_type.append("Inter2Sens")

        # Connect generative layers sequentially (Reverse direction: hidden_n+1 -> hidden_n)
        for i in range(len(generative_layers) - 1):
            for idx, node_from in enumerate(generative_layers[i]):
                # if idx % 2 == 0:  # Keep half the connections to ensure direction
                for node_to in generative_layers[i + 1]:
                    self.edge_index.append([node_to, node_from])
                    self.edge_type.append("Inter2Inter")

        # Connect last generative layer -> supervision
        for hidden_node in generative_layers[-1]:
            for sup_node in supervision_indices:
                self.edge_index.append([sup_node, hidden_node])  # Direction: supervision -> last generative hidden
                self.edge_type.append("Sup2Inter")



        # ------------ Generative -----------------
        # Connect first generative hidden layer -> sensory (parent to child)
        # for hidden_node in generative_layers[0]:
        #     for sensory_node in num_sensor_nodes:
        #         self.edge_index.append([hidden_node, sensory_node])  # hidden predicts sensory
        #         self.edge_type.append("Inter2Sens")

        # # Connect generative layers sequentially (Top-down: layer_n -> layer_n+1)
        # for i in range(len(generative_layers) - 1):
        #     for node_from in generative_layers[i]:
        #         for node_to in generative_layers[i + 1]:
        #             self.edge_index.append([node_from, node_to])  # parent to child
        #             self.edge_type.append("Inter2Inter")

        # # Connect supervision -> top generative layer (to keep consistent with backward correction)
        # for sup_node in supervision_indices:
        #     for hidden_node in generative_layers[-1]:
        #         self.edge_index.append([sup_node, hidden_node])  # supervision sends signal to generative layer
        #         self.edge_type.append("Sup2Inter")




        # ------------ Discriminative -----------------
        # Connect sensory -> first discriminative hidden layer
        for sensory_node in num_sensor_nodes:
            for hidden_node in discriminative_layers[0]:
                self.edge_index.append([sensory_node, hidden_node])  # Direction: sensory -> hidden
                self.edge_type.append("Sens2Inter")

        # Connect discriminative layers sequentially (Forward direction: hidden_n -> hidden_n+1)
        for i in range(len(discriminative_layers) - 1):
            for idx, node_from in enumerate(discriminative_layers[i]):
                # if idx % 2 == 0:  # Keep half the connections to ensure direction
                for node_to in discriminative_layers[i + 1]:
                    self.edge_index.append([node_from, node_to])
                    self.edge_type.append("Inter2Inter")


        # Connect last discriminative layer -> supervision
        for hidden_node in discriminative_layers[-1]:
            for sup_node in supervision_indices:
                self.edge_index.append([hidden_node, sup_node])  # Direction: last hidden -> supervision
                self.edge_type.append("Inter2Sup")

        # Add Sens2Sens connections if enabled
        if not no_sens2sens:
            for i in num_sensor_nodes:
                for j in num_sensor_nodes:
                    if i != j:
                        self.edge_index.append([i, j])
                        self.edge_type.append("Sens2Sens")

        # Add Sens2Sup connections if enabled
        if not no_sens2supervised:
            for sensory_node in num_sensor_nodes:
                for sup_node in supervision_indices:
                    self.edge_index.append([sensory_node, sup_node])
                    self.edge_type.append("Sens2Sup")

        print("Custom graph with shared internal nodes and multi-layer paths created.")

        # Convert to torch tensors
        # edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

        # Total number of nodes
        num_sensory_nodes = 784
        num_discriminative_nodes = sum(discriminative_hidden_layers)
        num_generative_nodes = sum(generative_hidden_layers)
        num_supervision_nodes = 10

        N = num_sensory_nodes + num_discriminative_nodes + num_generative_nodes + num_supervision_nodes


        # return edge_index, N

        

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

    def sbm_two_branch_chain(self, branch1_config=None, branch2_config=None):
        """
        Hardcoded version for correctness testing:
        Branch 1: Sensory → Internal1 (fully connected) → Supervision
        Branch 2: Supervision → Internal2 (fully connected) → Sensory
        """
        assert self.supervised_learning, "Supervision nodes are required for this graph type."

        self.edge_index = []
        self.edge_type = []

        sensory = list(range(self.SENSORY_NODES))
        supervision = list(self.supervision_indices)

        # === Internal Clusters ===
        internal1_start = self.SENSORY_NODES
        internal2_start = internal1_start + 50  # 50 nodes per cluster

        internal1 = list(range(internal1_start, internal1_start + 50))
        internal2 = list(range(internal2_start, internal2_start + 50))

        self.branch1_internal_indices = internal1
        self.branch2_internal_indices = internal2
        self.internal_indices = internal1 + internal2
        self.NUM_INTERNAL_NODES = len(self.internal_indices)

        # === Branch 1: Sensory → Internal1 → Supervision ===
        for s in sensory:
            for i in internal1:
                self.edge_index.append([s, i])
                self.edge_type.append("Sens2Inter")

        for i in internal1:
            for j in internal1:
                if i != j:
                    self.edge_index.append([i, j])
                    self.edge_type.append("Inter2Inter")

        for i in internal1:
            for sup in supervision:
                self.edge_index.append([i, sup])
                self.edge_type.append("Inter2Sup")

        # === Branch 2: Supervision → Internal2 → Sensory ===
        for sup in supervision:
            for i in internal2:
                self.edge_index.append([sup, i])
                self.edge_type.append("Sup2Inter")

        for i in internal2:
            for j in internal2:
                if i != j:
                    self.edge_index.append([i, j])
                    self.edge_type.append("Inter2Inter")

        for i in internal2:
            for s in sensory:
                self.edge_index.append([i, s])
                self.edge_type.append("Inter2Sens")

        print("✅ Hardcoded sbm_two_branch_chain constructed successfully.")


    def ensure_all_nodes_connected(self, total_nodes):
        used = set()
        for src, dst in self.edge_index:
            used.add(src)
            used.add(dst)
        all_nodes = set(range(total_nodes))
        missing = all_nodes - used
        for m in missing:
            self.edge_index.append([m, m])
            self.edge_type.append("Sup2Sup" if m in self.supervision_indices else "Inter2Inter")
        if missing:
            print(f"⚠️ Added self-loops for {len(missing)} disconnected nodes: {sorted(missing)}")
        else:
            print("✅ All nodes connected.")


    def fully_connected(self, self_connection, no_sens2sens=False, no_sens2supervised=False):
        # Initialize a set to track seen edges
        seen_edges = set()

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

        def add_edge(src, dest, edge_type):
            if (src, dest) not in seen_edges:
                self.edge_index.append([src, dest])
                self.edge_type.append(edge_type)
                seen_edges.add((src, dest))

        # Create fully connected edges with edge types
        if not no_sens2sens and not no_sens2supervised:
            for i in self.num_all_nodes:
                for j in self.num_all_nodes:
                    if i != j or self_connection:
                        add_edge(i, j, get_edge_type(i, j))
        else:
            print(f"Doing from scratch; Creating fully connected directed graph with self connections: {self_connection}")

            for i in self.sensory_indices:
                if not no_sens2sens:
                    for j in self.sensory_indices:
                        if i != j or self_connection:
                            add_edge(i, j, "Sens2Sens")

                for j in self.internal_indices:
                    add_edge(i, j, "Sens2Inter")
                    add_edge(j, i, "Inter2Sens")

                if self.supervised_learning and not no_sens2supervised:
                    for j in self.supervision_indices:
                        add_edge(i, j, "Sens2Sup")
                        add_edge(j, i, "Sup2Sens")

            for i in self.internal_indices:
                for j in self.internal_indices:
                    if i != j or self_connection:
                        add_edge(i, j, "Inter2Inter")

                for j in self.sensory_indices:
                    add_edge(i, j, "Inter2Sens")
                    add_edge(j, i, "Sens2Inter")

                if self.supervised_learning:
                    for j in self.supervision_indices:
                        add_edge(i, j, "Inter2Sup")
                        add_edge(j, i, "Sup2Inter")

            if self.supervised_learning:
                for i in self.supervision_indices:
                    if not no_sens2supervised:
                        for j in self.sensory_indices:
                            add_edge(i, j, "Sup2Sens")
                            add_edge(j, i, "Sens2Sup")

                    for j in self.internal_indices:
                        add_edge(i, j, "Sup2Inter")
                        add_edge(j, i, "Inter2Sup")

                    for j in self.supervision_indices:
                        if i != j or self_connection:
                            add_edge(i, j, "Sup2Sup")

        print("Fully connected graph creation complete.")



    def barabasi(self):
        num_nodes = len(self.num_all_nodes)
        m = self.graph_params.get("m", 5)  # Number of edges to attach from a new node to existing nodes
        G = nx.barabasi_albert_graph(num_nodes, m, seed=self.seed)
        self.edge_index = [[u, v] for u, v in G.edges()]
        print(f"Creating Barabási-Albert graph with {num_nodes} nodes and {m} edges to attach per new node")


    def stochastic_block(self, no_sens2sens, no_sens2supervised):
        import networkx as nx
        import importlib

        # Check if GPU backend (nx-cugraph) is available
        gpu_backend = False
        try:
            nxcg = importlib.import_module("nx_cugraph")
            cugraph = importlib.import_module("cugraph")
            gpu_backend = True
            print("✅ Using GPU backend with nx-cugraph.")
        except ImportError:
            print("⚠️ nx-cugraph not found, using default NetworkX (CPU).")

        # Load params
        num_communities = self.graph_params.get("num_communities", 40)
        community_size = self.graph_params.get("community_size", 50)
        p_intra = self.graph_params.get("p_intra", 0.5)
        p_inter = self.graph_params.get("p_inter", 0.1)

        assert (num_communities * community_size) == self.NUM_INTERNAL_NODES, "must be equal"

        sizes = [community_size] * num_communities
        self.INTERNAL = range(self.SENSORY_NODES, self.SENSORY_NODES + sum(sizes))
        self.num_internal_nodes = sum(self.INTERNAL)

        sizes.insert(0, self.SENSORY_NODES)
        sizes.append(10)
        community_sizes = sizes
        num_communities = len(community_sizes)

        # Build connectivity matrix
        p = np.full((num_communities, num_communities), p_inter)
        np.fill_diagonal(p, p_intra)

        if no_sens2sens:
            p[0, 0] = 0

        for i in range(1, num_communities):
            p[0, i] = 0.1
            p[i, 0] = 0.1

        if no_sens2supervised:
            p[0, -1] = 0
            p[-1, 0] = 0

        # For the last z internal clusters, increase supervision connection strength
        # z = 10
        fully_connect_last_cluster = self.graph_params.get("full_con_last_cluster_w_sup", False)
        z = self.graph_params.get("min_full_con_last_cluster_w_sup", 5)
        
        if fully_connect_last_cluster and self.supervised_learning:
            supervision_cluster_idx = num_communities - 1  # Supervision is the last cluster
            first_internal_cluster_idx = 1  # Internal clusters start after sensory
            last_internal_cluster_idx = num_communities - 2  # Before supervision

            # Clamp z if there are fewer than z internal clusters
            z = min(z, last_internal_cluster_idx - first_internal_cluster_idx + 1)

            # Apply stronger connection from last z internal clusters to supervision
            for i in range(last_internal_cluster_idx, last_internal_cluster_idx - z, -1):
                p[i, supervision_cluster_idx] = 0.7
                p[supervision_cluster_idx, i] = 0.7


        # Use GPU-based graph construction if possible
        if gpu_backend:
            # Use nx_cugraph's stochastic block model
            G = nxcg.stochastic_block_model(sizes, p, directed=True, seed=self.seed)
            print("✅ Created stochastic_block_model using GPU (nx-cugraph).")
        else:
            G = nx.stochastic_block_model(sizes, p, directed=True, seed=self.seed)
            print("ℹ️ Created stochastic_block_model using CPU (networkx).")

        # Convert to PyG format
        from torch_geometric.utils import from_networkx
        data = from_networkx(G)

        self.edge_index = []
        self.edge_type = []

        for u, v in G.edges():
            src_block = G.nodes[u]['block']
            dest_block = G.nodes[v]['block']

            if src_block == 0 and dest_block == 0:
                etype = "Sens2Sens"
            elif src_block == 0 and 1 <= dest_block <= num_communities - 2:
                etype = "Sens2Inter"
            elif src_block == 0 and dest_block == num_communities - 1:
                etype = "Sens2Sup"
            elif 1 <= src_block <= num_communities - 2 and dest_block == 0:
                etype = "Inter2Sens"
            elif 1 <= src_block <= num_communities - 2 and dest_block == num_communities - 1:
                etype = "Inter2Sup"
            elif src_block == num_communities - 1 and dest_block == 0:
                etype = "Sup2Sens"
            elif src_block == num_communities - 1 and 1 <= dest_block <= num_communities - 2:
                etype = "Sup2Inter"
            elif src_block == num_communities - 1 and dest_block == num_communities - 1:
                etype = "Sup2Sup"
            else:
                etype = "Inter2Inter"

            self.edge_index.append([u, v])
            self.edge_type.append(etype)

        print("✅ Finished assigning edge types for SBM.")



    def custom_two_branch(self, sensory_nodes, branch1_config, branch2_config, supervision_nodes):
        self.sensory_nodes = sensory_nodes
        self.branch1_config = branch1_config  # Branch 1 config: (layers, clusters per layer, nodes per cluster)
        self.branch2_config = branch2_config  # Branch 2 config: (layers, clusters per layer, nodes per cluster)
        self.supervision_nodes = supervision_nodes
        
        # Calculate total nodes based on branch configurations
        self.branch1_internal_nodes = branch1_config[0] * branch1_config[1] * branch1_config[2]
        self.branch2_internal_nodes = branch2_config[0] * branch2_config[1] * branch2_config[2]
        self.total_nodes = sensory_nodes + self.branch1_internal_nodes + self.branch2_internal_nodes + supervision_nodes
        self.edge_index = []
        self.edge_type = []
        
        # Define node lists for easy access
        self.sensory_indices = list(range(self.sensory_nodes))
        
        # Define internal node indices for each branch
        self.branch1_internal_indices = list(range(sensory_nodes, sensory_nodes + self.branch1_internal_nodes))
        self.branch2_internal_indices = list(range(sensory_nodes + self.branch1_internal_nodes, sensory_nodes + self.branch1_internal_nodes + self.branch2_internal_nodes))
        
        # Define supervision indices to start immediately after the last internal node in Branch 2
        # self.supervision_indices = list(range(sensory_nodes + self.branch1_internal_nodes + self.branch2_internal_nodes, self.total_nodes))
        
        self.supervision_indices = list(range(
            self.sensory_nodes + self.branch1_internal_nodes + self.branch2_internal_nodes,
            self.sensory_nodes + self.branch1_internal_nodes + self.branch2_internal_nodes + self.supervision_nodes
        ))
        
        # Connect sensory nodes fully within themselves
        self.connect_fully(self.sensory_indices, edge_type="Sens2Sens")

        # Build both branches based on configurations
        self.build_branch_2()
        self.build_branch_1()

        # Fully connect supervision (one-hot) nodes
        self.connect_fully(self.supervision_indices, edge_type="Sup2Sup")

        # Convert edge_index to tensor after graph construction
        # self.edge_index = torch.tensor(self.edge_index, dtype=torch.long).t().contiguous()

    def build_branch_1(self):
        """Builds Branch 1 based on its configuration."""
        layers, clusters_per_layer, nodes_per_cluster = self.branch1_config
        internal_layers = self.create_internal_layers(layers, clusters_per_layer, nodes_per_cluster, start_idx=self.sensory_nodes, branch="branch1")
        
        # Step 1: Sensory -> Internal Layer 1 (sparse)
        self.connect_clusters(self.sensory_indices, internal_layers[0], edge_type="Sens2Inter", branch_type="branch1", dense=False)
        
        # Connect internal layers within Branch 1
        for i in range(len(internal_layers) - 1):
            self.connect_clusters(internal_layers[i], internal_layers[i + 1], edge_type="Inter2Inter", branch_type="branch1", dense=False)
        
        # Step n: Internal Layer n -> Supervision (sparse)
        self.connect_clusters(internal_layers[-1], self.supervision_indices, edge_type="Inter2Sup", branch_type="branch1", dense=False,  verify_direction="Internal -> Supervision")

    def build_branch_2(self): 
        """Builds Branch 2 based on its configuration."""
        layers, clusters_per_layer, nodes_per_cluster = self.branch2_config
        start_idx = self.sensory_nodes + self.branch1_internal_nodes  # Offset by the internal nodes used in Branch 1
        internal_layers = self.create_internal_layers(layers, clusters_per_layer, nodes_per_cluster, start_idx=start_idx, branch="branch2")
        
        # Step 1: Sensory -> Internal Layer 1 (reverse direction for branch2)
        self.connect_clusters(self.sensory_indices, internal_layers[0], edge_type="Sens2Inter", branch_type="branch2", dense=False)
        
        # Step 2: Connect internal layers within Branch 2
        for i in range(len(internal_layers) - 1):
            self.connect_clusters(internal_layers[i], internal_layers[i + 1], edge_type="Inter2Inter", branch_type="branch2", dense=False)
        
        # Step 3: Internal Layer n -> Supervision (sparse, reverse direction for branch2)
        self.connect_clusters(internal_layers[-1], self.supervision_indices, edge_type="Inter2Sup", branch_type="branch2", dense=False)
    
    def create_internal_layers(self, num_layers, clusters_per_layer, cluster_size, start_idx, branch):
        """Creates layers of internal nodes organized into clusters for each branch."""
        layers = []
        for _ in range(num_layers):
            layer_clusters = []
            for _ in range(clusters_per_layer):
                cluster = list(range(start_idx, start_idx + cluster_size))
                layer_clusters.append(cluster)
                # Fully connect nodes within each internal cluster
                self.connect_fully(cluster, edge_type="Inter2Inter")
                start_idx += cluster_size
            layers.append(layer_clusters)

            # Add the cluster indices to the branch-specific list
            if branch == "branch1":
                self.branch1_internal_indices.extend([node for cluster in layer_clusters for node in cluster])
            elif branch == "branch2":
                self.branch2_internal_indices.extend([node for cluster in layer_clusters for node in cluster])

        return layers

    def connect_fully(self, nodes, edge_type):
        """Connects all nodes in the given list to each other (fully connected)."""
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                self.edge_index.append([nodes[i], nodes[j]])
                self.edge_index.append([nodes[j], nodes[i]])
                self.edge_type.append(edge_type)
                self.edge_type.append(edge_type)
 

    def connect_clusters(self, source_nodes, target_clusters, edge_type, branch_type=None, dense=False, verify_direction=None):
        """
        Connects clusters in dense or sparse configurations, ensuring directed connections based on the branch type and flow.
        Adds reverse connections for backward propagation and supports branch-specific handling.
        """
        # Flatten source_nodes if it contains clusters
        if isinstance(source_nodes, range) or isinstance(source_nodes[0], int):
            source_nodes = list(source_nodes)
        else:
            source_nodes = [node for cluster in source_nodes for node in cluster]

        # If target_clusters is a list of individual nodes, wrap it in another list to make it consistent
        if isinstance(target_clusters, range) or isinstance(target_clusters[0], int):
            target_clusters = [list(target_clusters)]

        # Handle connections based on density and branch type
        for source in source_nodes:
            target_list = [node for cluster in target_clusters for node in cluster]  # Flatten target clusters

            if dense:
                for target in target_list:
                    if source != target:
                        if branch_type == "branch2":
                            # Reverse direction for branch2
                            self.edge_index.append([target, source])
                            self.edge_type.append(edge_type)
                            # Add reverse edge (forward for branch2)
                            self.edge_index.append([source, target])
                            self.edge_type.append(edge_type)
                        else:
                            # Default forward direction for branch1
                            self.edge_index.append([source, target])
                            self.edge_type.append(edge_type)
                            # Add reverse edge
                            self.edge_index.append([target, source])
                            self.edge_type.append(edge_type)
            else:
                # Sparse connections (e.g., 20% of target nodes)
                sparse_targets = np.random.choice(target_list, size=int(0.2 * len(target_list)), replace=False)
                for target in sparse_targets:
                    if branch_type == "branch2":
                        # Reverse direction for branch2
                        self.edge_index.append([target, source])
                        self.edge_type.append(edge_type)
                        # Add reverse edge (forward for branch2)
                        self.edge_index.append([source, target])
                        self.edge_type.append(edge_type)
                    else:
                        # Default forward direction for branch1
                        self.edge_index.append([source, target])
                        self.edge_type.append(edge_type)
                        # Add reverse edge
                        self.edge_index.append([target, source])
                        self.edge_type.append(edge_type)



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