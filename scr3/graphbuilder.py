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
import wandb
import matplotlib.pyplot as plt
import math

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
                "add_residual": True  # <--- add this!

                # "remove_sens_2_sens": True, 
                # "remove_sens_2_sup": False, 
            }
        }, 

        
        "single_hidden_layer_clusters": {
            "params": {
                "discriminative_hidden_layers": [100],
                "generative_hidden_layers": [0],
                "num_clusters_per_layer": 10,
                "remove_sens_2_sens": True,
                "remove_sens_2_sup": True,
                "p_intra": 0.5,
                "p_inter": 0.5,
            }
        },


        "stochastic_block_hierarchy": {
            "params": {
                            # "remove_sens_2_sens": True, 
                            # "remove_sens_2_sup": False, 
                        }
        },

        
        # "stochastic_block": {
        #     "params": {
        #         "num_communities": 5,      # Number of communities (50)
        #         "community_size": 100,       # Size of each community (40)
        #         "p_intra": 0.25,             # Probability of edges within the same community
        #         "p_inter": 0.1,             # Probability of edges between different communities
        #         "full_con_last_cluster_w_sup": False,
        #         "min_full_con_last_cluster_w_sup": 0,
        #         # "remove_sens_2_sens": False, 
        #         # "remove_sens_2_sup": False, 
        #         }
        # },

          "stochastic_block": {
            "params": {
                "num_communities": 15,      # Number of communities (50)
                "community_size": 100,       # Size of each community (40)P
                "p_intra": 0.25,             # Probability of edges within the same community
                "p_inter": 0.1,             # Probability of edges between different communities
                "full_con_last_cluster_w_sup": True,
                "min_full_con_last_cluster_w_sup": 3,
                # "remove_sens_2_sens": False, 
                # "remove_sens_2_sup": False, 
                }
        },

            "dual_branch_sbm": {
                "params": {
                    "p_intra": 0.0,     # Probability of edges within the same community
                    "p_inter": 1,   # Probability of edges between different communities
                    "remove_sens_2_sens": True,  # Placeholder for removing sensory to sensory connections
                    "remove_sens_2_sup": True,   # Placeholder for removing sensory to supervision connections
                    "discriminative_layer": [
                        {"num_clusters": 49, "nodes_per_cluster": 16, "p_to_next": 1.0},
                        {"num_clusters": 40, "nodes_per_cluster": 1, "p_to_next": 1.0},
                        # {"num_clusters": 100, "nodes_per_cluster": 16, "p_to_next": 1.0},
                        {"num_clusters": 10, "nodes_per_cluster": 10, "p_to_next": 1.0}
                    ],
                    "generative_layer": [
                        {"num_clusters": 10, "nodes_per_cluster": 10, "p_to_next": 0.0},
                        {"num_clusters": 49, "nodes_per_cluster": 16, "p_to_next": 0.0}
                    ]
                    # "discriminative_layer": [
                    #     {"num_clusters": 49, "nodes_per_cluster": 16, "p_to_next": 0},
                    #     {"num_clusters": 10, "nodes_per_cluster": 10, "p_to_next": 0}
                    # ],
                    # "generative_layer": [
                    #     {"num_clusters": 10, "nodes_per_cluster": 10, "p_to_next": 1.0},
                    #     {"num_clusters": 1, "nodes_per_cluster": 16, "p_to_next": 1.0},
                    #     # {"num_clusters": 4, "nodes_per_cluster": 8, "p_to_next": 1.0},
                    #     {"num_clusters": 49, "nodes_per_cluster": 16, "p_to_next": 1.0}
                    # ]

                }
            }


        ,
        
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


        "sbm_interleave": {
            "params": {
                "patch_size": 4,
                "overlap_stride": 4,      # 13×13 patch grid
                "h1_filters": 4, "h1_per_filter": 50,
                "h2_filters": 4, "h2_per_filter": 50,
                "filter_connect": "all_to_all",   # all_to_all, matched
                "remove_sens_2_sens": True,
                "remove_sens_2_sup": True,
                "connect_h2_to_sup_dense": True,
                "connect_sup_to_h2_dense": True  # new, default off to preserve old behavior
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
                                    bidirectional_hidden=False,
                                    # add_residual=True,
                                    add_residual=False,
                                    # add_residual= self.graph_params.get("add_residual", False)  # Use the parameter from graph_params if available
                                            )

        elif self.graph_type["name"] == "single_hidden_layer_clusters":
            self.single_hidden_layer_clusters(
                discriminative_hidden_layers=self.graph_params["discriminative_hidden_layers"],
                generative_hidden_layers=self.graph_params["generative_hidden_layers"],
                num_clusters_per_layer=self.graph_params.get("num_clusters_per_layer", 10),
                no_sens2sens=self.graph_params["remove_sens_2_sens"],
                no_sens2supervised=self.graph_params["remove_sens_2_sup"],
                p_intra=self.graph_params.get("p_intra", 0.0),
                p_inter=self.graph_params.get("p_inter", 1.0),
            )

        elif self.graph_type["name"] == "sbm_interleave":
            self.sbm_interleave(
                patch_size=self.graph_params.get("patch_size", 4),
                super_block=self.graph_params.get("super_block", 2),

                # NEW (overlap + filters)
                overlap_stride=self.graph_params.get("overlap_stride", None),   # None → no overlap (stride=patch_size)
                h1_filters=self.graph_params.get("h1_filters", 1),
                h1_per_filter=self.graph_params.get("h1_per_filter",
                                                    self.graph_params.get("h1_per_cluster", 16)),  # backward-compat
                h2_filters=self.graph_params.get("h2_filters", 1),
                h2_per_filter=self.graph_params.get("h2_per_filter",
                                                    self.graph_params.get("h2_per_cluster", 16)),  # backward-compat
                filter_connect=self.graph_params.get("filter_connect", "matched"),  # or "all_to_all"

                no_sens2sens=self.graph_params.get("remove_sens_2_sens", True),
                no_sens2supervised=self.graph_params.get("remove_sens_2_sup", True),
                connect_h2_to_sup_dense=self.graph_params.get("connect_h2_to_sup_dense", True),
                connect_sup_to_h2_dense=self.graph_params.get("connect_sup_to_h2_dense", True),  # <-- add

            )


        elif self.graph_type["name"] == "barabasi":
            self.barabasi()
        elif self.graph_type["name"] == "stochastic_block":
            self.stochastic_block(no_sens2sens=self.graph_params["remove_sens_2_sens"], 
                                 no_sens2supervised=self.graph_params["remove_sens_2_sup"])
        elif self.graph_type["name"] == "dual_branch_sbm":
            self.generate_dual_sbm_graph(self.graph_params, image_size=28, patch_size=4,
                                                 remove_sens_2_sens=self.graph_params["remove_sens_2_sens"],
                                                 remove_sens_2_sup=self.graph_params["remove_sens_2_sup"])

                                   
                                    
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


    def log_graph_to_wandb(self):

        # ... after ensuring self.edge_index is [2, E] and self.num_vertices is set
        try:
            self.log_graph_metrics_to_wandb()
            print("✅ Graph metrics logged to wandb.")
        except Exception as e:
            print(f"⚠️ log_graph_metrics_to_wandb failed: {e}")

        # (Optional) Keep your detailed hop plots as well:
        try:
            self.log_hop_distribution_to_wandb()
            print("✅ Hop distribution logged to wandb.")
        except Exception as e:
            print(f"⚠️ log_hop_distribution_to_wandb failed: {e}")


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
    
        # ===== Graph metrics → wandb =====
    def _nx_digraph(self):
        G = nx.DiGraph()
        G.add_nodes_from(range(self.num_vertices))
        ei = self.edge_index
        if isinstance(ei, torch.Tensor):
            if ei.shape[0] == 2:  # [2, E]
                edges = ei.t().tolist()
            else:                 # [E, 2]
                edges = ei.tolist()
        else:
            edges = ei
        G.add_edges_from(edges)
        return G

    @staticmethod
    def _bincount_hist(deg_list):
        if len(deg_list) == 0:
            return [0]
        arr = np.asarray(deg_list, dtype=int)
        return np.bincount(arr, minlength=int(arr.max()) + 1).tolist()

    def log_graph_metrics_to_wandb(self, log_hop_hist=True, spectral_radius_max_n=3000):
        G = self._nx_digraph()
        N = G.number_of_nodes()
        M = G.number_of_edges()
        denom_pairs = N * (N - 1)

        # Degrees
        in_deg  = [d for _, d in G.in_degree()]
        out_deg = [d for _, d in G.out_degree()]

        # Clustering (directed, Fagiolo)
        try:
            dc = nx.directed_clustering(G)
            clust_mean = float(np.mean(list(dc.values()))) if len(dc) else 0.0
        except Exception:
            clust_mean = None

        # Shortest-path matrix (once), reachability & global efficiency
        spl = dict(nx.all_pairs_shortest_path_length(G))
        reachable_pairs = sum(len(d) - 1 for d in spl.values()) if N > 0 else 0
        reach_ratio = (reachable_pairs / denom_pairs) if denom_pairs else 0.0

        # Global efficiency (Latora–Marchiori, directed distances; unreachable→0)
        total_inv = 0.0
        if denom_pairs:
            for u in G:
                Lu = spl.get(u, {})
                for v in G:
                    if u == v:
                        continue
                    d = Lu.get(v)
                    if d is not None and d > 0:
                        total_inv += 1.0 / d
        global_eff = (total_inv / denom_pairs) if denom_pairs else 0.0

        # SCC structure & diameter on largest SCC
        sccs = list(nx.strongly_connected_components(G))
        num_scc = len(sccs)
        largest_scc_size = max((len(s) for s in sccs), default=0)
        diam_largest_scc = None
        if largest_scc_size > 1:
            try:
                H = G.subgraph(max(sccs, key=len)).copy()
                diam_largest_scc = nx.diameter(H.to_undirected())
            except Exception:
                diam_largest_scc = None

        # Reciprocity
        try:
            reciprocity = nx.reciprocity(G) if M else 0.0
        except Exception:
            reciprocity = None

        # Storage-capacity proxies
        Gu = G.to_undirected(as_view=False)
        cyclomatic = Gu.number_of_edges() - Gu.number_of_nodes() + nx.number_connected_components(Gu)
        # Approx. minimum feedback arc set size
        try:
            from networkx.algorithms import approximation as approx
            fas_size = len(approx.minimum_feedback_arc_set(G))
        except Exception:
            fas_size = None

        # Spectral radius of symmetrized adjacency (power iteration, skip if huge)
        rho_sym = None
        if N and N <= spectral_radius_max_n:
            try:
                A = nx.to_numpy_array(G, nodelist=range(N), dtype=float)
                As = (A + A.T) / 2.0
                x = np.random.rand(N)
                x /= np.linalg.norm(x) + 1e-12
                last = 0.0
                for _ in range(100):
                    y = As @ x
                    normy = np.linalg.norm(y)
                    if normy == 0:
                        last = 0.0
                        break
                    x_new = y / normy
                    if np.linalg.norm(x_new - x) < 1e-6:
                        last = normy
                        break
                    x = x_new
                    last = normy
                rho_sym = float(last)
            except Exception:
                rho_sym = None

        # ---- Sensory → Supervision hop stats ----
        hop_min_s2sup = None
        hop_mean_s2sup = None
        hop_samples_s2sup = []
        if getattr(self, "supervised_learning", False) and hasattr(self, "supervision_indices"):
            sens_nodes = list(self.sensory_indices)
            sup_nodes  = list(self.supervision_indices)
            if len(sens_nodes) and len(sup_nodes):
                for s in sens_nodes:
                    ds = spl.get(s, {})
                    for t in sup_nodes:
                        d = ds.get(t)
                        if d is not None and d > 0:
                            hop_samples_s2sup.append(d)
                if hop_samples_s2sup:
                    hop_min_s2sup  = int(np.min(hop_samples_s2sup))
                    hop_mean_s2sup = float(np.mean(hop_samples_s2sup))

        # ---- Log to wandb ----
        # ----- TABLE: single row with static metrics -----
        row = {
            "graph_name": self.graph_type["name"],
            "seed": self.seed,
            "n_nodes": N,
            "n_edges": M,
            "density": (M / denom_pairs) if denom_pairs else 0.0,
            "reciprocity": reciprocity,
            "clustering_mean_fagiolo": clust_mean,
            "reachability_ratio": reach_ratio,
            "global_efficiency": global_eff,
            "num_SCC": num_scc,
            "largest_SCC_size": largest_scc_size,
            "diameter_largest_SCC": diam_largest_scc,
            "cyclomatic_number": cyclomatic,
            "feedback_arc_set_size_approx": fas_size,
            "spectral_radius_sym": rho_sym,
            "mean_in_degree": float(np.mean(in_deg)) if N else 0.0,
            "mean_out_degree": float(np.mean(out_deg)) if N else 0.0,
            "sens_to_sup_hop_min": hop_min_s2sup,
            "sens_to_sup_hop_mean": hop_mean_s2sup,
            "sens_to_sup_hop_count_pairs": len(hop_samples_s2sup),
        }
        cols = list(row.keys())
        table = wandb.Table(columns=cols)
        table.add_data(*[row[c] for c in cols])
        wandb.log({"graph/metrics_table": table}, commit=True)

        # Also mirror key scalars to the summary for quick filtering
        try:
            wandb.run.summary.update({f"graph/{k}": v for k, v in row.items() if isinstance(v, (int, float))})
        except Exception:
            pass

        # ----- PLOTS: degree histograms (as images, not wandb.Histogram) -----
        def plot_hist(vals, title, xlab="Value", ylab="Count"):
            if not len(vals): return None
            fig, ax = plt.subplots()
            bins = np.arange(min(vals), max(vals) + 2) - 0.5
            ax.hist(vals, bins=bins, edgecolor="black")
            ax.set_title(title)
            ax.set_xlabel(xlab)
            ax.set_ylabel(ylab)
            ax.grid(True, alpha=0.3)
            img = wandb.Image(fig)
            plt.close(fig)
            return img

        in_img = plot_hist(in_deg,  "In-degree Histogram")
        out_img= plot_hist(out_deg, "Out-degree Histogram")

        payload = {}
        if in_img is not None:
            payload["graph/in_degree_hist_plot"] = in_img
        if out_img is not None:
            payload["graph/out_degree_hist_plot"] = out_img

        if payload:
            wandb.log(payload, commit=True)

        print("✅ Graph metrics table + degree hist plots logged to W&B.")


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


    def sbm_interleave(self,
                   patch_size=4,
                   super_block=2,
                   overlap_stride=None,       # None→no overlap; else < patch_size for overlap
                   h1_filters=1,
                   h1_per_filter=16,
                   h2_filters=1,
                   h2_per_filter=16,
                   filter_connect="matched",  # "matched" or "all_to_all"
                   no_sens2sens=True,
                   no_sens2supervised=True,
                   connect_h2_to_sup_dense=True,
                   connect_sup_to_h2_dense=True):


        IMG_H = IMG_W = 28
        S = overlap_stride or patch_size

        # ---- enumerate patch locations (top-left of each 4x4 window) ----
        patch_rows = list(range(0, IMG_H - patch_size + 1, S))
        patch_cols = list(range(0, IMG_W - patch_size + 1, S))
        G1r, G1c = len(patch_rows), len(patch_cols)
        num_patch_locs = G1r * G1c

        def patch_id(pr, pc): return pr * G1c + pc

        # ---- pixels per patch (and implicitly, pixels may belong to multiple patches if S < patch_size) ----
        pixels_in_patch = [[] for _ in range(num_patch_locs)]
        for pr, r0 in enumerate(patch_rows):
            for pc, c0 in enumerate(patch_cols):
                pid = patch_id(pr, pc)
                for r in range(r0, r0 + patch_size):
                    for c in range(c0, c0 + patch_size):
                        pixels_in_patch[pid].append(r * IMG_W + c)

        # ---- macro grouping on the (G1r x G1c) patch grid ----
        G2r = math.ceil(G1r / super_block)
        G2c = math.ceil(G1c / super_block)
        num_macro_locs = G2r * G2c

        def macro_id_from_patch(pr, pc):
            mr = pr // super_block
            mc = pc // super_block
            return mr * G2c + mc

        sensory = list(range(0, self.SENSORY_NODES))  # 0..783

        # ---- allocate H1: per patch-loc × h1_filters × h1_per_filter ----
        H1 = [[None for _ in range(h1_filters)] for _ in range(num_patch_locs)]
        next_id = self.SENSORY_NODES
        for pid in range(num_patch_locs):
            for f in range(h1_filters):
                cluster = list(range(next_id, next_id + h1_per_filter))
                H1[pid][f] = cluster
                next_id += h1_per_filter

        # ---- allocate H2: per macro-loc × h2_filters × h2_per_filter ----
        H2 = [[None for _ in range(h2_filters)] for _ in range(num_macro_locs)]
        for mid in range(num_macro_locs):
            for g in range(h2_filters):
                cluster = list(range(next_id, next_id + h2_per_filter))
                H2[mid][g] = cluster
                next_id += h2_per_filter

        # ---- supervision nodes (10) ----
        supervision = list(range(next_id, next_id + 10))
        next_id += 10

        # expose indices
        self.sensory_indices = sensory
        self.supervision_indices = supervision
        self.internal_indices = [n for pid in range(num_patch_locs) for f in range(h1_filters) for n in H1[pid][f]] + \
                                [n for mid in range(num_macro_locs) for g in range(h2_filters) for n in H2[mid][g]]
        self.NUM_INTERNAL_NODES = len(self.internal_indices)
        self.num_vertices = len(self.sensory_indices) + len(self.internal_indices) + len(self.supervision_indices)

        self.edge_index = []
        self.edge_type  = []

        # -------- 1) Sensory → H1 (to all filters at that patch-loc) --------
        for pid in range(num_patch_locs):
            src_pixels = pixels_in_patch[pid]
            for s in src_pixels:
                for f in range(h1_filters):
                    for t in H1[pid][f]:
                        self.edge_index.append([s, t])
                        self.edge_type.append("Sens2Inter")

        # optional Sens2Sens / Sens2Sup
        if not no_sens2sens:
            for i in sensory:
                for j in sensory:
                    if i != j:
                        self.edge_index.append([i, j]); self.edge_type.append("Sens2Sens")

        if not no_sens2supervised:
            for s in sensory:
                for sup in supervision:
                    self.edge_index.append([s, sup]); self.edge_type.append("Sens2Sup")

        # -------- 2) H1 → H2 (per macro). Filter wiring policy --------
        for pr in range(G1r):
            for pc in range(G1c):
                pid = patch_id(pr, pc)
                mid = macro_id_from_patch(pr, pc)

                if filter_connect == "matched":
                    # link H1 filter f → H2 filter f%h2_filters
                    for f in range(h1_filters):
                        tgt_filter = f % h2_filters
                        for u in H1[pid][f]:
                            for v in H2[mid][tgt_filter]:
                                self.edge_index.append([u, v])
                                self.edge_type.append("Inter2Inter")
                elif filter_connect == "all_to_all":
                    for f in range(h1_filters):
                        for g in range(h2_filters):
                            for u in H1[pid][f]:
                                for v in H2[mid][g]:
                                    self.edge_index.append([u, v])
                                    self.edge_type.append("Inter2Inter")
                else:
                    raise ValueError(f"Unknown filter_connect={filter_connect}")

        # -------- 3) H2 → Supervision (dense or not) --------
        if connect_h2_to_sup_dense:
            for mid in range(num_macro_locs):
                for g in range(h2_filters):
                    for u in H2[mid][g]:
                        for sup in supervision:
                            self.edge_index.append([u, sup])
                            self.edge_type.append("Inter2Sup")

        # -------- 3b) Supervision → H2 (top-down generative; optional) --------
        if connect_sup_to_h2_dense:
            for mid in range(num_macro_locs):
                for g in range(h2_filters):
                    for u in H2[mid][g]:
                        for sup in supervision:
                            self.edge_index.append([sup, u])
                            self.edge_type.append("Sup2Inter")


        print(f"✅ sbm_interleave+filters built: patches={G1r}x{G1c} (stride={S}), "
            f"H1 filters={h1_filters}×{h1_per_filter}, H2 filters={h2_filters}×{h2_per_filter}, "
            f"macros={G2r}x{G2c}, edges={len(self.edge_index)}")

        # keep meta
        self.sbm_interleave_meta = {
            "patch_size": patch_size,
            "overlap_stride": S,
            "super_block": super_block,
            "G1r": G1r, "G1c": G1c,
            "G2r": G2r, "G2c": G2c,
            "h1_filters": h1_filters, "h1_per_filter": h1_per_filter,
            "h2_filters": h2_filters, "h2_per_filter": h2_per_filter,
            "filter_connect": filter_connect,
        }


    def single_hidden_layer_clusters(self, discriminative_hidden_layers, generative_hidden_layers,
                                num_clusters_per_layer=10, p_intra=0.0, p_inter=0.1,
                                no_sens2sens=False, no_sens2supervised=False):
        """
        Like single_hidden_layer but layers are divided into clusters with probabilistic intra- and inter-cluster connections.
        """
        
        def build_clustered_layer(start_idx, total_nodes, clusters):
            """Divide layer into roughly equal-sized clusters."""
            if total_nodes == 0 or clusters == 0:
                return []

            nodes_per_cluster = max(1, total_nodes // clusters)
            cluster_nodes = []
            for i in range(clusters):
                start = start_idx + i * nodes_per_cluster
                end = start + nodes_per_cluster
                if start >= start_idx + total_nodes:
                    break
                end = min(end, start_idx + total_nodes)
                cluster = list(range(start, end))
                if cluster:
                    cluster_nodes.append(cluster)
            return cluster_nodes


        def add_probabilistic_edges(src_nodes, tgt_nodes, edge_type, p):
            """Add directed edges with probability p."""
            for src in src_nodes:
                for tgt in tgt_nodes:
                    if src != tgt and np.random.rand() < p:
                        self.edge_index.append([src, tgt])
                        self.edge_type.append(edge_type)

        
        # swap 
        # TODO; explain this??? something with edge_index (source, target) and edge_type??
        # also fix add_residual_connections_to_generative_layer() with the swap
        discriminative_hidden_layers, generative_hidden_layers = generative_hidden_layers, discriminative_hidden_layers

        current_idx = 784  # After sensory nodes

        # Step 1: Clustered discriminative layers
        discrim_layers = []
        for layer_size in discriminative_hidden_layers:
            layer = build_clustered_layer(current_idx, layer_size, num_clusters_per_layer)
            discrim_layers.append(layer)
            current_idx += layer_size

        # Step 2: Clustered generative layers
        gen_layers = []
        for layer_size in generative_hidden_layers:
            layer = build_clustered_layer(current_idx, layer_size, num_clusters_per_layer)
            gen_layers.append(layer)
            current_idx += layer_size

        # Step 3: Supervision nodes
        supervision_nodes = list(range(current_idx, current_idx + 10))
        self.supervision_indices = supervision_nodes
        current_idx += 10

        # Track internal node indices
        self.internal_indices = [n for layer in discrim_layers + gen_layers for cluster in layer for n in cluster]
        self.supervision_indices = supervision_nodes

        # === Discriminative path ===

        # Sensory → First discriminative layer
        for s in self.sensory_indices:
            for cluster in discrim_layers[0]:
                add_probabilistic_edges([s], cluster, "Sens2Inter", p_inter)

        # Discriminative hidden layers
        for i in range(len(discrim_layers) - 1):
            for src_cluster in discrim_layers[i]:
                for dst_cluster in discrim_layers[i + 1]:
                    add_probabilistic_edges(src_cluster, dst_cluster, "Inter2Inter", p_inter)

        # Discriminative last → supervision
        for cluster in discrim_layers[-1]:
            add_probabilistic_edges(cluster, supervision_nodes, "Inter2Sup", p_inter)

        # Intra-cluster edges in discriminative layers
        for layer in discrim_layers:
            for cluster in layer:
                add_probabilistic_edges(cluster, cluster, "Inter2Inter", p_intra)

        # === Generative path ===

        # Supervision → top generative layer
        for cluster in gen_layers[-1]:
            add_probabilistic_edges(supervision_nodes, cluster, "Sup2Inter", p_inter)

        # Generative hidden layers (reverse direction)
        for i in range(len(gen_layers) - 1, 0, -1):
            for src_cluster in gen_layers[i]:
                for dst_cluster in gen_layers[i - 1]:
                    add_probabilistic_edges(src_cluster, dst_cluster, "Inter2Inter", p_inter)

        # Bottom generative layer → sensory
        for cluster in gen_layers[0]:
            add_probabilistic_edges(cluster, self.sensory_indices, "Inter2Sens", p_inter)

        # Intra-cluster edges in generative layers
        for layer in gen_layers:
            for cluster in layer:
                add_probabilistic_edges(cluster, cluster, "Inter2Inter", p_intra)

        # === Sensory to sensory and supervision (optional) ===
        if not no_sens2sens:
            for i in self.sensory_indices:
                for j in self.sensory_indices:
                    if i != j:
                        self.edge_index.append([i, j])
                        self.edge_type.append("Sens2Sens")

        if not no_sens2supervised:
            for s in self.sensory_indices:
                for sup in supervision_nodes:
                    self.edge_index.append([s, sup])
                    self.edge_type.append("Sens2Sup")

        print("✅ single_hidden_layer_clusters (probabilistic) constructed.")
        self.NUM_INTERNAL_NODES = len(self.internal_indices)
        self.num_vertices = len(self.sensory_indices) + len(self.internal_indices)
        
        if self.supervised_learning:
            self.num_vertices += len(self.supervision_indices)




    def single_hidden_layer(self, discriminative_hidden_layers, generative_hidden_layers,
                                no_sens2sens=False, no_sens2supervised=False, bidirectional_hidden=False,
                                add_residual=False):
        """Creates a graph with shared internal nodes and layers for discriminative and generative paths."""
        # edge_index = []
        # edge_type = []

        def add_residual_connections_to_generative_layer(
            generative_layers,
            supervision_indices,
            edge_index,
            edge_type,
            edge_type_name="Sup2Inter",
            skip=1,
        ):
            """
            Adds residual connections from supervision nodes to earlier generative layers.
            For example, if skip=1 and there are 3 generative layers, it adds from sup to layer 1.
            """
            if len(generative_layers) < 2 or skip >= len(generative_layers):
                print("⚠️ Not enough layers or invalid skip value for residual connections.")
                return

            target_layer = generative_layers[-1 - skip]
            for sup in supervision_indices:
                for node in target_layer:
                    edge_index.append([sup, node])
                    edge_type.append(edge_type_name)

            print(f"✅ Added residual connections from supervision to generative layer {len(generative_layers)-1 - skip}")


        # swap 
        # TODO; explain this??? something with edge_index (source, target) and edge_type??
        # also fix add_residual_connections_to_generative_layer() with the swap
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


        # Add residual connections to generative layers if enabled


        # Optional residual connection from supervision → middle generative layer
        if add_residual:
            # log to wandb
            print("Adding residual connections to generative layers...")
            wandb.log({"add_residual_connections_to_generative_layer": True})
            
            add_residual_connections_to_generative_layer(
                # generative_layers=generative_layers,
                generative_layers=discriminative_layers,
                supervision_indices=supervision_indices,
                edge_index=self.edge_index,
                edge_type=self.edge_type,
                skip=1  # middle layer
            )
        print(self.graph_params["add_residual"], add_residual)
        print(self.graph_params)
                                            # add_residual=self.graph_params["add_residual"]


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



    def build_branch(self, layers_config, direction="forward", global_node_offset=0, shared_nodes=None,
                 p_intra=0.3, p_inter=0.05):
        G = nx.DiGraph()
        node_offset = global_node_offset
        layers = []

        for l, cfg in enumerate(layers_config):
            num_clusters = cfg['num_clusters']
            nodes_per_cluster = cfg['nodes_per_cluster']
            p_to_next = cfg.get('p_to_next', 0.0)
            layer_clusters = []

            for c in range(num_clusters):
                if l == 0:
                    if direction == "forward" and shared_nodes and 'sensory' in shared_nodes:
                        cluster_nodes = shared_nodes['sensory'][c]
                    elif direction == "backward" and shared_nodes and 'supervision' in shared_nodes:
                        cluster_nodes = shared_nodes['supervision'][c]
                    else:
                        cluster_nodes = list(range(node_offset, node_offset + nodes_per_cluster))
                        node_offset += nodes_per_cluster
                elif l == len(layers_config) - 1:
                    if direction == "forward" and shared_nodes and 'supervision' in shared_nodes:
                        cluster_nodes = shared_nodes['supervision'][c]
                    elif direction == "backward" and shared_nodes and 'sensory' in shared_nodes:
                        cluster_nodes = shared_nodes['sensory'][c]
                    else:
                        cluster_nodes = list(range(node_offset, node_offset + nodes_per_cluster))
                        node_offset += nodes_per_cluster
                else:
                    cluster_nodes = list(range(node_offset, node_offset + nodes_per_cluster))
                    node_offset += nodes_per_cluster

                G.add_nodes_from(cluster_nodes, layer=l, cluster=c)
                layer_clusters.append(cluster_nodes)

                # Intra-cluster connections
                for u in cluster_nodes:
                    for v in cluster_nodes:
                        if u != v and np.random.rand() < p_intra:
                            G.add_edge(u, v)

            # Inter-layer, inter-cluster connections
            if l > 0:
                prev_layer = layers[-1]
                for src_cluster in prev_layer:
                    for src_node in src_cluster:
                        for tgt_cluster in layer_clusters:
                            if np.random.rand() < p_inter:
                                for tgt_node in tgt_cluster:
                                    if np.random.rand() < layers_config[l - 1]['p_to_next']:
                                        G.add_edge(src_node, tgt_node)

            layers.append(layer_clusters)

        return G, node_offset, layers


    def generate_dual_sbm_graph(self, layers_config_dict, image_size=28, patch_size=4,
                            remove_sens_2_sens=False, remove_sens_2_sup=False):
        assert 'discriminative_layer' in layers_config_dict
        assert 'generative_layer' in layers_config_dict

        discrim_layers = layers_config_dict['discriminative_layer']
        generative_layers = layers_config_dict['generative_layer']
        p_intra = layers_config_dict.get('p_intra', 0.3)
        p_inter = layers_config_dict.get('p_inter', 0.05)

        num_patches = (image_size // patch_size) ** 2
        sensory_M = patch_size ** 2
        num_sensory_clusters = max(discrim_layers[0]['num_clusters'], num_patches)
        num_supervision_clusters = 10
        supervision_M = 10

        assert discrim_layers[0]['num_clusters'] >= num_patches, \
            f"Discriminative first layer must have ≥ {num_patches} clusters"

        assert (
            generative_layers[-1]['num_clusters'] == num_supervision_clusters or
            discrim_layers[-1]['num_clusters'] == num_supervision_clusters
        ), "Either discriminative or generative must end (or start) with 10 clusters"

        shared_sensory = []
        shared_supervision = []
        node_id = 0

        for i in range(num_sensory_clusters):
            nodes = list(range(node_id, node_id + sensory_M))
            shared_sensory.append(nodes)
            node_id += sensory_M

        for i in range(num_supervision_clusters):
            nodes = list(range(node_id, node_id + supervision_M))
            shared_supervision.append(nodes)
            node_id += supervision_M

        shared_nodes = {
            'sensory': shared_sensory,
            'supervision': shared_supervision
        }

        G_discrim, offset, discrim_clusters = self.build_branch(
            discrim_layers, "forward", node_id, shared_nodes, p_intra=p_intra, p_inter=p_inter)
        G_generative, _, gen_clusters = self.build_branch(
            generative_layers, "backward", node_id, shared_nodes, p_intra=p_intra, p_inter=p_inter)

        G = nx.compose(G_discrim, G_generative)

        # Sensory ↔ Sensory and Sensory → Supervision
        if not remove_sens_2_sens:
            for u in [n for cluster in shared_sensory for n in cluster]:
                for v in [n for cluster in shared_sensory for n in cluster]:
                    if u != v:
                        G.add_edge(u, v)

        if not remove_sens_2_sup:
            for u in [n for cluster in shared_sensory for n in cluster]:
                for v in [n for cluster in shared_supervision for n in cluster]:
                    G.add_edge(u, v)

        label_cluster_map = {}
        for i, cluster_nodes in enumerate(discrim_clusters[-1]):
            label_cluster_map[i] = cluster_nodes

        sensory_indices = [n for cluster in shared_sensory for n in cluster]
        supervision_indices = [n for nodes in label_cluster_map.values() for n in nodes]
        all_nodes = list(G.nodes)
        internal_indices = list(set(all_nodes) - set(sensory_indices) - set(supervision_indices))

        if remove_sens_2_sens:
            for u in sensory_indices:
                for v in sensory_indices:
                    if u != v and G.has_edge(u, v):
                        G.remove_edge(u, v)

        if remove_sens_2_sup:
            for u in sensory_indices:
                for v in supervision_indices:
                    if G.has_edge(u, v):
                        G.remove_edge(u, v)

        new_order = sensory_indices + internal_indices + supervision_indices
        old_to_new = {old: new for new, old in enumerate(new_order)}
        relabeled_edges = [(old_to_new[u], old_to_new[v]) for u, v in G.edges]
        self.edge_index = torch.tensor(relabeled_edges, dtype=torch.long).t().contiguous()

        self.sensory_indices = [old_to_new[n] for n in sensory_indices]
        self.supervision_indices = [old_to_new[n] for n in supervision_indices]
        self.internal_indices = [old_to_new[n] for n in internal_indices]
        self.label_cluster_map = {lbl: [old_to_new[n] for n in nodes] for lbl, nodes in label_cluster_map.items()}

        self.num_sensory_nodes = len(self.sensory_indices)
        self.num_supervision_nodes = len(self.supervision_indices)
        self.num_internal_nodes = len(self.internal_indices)
        self.num_vertices = len(new_order)
        self.num_all_nodes = list(range(self.num_vertices))

        adj_matrix = torch.zeros(self.num_vertices, self.num_vertices)
        for u, v in relabeled_edges:
            adj_matrix[u, v] = 1.0
        self.adj_matrix = adj_matrix

        return G









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

    
    def log_hop_distribution_to_wandb(self):
        """Logs hop length distribution between sensory and supervision nodes (both directions) to wandb."""

        def build_nx_graph(edge_index):
            G = nx.DiGraph()
            edge_list = edge_index.t().tolist()
            G.add_edges_from(edge_list)
            return G

        def get_all_hop_lengths(G, source_nodes, target_nodes):
            hop_lengths = []
            for src in source_nodes:
                lengths = nx.single_source_shortest_path_length(G, src)
                for dst in target_nodes:
                    if dst in lengths:
                        hop_lengths.append(lengths[dst])
            return hop_lengths

        def plot_histogram(hops, title, x_label="Hop Length", y_label="Count"):
            fig, ax = plt.subplots()
            ax.hist(hops, bins=np.arange(min(hops), max(hops) + 2) - 0.5, edgecolor='black')
            ax.set_title(title)
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.grid(True)
            return wandb.Image(fig)

        # === Build Graph ===
        G = build_nx_graph(self.edge_index)
        sensory = self.sensory_indices
        supervision = list(self.supervision_indices)

        hop_s2l = get_all_hop_lengths(G, sensory, supervision)
        hop_l2s = get_all_hop_lengths(G, supervision, sensory)

        log_dict = {}

        # ---- TABLE: summary stats for both directions ----
        def stat_dict(hops, prefix):
            if hops:
                return {
                    f"{prefix}_min": int(np.min(hops)),
                    f"{prefix}_max": int(np.max(hops)),
                    f"{prefix}_mean": float(np.mean(hops)),
                    f"{prefix}_count": int(len(hops)),
                }
            else:
                return {
                    f"{prefix}_min": None,
                    f"{prefix}_max": None,
                    f"{prefix}_mean": None,
                    f"{prefix}_count": 0,
                    f"{prefix}_unreachable": True,
                }

        def plot_hist(hops, title):
            if not hops: return None
            fig, ax = plt.subplots()
            bins = np.arange(min(hops), max(hops) + 2) - 0.5
            ax.hist(hops, bins=bins, edgecolor="black")
            ax.set_title(title)
            ax.set_xlabel("Hop Length")
            ax.set_ylabel("Count")
            ax.grid(True, alpha=0.3)
            img = wandb.Image(fig)
            plt.close(fig)
            return img
    
        row = {
            "graph_name": self.graph_type["name"],
            "seed": self.seed,
            **stat_dict(hop_s2l, "s2sup"),
            **stat_dict(hop_l2s, "sup2s"),
        }
        cols = list(row.keys())
        table = wandb.Table(columns=cols); table.add_data(*[row[c] for c in cols])
        wandb.log({"graph/hop_metrics_table": table}, commit=True)

        # ---- PLOTS: histograms as images ----
        payload = {}
        img_s2l = plot_hist(hop_s2l, "Sensory → Supervision Hop Distribution")
        img_l2s = plot_hist(hop_l2s, "Supervision → Sensory Hop Distribution")
        if img_s2l is not None:
            payload["graph/hops_sens_to_sup_hist_plot"] = img_s2l
        if img_l2s is not None:
            payload["graph/hops_sup_to_sens_hist_plot"] = img_l2s
        if payload:
            wandb.log(payload, commit=True)

        print("✅ Hop stats table + plots logged to W&B.")