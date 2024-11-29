import matplotlib.pyplot as plt
import numpy as np
from torch_geometric.utils import to_dense_adj, dense_to_sparse


import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_geometric.utils import to_dense_adj

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_geometric.utils import to_dense_adj
import wandb

def plot_adj_matrix(edge_index, model_dir, node_types=None):
    """
    Function to plot the adjacency matrix or node types (sensory, internal, supervision) as colored pixels.
    
    Parameters:
    edge_index (PyG edge_index): Edge index of the graph.
    model_dir (str): Directory to save the plot.
    node_types (tuple): Tuple containing sensory, internal, and supervision indices for overlay. 
                        If None, just plots the adjacency matrix.
    """
    # Convert edge_index to adjacency matrix
    adj_matrix_pyg = to_dense_adj(edge_index)[0].numpy()
    adj_matrix_size = adj_matrix_pyg.shape[0]

    if node_types:
        # Create a white empty grid for node types
        fig, ax = plt.subplots(figsize=(12, 12))  # Adjust size as needed
        grid = np.ones((adj_matrix_size, adj_matrix_size))  # White background
        ax.imshow(grid, cmap='gray', vmin=0, vmax=1, origin='upper')  # Display empty white grid with top-left origin

        # Extract sensory, internal, and supervision indices from node_types tuple
        sensory_indices, internal_indices, supervision_indices = node_types

        # Plot each node type as a single point/pixel
        if sensory_indices is not None:
            sensory_indices = np.array(sensory_indices)
            ax.scatter(sensory_indices, sensory_indices, color='red', label='Sensory Nodes', s=10)

        if internal_indices is not None:
            internal_indices = np.array(internal_indices)
            ax.scatter(internal_indices, internal_indices, color='blue', label='Internal Nodes', s=10)

        if supervision_indices is not None:
            supervision_indices = np.array(supervision_indices)
            ax.scatter(supervision_indices, supervision_indices, color='green', label='Supervision Nodes', s=10)

        # Set title and legend
        ax.set_title("Node Types: Sensory, Internal, and Supervision")
        ax.legend(loc='upper right')

        # Ensure the axis limits match the adjacency matrix size
        ax.set_xlim(0, adj_matrix_size - 1)
        ax.set_ylim(adj_matrix_size - 1, 0)  # Reverse the y-axis to match top-left origin

        # Save the figure with node types
        plt.tight_layout()
        if model_dir:
            fig.savefig(f'{model_dir}/node_types_grid.png')

            # Close the figure after saving
            plt.close(fig)
        else:
            plt.show()
    else:
        # Create figure and axis for the adjacency matrix
        fig, ax = plt.subplots(figsize=(30, 18))

        # Plot the adjacency matrix with top-left origin
        cax = ax.imshow(adj_matrix_pyg, cmap='viridis', origin='upper')
        ax.set_title("Adjacency Matrix")
        fig.colorbar(cax, ax=ax)

        # Ensure the axis limits match the adjacency matrix size
        ax.set_xlim(0, adj_matrix_size - 1)
        ax.set_ylim(adj_matrix_size - 1, 0)  # Reverse the y-axis to match top-left origin

        # Save the figure for the adjacency matrix
        plt.tight_layout()

        if model_dir:
            fig.savefig(f'{model_dir}/adj_matrix.png')

            # Close the figure after saving
            plt.close(fig)
        else:
            plt.show()  


def plot_connection_strength_dist(W):

    incomming = W[-10:, ].flatten()
    outgoing = W[:, -10:].flatten()
    plt.hist(incomming, alpha=0.5, bins=20, label="")
    plt.hist(outgoing, alpha=0.5, bins=20, label="")
    plt.show()



    incomming = W[-10:, 0:784 ].flatten()
    outgoing = W[0:784 :, -10:].flatten()
    plt.hist(incomming, alpha=0.5, bins=20)
    plt.hist(outgoing, alpha=0.5, bins=20)
    plt.show()



def plot_energy_during_training(internal_energy, sensory_energy, history, 
                                point1=(0, 0), point2=None, point3=None, model_dir=None, epoch="end"):



    # Assuming model and history are already defined and contain the required data

    # Create a subplot mosaic
    fig, ax = plt.subplot_mosaic([["A"], ["B"]], figsize=(12, 10))

    # Plot the first set of energy values with two y-axes
    ax["A"].plot(internal_energy, label="Internal Energy", color='blue')
    ax["A"].set_xlabel("Iterations")
    ax["A"].set_ylabel("Internal Energy", color='blue')
    ax["A"].tick_params(axis='y', labelcolor='blue')
    ax["A"].set_title("Energy over whole training")

    ax2 = ax["A"].twinx()
    ax2.plot(sensory_energy, label="Sensory Energy", color='orange')
    ax2.set_ylabel("Sensory Energy", color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')

    # Add legends to both y-axes
    ax["A"].legend(loc='upper left')
    ax2.legend(loc='upper right')

    if point1:
        ax["A"].plot(point1[0], point1[1], 'ro', label="Point 1 (0, 0)")
    if point2:
        ax["A"].plot(point2[0], point2[1], 'ro', label=f"Point 2 ({point2[0]}, {point2[1]})")
    if point3:
        ax["A"].plot(point3[0], point3[1], 'go', label=f"Point 3 ({point3[0]}, {point3[1]})")


    # Plot energy per epoch with two y-axes
    ax["B"].plot(history["internal_energy_per_epoch"], label="Internal Energy", color='blue')
    ax["B"].set_xlabel("Epochs")
    ax["B"].set_ylabel("Mean Internal Energy", color='blue')
    ax["B"].tick_params(axis='y', labelcolor='blue')
    ax["B"].set_title("Energy per Epoch")

    ax2_B = ax["B"].twinx()
    ax2_B.plot(history["sensory_energy_per_epoch"], label="Sensory Energy", color='orange')
    ax2_B.set_ylabel("Sensory Energy", color='orange')
    ax2_B.tick_params(axis='y', labelcolor='orange')

    # Add legends to both y-axes
    ax["B"].legend(loc='upper left')
    ax2_B.legend(loc='upper right')

    # Display the plots
    plt.tight_layout()

    if model_dir:
        fig.savefig(f'{model_dir}/energy/energy_{epoch}.png')
    else:
        plt.show()
    
    plt.close(fig)



import matplotlib.pyplot as plt
import os

def plot_energy_graphs(energy_vals, model_dir, window_size):
    # Ensure the model directory exists
    os.makedirs(model_dir, exist_ok=True)

    # Plot and save 'mean_internal_energy_sign'
    plt.plot(energy_vals["mean_internal_energy_sign"][:])
    plt.title("Mean Internal Energy Sign")
    plt.savefig(os.path.join(model_dir, "mean_internal_energy_sign.png"))
    plt.close()

    # Plot and save 'mean_sensory_energy_sign'
    plt.plot(energy_vals["mean_sensory_energy_sign"][:])
    plt.title("Mean Sensory Energy Sign")
    plt.savefig(os.path.join(model_dir, "mean_sensory_energy_sign.png"))
    plt.close()

    # Plot, save min, max, avg values for each type in ["sensory_energy", "internal_energy"]
    for tmp in ["sensory_energy", "internal_energy"]:
        t = energy_vals[tmp]
        plt.plot(t, label=f"{tmp}")

        v_min = []
        v_max = []
        v_avg = []

        for w in range(0, len(t)):
            window = t[w:w+window_size]
            if len(window) == 0:  # Avoid empty window
                continue
            w_min = min(window)
            w_max = max(window)
            w_avg = sum(window) / len(window)

            v_min.append(w_min)
            v_max.append(w_max)
            v_avg.append(w_avg)

        plt.plot(v_min, label=f"{tmp}_min")
        plt.plot(v_max, label=f"{tmp}_max")
        plt.plot(v_avg, label=f"{tmp}_avg")

        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.title(f"{tmp} with min, max, avg")
        plt.savefig(os.path.join(model_dir, f"{tmp}_min_max_avg.png"))
        plt.close()

    # Plot and save 'energy_drop'
    plt.plot(energy_vals['energy_drop'])
    plt.title("Energy Drop (positive best)")
    plt.savefig(os.path.join(model_dir, "energy_drop.png"))
    plt.close()

    # Plot and save 'weight_update_gain'
    plt.plot(energy_vals['weight_update_gain'])
    plt.title("Weight Update Gain (positive best)")
    plt.savefig(os.path.join(model_dir, "weight_update_gain.png"))
    plt.close()





def plot_effective_train_lr(model, model_dir):

    # Example: Plotting the effective learning rate
    import matplotlib.pyplot as plt
    plt.plot(model.pc_conv1.effective_learning["v_mean"], label='Mean Effective Learning Rate')
    plt.plot(model.pc_conv1.effective_learning["v_max"], label='Max Effective Learning Rate')
    plt.plot(model.pc_conv1.effective_learning["v_min"], label='Min Effective Learning Rate')
    plt.xlabel('Iteration')
    plt.ylabel('Effective Learning Rate')
    plt.legend()
    # plt.show()


    # Example: Plotting the effective learning rate
    import matplotlib.pyplot as plt
    plt.plot(model.pc_conv1.effective_learning["w_mean"], label='Mean Effective Learning Rate')
    plt.plot(model.pc_conv1.effective_learning["w_max"], label='Max Effective Learning Rate')
    plt.plot(model.pc_conv1.effective_learning["w_min"], label='Min Effective Learning Rate')
    plt.xlabel('Iteration')
    plt.ylabel('Effective Learning Rate')
    plt.legend()
    
    # plt.show()



import matplotlib.pyplot as plt
import torch
import scipy.sparse as sp


import os
import matplotlib.pyplot as plt
import scipy.sparse as sp

def plot_model_weights(model, GRAPH_TYPE=None, model_dir=None, save_wandb=False):
    # Ensure the output directory exists
    # os.makedirs(model_dir, exist_ok=True)

    # Extract the edge indices and weights
    edge_index = model.pc_conv1.edge_index_single_graph.cpu().numpy()
    weights = model.pc_conv1.weights.cpu().detach().numpy()

    # Create a sparse matrix using the edge indices and weights
    W_sparse = sp.coo_matrix((weights, (edge_index[0], edge_index[1])), shape=(model.pc_conv1.num_vertices, model.pc_conv1.num_vertices))

    # Convert to dense for detailed visualization (if the graph is not too large)
    W = W_sparse.toarray()

    # Create a figure with multiple subplots
    fig, axes = plt.subplots(3, 2, figsize=(20, 30))
    if GRAPH_TYPE:
        fig.suptitle(f'Visualization of Weight Matrix of {GRAPH_TYPE}', fontsize=20)
    else:
        fig.suptitle(f'Visualization of Weight Matrix', fontsize=20)

    # Subplot 1: Full weight matrix
    zero_weights = (W == 0).astype(int)  # Create a binary mask of zero weights
    im1 = axes[0, 0].imshow(zero_weights, cmap='viridis', aspect='auto')
    fig.colorbar(im1, ax=axes[0, 0], label='Weight value')
    axes[0, 0].set_title('Zero weights')


    # Subplot 2: Full weight matrix
    im1 = axes[0, 1].imshow(W, cmap='viridis', aspect='auto')
    fig.colorbar(im1, ax=axes[0, 1], label='Weight value')
    axes[0, 1].set_title('Full Weight Matrix')

    # Subplot 3: Weights > 0.001
    im2 = axes[1, 0].imshow(W > 0.001, cmap='viridis', aspect='auto')
    fig.colorbar(im2, ax=axes[1, 0], label='Weight value')
    axes[1, 0].set_title('Weights > 0.001')

    # Subplot 4: Weights > 0.0001
    im3 = axes[1, 1].imshow(W > 0.0001, cmap='viridis', aspect='auto')
    fig.colorbar(im3, ax=axes[1, 1], label='Weight value')
    axes[1, 1].set_title('Weights > 0.0001')

    # Subplot 5: Weights > 0.00001
    im4 = axes[2, 0].imshow(W > 0.00001, cmap='viridis', aspect='auto')
    fig.colorbar(im4, ax=axes[2, 0], label='Weight value')
    axes[2, 0].set_title('Weights > 0.00001')

    # Subplot 6: Weights > 0.000001
    im5 = axes[2, 1].imshow(W > 0.000001, cmap='viridis', aspect='auto')
    fig.colorbar(im5, ax=axes[2, 1], label='Weight value')
    axes[2, 1].set_title('Weights > 0.000001')

    # Adjust layout and save the figure
    plt.tight_layout(rect=[0, 0, 1, 0.97])  # Adjust layout to fit the title

    if save_wandb:

        # save_path = os.path.join(model_dir, 'parameter_info/weight_matrix_visualization_epoch0.png')
        epoch_x = model_dir.split("_")[-1]
        wandb.log({f"Weights/weights_{epoch_x}": [wandb.Image(fig)]})

    if model_dir:
        plt.savefig(model_dir)
        plt.close(fig)

        print(f'Figure saved to {model_dir}')
    else:
        plt.show()

    return W





import matplotlib.lines as mlines

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_geometric.utils import to_dense_adj

def plot_graph_with_edge_types(N, edge_index, edge_types, edge_type_map):
    """
    Plot an NxN adjacency matrix with edge types represented as distinct colors.

    Parameters:
    - N: Number of nodes in the graph.
    - edge_index: Tensor of shape (2, num_edges), edge connections.
    - edge_types: Tensor of edge types corresponding to each edge.
    - edge_type_map: Dictionary mapping edge type names to indices.
    """

    print("----plot_graph_with_edge_types-----")
    # Create a dense adjacency matrix for edge types
    adj_matrix = torch.zeros((N, N), dtype=torch.long)
    for (src, tgt), etype in zip(edge_index.T, edge_types):
        adj_matrix[src, tgt] = etype

    # Prepare a colormap for the edge types
    edge_colors = plt.cm.get_cmap("tab10", len(edge_type_map))

    # Plot the adjacency matrix with colors for edge types
    fig, ax = plt.subplots(figsize=(10, 10))
    cax = ax.imshow(adj_matrix.numpy(), cmap=edge_colors, origin="upper")
    ax.set_title("Graph with Edge Types", fontsize=16)
    ax.set_xlabel("Target Node")
    ax.set_ylabel("Source Node")

    # Add color bar with edge type labels
    cbar = fig.colorbar(cax, ticks=range(len(edge_type_map)))
    cbar.ax.set_yticklabels(list(edge_type_map.keys()))
    cbar.set_label("Edge Types", rotation=270, labelpad=20)

    plt.tight_layout()

    # log to wandb 
    wandb.log({"delta_w/Graph_with_Edge_Types": [wandb.Image(fig)]})

    # close the figure
    plt.close(fig)
    