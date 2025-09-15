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

def plot_adj_matrix(edge_index, model_dir=None, node_types=None):
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
    return adj_matrix_pyg


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
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm, TwoSlopeNorm


import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
from matplotlib.colors import SymLogNorm

def plot_weights_symlog_adaptive(
    model,
    title="Weights (symlog)",
    max_dense_pixels=1_200_000,   # if N*N exceeds this, bin instead of imshow dense
    target_bins=768,              # coarse image size for binned view
    clip_percentiles=(5, 95),     # robust clipping for tails
    linthresh_quantile=0.70,      # linear window around zero from |w| quantile
    force_bins=None,              # None=auto, True/False to override
    return_img=False
):
    """
    Renders a symlog heatmap that adapts to very sparse/large matrices.

    - If N*N is small enough, draws dense symlog heatmap with masked zeros.
    - Else, bins edges into a smaller grid and draws symlog on that.
    - linthresh/vmax/vmin computed on nonzero entries only (robust).
    """

    # --- pull weights and edges ---
    N = model.num_vertices
    edge_index = model.edge_index_single_graph.detach().cpu().numpy()
    w = model.w.detach().cpu().numpy()

    # If w is dense NxN, convert to COO edges; if it is 1D per-edge, assemble sparse
    if w.ndim == 2 and w.shape == (N, N):
        # Use sparsity if possible
        W_sparse = sp.coo_matrix(w)  # cheap view if already sparse-like
        rows, cols, vals = W_sparse.row, W_sparse.col, W_sparse.data
    else:
        # Per-edge weights assumed at indices (edge_index[0], edge_index[1])
        rows, cols = edge_index[0], edge_index[1]
        vals = w[rows, cols] if w.ndim == 2 else w  # handle both cases

    nnz = vals.size
    total_pixels = N * N

    # Decide dense vs binned
    if force_bins is None:
        use_bins = (total_pixels > max_dense_pixels) or (nnz == 0)
    else:
        use_bins = bool(force_bins)

    if use_bins:
        # ----- BIN into a target grid -----
        bins = min(target_bins, max(N, 1))
        rbin = (rows * bins // max(N, 1)).astype(np.int32)
        cbin = (cols * bins // max(N, 1)).astype(np.int32)
        acc = np.zeros((bins, bins), dtype=np.float32)
        cnt = np.zeros((bins, bins), dtype=np.float32)

        # Sum and counts per bin (fast, vectorized)
        np.add.at(acc, (rbin, cbin), vals)
        np.add.at(cnt, (rbin, cbin), 1.0)
        # Use mean per bin (smoother); switch to sum by replacing this line
        img = acc / np.maximum(cnt, 1e-12)
        img_mask = cnt == 0  # true zeros = empty bins

        data_for_stats = img[~img_mask]
        mode_str = f"binned {bins}×{bins}"
    else:
        # ----- DENSE ARRAY but keep it memory-conscious -----
        # Build sparse and toarray only if it fits, else fall back to bins
        if total_pixels > max_dense_pixels:
            # safety net — should not happen given the earlier condition
            return plot_weights_symlog_adaptive(
                model, title, max_dense_pixels, target_bins, clip_percentiles,
                linthresh_quantile, force_bins=True, return_img=return_img
            )
        W = sp.coo_matrix((vals, (rows, cols)), shape=(N, N)).toarray()
        img = W
        img_mask = (img == 0)
        data_for_stats = img[~img_mask]
        mode_str = f"dense {N}×{N}"

    # --- stats on nonzeros only ---
    if data_for_stats.size == 0:
        # nothing to show — create a blank and return
        fig, ax = plt.subplots(figsize=(8, 7))
        ax.imshow(np.zeros((10,10)), cmap='RdBu_r', vmin=-1, vmax=1)
        ax.set_title(f"{title} — empty")
        ax.axis("off")
        fig.tight_layout()
        return (fig, img) if return_img else fig

    lo, hi = np.percentile(data_for_stats, clip_percentiles)
    img_clipped = np.clip(img, lo, hi)

    abs_data = np.abs(data_for_stats)
    vmax = np.percentile(abs_data, clip_percentiles[1])  # symmetric limit
    vmax = float(max(vmax, 1e-12))
    linthresh = np.percentile(abs_data, 100 * linthresh_quantile)
    linthresh = float(max(linthresh, 1e-9))

    # --- masked array for clean background ---
    img_ma = np.ma.masked_where(img_mask, img_clipped)

    # --- symlog norm ---
    norm = SymLogNorm(vmin=-vmax, vmax=vmax, linthresh=linthresh, base=10)

    # # --- plot ---
    # fig, ax = plt.subplots(figsize=(10, 10))
    # im = ax.imshow(img_ma, cmap='RdBu_r', norm=norm, origin='upper', aspect='auto')

    fig, ax = plt.subplots(figsize=(15, 15))
    im = ax.imshow(
        img_ma,
        cmap='RdBu_r',
        norm=norm,
        origin='upper',
        aspect='auto',
        interpolation='nearest',
        extent=(-0.5, N-0.5, N-0.5, -0.5)  # <<< map image -> node index range
    )
    ax.set_xlim(-0.5, N-0.5)
    ax.set_ylim(N-0.5, -0.5)

    # increase title font size and xy labels size 
    ax.title.set_fontsize(16)
    ax.xaxis.label.set_fontsize(14)
    ax.yaxis.label.set_fontsize(14)
    
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("weight")
    # ax.set_title(f"{title} — {mode_str}\nlinthresh≈{linthresh:.2g}, vmax≈{vmax:.2g}, clip={clip_percentiles}")
    ax.set_title(f"{title}")
    ax.set_xlabel("j (columns)")
    ax.set_ylabel("i (rows)")
    fig.tight_layout()

    return (fig, img_ma) if return_img else fig


def plot_model_weights(model, GRAPH_TYPE=None, model_dir=None, save_wandb=None):
    # Extract the edge indices and weights
    edge_index = model.edge_index_single_graph.cpu().numpy()
    weights = model.w.detach().cpu().numpy()

    if weights.ndim == 1:
        W_sparse = sp.coo_matrix(
            (weights, (edge_index[0], edge_index[1])),
            shape=(model.num_vertices, model.num_vertices)
        )
    else:
        W_sparse = sp.coo_matrix(weights)

    W = W_sparse.toarray()

    # ---------------------------
    # Main 4x2 panel (your plot)
    # ---------------------------
    fig, axes = plt.subplots(4, 2, figsize=(20, 30))
    title = f'Visualization of Weight Matrix of {GRAPH_TYPE}' if GRAPH_TYPE else 'Visualization of Weight Matrix'
    fig.suptitle(title, fontsize=20)

    zero_weights = (W == 0).astype(int)
    im1 = axes[0, 0].imshow(zero_weights, cmap='viridis', aspect='auto')
    fig.colorbar(im1, ax=axes[0, 0], label='Weight value')
    axes[0, 0].set_title('Zero weights')

    im2 = axes[0, 1].imshow(W, cmap='viridis', aspect='auto')
    fig.colorbar(im2, ax=axes[0, 1], label='Weight value')
    axes[0, 1].set_title('Full Weight Matrix')

    thresholds = [0.001, 0.0001, 0.00001, 0.000001, 0.0000001, 0.00000001]
    for idx, thresh in enumerate(thresholds):
        row, col = divmod(idx + 2, 2)
        im = axes[row, col].imshow(W > thresh, cmap='viridis', aspect='auto')
        fig.colorbar(im, ax=axes[row, col], label='Weight value')
        axes[row, col].set_title(f'Weights > {thresh}')

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    # ---------------------------
    # Separate symlog figure
    # ---------------------------
    # percentile clip to stop outliers from crushing the scale
    # lo, hi = np.percentile(W, 1), np.percentile(W, 99)
    # Wc = np.clip(W, lo, hi)
    # v = np.max(np.abs(Wc))
    # # pick a small linear window around zero (median abs magnitude works well)
    # linthresh = np.percentile(np.abs(Wc[Wc != 0]), 50) if np.any(Wc != 0) else 1e-6
    # norm = SymLogNorm(vmin=-v, vmax=v, linthresh=linthresh, base=10)

    # fig_symlog, ax_symlog = plt.subplots(figsize=(10, 10))
    # im_symlog = ax_symlog.imshow(Wc, cmap='RdBu_r', norm=norm, aspect='auto')
    # cb = fig_symlog.colorbar(im_symlog, ax=ax_symlog)
    # cb.set_label('weight')
    # ax_symlog.set_title(f'Weights (signed symlog) — linthresh ≈ {linthresh:.2g}')
    # ax_symlog.set_xlabel('j (columns)')
    # ax_symlog.set_ylabel('i (rows)')
    # fig_symlog.tight_layout()

    # ---------------------------
    # Log to Weights & Biases
    # ---------------------------
    if save_wandb:
        epoch_x = save_wandb
        wandb.log({f"Weights/weights_{epoch_x}": [wandb.Image(fig)]})

        fig_symlog = plot_weights_symlog_adaptive(model, title=f"Weights (symlog) — {GRAPH_TYPE}")
        if save_wandb:
            wandb.log({f"Weights/weights_symlog_{save_wandb}_{epoch_x}": wandb.Image(fig_symlog)})

        # wandb.log({f"Weights/weights_symlog_{epoch_x}": [wandb.Image(fig_symlog)]})

        # Negative-only quick view (bugfix: log the correct figure)
        if (W < 0).any():
            fig_neg, ax_neg = plt.subplots(figsize=(10, 8))
            neg = W * (W < 0)
            im_neg = ax_neg.imshow(neg, cmap='coolwarm', aspect='auto')
            fig_neg.colorbar(im_neg, ax=ax_neg, label='Negative weight magnitude')
            ax_neg.set_title('Negative Weights and Their Magnitudes')
            ax_neg.set_xlabel('j (columns)'); ax_neg.set_ylabel('i (rows)')
            fig_neg.tight_layout()
            wandb.log({f"Weights_neg/weights_negative_{epoch_x}": [wandb.Image(fig_neg)]})
            plt.close(fig_neg)

        # Bias distribution if present
        if getattr(model, 'use_bias', False):
            bias = model.b.detach().cpu().numpy().flatten()
            fig_bias, ax_bias = plt.subplots(figsize=(10, 4))
            ax_bias.hist(bias, bins=100, alpha=0.8)
            ax_bias.set_title('Distribution of Bias'); ax_bias.set_xlabel('Bias'); ax_bias.set_ylabel('Count')
            fig_bias.tight_layout()
            wandb.log({f"Weights/bias_distribution_{epoch_x}": wandb.Image(fig_bias)})
            plt.close(fig_bias)

    # ---------------------------
    # Save to disk (two files)
    # ---------------------------
    # if model_dir:
    #     # main grid
    #     fig.savefig(model_dir, dpi=200)
    #     # symlog figure (adds suffix)
    #     symlog_path = model_dir.rsplit('.', 1)[0] + '_symlog.png'
    #     fig_symlog.savefig(symlog_path, dpi=200)
    #     print(f'Figures saved to:\n  • {model_dir}\n  • {symlog_path}')
    #     plt.close(fig_symlog)
    #     plt.close(fig)

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
    

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_geometric.utils import to_dense_adj
import wandb

def plot_updated_edges(N, edge_index, edges_2_update, delta_w_selection, model_dir=None, show=True, sample_size=1000):
    """
    Plot a subset of the graph's adjacency matrix with updated edges highlighted.
    
    Parameters:
    - N: Number of nodes in the graph.
    - edge_index: Tensor of shape (2, num_edges), edge connections.
    - edges_2_update: Boolean mask indicating which edges are being updated.
    - model_dir: Directory to save the plot (optional).
    - show: Whether to display the plot (default: True).
    - sample_size: Number of edges to sample for plotting to improve performance.
    """
    print("------plot_updated_edges-----")
    
    num_edges = edge_index.size(1)
    
    # Sample edges if the graph is too large
    if num_edges > sample_size:
        sampled_indices = torch.randperm(num_edges)[:sample_size]
        edge_index_sampled = edge_index[:, sampled_indices]
        edges_2_update_sampled = edges_2_update[sampled_indices]
    else:
        edge_index_sampled = edge_index
        edges_2_update_sampled = edges_2_update
    
    # Convert sampled edges to numpy
    edge_index_np = edge_index_sampled.cpu().numpy()
    edges_2_update_np = edges_2_update_sampled.cpu().numpy()
    
    # Create adjacency matrix for sampled edges
    adj_matrix = np.zeros((N, N), dtype=np.float32)
    src, tgt = edge_index_np
    adj_matrix[src, tgt] = 1.0
    
    # Identify updated edges
    updated_src = src[edges_2_update_np]
    updated_tgt = tgt[edges_2_update_np]
    
    # Plot the adjacency matrix
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(adj_matrix, cmap="Greys", origin="upper", alpha=0.3)

    # add delta_w_selection to title
    ax.set_title(f"Graph Adjacency Matrix with Updated Edges Highlighted\nDelta W Selection: {delta_w_selection}", fontsize=16)
    ax.set_xlabel("Target Node")
    ax.set_ylabel("Source Node")
    
    # Overlay the updated edges in red using vectorized plotting
    if len(updated_src) > 0:
        ax.scatter(updated_tgt, updated_src, color="red", s=10, label="Updated Edge")
    
    # Adjust axis limits
    ax.set_xlim(-0.5, N - 0.5)
    ax.set_ylim(N - 0.5, -0.5)
    
    # Add legend if there are updated edges
    if len(updated_src) > 0:
        ax.legend(loc="upper right")
    
    
    plt.tight_layout()
    
    # Log to WandB
    if wandb.run and model_dir:
        wandb.log({"delta_w/Graph_with_Updated_Edges": [wandb.Image(fig)]})
    
    # # Save the figure
    # if model_dir:
    #     save_path = f"{model_dir}/updated_edges_plot.png"
    #     plt.savefig(save_path, dpi=300)
    #     print(f"Figure saved to {save_path}")
    
    # # Show or close the figure
    # if show:
    #     plt.show()
    # else:
    #     plt.close(fig)


import networkx as nx

# Convert PyTorch edge_index to a NetworkX DiGraph
def build_nx_graph(edge_index):
    G = nx.DiGraph()
    edge_list = edge_index.t().tolist()
    G.add_edges_from(edge_list)
    return G

def compute_min_max_hops(G, source_nodes, target_nodes):
    hop_lengths = []

    for src in source_nodes:
        lengths = nx.single_source_shortest_path_length(G, src)
        relevant_lengths = [lengths[dst] for dst in target_nodes if dst in lengths]
        hop_lengths.extend(relevant_lengths)

    if not hop_lengths:    
        # A hop length of 0 means the source and target are the same node, or there is a direct self-loop.

        return float('inf'), float('-inf')  # No path found

    return min(hop_lengths), max(hop_lengths)




import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
import wandb


# args = {    "discriminative_hidden_layers": [32, 16],  # Example hidden layers
#             "batch_size": 64,
#             "epochs": 10,
#             "lr_weights": 0.001,
#             "w_decay_lr_weights": 0.0001,
#             "activation_func": "relu",  # Options: "relu", "swish", "tanh", "sigmoid"
#             "seed": 42,
#             "graph_type": "mlp"  # Example graph type
#         }


def train_mlp_classifier(run_id, PC_weights, hidden_layer, activation_fn, 
               base_train_dataset, base_test_dataset, batch_size, epochs,
                lr_w, weight_decay, seed, graph_type):

    # TODO MOVE THIS FUNCTION TO TRAIN_MLP.py
    # Set random seed for reproducibility
    torch.manual_seed(seed)

    # ----------------- Dataset ----------------- #
    # base dataset already has the same transformations applied
    train_loader = DataLoader(base_train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader = DataLoader(base_test_dataset, batch_size=batch_size, shuffle=False)

    # ----------------- Hyperparams from args ----------------- #
    input_size = 784
    hidden_layers = hidden_layer  # e.g., [32, 16]
    output_size = 10
    batch_size = batch_size
    epochs = epochs
    learning_rate = lr_w  # for backprop
    weight_decay = weight_decay
    activation_fn = activation_fn.lower()  # "relu", "swish", etc.
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # wandb.init(project="MLP_MNIST", name="MLP_discrim_baseline", config={
    #     "hidden_layers": hidden_layers,
    #     "lr": learning_rate,
    #     "weight_decay": weight_decay,
    #     "activation": activation_fn
    # })

    # wandb.disable_wandb()

    # ----------------- Activation Selection ----------------- #
    def get_activation(act):
        if act == "swish":
            return lambda x: x * torch.sigmoid(x)
        elif act == "relu":
            return nn.ReLU()
        elif act == "tanh":
            return nn.Tanh()
        elif act == "sigmoid":
            return nn.Sigmoid()
        else:
            raise ValueError(f"Unsupported activation: {act}")

    activation = get_activation(activation_fn.lower())

    # ----------------- MLP Model ----------------- #
    layers = [nn.Flatten()]
    prev = input_size
    for h in hidden_layers:
        layers.append(nn.Linear(prev, h))
        layers.append(activation if isinstance(activation, nn.Module) else nn.ReLU())  # swish is function
        prev = h
    layers.append(nn.Linear(prev, output_size))
    mlp = nn.Sequential(*layers).to(DEVICE)

    # ----------------- Optimizer and Loss ----------------- #
    optimizer = torch.optim.Adam(mlp.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()


    # ----------------- Training Loop ----------------- #
    test_acc = []
    for epoch in range(epochs):
        mlp.train()
        for X, y in train_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out = mlp(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

        # Validation
        mlp.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.to(DEVICE), y.to(DEVICE)
                out = mlp(X)
                preds = out.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        acc = correct / total
        test_acc.append(acc)
        wandb.log({"epoch": epoch, "val_accuracy": acc, "train_loss": loss.item()})
        print(f"Epoch {epoch}: Acc={acc:.4f} | Loss={loss.item():.4f}")

    # ----------------- Save ----------------- #
    # torch.save(mlp.state_dict(), f"trained_models/{graph_type}/weights/mlp_last.pt")
    # wandb.save(f"trained_models/{graph_type}/weights/mlp_last.pt")
    # wandb.finish()

    # create new folder in 'scr3/jobs"
    import os

    # Ensure directory exists
    # dir path include training parameters: mlp_{layers}_  epochs, lr_w, w_decay_lr_w, activation_fn, batch_size, seed
    # unpack_layers = "_".join(map(str, layers))
    unpack_hidden_layers = "_".join(map(str, hidden_layers))
    # make dir_path have all the parameters but string as short as possible
    run_id = str(run_id) if run_id is not None else "mlp_run"
    dir_path = f"trained_models/mlp/weights/mlp_{run_id}_{unpack_hidden_layers}_ep_{epochs}_bs_{batch_size}_lr_{learning_rate}_wd_{weight_decay}_{activation_fn}_sed_{seed}"
    
    os.makedirs(dir_path, exist_ok=True)

    os.makedirs(f"{dir_path}/weights", exist_ok=True)
    # os.makedirs(f"scr3/jobs/{graph_type}/weights", exist_ok=True)
    # torch.save(mlp.state_dict(), f"scr3/jobs/{graph_type}/weights/mlp_last.pt")
    torch.save(mlp.state_dict(), f"{dir_path}/weights/mlp_last.pt")
    print(f"Model saved to scr3/jobs/{graph_type}/weights/mlp_last.pt") 

    # in a file called "test_acc.txt" in the same folder
    with open(f"{dir_path}/test_acc.txt", "w") as f:
        for acc in test_acc:
            f.write(f"{acc}\n")
    print(f"Test accuracies saved to {dir_path}/test_acc.txt")

    # save fig
    # prompt: plot the model weights, in a imshow

    import matplotlib.pyplot as plt

    # Extract the weights of the first linear layer (after Flatten)
    first_linear_layer = None
    for layer in mlp.modules():
        if isinstance(layer, nn.Linear):
            first_linear_layer = layer
            break

    if first_linear_layer is not None:
        weights = first_linear_layer.weight.cpu().detach().numpy()

        # Since the input is flattened MNIST (28x28 = 784), we can reshape the weights
        # for the first layer to visualize them as images.
        # The weights have shape (output_features, input_features).
        # We can reshape each row (which corresponds to the weights connecting to one output neuron)
        # into a 28x28 image.
        num_output_features = weights.shape[0]
        num_input_features = weights.shape[1] # Should be 784 for the first linear layer

        if num_input_features == 784:
            print(f"Plotting weights for the first linear layer ({num_output_features} output features).")
            # Determine the grid size for plotting
            grid_cols = min(num_output_features, 8) # Plot up to 8 columns
            grid_rows = (num_output_features + grid_cols - 1) // grid_cols

            plt.figure(figsize=(grid_cols * 2, grid_rows * 2))

            for i in range(num_output_features):
                plt.subplot(grid_rows, grid_cols, i + 1)
                # Reshape the weights for this output neuron into a 28x28 image
                weight_image = weights[i].reshape(28, 28)
                plt.imshow(weight_image, cmap='viridis') # Use 'viridis' or another colormap
                plt.title(f'Neuron {i+1}')
                plt.axis('off')

            plt.tight_layout()
            # plt.show()
            # plt.savefig to dir_path
            plt.savefig(f"{dir_path}/mlp_weights.png")

            # plt.savefig(f"scr3/jobs/{graph_type}/weights/mlp_weights.png")
            print(f"Weights image saved to scr3/jobs/{graph_type}/weights/mlp_weights.png")
            # close 
            plt.close()

        else:
            print(f"First linear layer has input features {num_input_features}, not 784. Cannot reshape to 28x28 for imshow.")
    else:
        print("No linear layer found in the model after Flatten.")

    import numpy as np
    import matplotlib.pyplot as plt
    import os


    # Calculate the total number of neurons
    total_neurons = input_size + sum(hidden_layers) + output_size

    # Create adjacency matrix
    adj_matrix = np.zeros((total_neurons, total_neurons))

    # Index tracking
    current_idx = 0
    input_indices = list(range(current_idx, current_idx + input_size))
    current_idx += input_size

    hidden_layer_indices = []
    for h in hidden_layers:
        indices = list(range(current_idx, current_idx + h))
        hidden_layer_indices.append(indices)
        current_idx += h

    output_indices = list(range(current_idx, current_idx + output_size))

    # Wiring: input -> hidden -> output
    layer_input_indices = input_indices
    layer_output_indices_list = hidden_layer_indices + [output_indices]

    # TODO; check double transpose
    for layer in mlp.modules():
        if isinstance(layer, nn.Linear):
            weights = layer.weight.cpu().detach().numpy().T  # ← TRANSPOSE HERE
            current_layer_output_indices = layer_output_indices_list.pop(0)

            for i_in, from_idx in enumerate(layer_input_indices):
                for i_out, to_idx in enumerate(current_layer_output_indices):
                    adj_matrix[from_idx, to_idx] = weights[i_in, i_out]

            layer_input_indices = current_layer_output_indices

            # Plot
            plt.figure(figsize=(10, 10))
            plt.imshow(adj_matrix, cmap='viridis', origin='upper')
            plt.title('Transposed Weights as Adjacency Matrix')
            plt.xlabel('To Neuron Index')
            plt.ylabel('From Neuron Index')
            plt.colorbar(label='Weight Value')
            plt.tight_layout()

            # save fig to folder dir
            plt.savefig(f"{dir_path}/mlp_weights_adj_matrix.png")
            print(f"Adjacency matrix saved to trained_models/mlp/weights/mlp_weights_adj_matrix.png")
            plt.close()

    # shoud be an NxN matrix, where N is the total number of neurons (N= 784 + sum(hidden_layers) + 10)
    # mlp_model_weights = adj_matrix
    weights = adj_matrix.T

    ############################



    # compare the matrix "weights" with the "PC_weights" by calculating the abs avg difference in percentage
    if PC_weights is not None:
        PC_weights = PC_weights  # .cpu().detach().numpy()
        # Calculate the absolute average difference in percentage
        abs_diff = np.abs(weights - PC_weights)
        avg_diff = np.mean(abs_diff)
        avg_diff_percentage = (avg_diff / np.mean(np.abs(PC_weights))) * 100

        print(f"Average absolute difference between model weights and PC weights: {avg_diff_percentage:.2f}%")
    else:
        print("PC_weights is None, skipping comparison.")

    # log the average difference to txt file 
    with open(f"{dir_path}/avg_diff.txt", "w") as f:
        f.write(f"Average absolute difference between model weights and PC weights: {avg_diff_percentage:.2f}%\n")

    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity

    # ------------------ Inputs ---------------------
    # weights:      NxN matrix from model A (e.g., MLP)
    # PC_weights:   NxN matrix from model B (e.g., Predictive Coding)
    # ------------------------------------------------

    assert weights.shape == PC_weights.shape, "Weight shapes do not match!"

    # Threshold to consider a value as "non-zero"
    # threshold = 1e-6
    threshold = 1e-2
    # threshold = max(np.mean(np.abs(weights)) * 0.01, 1e-6)
    print("Using threshold for non-zero weights:", threshold)


    # 1. ---------- Non-zero mask (both matrices) ----------
    common_non_zero_mask = (np.abs(PC_weights) > threshold) & (np.abs(weights) > threshold)

    if np.sum(common_non_zero_mask) == 0:
        print("No common non-zero weights to compare.")

        # save imshow plot of both weights
        fig, ax = plt.subplots(1, 2, figsize=(20, 10))
        ax[0].imshow(weights, cmap='viridis', origin='upper')
        ax[0].set_title("Model Weights")
        ax[1].imshow(PC_weights, cmap='viridis', origin='upper')
        ax[1].set_title("PC Weights")
        plt.tight_layout()
        # save dir_path
        plt.savefig(f"{dir_path}/weights_comparison_no_common_non_zero.png")
        print(f"Saved weights comparison plot to {dir_path}/weights_comparison_no_common_non_zero.png")
        plt.close(fig)
    else:
        # Extract only relevant weights
        W1 = weights[common_non_zero_mask]
        W2 = PC_weights[common_non_zero_mask]

        # 2. ---------- Average Absolute Percentage Difference ----------
        abs_diff = np.abs(W1 - W2)
        avg_abs_diff = np.mean(abs_diff)
        avg_pct_diff = (avg_abs_diff / np.mean(np.abs(W2))) * 100

        # 3. ---------- RMSE ----------
        rmse = np.sqrt(np.mean((W1 - W2) ** 2))

        # 4. ---------- Cosine Similarity ----------
        # Reshape into vectors
        W1_vec = W1.reshape(1, -1)
        W2_vec = W2.reshape(1, -1)
        cosine_sim = cosine_similarity(W1_vec, W2_vec)[0, 0]

        # 5. ---------- Normalized weights comparison ----------
        norm_W1 = W1 / np.linalg.norm(W1)
        norm_W2 = W2 / np.linalg.norm(W2)

        norm_abs_diff = np.mean(np.abs(norm_W1 - norm_W2))

        # ----------- Print Results ------------
        print(f"🔍 Common non-zero weights: {common_non_zero_mask.sum()}")
        print(f"📏 Avg absolute % difference: {avg_pct_diff:.2f}%")
        print(f"📉 RMSE: {rmse:.6f}")
        print(f"🎯 Cosine Similarity: {cosine_sim:.4f}")
        print(f"📐 Normalized Avg Abs Difference: {norm_abs_diff:.6f}")

        # ----------- Optionally: Save to File ------------
        with open(f"{dir_path}/weight_comparison_metrics.txt", "w") as f:
            f.write(f"Common non-zero weights: {common_non_zero_mask.sum()}\n")
            f.write(f"Average absolute % difference: {avg_pct_diff:.2f}%\n")
            f.write(f"RMSE: {rmse:.6f}\n")
            f.write(f"Cosine Similarity: {cosine_sim:.4f}\n")
            f.write(f"Normalized Avg Abs Difference: {norm_abs_diff:.6f}\n")


    import numpy as np
    import matplotlib.pyplot as plt

    # Parameters (same as for your first layer in MLP)
    num_hidden_neurons = hidden_layer[0]  # e.g. like [32, 16] → use 16 for second layer, or 32 for first
    input_size = 784  # 28x28 flattened image


    # Architecture setup
    input_size = 784
    hidden_size = hidden_layer[0]   # Change as needed
    output_size = 10
    N = input_size + hidden_size + output_size

    # Create full weight matrix: N x N
    # Create random weights matrix
    W = np.random.randn(N, N) 
    W = PC_weights if PC_weights is not None else W  # Use PC_weights if provided

    # Get hidden node indices
    hidden_start = input_size
    hidden_end = input_size + hidden_size

    # Extract weights from inputs to hidden layer
    W_input_to_hidden = W[hidden_start:hidden_end, :input_size]  # Shape: [hidden_size, 784]

    # Plot each hidden neuron's incoming weights as a 28x28 image
    grid_cols = min(hidden_size, 8)
    grid_rows = (hidden_size + grid_cols - 1) // grid_cols

    plt.figure(figsize=(grid_cols * 2, grid_rows * 2))
    for i in range(hidden_size):
        weight_img = W_input_to_hidden[i].reshape(28, 28)
        plt.subplot(grid_rows, grid_cols, i + 1)
        plt.imshow(weight_img, cmap='viridis')
        # plt.title(f'Hidden {i+1}')
        plt.axis('off')

    # plt tight_layout()
    # no space between subplots
    plt.subplots_adjust(wspace=0, hspace=0)
    # main title for the whole figure
    # plt.suptitle(f"Hidden Layer Weights Visualization\nHidden Neurons: {hidden_size}, Input Size: {input_size}", fontsize=16)
    # make main title higher such that not overlapping with subplots
    plt.suptitle(f"Hidden Layer Weights Visualization\nHidden Neurons: {hidden_size}, Input Size: {input_size}", fontsize=16, y=1.02)

    plt.tight_layout()
    # Save the figure of PC_weights along with the MLP weights
    plt.savefig(f"{dir_path}/PC_weights.png")
    plt.close()

    # Return the model and test accuracies
    return mlp, test_acc

# End of file scr3/train_mlp
# End of recent edits
# End of file scr3/train_mlp 


