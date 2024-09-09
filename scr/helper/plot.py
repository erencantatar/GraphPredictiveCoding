import matplotlib.pyplot as plt
import numpy as np
from torch_geometric.utils import to_dense_adj, dense_to_sparse

def plot_adj_matrix(edge_index, model_dir, overlay=None):
    # Convert edge_index to adjacency matrix
    adj_matrix_pyg = to_dense_adj(edge_index)[0].numpy()

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(30, 18))

    # Plot the adjacency matrix
    cax = ax.imshow(adj_matrix_pyg, cmap='viridis')
    ax.set_title("Adjacency Matrix with Sensory, Internal, and Supervision Blocks")
    fig.colorbar(cax, ax=ax)

    if overlay:
        sensory_indices, internal_indices, supervision_indices = overlay
        # Overlay blocks for sensory, internal, and supervision nodes
        def overlay_block(indices, color, label):
            for idx in indices:

                # print("idxx overlay", idx)
                if isinstance(idx, (torch.Tensor, np.ndarray)):  # Convert if it's a tensor or array
                    idx = idx.item()  # Extract the integer value
                elif isinstance(idx, tuple):  # Handle if idx is a tuple
                    idx = idx[0]  # Choose the first element, or modify based on your data structure
                ax.add_patch(plt.Rectangle((idx, idx), 1, 1, fill=False, edgecolor=color, lw=2))
            # Adding legend entries for each block
            ax.plot([], [], color=color, label=label)

        # Overlay sensory block
        overlay_block(sensory_indices, 'red', 'Sensory Nodes')

        # Overlay internal block
        overlay_block(internal_indices, 'blue', 'Internal Nodes')

        # Overlay supervision block
        overlay_block(supervision_indices, 'green', 'Supervision Nodes')

        # Add legend on top
        ax.legend(loc='upper right')


    # Save the figure to model_dir
    plt.tight_layout()
    
    if overlay:
        fig.savefig(f'{model_dir}/adj_matrix_overlay.png')
    else:
        fig.savefig(f'{model_dir}/adj_matrix_.png')

    plt.close(fig)



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



def plot_energy_during_training(internal_energy, sensory_energy, history,model_dir, epoch):



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

    # Plot energy per epoch with two y-axes
    ax["B"].plot(history["internal_energy_per_epoch"], label="Internal Energy", color='blue')
    ax["B"].set_xlabel("Epochs")
    ax["B"].set_ylabel("Internal Energy", color='blue')
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
    # plt.show()

    fig.savefig(f'{model_dir}/energy/energy_{epoch}.png')
    plt.close(fig)







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

def plot_model_weights(model, model_dir):
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
    fig.suptitle('Visualization of Weight Matrix', fontsize=20)

    # Subplot 1: Sparsity pattern
    axes[0, 0].spy(W_sparse, markersize=1)
    axes[0, 0].set_title('Sparsity Pattern of the Weight Matrix')

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
    save_path = model_dir

    plt.savefig(save_path)
    plt.close(fig)

    print(f'Figure saved to {save_path}')
