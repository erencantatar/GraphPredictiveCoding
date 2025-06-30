# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torchvision
# from torch.utils.data import DataLoader
# from torchvision import transforms
# import wandb


# # args = {    "discriminative_hidden_layers": [32, 16],  # Example hidden layers
# #             "batch_size": 64,
# #             "epochs": 10,
# #             "lr_weights": 0.001,
# #             "w_decay_lr_weights": 0.0001,
# #             "activation_func": "relu",  # Options: "relu", "swish", "tanh", "sigmoid"
# #             "seed": 42,
# #             "graph_type": "mlp"  # Example graph type
# #         }


# def train_mlp_classifier(PC_weights, hidden_layer, activation_fn, 
#                base_train_dataset, base_test_dataset, batch_size, epochs,
#                 lr_w, weight_decay, seed, graph_type):
#     # Set random seed for reproducibility
#     torch.manual_seed(seed)

#     # ----------------- Dataset ----------------- #
#     # base dataset already has the same transformations applied
#     train_loader = DataLoader(base_train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
#     test_loader = DataLoader(base_test_dataset, batch_size=batch_size, shuffle=False)

#     # ----------------- Hyperparams from args ----------------- #
#     input_size = 784
#     hidden_layers = hidden_layer  # e.g., [32, 16]
#     output_size = 10
#     batch_size = batch_size
#     epochs = epochs
#     learning_rate = lr_w  # for backprop
#     weight_decay = weight_decay
#     activation_fn = activation_fn.lower()  # "relu", "swish", etc.
#     DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # wandb.init(project="MLP_MNIST", name="MLP_discrim_baseline", config={
#     #     "hidden_layers": hidden_layers,
#     #     "lr": learning_rate,
#     #     "weight_decay": weight_decay,
#     #     "activation": activation_fn
#     # })

#     # wandb.disable_wandb()

#     # ----------------- Activation Selection ----------------- #
#     def get_activation(act):
#         if act == "swish":
#             return lambda x: x * torch.sigmoid(x)
#         elif act == "relu":
#             return nn.ReLU()
#         elif act == "tanh":
#             return nn.Tanh()
#         elif act == "sigmoid":
#             return nn.Sigmoid()
#         else:
#             raise ValueError(f"Unsupported activation: {act}")

#     activation = get_activation(activation_fn.lower())

#     # ----------------- MLP Model ----------------- #
#     layers = [nn.Flatten()]
#     prev = input_size
#     for h in hidden_layers:
#         layers.append(nn.Linear(prev, h))
#         layers.append(activation if isinstance(activation, nn.Module) else nn.ReLU())  # swish is function
#         prev = h
#     layers.append(nn.Linear(prev, output_size))
#     mlp = nn.Sequential(*layers).to(DEVICE)

#     # ----------------- Optimizer and Loss ----------------- #
#     optimizer = torch.optim.Adam(mlp.parameters(), lr=learning_rate, weight_decay=weight_decay)
#     criterion = nn.CrossEntropyLoss()


#     # ----------------- Training Loop ----------------- #
#     test_acc = []
#     for epoch in range(epochs):
#         mlp.train()
#         for X, y in train_loader:
#             X, y = X.to(DEVICE), y.to(DEVICE)
#             optimizer.zero_grad()
#             out = mlp(X)
#             loss = criterion(out, y)
#             loss.backward()
#             optimizer.step()

#         # Validation
#         mlp.eval()
#         correct = 0
#         total = 0
#         with torch.no_grad():
#             for X, y in test_loader:
#                 X, y = X.to(DEVICE), y.to(DEVICE)
#                 out = mlp(X)
#                 preds = out.argmax(dim=1)
#                 correct += (preds == y).sum().item()
#                 total += y.size(0)
#         acc = correct / total
#         test_acc.append(acc)
#         wandb.log({"epoch": epoch, "val_accuracy": acc, "train_loss": loss.item()})
#         print(f"Epoch {epoch}: Acc={acc:.4f} | Loss={loss.item():.4f}")

#     # ----------------- Save ----------------- #
#     # torch.save(mlp.state_dict(), f"trained_models/{graph_type}/weights/mlp_last.pt")
#     # wandb.save(f"trained_models/{graph_type}/weights/mlp_last.pt")
#     # wandb.finish()

#     # create new folder in 'scr3/jobs"
#     import os

#     # Ensure directory exists
#     # dir path include training parameters: mlp_{layers}_  epochs, lr_w, w_decay_lr_w, activation_fn, batch_size, seed
#     unpack_layers = "_".join(map(str, layers))
#     # make dir_path have all the parameters but string as short as possible
#     dir_path = f"trained_models/mlp/weights/mlp_{unpack_layers}_ep_{epochs}_bs_{batch_size}_lr_{learning_rate}_wd_{weight_decay}_{activation_fn}_seed_{seed}"
    
#     os.makedirs(dir_path, exist_ok=True)

#     os.makedirs(f"{dir_path}/weights", exist_ok=True)
#     # os.makedirs(f"scr3/jobs/{graph_type}/weights", exist_ok=True)
#     # torch.save(mlp.state_dict(), f"scr3/jobs/{graph_type}/weights/mlp_last.pt")
#     torch.save(mlp.state_dict(), f"{dir_path}/weights/mlp_last.pt")
#     print(f"Model saved to scr3/jobs/{graph_type}/weights/mlp_last.pt") 

#     # in a file called "test_acc.txt" in the same folder
#     with open(f"{dir_path}/test_acc.txt", "w") as f:
#         for acc in test_acc:
#             f.write(f"{acc}\n")
#     print(f"Test accuracies saved to {dir_path}/test_acc.txt")

#     # save fig
#     # prompt: plot the model weights, in a imshow

#     import matplotlib.pyplot as plt

#     # Extract the weights of the first linear layer (after Flatten)
#     first_linear_layer = None
#     for layer in mlp.modules():
#         if isinstance(layer, nn.Linear):
#             first_linear_layer = layer
#             break

#     if first_linear_layer is not None:
#         weights = first_linear_layer.weight.cpu().detach().numpy()

#         # Since the input is flattened MNIST (28x28 = 784), we can reshape the weights
#         # for the first layer to visualize them as images.
#         # The weights have shape (output_features, input_features).
#         # We can reshape each row (which corresponds to the weights connecting to one output neuron)
#         # into a 28x28 image.
#         num_output_features = weights.shape[0]
#         num_input_features = weights.shape[1] # Should be 784 for the first linear layer

#         if num_input_features == 784:
#             print(f"Plotting weights for the first linear layer ({num_output_features} output features).")
#             # Determine the grid size for plotting
#             grid_cols = min(num_output_features, 8) # Plot up to 8 columns
#             grid_rows = (num_output_features + grid_cols - 1) // grid_cols

#             plt.figure(figsize=(grid_cols * 2, grid_rows * 2))

#             for i in range(num_output_features):
#                 plt.subplot(grid_rows, grid_cols, i + 1)
#                 # Reshape the weights for this output neuron into a 28x28 image
#                 weight_image = weights[i].reshape(28, 28)
#                 plt.imshow(weight_image, cmap='viridis') # Use 'viridis' or another colormap
#                 plt.title(f'Neuron {i+1}')
#                 plt.axis('off')

#             plt.tight_layout()
#             # plt.show()
#             # plt.savefig to dir_path
#             plt.savefig(f"{dir_path}/mlp_weights.png")

#             # plt.savefig(f"scr3/jobs/{graph_type}/weights/mlp_weights.png")
#             print(f"Weights image saved to scr3/jobs/{graph_type}/weights/mlp_weights.png")
#             # close 
#             plt.close()

#         else:
#             print(f"First linear layer has input features {num_input_features}, not 784. Cannot reshape to 28x28 for imshow.")
#     else:
#         print("No linear layer found in the model after Flatten.")

#     import numpy as np
#     import matplotlib.pyplot as plt
#     import os


#     # Calculate the total number of neurons
#     total_neurons = input_size + sum(hidden_layers) + output_size

#     # Create adjacency matrix
#     adj_matrix = np.zeros((total_neurons, total_neurons))

#     # Index tracking
#     current_idx = 0
#     input_indices = list(range(current_idx, current_idx + input_size))
#     current_idx += input_size

#     hidden_layer_indices = []
#     for h in hidden_layers:
#         indices = list(range(current_idx, current_idx + h))
#         hidden_layer_indices.append(indices)
#         current_idx += h

#     output_indices = list(range(current_idx, current_idx + output_size))

#     # Wiring: input -> hidden -> output
#     layer_input_indices = input_indices
#     layer_output_indices_list = hidden_layer_indices + [output_indices]

#     for layer in mlp.modules():
#         if isinstance(layer, nn.Linear):
#             weights = layer.weight.cpu().detach().numpy().T  # ← TRANSPOSE HERE
#             current_layer_output_indices = layer_output_indices_list.pop(0)

#             for i_in, from_idx in enumerate(layer_input_indices):
#                 for i_out, to_idx in enumerate(current_layer_output_indices):
#                     adj_matrix[from_idx, to_idx] = weights[i_in, i_out]

#             layer_input_indices = current_layer_output_indices

#             # Plot
#             plt.figure(figsize=(10, 10))
#             plt.imshow(adj_matrix, cmap='viridis', origin='upper')
#             plt.title('Transposed Weights as Adjacency Matrix')
#             plt.xlabel('To Neuron Index')
#             plt.ylabel('From Neuron Index')
#             plt.colorbar(label='Weight Value')
#             plt.tight_layout()

#             # save fig to folder dir
#             plt.savefig(f"{dir_path}/mlp_weights_adj_matrix.png")
#             print(f"Adjacency matrix saved to trained_models/mlp/weights/mlp_weights_adj_matrix.png")
#             plt.close()


#             # compare the matrix "weights" with the "PC_weights" by calculating the abs avg difference in percentage
#             if PC_weights is not None:
#                 PC_weights = PC_weights.cpu().detach().numpy()
#                 # Calculate the absolute average difference in percentage
#                 abs_diff = np.abs(weights - PC_weights)
#                 avg_diff = np.mean(abs_diff)
#                 avg_diff_percentage = (avg_diff / np.mean(np.abs(PC_weights))) * 100

#                 print(f"Average absolute difference between model weights and PC weights: {avg_diff_percentage:.2f}%")
#             else:
#                 print("PC_weights is None, skipping comparison.")

#             # log the average difference to txt file 
#             with open(f"{dir_path}/avg_diff.txt", "w") as f:
#                 f.write(f"Average absolute difference between model weights and PC weights: {avg_diff_percentage:.2f}%\n")


#     # Return the model and test accuracies
#     return mlp, test_acc

# # End of file scr3/train_mlp
# # End of recent edits
# # End of file scr3/train_mlp 


