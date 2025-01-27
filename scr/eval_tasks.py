
import torch
import torchvision
import torchvision.transforms as transforms

import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch_geometric.data import Data

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

import torch.nn.init as init
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
# from icecream import ic

import matplotlib.pyplot as plt 

from torch_scatter import scatter_mean
from torch_geometric.data import Data
from torch.utils.data import Dataset

from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
import random

from torch_geometric.nn import MessagePassing
import torch.nn.functional as F
import os
import wandb
    
import gc 
from helper.error import calculate_mse

# maps = RdBu, RdBu_r, RdBu_r, RdBu, BrBG, BrBG_r, coolwarm, coolwarm_r, bwr, bwr_r, seismic, seismic_r
# cmap = "RdBu_r"
cmap = "gray"

# -----------generation--------------
from skimage.metrics import structural_similarity as ssim
    
def mse(imageA, imageB):
    """Calculate the Mean Squared Error (MSE) between two images."""
    return np.mean((imageA - imageB) ** 2)

# --------classification------------
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

"""
def task():
- comment

- setup:
    - restart_act()
    - pc_conv1.set_mode("testing", task="")
    - set nodes_2_update
    - trace = []
    - metric = []

- for img, clean in loader:

    - alter sensory_: value, error, preds

    - alter internal: value, error, preds

    - set supervision nodes (already done in dataset)
        (only remove label if needed)

    - Inference 

    - metric 

    - log 

    - plot

    - return metric
"""


"""
dynamic_task is a decorator function that dynamically sets up 
a task for a model based on the function name, allowing for 
flexible and modular task-specific configurations
"""
def dynamic_task(func):
    def wrapper(test_loader, model, test_params, *args, **kwargs):
        # Dynamically set the task based on the function name
        task_name = func.__name__

        # base setup
        global cmap  # Ensures the wrapper can access the global cmap variable
        model.pc_conv1.debug = False # True 

        model.pc_conv1.batchsize = 1
        model.pc_conv1.restart_activity()
        model.pc_conv1.set_mode("testing", task=task_name)

        # Default trace setup for most tasks
        model.pc_conv1.trace['values'] = []
        model.pc_conv1.trace['preds'] = []

        # Dynamically set T if present in test_params
        if "T" in test_params:
            model.pc_conv1.T = test_params["T"]

        #------task specifics------ 
        if task_name in ["generation"]:
            # model.pc_conv1.nodes_2_update = # all good
            pass 
        if task_name in ["occulusion"]:
            model.pc_conv1.nodes_2_update = list(model.pc_conv1.sensory_indices[len(model.pc_conv1.sensory_indices)//2:]) +  list(model.pc_conv1.internal_indices)

        # Pass along any additional arguments or task-specific tweaks
        return func(test_loader, model, test_params, *args, **kwargs)

    return wrapper


@dynamic_task
def occlusion(test_loader, model, test_params, verbose=0):


    """We compare the generated number with the first N=10 clean images of the same class,
    using SSIM, and MSE metric. Better generation is higher SSIM (0-1), and lower MSE (0,+inf)
    +- 2sec    
    """
    print("IMPORTANT SEE OLD OCCLUSION func; ")

    # model.pc_conv1.nodes_2_update = list(model.pc_conv1.sensory_indices[len(model.pc_conv1.sensory_indices)//2:]) +  list(model.pc_conv1.internal_indices)

    # assert test_params["supervised_learning"] == True 

    torch.cuda.empty_cache()
    gc.collect()

    print("No vmin vmax")

    avg_SSIM_mean, avg_SSIM_max = [], []    
    avg_MSE_mean, avg_MSE_max   = [], []
 
    MSE_values = []

    for idx, (noisy_batch, clean_image) in enumerate(test_loader, start=1):

        # noisy_batch, clean_image = noisy_batch[0], clean_image[0]
        # print(noisy_batch.y)

        # Perform inference to denoise
        noisy_batch = noisy_batch.to(model.pc_conv1.device)

        noisy_batch.x[:, 0][model.pc_conv1.internal_indices] = torch.rand(noisy_batch.x[:, 0][model.pc_conv1.internal_indices].shape).to(model.pc_conv1.device)
        # noisy_batch.x[:, 1][model.pc_conv1.internal_indices]  = 0
        # noisy_batch.x[:, 2][model.pc_conv1.internal_indices]  = 0

        white = torch.ones_like(noisy_batch.x)
        black = torch.zeros_like(noisy_batch.x[:, 0][0:-10])

        random = torch.rand(noisy_batch.x[:, 0][0:-10].shape)

        noisy_batch.x[:, 0][784 // 2: ] = 0 
        # noisy_batch.x[:, 0][model.pc_conv1.sensory_indices] = torch.rand(noisy_batch.x[:, 0][model.pc_conv1.sensory_indices].shape).to(model.pc_conv1.device)
    
         
        if test_params["add_sens_noise"]:
            noisy_batch.x[:, 0][784 // 2: ] = torch.rand( noisy_batch.x[:, 0][784 // 2: ].shape) 

    
        # noisy_batch.x[:, 2][model.pc_conv1.sensory_indices] = torch.rand(noisy_batch.x[:, 2][model.pc_conv1.sensory_indices].shape).to(model.pc_conv1.device)
        # noisy_batch.x[:, 0][0:-10] = random
        
        if model.pc_conv1.trace_activity_preds:
            model.pc_conv1.trace["preds"].append(noisy_batch.x[:, 2][0:784].detach())
        if model.pc_conv1.trace_activity_values:
            model.pc_conv1.trace["values"].append(noisy_batch.x[:, 0][0:784].detach())
        
        #### SURVERVISED ##########
        # we can alter the supervision signal during testing as an experiment
        if test_params["supervised_learning"]:
            # correct supervision label already set in dataset
            # if during training the label for the supervised node was x, also here
            # model.pc_conv1.values[model.pc_conv1.supervised_labels].data = noisy_batch.x[model.pc_conv1.supervised_labels]
            pass
        else:
            noisy_batch.x[:, 0][model.pc_conv1.supervised_labels] = 0
        
        print("labels model", noisy_batch.x[:, 0][model.pc_conv1.supervised_labels] )

        # model.pc_conv1.values[model.pc_conv1.supervised_labels] = noisy_batch.x[model.pc_conv1.supervised_labels]
        # model.inference()
        
        noisy_image = noisy_batch.x[:, 0][0:784].view(28,28).cpu().detach().numpy()

        values, predictions, labels = model.query(method="pass", 
                                                  random_internal=True,
                                                  data=noisy_batch)  # query_by_conditioning


        # values, predictions = values[batch_idx, :, 0], predictions[batch_idx, :, 0]
        # print("CHECK", noisy_batch.x[:,0][-10:])
        
        denoised_output = predictions[0:784].view(28,28).cpu().detach().numpy()

        clean_image = clean_image.view(28,28).cpu().numpy()  # Adjust shape as necessary

 
        # Plotting both images side by side
    

        fig, ax = plt.subplot_mosaic([
            ["A", "B", "C", "D", "E"],   # First row for clean image, noisy image, and denoised output
            ["1", "2", "3", "4", "5"],   # Second row for values with label
            ["H", "I", "J", "K", "L"],   # Third row for predictions with label
            ["F", "F", "F", "G", "G"],   # Fourth row for energy plots and additional info
        ], figsize=(15, 10))

        # Adding text labels to the left of the rows for values and predictions
        fig.text(0.02, 0.67, "Values [0-T]", ha='center', va='center', fontsize=12, rotation='vertical', fontweight='bold')
        fig.text(0.02, 0.47, "error/preds [0-T]", ha='center', va='center', fontsize=12, rotation='vertical', fontweight='bold')
        fig.text(0.02, 0.87, "at t=0", ha='center', va='center', fontsize=12, rotation='vertical', fontweight='bold')

        # Plotting the images
        ax["A"].imshow(clean_image, vmin=0, vmax=1, cmap=cmap)
        ax["A"].set_title(f"Clean Image of a {noisy_batch.y.item()}")

        ax["B"].imshow(noisy_image, vmin=0, vmax=1, cmap=cmap)
        ax["B"].set_title("Noisy Input")

        ax["C"].imshow(denoised_output, vmin=0, vmax=1, cmap=cmap)
        ax["C"].set_title("Predic. at T")

        ax["D"].imshow(denoised_output, cmap=cmap)
        ax["D"].set_title("Predic. at T, no vmin vmax")
        print("Predic. at T, no vmin vmax", max(denoised_output.flatten()), min(denoised_output.flatten()))

        denoised_output_scaled = (denoised_output - min(denoised_output.flatten())) / (max(denoised_output.flatten()) - min(denoised_output.flatten()))
        ax["E"].imshow(clean_image - denoised_output_scaled, vmin=0, vmax=1, cmap=cmap)
        ax["E"].set_title("Diff clean - denoised_scaled")
        print("Denoised val", max(values[0:784].view(28, 28).cpu().detach().numpy().flatten()), min(values[0:784].view(28, 28).cpu().detach().numpy().flatten()))

        # Plotting the values
        tr = model.pc_conv1.trace["values"]
        tmp = int(model.pc_conv1.T // 5)

        ax["1"].imshow(tr[0][0:784].view(28,28).cpu().detach().numpy(), cmap=cmap)
        ax["2"].imshow(tr[1][0:784].view(28,28).cpu().detach().numpy(), cmap=cmap)
        ax["3"].imshow(tr[2*tmp][0:784].view(28,28).cpu().detach().numpy(), cmap=cmap)
        ax["4"].imshow(tr[-2*tmp][0:784].view(28,28).cpu().detach().numpy(), cmap=cmap)
        ax["5"].imshow(tr[-1][0:784].view(28,28).cpu().detach().numpy(), cmap=cmap)

        # Plotting the predictions
        tr_preds = model.pc_conv1.trace["preds"]
        ax["H"].imshow(tr_preds[0][0:784].view(28,28).cpu().detach().numpy(), cmap=cmap)
        ax["I"].imshow(tr_preds[1][0:784].view(28,28).cpu().detach().numpy(), cmap=cmap)
        ax["J"].imshow(tr_preds[2*tmp][0:784].view(28,28).cpu().detach().numpy(), cmap=cmap)
        ax["K"].imshow(tr_preds[-2*tmp][0:784].view(28,28).cpu().detach().numpy(), cmap=cmap)
        ax["L"].imshow(tr_preds[-1][0:784].view(28,28).cpu().detach().numpy(), cmap=cmap)

        # Hide axis for all image subplots
        for a in ["A", "B", "C", "D", "E", "1", "2", "3", "4", "5", "H", "I", "J", "K", "L", "G"]:
            ax[a].axis('off')

        # Apply tight layout
        plt.tight_layout()

        # Adjust subplot spacing if needed
        plt.subplots_adjust(wspace=0.05, hspace=0.05)  # Adjust spacing between subplots

        # Plotting the line graphs
        ax["F"].plot(model.pc_conv1.energy_vals["internal_energy_testing"][-model.pc_conv1.T:], label="Internal energy")
        ax["F"].plot(model.pc_conv1.energy_vals["sensory_energy_testing"][-model.pc_conv1.T:], label="Sensory energy")  # Replace with actual values
        ax["F"].legend()


        ax["G"].imshow(values[0:784].view(28,28).cpu().detach().numpy(), 
                    #    vmin=0, vmax=1, 
                       cmap=cmap)
        ax["G"].set_title("values no vmix/max")

        # Apply tight layout
        plt.tight_layout()

        # Adjust subplot spacing if needed
        plt.subplots_adjust(wspace=0.05, hspace=0.05)  # Adjust spacing between subplots

        if test_params["model_dir"] and test_params["num_wandb_img_log"] < idx:
            fig.savefig(f'{test_params["model_dir"]}eval/generation/generation_condition_label_T_{idx}_{model.pc_conv1.T}_{noisy_batch.y.item()}.png')


        if test_params["num_wandb_img_log"] < idx:
            # log fig to wandb
            if test_params["supervised_learning"]:
                wandb.log({"occlusion_sup/occlusion_IMG": wandb.Image(fig)})
            else:
                wandb.log({"occlusion_unsup/occlusion_IMG": wandb.Image(fig)})
            plt.close(fig)

        if idx >= test_params["num_samples"]:
           
            return MSE_values

    
@dynamic_task
def denoise(test_loader, model, test_params, sigma=0.1):
    
    model.pc_conv1.debug = False
    model.pc_conv1.set_mode("testing", task="reconstruction")

    # model.pc_conv1.nodes_2_update +=  list(model.pc_conv1.supervised_labels)

    model.pc_conv1.restart_activity()
    model.pc_conv1.set_mode("testing", task="reconstruction")
    model.pc_conv1.batchsize = 1

    MSE_values = []

    for idx, (noisy_batch, clean_image) in enumerate(test_loader, start=1):

        # Perform inference to denoise
        noisy_batch = noisy_batch.to(model.pc_conv1.device)


        noisy_batch.x[:, 0][model.pc_conv1.internal_indices] = torch.rand(noisy_batch.x[:, 0][model.pc_conv1.internal_indices].shape).to(model.pc_conv1.device)

        noisy_batch.x[:, 0][model.pc_conv1.sensory_indices ] += torch.rand(noisy_batch.x[:, 0][model.pc_conv1.sensory_indices].shape).to(model.pc_conv1.device)

        #### SURVERVISED ##########
        if test_params["supervised_learning"]:
            # if during training the label for the supervised node was 60, also here
            # model.pc_conv1.values[model.pc_conv1.supervised_labels].data = noisy_batch.x[model.pc_conv1.supervised_labels]
            pass
        else:
            noisy_batch.x[:, 0][model.pc_conv1.supervised_labels] = 0
        
        
        print("labels model", noisy_batch.x[:, 0][model.pc_conv1.supervised_labels] )

        # model.inference()
        
        # remove label
        # noisy_batch.x[:, 0][-10:] = 0

        # noisy_batch.x[:, 0][0:784] += 2 * torch.rand(noisy_batch.x[:, 0][0:784].shape).to(device)

        # Extract the denoised output from the sensory nodes
        noisy_image = noisy_batch.x[:, 0][0:784].view(28,28).cpu().detach().numpy()
        
        
        values, predictions, labels = model.query(method="pass", 
                                                random_internal=True,
                                                  data=noisy_batch)  # query_by_conditioning
        # values, predictions = values[batch_idx, :, 0], predictions[batch_idx, :, 0]
        
        denoised_output = predictions[0:784].view(28,28).cpu().detach().numpy()

        # denoised_output = model.reconstruction().view(28,28).cpu().detach().numpy()

        clean_image = clean_image.view(28,28).cpu().numpy()  # Adjust shape as necessary


        # Creating a subplot mosaic
        fig, ax = plt.subplot_mosaic([
            ["A", "B", "C", "D", "E"],
            ["F", "F", "F", "G", "G"]
        ], figsize=(15, 8)) 

        MSE = round(calculate_mse(clean_image, denoised_output), 4)
        fig.suptitle(f"MSE {MSE} clean/denoised_output")



        # Plotting the images
        ax["A"].imshow(clean_image, vmin=0, vmax=1, cmap=cmap)
        ax["A"].set_title(f"Clean Image of a {noisy_batch.y.item()}")

        ax["B"].imshow(noisy_image, vmin=0, vmax=1, cmap=cmap)
        ax["B"].set_title("Noisy Input")

        ax["C"].imshow(denoised_output, vmin=0, vmax=1, cmap=cmap)
        ax["C"].set_title("Predic. at T")

        ax["D"].imshow(denoised_output, cmap=cmap)
        ax["D"].set_title("Predic. at T, no vmin vmax")
        print("Predic. at T, no vmin vmax", max(denoised_output.flatten()), min(denoised_output.flatten()))

        denoised_output_scaled = (denoised_output - min(denoised_output.flatten())) / (max(denoised_output.flatten()) - min(denoised_output.flatten()))
        ax["E"].imshow(clean_image - denoised_output_scaled, vmin=0, vmax=1, cmap=cmap)
        ax["E"].set_title("Diff clean - denoised_scaled")
        print("Denoised val", max(values[0:784].view(28, 28).cpu().detach().numpy().flatten()), min(values[0:784].view(28, 28).cpu().detach().numpy().flatten()))

        for a in ["A", "B", "C", "D", "E", "G"]:
            ax[a].axis('off')

        # Plotting the line graphs
        ax["F"].plot(model.pc_conv1.energy_vals["internal_energy"][-model.pc_conv1.T:], label="Internal energy")
        ax["F"].plot(model.pc_conv1.energy_vals["sensory_energy"][-model.pc_conv1.T:], label="Sensory energy")  # Replace with actual values
        ax["F"].legend()


        ax["G"].imshow(values[0:784].view(28,28).cpu().detach().numpy(), cmap=cmap)
        ax["G"].set_title("value")

        if test_params["model_dir"] and test_params["num_wandb_img_log"] < idx:
            fig.savefig(f'{test_params["model_dir"]}eval/denoise/denoise_{idx}_T_{model.pc_conv1.T}_{noisy_batch.y.item()}.png')

        labels = values[model.pc_conv1.supervised_labels]
        print(labels)

        difference_image = clean_image - denoised_output_scaled

        MSE_values.append(MSE)

        if test_params["num_wandb_img_log"] < idx:
            # log fig to wandb
            if test_params["supervised_learning"]:
                wandb.log({"denoise_sup/denoise_IMG": wandb.Image(fig)})
            else:
                wandb.log({"denoise_unsup/denoise_IMG": wandb.Image(fig)})

            plt.close(fig)

        if idx >= test_params["num_samples"]:
            break 

    return MSE_values


@dynamic_task
def generation(test_loader, model, test_params, clean_images, num_samples=8, verbose=0):


    """We compare the generated number with the first N=10 clean images of the same class,
    using SSIM, and MSE metric. Better generation is higher SSIM (0-1), and lower MSE (0,+inf)
    +- 2sec    
    """

    # ------------ Setup done by Decorator --------------
    # 
    assert test_params["supervised_learning"] == True, "Need to know what num. to generate"

    avg_SSIM_mean, avg_SSIM_max = [], []    
    avg_MSE_mean, avg_MSE_max   = [], []
    torch.cuda.empty_cache()
    gc.collect()
    # ---------------------------------

    
    for idx, (noisy_batch, clean_image) in enumerate(test_loader, start=1):


        # remove the whole thing (also label)

        noisy_batch = noisy_batch.to(model.pc_conv1.device)
        clean_image = clean_image.view(28,28).cpu().numpy()  # Adjust shape as necessary
        noisy_image = noisy_batch.x[:, 0][0:784].view(28,28).cpu().detach().numpy()


        # ------------ Alter sensory --------------
        # white = torch.ones_like(noisy_batch.x)
        # black = torch.zeros_like(noisy_batch.x[:, 0][0:-10])
        # random = torch.rand(noisy_batch.x[:, 0][0:-10].shape)
        # values 
        noisy_batch.x[:, 0][model.pc_conv1.sensory_indices] = torch.rand(noisy_batch.x[:, 0][model.pc_conv1.sensory_indices].shape).to(model.pc_conv1.device)
        # errors 
        noisy_batch.x[:, 1][model.pc_conv1.sensory_indices] = torch.zeros_like(noisy_batch.x[:, 1][model.pc_conv1.sensory_indices]).to(model.pc_conv1.device)
        # preds 
        noisy_batch.x[:, 2][model.pc_conv1.sensory_indices] = torch.zeros_like(noisy_batch.x[:, 2][model.pc_conv1.sensory_indices]).to(model.pc_conv1.device)
    
        
        if model.pc_conv1.trace_activity_preds:
            model.pc_conv1.trace["preds"].append(noisy_batch.x[:, 2][0:784].detach())
        if model.pc_conv1.trace_activity_values:
            model.pc_conv1.trace["values"].append(noisy_batch.x[:, 0][0:784].detach())

        plt.imshow(model.pc_conv1.trace["values"][0][0:784].view(28,28).cpu())
        plt.imshow(model.pc_conv1.trace["preds"][0][0:784].view(28,28).cpu())
        # plt.show()

        # ------------ Alter internal --------------
        # ... 

        # ------------ Alter supervision --------------
        #### SURVERVISED ##########
        # we can alter the supervision signal during testing as an experiment
        if test_params["supervised_learning"]:
            # correct supervision label already set in dataset
            # if during training the label for the supervised node was x, also here
            # model.pc_conv1.values[model.pc_conv1.supervised_labels].data = noisy_batch.x[model.pc_conv1.supervised_labels]
            pass
        else:
            noisy_batch.x[:, 0][model.pc_conv1.supervised_labels] = 0
        
        print("labels model", noisy_batch.x[:, 0][model.pc_conv1.supervised_labels] )

        
        # ----------- Inference ---------------
        # output
        values, predictions, labels = model.query(method="pass", data=noisy_batch)  # query_by_conditioning
        denoised_output = predictions[0:784].view(28,28).cpu().detach().numpy()


        # ----------- Plotting ---------------
        
        # Creating a subplot mosaic with the row for values and a row for predictions
        fig, ax = plt.subplot_mosaic([
            ["A", "B", "C", "D", "E"],   # First row for clean image, noisy image, and denoised output
            ["1", "2", "3", "4", "5"],   # Second row for values with label
            ["H", "I", "J", "K", "L"],   # Third row for predictions with label
            ["F", "F", "F", "G", "G"],   # Fourth row for energy plots and additional info
        ], figsize=(15, 10))

        # Adding text labels to the left of the rows for values and predictions
        fig.text(0.02, 0.67, "Values [0-T]", ha='center', va='center', fontsize=12, rotation='vertical', fontweight='bold')
        fig.text(0.02, 0.47, "error/preds [0-T]", ha='center', va='center', fontsize=12, rotation='vertical', fontweight='bold')
        fig.text(0.02, 0.87, "at t=0", ha='center', va='center', fontsize=12, rotation='vertical', fontweight='bold')

        # Plotting the images
        ax["A"].imshow(clean_image, vmin=0, vmax=1, cmap=cmap)
        ax["A"].set_title(f"Clean Image of a {noisy_batch.y.item()}")

        ax["B"].imshow(noisy_image, vmin=0, vmax=1, cmap=cmap)
        ax["B"].set_title("Noisy Input")

        ax["C"].imshow(denoised_output, vmin=0, vmax=1, cmap=cmap)
        ax["C"].set_title("Predic. at T")

        ax["D"].imshow(denoised_output, cmap=cmap)
        ax["D"].set_title("Predic. at T, no vmin vmax")
        print("Predic. at T, no vmin vmax", max(denoised_output.flatten()), min(denoised_output.flatten()))

        denoised_output_scaled = (denoised_output - min(denoised_output.flatten())) / (max(denoised_output.flatten()) - min(denoised_output.flatten()))
        ax["E"].imshow(clean_image - denoised_output_scaled, vmin=0, vmax=1, cmap=cmap)
        ax["E"].set_title("Diff clean - denoised_scaled")
        print("Denoised val", max(values[0:784].view(28, 28).cpu().detach().numpy().flatten()), min(values[0:784].view(28, 28).cpu().detach().numpy().flatten()))

        # Plotting the values
        tr = model.pc_conv1.trace["values"]

        # step_size
        tmp = int(model.pc_conv1.T // 5)

        ax["1"].imshow(tr[0][0:784].view(28,28).cpu().detach().numpy(), cmap=cmap)
        ax["2"].imshow(tr[tmp][0:784].view(28,28).cpu().detach().numpy(), cmap=cmap)
        ax["3"].imshow(tr[2*tmp][0:784].view(28,28).cpu().detach().numpy(), cmap=cmap)
        ax["4"].imshow(tr[-2*tmp][0:784].view(28,28).cpu().detach().numpy(), cmap=cmap)
        ax["5"].imshow(tr[-1][0:784].view(28,28).cpu().detach().numpy(), cmap=cmap)

        # Plotting the predictions
        tr_preds = model.pc_conv1.trace["preds"]
        ax["H"].imshow(tr_preds[0][0:784].view(28,28).cpu().detach().numpy(), cmap=cmap)
        ax["I"].imshow(tr_preds[tmp][0:784].view(28,28).cpu().detach().numpy(), cmap=cmap)
        ax["J"].imshow(tr_preds[2*tmp][0:784].view(28,28).cpu().detach().numpy(), cmap=cmap)
        ax["K"].imshow(tr_preds[-2*tmp][0:784].view(28,28).cpu().detach().numpy(), cmap=cmap)
        ax["L"].imshow(tr_preds[-1][0:784].view(28,28).cpu().detach().numpy(), cmap=cmap)

        # Hide axis for all image subplots
        for a in ["A", "B", "C", "D", "E", "1", "2", "3", "4", "5", "H", "I", "J", "K", "L", "G"]:
            ax[a].axis('off')

        # Apply tight layout
        plt.tight_layout()

        # Adjust subplot spacing if needed
        plt.subplots_adjust(wspace=0.05, hspace=0.05)  # Adjust spacing between subplots

        # Plotting the line graphs
        ax["F"].plot(model.pc_conv1.energy_vals["internal_energy_testing"][-model.pc_conv1.T:], label="Internal energy")
        ax["F"].plot(model.pc_conv1.energy_vals["sensory_energy_testing"][-model.pc_conv1.T:], label="Sensory energy")  # Replace with actual values
        ax["F"].legend()


        ax["G"].imshow(values[0:784].view(28,28).cpu().detach().numpy(), 
                    #    vmin=0, vmax=1, 
                       cmap=cmap)
        ax["G"].set_title("values no vmix/max")

        # Apply tight layout
        plt.tight_layout()

        # Adjust subplot spacing if needed
        plt.subplots_adjust(wspace=0.05, hspace=0.05)  # Adjust spacing between subplots

        if test_params["model_dir"] and test_params["num_wandb_img_log"] < idx:
            fig.savefig(f'{test_params["model_dir"]}eval/generation/generation_condition_label_T_{idx}_{model.pc_conv1.T}_{noisy_batch.y.item()}.png')



        # ----------- Metric ---------------
        label = noisy_batch.y.item()
        label_clean_images = clean_images[label]

        # Initialize lists for SSIM and MSE values
        ssim_values = []
        mse_values = []

        # Compare the denoised output with each clean image of the same class
        for clean_image in label_clean_images:
            # Calculate SSIM and MSE
            denoised_output = (denoised_output - denoised_output.min()) / (denoised_output.max() - denoised_output.min())
            clean_image = (clean_image - clean_image.min()) / (clean_image.max() - clean_image.min())
            ssim_index = ssim(denoised_output, clean_image, data_range=1.0)
            mse_value = mse(denoised_output, clean_image)

            ssim_values.append(ssim_index)
            mse_values.append(mse_value)

            if verbose == 1:
                print("ssim_values", ssim_values)
                print("mse_values", mse_values)

        avg_SSIM_mean.append(np.mean(ssim_values))
        avg_SSIM_max.append(np.max(ssim_values))
        avg_MSE_mean.append(np.mean(mse_values))
        avg_MSE_max.append(np.max(mse_values))
    
        if idx >= test_params["num_samples"]:
            
            return np.mean(avg_SSIM_mean), np.max(avg_SSIM_max), np.mean(avg_MSE_mean), np.max(avg_MSE_max)  

        if test_params["num_wandb_img_log"] < idx:
            # log fig to wandb
            wandb.log({"generation/generation_IMG": wandb.Image(fig)})
            plt.close(fig)



@dynamic_task
def plot_digits_vertically(test_loader, model, test_params, numbers_list):
    """
    Generate and plot digits specified in test_loader.numbers_list using the model.

    Parameters:
    - test_loader: DataLoader with numbers_list attribute containing digits to generate.
    - model: Model to use for generating the digits.
    - test_params: Parameters for the test process.
    """
    # Ensure numbers_list exists in the test_loader
    numbers_list = numbers_list

    # Prepare to store generated digits
    generated_digits = {digit: None for digit in numbers_list}
    generated_digits_normalized = {digit: None for digit in numbers_list}

    for idx, (noisy_batch, _) in enumerate(test_loader, start=1):
        # Move batch to the correct device
        noisy_batch = noisy_batch.to(model.pc_conv1.device)

        # Set random noise in sensory nodes
        noisy_batch.x[:, 0][model.pc_conv1.sensory_indices] = torch.rand(
            noisy_batch.x[:, 0][model.pc_conv1.sensory_indices].shape
        ).to(model.pc_conv1.device)
        noisy_batch.x[:, 1][model.pc_conv1.sensory_indices] = 0  # Error nodes
        noisy_batch.x[:, 2][model.pc_conv1.sensory_indices] = 0  # Prediction nodes

        # Set the supervision signal for the target digits
        target_digit = noisy_batch.y.item()
        if target_digit in numbers_list and generated_digits[target_digit] is None:
            # Perform inference to generate the digit
            values, predictions, labels = model.query(method="pass", data=noisy_batch)
            generated_image = values[0:784].view(28, 28).cpu().detach().numpy()
            generated_digits[target_digit] = generated_image

            # Normalize the image individually
            min_val = generated_image.min()
            max_val = generated_image.max()
            if max_val > min_val:  # Avoid division by zero
                generated_image_normalized = (generated_image - min_val) / (max_val - min_val)
            else:
                generated_image_normalized = generated_image
            generated_digits_normalized[target_digit] = generated_image_normalized

        # Stop if all requested digits are generated
        if all(v is not None for v in generated_digits.values()):
            break

    # Ensure all requested digits are generated
    missing_digits = [digit for digit, img in generated_digits.items() if img is None]
    if missing_digits:
        raise ValueError(f"Could not generate images for the following digits: {missing_digits}")

    # Plot the generated digits in a vertical layout
    fig, axs = plt.subplots(nrows=1, ncols=len(numbers_list), figsize=(len(numbers_list) * 2, 4))
    if len(numbers_list) == 1:  # Ensure axs is iterable for a single-column layout
        axs = [axs]

    for ax, digit in zip(axs, numbers_list):
        ax.imshow(generated_digits[digit], cmap='gray', vmin=0, vmax=1)
        ax.set_title(f"Digit: {digit}")
        ax.axis('off')

    plt.tight_layout()
    wandb.log({"generation/generated_num_list": wandb.Image(fig)})
    plt.close(fig)

    # Plot the normalized generated digits in a vertical layout
    fig_normalized, axs_normalized = plt.subplots(nrows=1, ncols=len(numbers_list), figsize=(len(numbers_list) * 2, 4))
    if len(numbers_list) == 1:  # Ensure axs is iterable for a single-column layout
        axs_normalized = [axs_normalized]

    for ax, digit in zip(axs_normalized, numbers_list):
        ax.imshow(generated_digits_normalized[digit], cmap='gray', vmin=0, vmax=1)
        ax.set_title(f"Digit: {digit} (Normalized)")
        ax.axis('off')

    plt.tight_layout()
    wandb.log({"generation/generated_num_list_normalized": wandb.Image(fig_normalized)})
    plt.close(fig_normalized)


from torch_geometric.data import Data

@dynamic_task
def progressive_digit_generation(test_loader, model, test_params, numbers_list):
    """
    Generate digits iteratively, starting from random noise and transitioning through supervision labels.
    The last trace value is used as the input for the next label generation. 
    Focus more on the transitioning before and after the switch to a new supervised label.
    Visualize the results in a grid plot and log to WandB.

    Parameters:
    - test_loader: DataLoader to provide graph samples.
    - model: The predictive coding model.
    - test_params: Test parameters for evaluation.
    - numbers_list: List of digits to generate.
    """
    import wandb
    import matplotlib.pyplot as plt
    import torch

    # Get a single batch from the test_loader
    for noisy_batch, _ in test_loader:
        noisy_batch = noisy_batch.to(model.pc_conv1.device)
        break  # Only take the first batch

    # Modify x to be random for sensory nodes
    noisy_batch.x[:, 0][model.pc_conv1.sensory_indices] = torch.rand(
        noisy_batch.x[:, 0][model.pc_conv1.sensory_indices].shape, device=model.pc_conv1.device
    )

    generated_images = []  # Store generated outputs
    start_image = noisy_batch.x[:, 0][model.pc_conv1.sensory_indices].view(28, 28).cpu().detach().numpy()
    total_images = [start_image]  # Initialize with the starting random noise

    for target_digit in numbers_list:
        # Set the supervision signal for the current target digit
        supervision_label = torch.zeros((10,), device=model.pc_conv1.device)
        supervision_label[target_digit] = 1.0
        supervision_label = supervision_label.unsqueeze(1)  # Change shape to [10, 1]

        noisy_batch.x[:, 0][model.pc_conv1.supervised_labels] = supervision_label

        # Perform inference to generate the target digit
        values, predictions, labels = model.query(method="pass", data=noisy_batch)

        # Store intermediate snapshots from the trace with focus on transitioning
        intermediate_images = []
        if model.pc_conv1.trace_activity_values:
            trace_length = len(model.pc_conv1.trace["values"])
            # Capture more points before and after the transition
            step_size = trace_length // 10
            transition_focus_indices = list(range(0, trace_length, step_size))[-10:]
            for idx in transition_focus_indices:
                trace_snapshot = model.pc_conv1.trace["values"][idx][0:784].view(28, 28).cpu().detach().numpy()
                trace_snapshot = (trace_snapshot - trace_snapshot.min()) / (trace_snapshot.max() - trace_snapshot.min())
                intermediate_images.append(trace_snapshot)

        # Normalize the generated image and store it
        generated_image = values[0:784].view(28, 28).cpu().detach().numpy()
        min_val, max_val = generated_image.min(), generated_image.max()
        if max_val > min_val:  # Avoid division by zero
            generated_image = (generated_image - min_val) / (max_val - min_val)
        generated_images.append(generated_image)

        # Add intermediate and final generated images to the total
        total_images += intermediate_images + [generated_image]

        # Use the last trace value as the next input
        noisy_batch.x[:, 0][model.pc_conv1.sensory_indices] = values[0:784]

    # Determine grid dimensions: one row per digit
    grid_rows = len(numbers_list)
    grid_cols = (len(total_images) + grid_rows - 1) // grid_rows

    # Create the figure with a rectangular grid layout
    fig, axs = plt.subplots(nrows=grid_rows, ncols=grid_cols, figsize=(grid_cols * 2, grid_rows * 2))

    # Flatten axes for easy iteration if the grid is larger than 1x1
    axs = axs.ravel() if grid_rows * grid_cols > 1 else [axs]

    for ax, img in zip(axs, total_images):
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.axis("off")

    # Remove unused subplots
    for ax in axs[len(total_images):]:
        ax.axis("off")

    plt.tight_layout()

    # Log to WandB
    wandb.log({"generation/progressive_digit_generation_norm": wandb.Image(fig)})
    plt.close(fig)


@dynamic_task
def classification(test_loader, model, test_params, num_samples=5):
    

    assert test_params["supervised_learning"] == False

    # metrics 
    y_true, y_pred = [], []

    for idx, (noisy_batch, clean_image) in enumerate(test_loader):

        noisy_batch = noisy_batch.to(model.pc_conv1.device)
        noisy_image = noisy_batch.x[:, 0][0:784].view(28,28).cpu().detach().numpy()
        clean_image = clean_image.view(28,28).cpu().numpy()  # Adjust shape as necessary


        # ------------ Alter sensory --------------
        noisy_batch.x[:, 0][model.pc_conv1.internal_indices] = torch.rand(noisy_batch.x[:, 0][model.pc_conv1.internal_indices].shape).to(model.pc_conv1.device)
        # ------------ Alter internal --------------
        # ... 
        # ------------ Alter supervision --------------
        #### SURVERVISED ##########
        # we can alter the supervision signal during testing as an experiment
        if test_params["supervised_learning"]:
            # correct supervision label already set in dataset
            # if during training the label for the supervised node was x, also here
            # model.pc_conv1.values[model.pc_conv1.supervised_labels].data = noisy_batch.x[model.pc_conv1.supervised_labels]
            pass
        else:
            noisy_batch.x[:, 0][model.pc_conv1.supervised_labels] = 0
        
        # assert all zeros
        assert torch.all(torch.isclose(noisy_batch.x[:, 0][-10:], torch.zeros_like(noisy_batch.x[:, 0][-10:]), atol=1e-6)), "Last 10 elements are not close to zero!"

      
        # ----------- Inference ---------------
        # output
        values, predictions, labels = model.query(method="pass", data=noisy_batch)  # query_by_conditioning
        denoised_output = predictions[0:784].view(28,28).cpu().detach().numpy()

        
   
        # ----------- Plotting ---------------

        # Creating a subplot mosaic
        fig, ax = plt.subplot_mosaic([
            ["A", "B", "C", "D", "E"],
            ["F", "F", "F", "G", "G"]
        ], figsize=(15, 8))

        # Plotting the images
        ax["A"].imshow(clean_image, vmin=0, vmax=1, cmap=cmap)
        ax["A"].set_title(f"Clean Image of a {noisy_batch.y.item()}")

        ax["B"].imshow(noisy_image, vmin=0, vmax=1, cmap=cmap)
        ax["B"].set_title("Noisy Input")

        ax["C"].imshow(denoised_output, vmin=0, vmax=1, cmap=cmap)
        ax["C"].set_title("Predic. at T")

        ax["D"].imshow(denoised_output, cmap=cmap)
        ax["D"].set_title("Predic. at T, no vmin vmax")
        print("Predic. at T, no vmin vmax", max(denoised_output.flatten()), min(denoised_output.flatten()))

        denoised_output_scaled = (denoised_output - min(denoised_output.flatten())) / (max(denoised_output.flatten()) - min(denoised_output.flatten()))
        # ax["E"].imshow(clean_image - denoised_output_scaled, vmin=0, vmax=1, cmap=cmap)
        # ax["E"].set_title("---")
        print("Denoised val", max(values[0:784].view(28, 28).cpu().detach().numpy().flatten()), min(values[0:784].view(28, 28).cpu().detach().numpy().flatten()))

        for a in ["A", "B", "C", "D", "E"]:
            ax[a].axis('off')

        # Plotting the line graphs
        ax["F"].plot(model.pc_conv1.energy_vals["internal_energy_testing"][-model.pc_conv1.T:], label="Internal energy")
        ax["F"].plot(model.pc_conv1.energy_vals["sensory_energy_testing"][-model.pc_conv1.T:], label="Sensory energy")  # Replace with actual values
        ax["F"].plot(model.pc_conv1.energy_vals["supervised_energy_testing"][-model.pc_conv1.T:], label="supervised_energy energy")  # Replace with actual values 
        ax["F"].legend()


        ax["E"].imshow(values[0:784].view(28,28).cpu().detach().numpy(), vmin=0, vmax=1, cmap=cmap)
        ax["E"].set_title("Diff clean - denoised_scaled")


        # fig.savefig(f'{model_dir}/reconstruction/recon_T_{model.pc_conv1.T}_{noisy_batch.y.item()}.png')

        labels = values[model.pc_conv1.supervised_labels].squeeze()
        print(labels)

        difference_image = clean_image - denoised_output_scaled

        softmax_labels = torch.nn.Softmax(dim=0)(labels)

        if sum(labels) == 0:
            print("-----------NO prediction on labels-----------")
            break 
        else: 
            label_pred = torch.argmax(softmax_labels)
            label_pred = torch.arange(10)[label_pred]
            
            softmax_labels_np = softmax_labels.cpu().detach().numpy()

            y_t, y_p = int(noisy_batch.y.item()), int(label_pred.item())
            y_true.append(y_t)
            y_pred.append(y_p)

            if y_p == y_t:
                # Define the color for each bar, defaulting to blue, with the max value bar colored red
                colors = ['blue' if i != torch.argmax(softmax_labels).item() else 'green' for i in range(len(softmax_labels_np))]
            else:
                colors = ['blue' if i != torch.argmax(softmax_labels).item() else 'red' for i in range(len(softmax_labels_np))]

            # Plot the bars with the specified colors
            ax["G"].bar([str(x) for x in range(10)], softmax_labels_np, color=colors)
            ax["G"].set_title("Softmax Probability Distribution")



        if test_params["model_dir"] and test_params["num_wandb_img_log"] < idx:
            fig_path = f'{test_params["model_dir"]}eval/classification/classification_{idx}_condition_label_T_{model.pc_conv1.T}_{noisy_batch.y.item()}.png'
            fig.savefig(fig_path)

        if test_params["num_wandb_img_log"] < idx:
            # log fig to wandb
            wandb.log({"classification/classification_IMG": wandb.Image(fig)})
            plt.close(fig)

        if len(y_true) >= test_params["num_samples"]:
            
            break 

    print(y_true, y_pred)

    # Calculate classification metrics with 'macro' averaging for multiclass
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    # Print classification metrics
    print("Classification Metrics:")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")

    # # Print classification report
    # print("\nClassification Report:")
    # print(classification_report(y_true, y_pred, zero_division=0))

    # # Calculate confusion matrix
    # conf_matrix = confusion_matrix(y_true, y_pred)

    # # Print confusion matrix
    # print("\nConfusion Matrix:")
    # print(conf_matrix)

    # # Plot confusion matrix
    # plt.figure(figsize=(8, 6))
    # sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=sorted(set(y_true)), yticklabels=sorted(set(y_true)))
    # plt.xlabel('Predicted Labels')
    # plt.ylabel('True Labels')
    # plt.title('Confusion Matrix')

    return y_true, y_pred, accuracy
    

# --------------------------------------


# def occlusion(test_loader, model, test_params):
    
#     model.pc_conv1.debug = False
#     model.pc_conv1.set_mode("testing", task="reconstruction")
#     model.pc_conv1.nodes_2_update = list(model.pc_conv1.sensory_indices[len(model.pc_conv1.sensory_indices)//2:]) +  list(model.pc_conv1.internal_indices)

#     # model.pc_conv1.nodes_2_update +=  list(model.pc_conv1.supervised_labels)
#     model.pc_conv1.restart_activity()

#     model.pc_conv1.debug = False
#     model.pc_conv1.T = test_params["T"] 
#     MSE_values = []

#     for idx, (noisy_batch, clean_image) in enumerate(test_loader, start=1):

#         print("idx", idx)
#         # Perform inference to denoise
#         noisy_batch = noisy_batch.to(model.pc_conv1.device)

#         # init all internal nodes random
#         noisy_batch.x[:, 0][model.pc_conv1.internal_indices] = torch.rand(noisy_batch.x[:, 0][model.pc_conv1.internal_indices].shape).to(model.pc_conv1.device)

#         # make occlusion
#         noisy_batch.x[:, 0][784 // 2: ] = 0 
        
#         if test_params["add_sens_noise"]:
#             noisy_batch.x[:, 0][784 // 2: ] = torch.rand( noisy_batch.x[:, 0][784 // 2: ].shape) 

#         #### SURVERVISED ##########
#         if test_params["supervised_learning"]:
#             # if during training the label for the supervised node was 60, also here
#             # model.pc_conv1.values[model.pc_conv1.supervised_labels].data = noisy_batch.x[model.pc_conv1.supervised_labels]
#             pass 
#         else:
#             noisy_batch.x[:, 0][model.pc_conv1.supervised_labels] = 0
        
#         print("labels model", noisy_batch.x[:, 0][model.pc_conv1.supervised_labels] )

#         # Extract the denoised output from the sensory nodes
#         noisy_image = noisy_batch.x[:, 0][0:784].view(28,28).cpu().detach().numpy()
        
#         values, predictions, labels = model.query(method="pass", data=noisy_batch)  # query_by_conditioning
        
        
#         denoised_output = predictions[0:784].view(28,28).cpu().detach().numpy()

        
#         # Creating a subplot mosaic
#         fig, ax = plt.subplot_mosaic([
#             ["A", "B", "C", "D", "E"],
#             ["F", "F", "F", "G", "G"]
#         ], figsize=(15, 8)) 


#         # WANT TO ONLY COMPARE THE OCClUDED PART WITH THE MODELS CREATIONS 
#         # Assuming occlusion was applied to the first half of the image
#         occluded_part = 784 // 2  # Adjust this based on the actual occluded region size

#         # Flatten the images (if not already flattened)
#         clean_image_flat = clean_image.flatten()
#         denoised_output_flat = denoised_output.flatten()

#         # Compare only the occluded region
#         MSE = round(calculate_mse(clean_image_flat[:occluded_part], denoised_output_flat[:occluded_part]), 4)

#         fig.suptitle(f"MSE {MSE} clean/denoised_output")

#         clean_image = clean_image.view(28,28).cpu().numpy()  # Adjust shape as necessary
#         # Plotting the images
#         ax["A"].imshow(clean_image, vmin=0, vmax=1, cmap=cmap)
#         ax["A"].set_title(f"Clean Image of a {noisy_batch.y.item()}")

#         ax["B"].imshow(noisy_image, vmin=0, vmax=1, cmap=cmap)
#         ax["B"].set_title("Noisy Input")

#         ax["C"].imshow(denoised_output, vmin=0, vmax=1, cmap=cmap)
#         ax["C"].set_title("Predic. at T")

#         ax["D"].imshow(denoised_output, cmap=cmap)
#         ax["D"].set_title("Predic. at T, no vmin vmax")
#         print("Predic. at T, no vmin vmax", max(denoised_output.flatten()), min(denoised_output.flatten()))

#         denoised_output_scaled = (denoised_output - min(denoised_output.flatten())) / (max(denoised_output.flatten()) - min(denoised_output.flatten()))
#         ax["E"].imshow(clean_image - denoised_output_scaled, vmin=0, vmax=1, cmap=cmap)
#         ax["E"].set_title("Diff clean - denoised_scaled")
#         print("Denoised val", max(values[0:784].view(28, 28).cpu().detach().numpy().flatten()), min(values[0:784].view(28, 28).cpu().detach().numpy().flatten()))

#         for a in ["A", "B", "C", "D", "E", "G"]:
#             ax[a].axis('off')

#         # Plotting the line graphs
#         ax["F"].plot(model.pc_conv1.energy_vals["internal_energy"][-model.pc_conv1.T:], label="Internal energy")
#         ax["F"].plot(model.pc_conv1.energy_vals["sensory_energy"][-model.pc_conv1.T:], label="Sensory energy")  # Replace with actual values
#         ax["F"].legend()


#         ax["G"].imshow(values[0:784].view(28,28).cpu().detach().numpy(), vmin=0, vmax=1, cmap=cmap)
#         ax["G"].set_title("value")

#         if test_params["model_dir"] and test_params["num_wandb_img_log"] < idx:
#             fig.savefig(f'{test_params["model_dir"]}eval/occlusion/occ_{idx}_T_{model.pc_conv1.T}_{noisy_batch.y.item()}.png')
        
#         if not test_params["model_dir"]:
#             plt.show()
            
#         labels = values[model.pc_conv1.supervised_labels]
#         print(labels)

#         MSE_values.append(MSE)

#         if test_params["num_wandb_img_log"] < idx:
#             # log fig to wandb
#             wandb.log({"occlusion_IMG": wandb.Image(fig)})
#             plt.close(fig)

#         if idx >= test_params["num_samples"]:
#             break 

#     return MSE_values





















# --------------------------------------


# def reconstruction(test_loader, model, test_params,
#                    num_samples=1):
    

     
#     model.pc_conv1.debug = False
#     # model.pc_conv1.T = 150
#     model.pc_conv1.T = test_params["T"] 
#     # model.pc_conv1.lr_gamma = 10
#     # model.pc_conv1.lr_alpha = 10

#     model.pc_conv1.set_mode("testing", task="reconstruction")


#     # model.pc_conv1.nodes_2_update +=  list(model.pc_conv1.supervised_labels)

#     model.pc_conv1.batchsize = 1

#     model.pc_conv1.restart_activity()

#     batch_idx = 0.


#     model.pc_conv1.debug = False
#     # model.pc_conv1.T = 100
#     # model.pc_conv1.lr_values = 0.5
#     model.pc_conv1.set_mode("testing", task="reconstruction")
#     model.pc_conv1.batchsize = 1

#     batch_idx = 0

#     print("No vmin vmax")
#     for idx, (noisy_batch, clean_image) in enumerate(test_loader):

#         # noisy_batch, clean_image = noisy_batch[0], clean_image[0]
#         print(noisy_batch.y)

#         # model.pc_conv1.restart_activity()

#         # Perform inference to denoise
#         noisy_batch = noisy_batch.to(model.pc_conv1.device)

#         # TEST: set x of noisy_batch to equal all zeros
#         # noisy_batch.x += torch.rand(noisy_batch.x.shape).to(model.pc_conv1.device)

#         # noisy_batch.x /= noisy_batch.x.max() 

#         print("X shape", noisy_batch.x.shape)

#         # set bottom half of the image to zero
#         # print("Make occuled")

#         # model.pc_conv1.set_sensory_nodes(noisy_batch.x)

#         #### SURVERVISED ##########

        

            # #### SURVERVISED ##########
            # if test_params["supervised_learning"]:
            #     # if during training the label for the supervised node was 60, also here
            #     # model.pc_conv1.values[model.pc_conv1.supervised_labels].data = noisy_batch.x[model.pc_conv1.supervised_labels]
            #     continue
            # else:
            #     noisy_batch.x[:, 0][model.pc_conv1.supervised_labels] = 0
            
#         noisy_batch.x[:, 0][model.pc_conv1.internal_indices] = torch.rand(noisy_batch.x[:, 0][model.pc_conv1.internal_indices].shape).to(model.pc_conv1.device)

#         if test_params["supervised_learning"]: # dataset_params["supervised_learning"]:
#             print("labels dataloader", noisy_batch.x.view(-1)[model.pc_conv1.supervised_labels])
#             one_hot = torch.zeros(10, device=model.pc_conv1.device)
#             one_hot[noisy_batch.y] = 1 
            
#             # one_hot[noisy_batch.y] = 0 

#             one_hot = one_hot.view(-1, 1)
#             print(model.pc_conv1.values.data[model.pc_conv1.supervised_labels].shape,  one_hot.shape)

            
#             # HERE 
#             # model.pc_conv1.values.data[supervised_labels] = one_hot

#             # for i in model.pc_conv1.supervised_labels: 
#             #     model.pc_conv1.values.data[i] = noisy_batch.x.view(-1)[i]
            
#             # model.pc_conv1.values[model.pc_conv1.supervised_labels].data = noisy_batch.x[model.pc_conv1.supervised_labels]
            
#         else:
#             noisy_batch.x[:, 0][model.pc_conv1.supervised_labels] = 0
        
#         print("labels model", noisy_batch.x[:, 0][model.pc_conv1.supervised_labels] )
#         # model.inference()
        
#         # remove label

#         # noisy_batch.x[:, 0][0:784] += 2 * torch.rand(noisy_batch.x[:, 0][0:784].shape).to(device)

#         # Extract the denoised output from the sensory nodes
#         noisy_image = noisy_batch.x[:, 0][0:784].view(28,28).cpu().detach().numpy()
        
        
#         values, predictions, labels = model.query(method="pass", data=noisy_batch)  # query_by_conditioning
#         # values, predictions = values[batch_idx, :, 0], predictions[batch_idx, :, 0]
        
#         denoised_output = predictions[0:784].view(28,28).cpu().detach().numpy()

#         # denoised_output = model.reconstruction().view(28,28).cpu().detach().numpy()

#         clean_image = clean_image.view(28,28).cpu().numpy()  # Adjust shape as necessary

        
#         # Creating a subplot mosaic
#         fig, ax = plt.subplot_mosaic([
#             ["A", "B", "C", "D", "E"],
#             ["F", "F", "F", "G", "G"]
#         ], figsize=(15, 8))

#         # Plotting the images
#         ax["A"].imshow(clean_image, vmin=0, vmax=1, cmap=cmap)
#         ax["A"].set_title(f"Clean Image of a {noisy_batch.y.item()}")

#         ax["B"].imshow(noisy_image, vmin=0, vmax=1, cmap=cmap)
#         ax["B"].set_title("Noisy Input")

#         ax["C"].imshow(denoised_output, vmin=0, vmax=1, cmap=cmap)
#         ax["C"].set_title("Predic. at T")

#         ax["D"].imshow(denoised_output, cmap=cmap)
#         ax["D"].set_title("Predic. at T, no vmin vmax")
#         print("Predic. at T, no vmin vmax", max(denoised_output.flatten()), min(denoised_output.flatten()))

#         denoised_output_scaled = (denoised_output - min(denoised_output.flatten())) / (max(denoised_output.flatten()) - min(denoised_output.flatten()))
#         ax["E"].imshow(clean_image - denoised_output_scaled, vmin=0, vmax=1, cmap=cmap)
#         ax["E"].set_title("Diff clean - denoised_scaled")
#         print("Denoised val", max(values[0:784].view(28, 28).cpu().detach().numpy().flatten()), min(values[0:784].view(28, 28).cpu().detach().numpy().flatten()))

#         for a in ["A", "B", "C", "D", "E", "G"]:
#             ax[a].axis('off')

#         # Plotting the line graphs
#         ax["F"].plot(model.pc_conv1.energy_vals["internal_energy"][-model.pc_conv1.T:], label="Internal energy")
#         ax["F"].plot(model.pc_conv1.energy_vals["sensory_energy"][-model.pc_conv1.T:], label="Sensory energy")  # Replace with actual values
#         ax["F"].legend()


#         ax["G"].imshow(values[0:784].view(28,28).cpu().detach().numpy(), cmap=cmap)
#         ax["G"].set_title("value")


#         # fig.savefig(f'{model_dir}/reconstruction/recon_T_{model.pc_conv1.T}_{noisy_batch.y.item()}.png')

#         labels = values[model.pc_conv1.supervised_labels]
#         print(labels)

#         difference_image = clean_image - denoised_output_scaled

#         if idx >= num_samples:
#             break