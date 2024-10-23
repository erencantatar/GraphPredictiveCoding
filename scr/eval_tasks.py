
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


def occlusion(test_loader, model, test_params):
    
    model.pc_conv1.debug = False
    model.pc_conv1.set_mode("testing", task="reconstruction")
    model.pc_conv1.nodes_2_update = list(model.pc_conv1.sensory_indices[len(model.pc_conv1.sensory_indices)//2:]) +  list(model.pc_conv1.internal_indices)

    # model.pc_conv1.nodes_2_update +=  list(model.pc_conv1.supervised_labels)
    model.pc_conv1.restart_activity()

    model.pc_conv1.debug = False
    model.pc_conv1.T = test_params["T"] 
    MSE_values = []

    for idx, (noisy_batch, clean_image) in enumerate(test_loader, start=1):

        print("idx", idx)
        # Perform inference to denoise
        noisy_batch = noisy_batch.to(model.pc_conv1.device)

        # init all internal nodes random
        noisy_batch.x[:, 0][model.pc_conv1.internal_indices] = torch.rand(noisy_batch.x[:, 0][model.pc_conv1.internal_indices].shape).to(model.pc_conv1.device)

        # make occlusion
        noisy_batch.x[:, 0][784 // 2: ] = 0 
        
        if test_params["add_sens_noise"]:
            noisy_batch.x[:, 0][784 // 2: ] = torch.rand( noisy_batch.x[:, 0][784 // 2: ].shape) 

        #### SURVERVISED ##########
        if test_params["supervised_learning"]:
            # if during training the label for the supervised node was 60, also here
            # model.pc_conv1.values[model.pc_conv1.supervised_labels].data = noisy_batch.x[model.pc_conv1.supervised_labels]
            pass 
        else:
            noisy_batch.x[:, 0][model.pc_conv1.supervised_labels] = 0
        
        print("labels model", noisy_batch.x[:, 0][model.pc_conv1.supervised_labels] )

        # Extract the denoised output from the sensory nodes
        noisy_image = noisy_batch.x[:, 0][0:784].view(28,28).cpu().detach().numpy()
        
        values, predictions, labels = model.query(method="pass", data=noisy_batch)  # query_by_conditioning
        
        
        denoised_output = predictions[0:784].view(28,28).cpu().detach().numpy()


        cmap = "gray"
        
        # Creating a subplot mosaic
        fig, ax = plt.subplot_mosaic([
            ["A", "B", "C", "D", "E"],
            ["F", "F", "F", "G", "G"]
        ], figsize=(15, 8)) 


        # WANT TO ONLY COMPARE THE OCClUDED PART WITH THE MODELS CREATIONS 
        # Assuming occlusion was applied to the first half of the image
        occluded_part = 784 // 2  # Adjust this based on the actual occluded region size

        # Flatten the images (if not already flattened)
        clean_image_flat = clean_image.flatten()
        denoised_output_flat = denoised_output.flatten()

        # Compare only the occluded region
        MSE = round(calculate_mse(clean_image_flat[:occluded_part], denoised_output_flat[:occluded_part]), 4)

        fig.suptitle(f"MSE {MSE} clean/denoised_output")

        clean_image = clean_image.view(28,28).cpu().numpy()  # Adjust shape as necessary
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


        ax["G"].imshow(values[0:784].view(28,28).cpu().detach().numpy(), vmin=0, vmax=1, cmap=cmap)
        ax["G"].set_title("value")

        if test_params["model_dir"]:
            fig.savefig(f'{test_params["model_dir"]}eval/occlusion/occ_{idx}_T_{model.pc_conv1.T}_{noisy_batch.y.item()}.png')

        labels = values[model.pc_conv1.supervised_labels]
        print(labels)

        MSE_values.append(MSE)

        if test_params["num_wandb_img_log"] < idx:
            # log fig to wandb
            wandb.log({"occlusion_IMG": wandb.Image(fig)})
            plt.close(fig)

        if idx >= test_params["num_samples"]:
            break 

    return MSE_values

    

def denoise(test_loader, model, test_params, sigma=0.1):
    
    model.pc_conv1.debug = False
    model.pc_conv1.set_mode("testing", task="reconstruction")

    # model.pc_conv1.nodes_2_update +=  list(model.pc_conv1.supervised_labels)

    model.pc_conv1.restart_activity()


    model.pc_conv1.T = test_params["T"] 

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
        
        
        values, predictions, labels = model.query(method="pass", data=noisy_batch)  # query_by_conditioning
        # values, predictions = values[batch_idx, :, 0], predictions[batch_idx, :, 0]
        
        denoised_output = predictions[0:784].view(28,28).cpu().detach().numpy()

        # denoised_output = model.reconstruction().view(28,28).cpu().detach().numpy()

        clean_image = clean_image.view(28,28).cpu().numpy()  # Adjust shape as necessary

        # all cmaps starting with Red, blue with
        # maps = RdBu, RdBu_r, RdBu_r, RdBu, BrBG, BrBG_r, coolwarm, coolwarm_r, bwr, bwr_r, seismic, seismic_r
        # cmap = "RdBu_r"
        cmap = "gray"
        


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

        if test_params["model_dir"]:
            fig.savefig(f'{test_params["model_dir"]}eval/denoise/denoise_{idx}_T_{model.pc_conv1.T}_{noisy_batch.y.item()}.png')

        labels = values[model.pc_conv1.supervised_labels]
        print(labels)

        difference_image = clean_image - denoised_output_scaled

        MSE_values.append(MSE)

        if test_params["num_wandb_img_log"] < idx:
            # log fig to wandb
            wandb.log({"denoise_IMG": wandb.Image(fig)})
            plt.close(fig)

        if idx >= test_params["num_samples"]:
            break 

    return MSE_values


from skimage.metrics import structural_similarity as ssim
    
def mse(imageA, imageB):
    """Calculate the Mean Squared Error (MSE) between two images."""
    return np.mean((imageA - imageB) ** 2)

def generation(test_loader, model, test_params, clean_images, num_samples=8, verbose=0):


    """We compare the generated number with the first N=10 clean images of the same class,
    using SSIM, and MSE metric. Better generation is higher SSIM (0-1), and lower MSE (0,+inf)
    +- 2sec    
    """


    # model.pc_conv1.trace_activity_predictions = True 
    model.pc_conv1.restart_activity()
    model.pc_conv1.trace['values'] = []
    model.pc_conv1.trace['preds'] = []

    model.pc_conv1.T = test_params["T"]

    model.pc_conv1.mode = "testing"
    model.pc_conv1.set_mode("testing", task="generation")


    
    print("model.pc_conv1.values.shape", model.pc_conv1.values.shape)
    model.pc_conv1.restart_activity()

    batch_idx = 0

    assert test_params["supervised_learning"] == True 

    torch.cuda.empty_cache()
    gc.collect()

    print("No vmin vmax")

    avg_SSIM_mean, avg_SSIM_max = [], []    
    avg_MSE_mean, avg_MSE_max   = [], []
 
    for idx, (noisy_batch, clean_image) in enumerate(test_loader, start=1):

        # noisy_batch, clean_image = noisy_batch[0], clean_image[0]
        print(noisy_batch.y)

        # Perform inference to denoise
        noisy_batch = noisy_batch.to(model.pc_conv1.device)

        noisy_batch.x[:, 0][model.pc_conv1.internal_indices] = torch.rand(noisy_batch.x[:, 0][model.pc_conv1.internal_indices].shape).to(model.pc_conv1.device)
        # noisy_batch.x[:, 1][model.pc_conv1.internal_indices]  = 0
        # noisy_batch.x[:, 2][model.pc_conv1.internal_indices]  = 0

        plt.imshow(noisy_batch.x[:, 1][0:784].view(28,28).cpu())
        plt.show()


        # TEST: set x of noisy_batch to equal all zeros
        # noisy_batch.x = torch.rand(noisy_batch.x.shape).to(device)
        print(noisy_batch.x.shape)

        # set bottom half of the image to zero
        # print("Make occuled")
        # noisy_batch.x[0:784//2] = 0

        # set sensory
        
        # noisy_batch.x[0:784//2] = torch.zeros_like(noisy_batch.x[0:784//2])
        # model.pc_conv1.set_sensory_nodes(noisy_batch.x)

        print("omitting setting sensory nodes")

        white = torch.ones_like(noisy_batch.x)
        black = torch.zeros_like(noisy_batch.x[:, 0][0:-10])

        random = torch.rand(noisy_batch.x[:, 0][0:-10].shape)

        noisy_batch.x[:, 0][model.pc_conv1.sensory_indices] = torch.rand(noisy_batch.x[:, 0][model.pc_conv1.sensory_indices].shape).to(model.pc_conv1.device)
        # noisy_batch.x[:, 0][0:-10] = random
        

        # self.values, self.errors, self.predictions, = self.data.x[:, 0], self.data.x[:, 1], self.data.x[:, 2]

        # noisy_batch.x[:, 0] = random
        # noisy_batch.x[:, 0][0:-10] = random
        
        # print("aa", noisy_batch.x[:, 0].shape)
        # model.pc_conv1.set_sensory_nodes(noisy_batch.x)
        # model.pc_conv1.set_sensory_nodes(random)
        # model.pc_conv1.set_sensory_nodes(white)

        # black[30:60] = 1
        # model.pc_conv1.set_sensory_nodes(black)

        # print("TST", noisy_batch.x.shape)
        # noisy_batch.x[sensory_indices] = 0
        # noisy_batch.x[sensory_indices] = 1
        
        if test_params["supervised_learning"]:
            a = noisy_batch.x.view(-1)[model.pc_conv1.supervised_labels]
            print("labels before", a.shape, a)

            one_hot = torch.zeros(10)
            one_hot[noisy_batch.y] = 10
            # one_hot[noisy_batch.y + 1] = 0.5
            # one_hot[noisy_batch.y] = 0
            one_hot = one_hot.view(-1, 1)
            one_hot = one_hot.to(model.pc_conv1.device)
            model.pc_conv1.values.data[model.pc_conv1.supervised_labels] = one_hot

            # for i in model.pc_conv1.supervised_labels: 
            #     model.pc_conv1.values.data[i] = noisy_batch.x.view(-1)[i]


            print("labels after", noisy_batch.x.view(-1)[model.pc_conv1.supervised_labels])
            print(model.pc_conv1.values.data[model.pc_conv1.supervised_labels])

        else:
            noisy_batch.x[:, 0][model.pc_conv1.supervised_labels] = 0
        
        print("labels model", noisy_batch.x[:, 0][model.pc_conv1.supervised_labels] )

        # model.pc_conv1.values[model.pc_conv1.supervised_labels] = noisy_batch.x[model.pc_conv1.supervised_labels]
        # model.inference()
        
        noisy_image = noisy_batch.x[:, 0][0:784].view(28,28).cpu().detach().numpy()

        print("aaa", model.pc_conv1.mode)
        # Extract the denoised output from the sensory nodes

        print("CHECK", noisy_batch.x[:,0][-10:])
        values, predictions, labels = model.query(method="pass", data=noisy_batch)  # query_by_conditioning
        # values, predictions = values[batch_idx, :, 0], predictions[batch_idx, :, 0]
        print("CHECK", noisy_batch.x[:,0][-10:])
        
        denoised_output = predictions[0:784].view(28,28).cpu().detach().numpy()

        # denoised_output = model.reconstruction().view(28,28).cpu().detach().numpy()

        clean_image = clean_image.view(28,28).cpu().numpy()  # Adjust shape as necessary

        # all cmaps starting with Red, blue with
        # maps = RdBu, RdBu_r, RdBu_r, RdBu, BrBG, BrBG_r, coolwarm, coolwarm_r, bwr, bwr_r, seismic, seismic_r
        # cmap = "RdBu_r"
        cmap = "gray"

        # Plotting both images side by side
    
        # Creating a subplot mosaic
        # Creating a subplot mosaic with the row for values and a row for predictions
        # Creating a subplot mosaic with titles for values and predictions
        fig, ax = plt.subplot_mosaic([
            ["A", "B", "C", "D", "E"],   # First row for clean image, noisy image, and denoised output
            ["1", "2", "3", "4", "5"],   # Second row for values with label
            ["H", "I", "J", "K", "L"],   # Third row for predictions with label
            ["F", "F", "F", "G", "G"],   # Fourth row for energy plots and additional info
        ], figsize=(15, 10))

        # Adding text labels to the left of the rows for values and predictions
        fig.text(0.02, 0.67, "Values", ha='center', va='center', fontsize=12, rotation='vertical', fontweight='bold')
        fig.text(0.02, 0.47, "Predictions", ha='center', va='center', fontsize=12, rotation='vertical', fontweight='bold')


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
        ax["F"].plot(model.pc_conv1.energy_vals["internal_energy"][-model.pc_conv1.T:], label="Internal energy")
        ax["F"].plot(model.pc_conv1.energy_vals["sensory_energy"][-model.pc_conv1.T:], label="Sensory energy")  # Replace with actual values
        ax["F"].legend()


        ax["G"].imshow(values[0:784].view(28,28).cpu().detach().numpy(), 
                    #    vmin=0, vmax=1, 
                       cmap=cmap)
        ax["G"].set_title("values no vmix/max")

        # Apply tight layout
        plt.tight_layout()

        # Adjust subplot spacing if needed
        plt.subplots_adjust(wspace=0.05, hspace=0.05)  # Adjust spacing between subplots

        if test_params["model_dir"]:
            fig.savefig(f'{test_params["model_dir"]}eval/generation/generation_condition_label_T_{idx}_{model.pc_conv1.T}_{noisy_batch.y.item()}.png')

   

        # save 
        print(f"generation_task_{noisy_batch.y.item()}.png")
        # plt.savefig(f"generation_task_{noisy_batch.y.item()}.png")        

        labels = values[model.pc_conv1.supervised_labels]

        # print(model.pc_conv1.values.data[model.pc_conv1.supervised_labels] )
        # print(labels)

        print("CHECK", noisy_batch.x[:,0][-10:])
        
        label = noisy_batch.y.item()
        label_clean_images = clean_images[label]

        # Initialize lists for SSIM and MSE values
        ssim_values = []
        mse_values = []

        # Compare the denoised output with each clean image of the same class
        for clean_image in label_clean_images:
            # Calculate SSIM and MSE
            ssim_index = ssim(denoised_output, clean_image, data_range=1.0)
            mse_value = mse(denoised_output, clean_image)

            ssim_values.append(ssim_index)
            mse_values.append(mse_value)

            if verbose == 1:
                print("ssim_values", ssim_values)
                print("mse_values", mse_values)

        avg_SSIM_mean.append(np.mean(ssim_values))
        avg_SSIM_max.append(np.max(ssim_values))
        avg_MSE_mean.append(np.mean(ssim_values))
        avg_MSE_max.append(np.max(ssim_values))
    
        if idx >= test_params["num_samples"]:
            
            return np.mean(avg_SSIM_mean), np.max(avg_SSIM_max), np.mean(avg_MSE_mean), np.max(avg_MSE_max)  

        if test_params["num_wandb_img_log"] < idx:
            # log fig to wandb
            wandb.log({"generation_IMG": wandb.Image(fig)})
            plt.close(fig)

        # model_dir=None,




from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

def classification(test_loader, model, 
                   test_params, num_samples=5):
        
        
    model.pc_conv1.set_mode("testing", task="classification")
    

    # model.pc_conv1.nodes_2_update += list(model.pc_conv1.sensory_indices) 


    # custom_dataset_test = CustomGraphDataset(mnist_dataset=mnist_dataset_test, **dataset_params)
    # test_loader = DataLoader(custom_dataset_test, batch_size=1, shuffle=True)

    model.pc_conv1.batchsize = 1

    print("model.pc_conv1.values.shape", model.pc_conv1.values.shape)
    # model.pc_conv1.restart_activity()

    print("AAAAAAAAAAAA")

    batch_idx = 0.

    assert test_params["supervised_learning"] == False

    
    model.pc_conv1.debug = False
    model.pc_conv1.T = test_params["T"]   
    # model.pc_conv1.T = 50 
    # model.pc_conv1.lr_values = 1.5
    # model.pc_conv1.lr_values = 0.1
    # model.pc_conv1.set_mode("testing", task="reconstruction")
    model.pc_conv1.batchsize = 1

    print("model.pc_conv1.values.shape", model.pc_conv1.values.shape)

    batch_idx = 0

    y_true, y_pred = [], []

    print("CHECK 1 ",  model.pc_conv1.values.data[model.pc_conv1.supervised_labels] )


    print("No vmin vmax")
    for idx, (noisy_batch, clean_image) in enumerate(test_loader):

        # noisy_batch, clean_image = noisy_batch[0], clean_image[0]
        print(noisy_batch.y)

        # model.pc_conv1.restart_activity()

        # Perform inference to denoise
        noisy_batch = noisy_batch.to(model.pc_conv1.device)

        # TEST: set x of noisy_batch to equal all zeros
        # noisy_batch.x += torch.rand(noisy_batch.x.shape).to(device)
        print(noisy_batch.x.shape)

        # set bottom half of the image to zero
        # print("Make occuled")

        # model.pc_conv1.set_sensory_nodes(noisy_batch.x)

        #### SURVERVISED ##########

        noisy_batch.x[:, 0][model.pc_conv1.internal_indices] = torch.rand(noisy_batch.x[:, 0][model.pc_conv1.internal_indices].shape).to(model.pc_conv1.device)

        print("CHECK 2 ",  model.pc_conv1.values.data[model.pc_conv1.supervised_labels] )

        #### SURVERVISED ##########
        if test_params["supervised_learning"]:
            # if during training the label for the supervised node was 60, also here
            # model.pc_conv1.values[model.pc_conv1.supervised_labels].data = noisy_batch.x[model.pc_conv1.supervised_labels]
            pass
        else:
            noisy_batch.x[:, 0][model.pc_conv1.supervised_labels] = 0
        
        print("CHECK 3 ",  model.pc_conv1.values.data[model.pc_conv1.supervised_labels] )

        print("labels model", noisy_batch.x[:, 0][model.pc_conv1.supervised_labels] )

        # model.inference()
        
        noisy_batch.x[:, 0][-10:] = 0 
        
        print("CHECK 4 ",  model.pc_conv1.values.data[model.pc_conv1.supervised_labels] )

        print("CHECK",  model.pc_conv1.values.data[model.pc_conv1.supervised_labels] )
        # Extract the denoised output from the sensory nodes
        noisy_image = noisy_batch.x[:, 0][0:784].view(28,28).cpu().detach().numpy()
        
        
        values, predictions, labels = model.query(method="pass", data=noisy_batch)  # query_by_conditioning
        # values, predictions = values[batch_idx, :, 0], predictions[batch_idx, :, 0]
        

        denoised_output = predictions[0:784].view(28,28).cpu().detach().numpy()

        # denoised_output = model.reconstruction().view(28,28).cpu().detach().numpy()

        clean_image = clean_image.view(28,28).cpu().numpy()  # Adjust shape as necessary

        # all cmaps starting with Red, blue with
        # maps = RdBu, RdBu_r, RdBu_r, RdBu, BrBG, BrBG_r, coolwarm, coolwarm_r, bwr, bwr_r, seismic, seismic_r
        # cmap = "RdBu_r"
        cmap = "gray"
        
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
        ax["F"].plot(model.pc_conv1.energy_vals["internal_energy"][-model.pc_conv1.T:], label="Internal energy")
        ax["F"].plot(model.pc_conv1.energy_vals["sensory_energy"][-model.pc_conv1.T:], label="Sensory energy")  # Replace with actual values
        ax["F"].plot(model.pc_conv1.energy_vals["supervised_energy"][-model.pc_conv1.T:], label="supervised_energy energy")  # Replace with actual values 
        ax["F"].legend()


        ax["E"].imshow(values[0:784].view(28,28).cpu().detach().numpy(), vmin=0, vmax=1, cmap=cmap)
        ax["E"].set_title("Diff clean - denoised_scaled")


        # fig.savefig(f'{model_dir}/reconstruction/recon_T_{model.pc_conv1.T}_{noisy_batch.y.item()}.png')

        labels = values[model.pc_conv1.supervised_labels].squeeze()
        print(labels)

        difference_image = clean_image - denoised_output_scaled

        


        # save 
        # print(f"generation_task_{noisy_batch.y.item()}.png")
        # plt.savefig(f"generation_task_{noisy_batch.y.item()}.png")        

        print(labels)
        print(model.pc_conv1.values.data[model.pc_conv1.supervised_labels])

        softmax_labels = torch.nn.Softmax(dim=0)(labels)
        print(softmax_labels)

        if sum(labels) == 0:
            print("-----------NO prediction on labels-----------")
            break 
        else: 
            label_pred = torch.argmax(softmax_labels)
            label_pred = torch.arange(10)[label_pred]
            
            print(label_pred)

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

          


        if test_params["model_dir"]:
            fig_path = f'{test_params["model_dir"]}eval/classification/classification_{idx}_condition_label_T_{model.pc_conv1.T}_{noisy_batch.y.item()}.png'
            fig.savefig(fig_path)

        if test_params["num_wandb_img_log"] < idx:
            # log fig to wandb
            wandb.log({"classification_IMG": wandb.Image(fig)})
            plt.close(fig)

        # fig.savefig(f'{model_dir}/classification/classification_{idx}_condition_label_T_{model.pc_conv1.T}_{noisy_batch.y.item()}.png')

            # plt.savefig(f"generation_task_{noisy_batch.y.item()}.png")        

        # wandb.log({"classification_figure": wandb.Image(fig_path)})


        if len(y_true) >= test_params["num_samples"]:
            
            break 


        # softmax_labels_np = softmax_labels.cpu().detach().numpy()
        # probabilities = [softmax_prob.cpu().detach().numpy() for softmax_prob in softmax_labels]  # Assuming softmax_prob contains the probabilities

        # # Create data for the table
        # data = [[str(label), prob] for label, prob in zip(softmax_labels_np, probabilities)]

        # # Create a table
        # table = wandb.Table(data=data, columns=["Category", "Probability"])

        # # Create the bar plot
        # softmax_bar = wandb.plot.bar(
        #     table, "Category", "Probability", title="Softmax Probability Distribution"
        # )

        # classification_table.add_data(
        #     epoch,  # Assuming 'epoch' is defined elsewhere in your script
        #     noisy_batch.y.item(),
        #     label_pred.item(),
        #     wandb.Image(noisy_image, caption="Noisy Input"), 
        #     wandb.Image(denoised_output, caption="Predic. at T (vmin/max)"), 
        #     wandb.Image(values[0:784].view(28,28).cpu().detach().numpy(), caption="Value"),
        #     softmax_bar, 
        #     # wandb.Image(softmax_labels.cpu().detach().numpy(), caption="softmax")
        # )


    # wandb.log({"classification_table": classification_table})

    print("---------Done-----------------")

    print("TODO output y_pred and y_true to file ")

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
    
    # plt.show()





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

#         # all cmaps starting with Red, blue with
#         # maps = RdBu, RdBu_r, RdBu_r, RdBu, BrBG, BrBG_r, coolwarm, coolwarm_r, bwr, bwr_r, seismic, seismic_r
#         # cmap = "RdBu_r"
#         cmap = "gray"
        
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