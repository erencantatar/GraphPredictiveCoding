import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import wandb

# --- Configuration ---
use_all_runs = False



run_ids = ["uqqkm2uq","p3mvfpe4","1rx1pf9f", "svloxh0l", "lp3o6d1s","jozn66av"]


# run_ids = ["5h1w3x3c", "ri8vip0i", "sr0k06o9"]
# ....
# run_ids = ["gd8ehejb", "yak2adzi", "dqfwat8j", "7dhluif7", "12p7k1e2",
#             "h3tmeag1", "d6r48s7b"]


run_ids = list(set(run_ids))

base_dir = "jobs/plotting/trajectories"
out_dir = "jobs/plotting/outputs_trajectories/converge_instability2/"
os.makedirs(out_dir, exist_ok=True)

if use_all_runs:
    run_ids = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

num_runs = len(run_ids)
alpha_value = max(0.2, min(0.8, 1.5 / num_runs))
arrow_width = max(0.002, 0.01 / num_runs)
head_width = max(2, 8 - 0.5 * num_runs)
arrow_scale = max(0.4, min(1.5, 3.0 / num_runs))

# --- W&B API setup ---
api = wandb.Api()
ENTITY = "etatar-atdamen"
PROJECT = "PredCod"

# --- Load weights and W&B config ---
flat_weights_all = []
flat_weights_by_run = []
std_mean_data = []

for run_id in run_ids:
    try:
        # run = api.run(f"{ENTITY}/{PROJECT}/{run_id}")
        # print(run_id, "use_grokfast", run.config["use_grokfast"])
        # use_grok = bool(run.config.get("use_grokfast", False))

        run = api.run(f"{ENTITY}/{PROJECT}/{run_id}")
        raw_flag = run.config.get("use_grokfast", "False")
        use_grok = raw_flag.strip().lower() == "true"
        print(f"{run_id} use_grokfast: {raw_flag} → parsed as {use_grok}")

    except Exception as e:
        print(f"⚠️ Could not fetch W&B run {run_id}: {e}")
        use_grok = False

    weight_path = os.path.join(base_dir, run_id, "weights.npy")
    if not os.path.exists(weight_path):
        print(f"✗ Missing: {weight_path}")
        continue

    try:
        weights = np.load(weight_path)  # shape: [T, N, N]
        flat = weights.reshape(weights.shape[0], -1)
        flat_weights_by_run.append((run_id, flat, use_grok))
        flat_weights_all.append(flat)

        stds = np.std(flat, axis=1)
        means = np.mean(flat, axis=1)
        std_mean_data.append((run_id, stds, means, use_grok))
        print(f"✓ Loaded {run_id}, use_grokfast={use_grok}")
        print(f"  → weights shape: {weights.shape}")

    except Exception as e:
        print(f"✗ Failed to load {run_id}: {e}")

if len(flat_weights_all) == 0:
    print("✗ No valid weight files found. Exiting.")
    exit()

flat_weights_all = np.concatenate(flat_weights_all, axis=0)
flat_weights_all -= flat_weights_all.mean(axis=0)

# --- PCA on combined data ---
pca = PCA(n_components=10)
pca_vals_all = pca.fit_transform(flat_weights_all)
explained_var = pca.explained_variance_ratio_

# --- Project each run ---
projected_pca_data = []
global_mean = flat_weights_all.mean(axis=0)
for run_id, flat, use_grok in flat_weights_by_run:
    centered = flat - global_mean
    proj = pca.transform(centered)
    projected_pca_data.append((run_id, proj, use_grok))

# --- Plot settings ---
color_map = {True: "blue", False: "red"}
label_map = {True: "Grokfast", False: "No Grokfast"}

# --- Explained variance plot ---
plt.figure(figsize=(6, 4))
plt.plot(np.arange(1, len(explained_var)+1), explained_var, 'o-', linewidth=2)
plt.title("Explained Variance Spectrum (Combined Runs)")
plt.xlabel("Principal Component")
plt.ylabel("Explained Variance Ratio")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "pca_explained_variance_combined.png"))
plt.close()
print("✓ Saved explained variance plot.")

# --- PCA Trajectories ---
plt.figure(figsize=(8, 6))
added_labels = set()
for run_id, data, use_grok in projected_pca_data:
    color = color_map[use_grok]
    label = label_map[use_grok]
    x, y = data[:-1, 0], data[:-1, 1]
    u, v = data[1:, 0] - x, data[1:, 1] - y
    plt.plot(data[:, 0], data[:, 1], 'o--', color=color, alpha=alpha_value * 0.8,
             markersize=2, linewidth=0.8)
    if label not in added_labels:
        plt.quiver(x, y, u, v, angles='xy', scale_units='xy',
                   scale=arrow_scale, width=arrow_width, headwidth=head_width,
                   color=color, alpha=alpha_value, label=label)
        added_labels.add(label)
    else:
        plt.quiver(x, y, u, v, angles='xy', scale_units='xy',
                   scale=arrow_scale, width=arrow_width, headwidth=head_width,
                   color=color, alpha=alpha_value)
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.title("PCA Trajectories (Combined Space)")
plt.grid()
plt.legend(fontsize=8, loc="upper left")
plt.tight_layout(rect=[0, 0, 0.8, 1])
plt.savefig(os.path.join(out_dir, "pca_trajectory_combined.png"))
plt.close()
print("✓ Saved PCA trajectory plot.")

# --- Std/Mean Trajectories ---
plt.figure(figsize=(8, 6))
added_labels = set()
for run_id, stds, means, use_grok in std_mean_data:
    color = color_map[use_grok]
    label = label_map[use_grok]
    x, y = stds[:-1], means[:-1]
    u, v = stds[1:] - x, means[1:] - y
    plt.plot(stds, means, 'o--', color=color, alpha=alpha_value * 0.8,
             markersize=2, linewidth=0.8)
    if label not in added_labels:
        plt.quiver(x, y, u, v, angles='xy', scale_units='xy',
                   scale=arrow_scale, width=arrow_width, headwidth=head_width,
                   color=color, alpha=alpha_value, label=label)
        added_labels.add(label)
    else:
        plt.quiver(x, y, u, v, angles='xy', scale_units='xy',
                   scale=arrow_scale, width=arrow_width, headwidth=head_width,
                   color=color, alpha=alpha_value)
plt.xlabel("Std")
plt.ylabel("Mean")
plt.title("Std/Mean Trajectories")
plt.grid()
plt.legend(fontsize=8, loc="upper left")
plt.tight_layout(rect=[0, 0, 0.8, 1])
plt.savefig(os.path.join(out_dir, "std_mean_trajectory.png"))
plt.close()
print("✓ Saved Std/Mean plot.")
