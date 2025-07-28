import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# --- Configuration ---
use_all_runs = False
run_ids = ["t8vrr9mf", "wmbuow8a"]

base_dir = "jobs/plotting/trajectories"
out_dir = "jobs/plotting/outputs_trajectories"
os.makedirs(out_dir, exist_ok=True)

if use_all_runs:
    run_ids = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

num_runs = len(run_ids)

alpha_value = max(0.2, min(0.8, 1.5 / num_runs))
arrow_width = max(0.002, 0.01 / num_runs)
head_width = max(2, 8 - 0.5 * num_runs)
arrow_scale = max(0.4, min(1.5, 3.0 / num_runs))

# --- Load weights ---
flat_weights_all = []
std_mean_data = []
flat_weights_by_run = []

for run_id in run_ids:
    weight_path = os.path.join(base_dir, run_id, "weights.npy")
    if not os.path.exists(weight_path):
        print(f"✗ Missing: {weight_path}")
        continue
    try:
        weights = np.load(weight_path)  # shape: [T, N, N]
        flat = weights.reshape(weights.shape[0], -1)
        flat_weights_by_run.append((run_id, flat))
        flat_weights_all.append(flat)

        stds = np.std(flat, axis=1)
        means = np.mean(flat, axis=1)
        std_mean_data.append((run_id, np.stack([stds, means], axis=1)))
        print(f"✓ Loaded: {weight_path}")
    except Exception as e:
        print(f"✗ Failed to read {weight_path}: {e}")

flat_weights_all = np.concatenate(flat_weights_all, axis=0)
flat_weights_all -= flat_weights_all.mean(axis=0)  # ✅ center weights

# --- PCA on combined data ---
pca = PCA(n_components=10)
pca_vals_all = pca.fit_transform(flat_weights_all)

# --- Plot explained variance spectrum ---
explained_var = pca.explained_variance_ratio_
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

# # Save as CSV (optional)
# pd.DataFrame({
#     "PC": np.arange(1, len(explained_var)+1),
#     "explained_variance_ratio": explained_var
# }).to_csv(os.path.join(out_dir, "pca_explained_variance_combined.csv"), index=False)

# --- Project each run into shared PCA space ---
projected_pca_data = []
for run_id, flat in flat_weights_by_run:
    centered = flat - flat_weights_all.mean(axis=0)  # use same mean
    pca_proj = pca.transform(centered)
    projected_pca_data.append((run_id, pca_proj))

# --- Plot PCA trajectories ---
plt.figure(figsize=(8, 6))
for run_id, data in projected_pca_data:
    x, y = data[:-1, 0], data[:-1, 1]
    u, v = data[1:, 0] - x, data[1:, 1] - y
    plt.quiver(x, y, u, v, angles='xy', scale_units='xy',
               scale=arrow_scale, width=arrow_width, headwidth=head_width,
               label=run_id, alpha=alpha_value)
    plt.plot(data[:, 0], data[:, 1], 'o--', markersize=2, linewidth=0.8,
             alpha=alpha_value * 0.8)
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.title("PCA Trajectories (Combined Space)")
plt.grid()
plt.legend(fontsize=7, bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout(rect=[0, 0, 0.8, 1])
plt.savefig(os.path.join(out_dir, "pca_trajectory_combined.png"))
plt.close()
print("✓ Saved PCA trajectory plot.")

# --- Plot Std/Mean Trajectories ---
plt.figure(figsize=(8, 6))
for run_id, data in std_mean_data:
    x, y = data[:-1, 0], data[:-1, 1]
    u, v = data[1:, 0] - x, data[1:, 1] - y
    plt.quiver(x, y, u, v, angles='xy', scale_units='xy',
               scale=arrow_scale, width=arrow_width, headwidth=head_width,
               label=run_id, alpha=alpha_value)
    plt.plot(data[:, 0], data[:, 1], 'o--', markersize=2, linewidth=0.8,
             alpha=alpha_value * 0.8)
plt.xlabel("Std")
plt.ylabel("Mean")
plt.title("Std/Mean Trajectories")
plt.grid()
plt.legend(fontsize=7, bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout(rect=[0, 0, 0.8, 1])
plt.savefig(os.path.join(out_dir, "std_mean_trajectory.png"))
plt.close()
print("✓ Saved Std/Mean plot.")
