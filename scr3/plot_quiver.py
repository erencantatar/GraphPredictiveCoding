import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
import wandb

# --- Configuration ---
use_all_runs = False

# run_ids = ["uqqkm2uq", "1rx1pf9f"]  # "svloxh0l", "lp3o6d1s", "jozn66av"

# legend_replacements = {
#     "uqqkm2uq": "Acc: 0.84",
#     "1rx1pf9f": "Acc: 0.94",
#     # Add more replacements as needed
# }



# run_ids = ["9ojg2wg4", "8bbuenop", "nq9qdym2", "7swwe5np", "z3gjfool"]
# run_ids = ["z3gjfool", "8bbuenop", "nq9qdym2", "igrcow02", "9ojg2wg4", "7ci6q50q", "yykum5ef", "dgsl86md"]
run_ids = ["z3gjfool", "8bbuenop", "nq9qdym2", "igrcow02", "9ojg2wg4", "7ci6q50q"]

legend_replacements = {
    # "uqqkm2uq": "Acc: 0.84",
    # "1rx1pf9f": "Acc: 0.94",
    "z3gjfool": "baseline",
    "8bbuenop": "lr2",
    "nq9qdym2": "lr3",
    "8bbuenop": "T5",
    "igrcow02": "T6",
    "9ojg2wg4": "init_8",
    "7ci6q50q": "10_Grokfast_T",
    "yykum5ef": "12_MPL like",
    "dgsl86md": "12_Weight 0.05",

    

    # Add more replacements as needed
}


# 1 LR ----0.94----
# 2
# 3
# 4 T 15: ----0.94----
# 5 50 : 8bbuenop: 0.97
# 6 100: igrcow02: 0.97
# 7 init  ----0.94----
# 8   0.94:   9ojg2wg4
# 9 Grokfast False ----0.94----
# 10 True: 7ci6q50q : 0.94
# 11  Weight              0.005 ----0.94----
# 12  MPL like: yykum5ef
# 13  0.05 dgsl86md

run_ids = list(dict.fromkeys(run_ids))  # preserve order, dedupe
display_label_map = {rid: legend_replacements.get(rid, rid) for rid in run_ids}

base_dir = "jobs/plotting/trajectories"
out_dir = "jobs/plotting/outputs_trajectories/converge_instability8/"


os.makedirs(out_dir, exist_ok=True)

if use_all_runs:
    run_ids = [
        d for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d))
    ]
    # refresh display map if we changed run_ids
    display_label_map = {rid: legend_replacements.get(rid, rid) for rid in run_ids}

num_runs = len(run_ids)
alpha_value = max(0.2, min(0.8, 1.5 / num_runs))
arrow_width = max(0.002, 0.01 / num_runs)
head_width = max(2, 8 - 0.5 * num_runs)
arrow_scale = max(0.4, min(1.5, 3.0 / num_runs))

# --- W&B API (optional, used for logging but not for grouping) ---
api = wandb.Api()
ENTITY = "etatar-atdamen"
PROJECT = "PredCod"

# --- Load weights ---
flat_weights_all = []
flat_weights_by_run = []
std_mean_data = []

print("Collecting runs:")
for run_id in run_ids:
    # Optional: fetch and print metadata
    try:
        run = api.run(f"{ENTITY}/{PROJECT}/{run_id}")
        raw_flag = run.config.get("use_grokfast", "False")
        print(f"  {run_id}: use_grokfast={raw_flag}")
    except Exception as e:
        print(f"  {run_id}: (no W&B metadata) {e}")

    weight_path = os.path.join(base_dir, run_id, "weights.npy")
    if not os.path.exists(weight_path):
        print(f"    ✗ Missing: {weight_path}")
        continue

    try:
        weights = np.load(weight_path)  # shape: [T, N, N]
        flat = weights.reshape(weights.shape[0], -1)  # [T, N*N]
        flat_weights_by_run.append((run_id, flat))
        flat_weights_all.append(flat)

        stds = np.std(flat, axis=1)   # [T]
        means = np.mean(flat, axis=1) # [T]
        std_mean_data.append((run_id, stds, means))

        print(f"    ✓ Loaded {run_id} → weights shape: {weights.shape}")
    except Exception as e:
        print(f"    ✗ Failed to load {run_id}: {e}")

if len(flat_weights_all) == 0:
    print("✗ No valid weight files found. Exiting.")
    raise SystemExit

# --- PCA on combined data (let sklearn handle centering) ---
flat_weights_all = np.concatenate(flat_weights_all, axis=0)  # [sum_T, N*N]

pca = PCA(n_components=10)
pca_vals_all = pca.fit_transform(flat_weights_all)
explained_var = pca.explained_variance_ratio_

# --- Project each run with the same PCA model ---
projected_pca_data = []
for run_id, flat in flat_weights_by_run:
    proj = pca.transform(flat)  # [T, 10]
    projected_pca_data.append((run_id, proj))

# --- Build a distinct color per run-id ---
def make_colors(n):
    # Prefer tab20 up to 20; otherwise use hsv for more distinct hues
    if n <= 20:
        cmap = plt.cm.get_cmap("tab20", n)
    else:
        cmap = plt.cm.get_cmap("hsv", n)
    return [cmap(i) for i in range(n)]

colors = make_colors(num_runs)
color_map_run = {rid: colors[i] for i, rid in enumerate(run_ids)}

# ===========================
# Explained variance plot
# ===========================
plt.figure(figsize=(6, 4))
plt.plot(np.arange(1, len(explained_var) + 1), explained_var, 'o-', linewidth=2)
plt.title("Explained Variance Spectrum (Combined Runs)")
plt.xlabel("Principal Component")
plt.ylabel("Explained Variance Ratio")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "pca_explained_variance_combined.png"))
plt.close()
print("✓ Saved explained variance plot.")

# ===========================
# PCA Trajectories (per-run) – individual figure
# ===========================
plt.figure(figsize=(8, 6))
added_labels = set()

for run_id, data in projected_pca_data:
    color = color_map_run[run_id]
    label_txt = display_label_map[run_id]

    # Lines
    plt.plot(
        data[:, 0], data[:, 1],
        'o--', color=color,
        alpha=alpha_value * 0.9,
        markersize=2, linewidth=0.8
    )

    # Arrows (quiver)
    x, y = data[:-1, 0], data[:-1, 1]
    u, v = data[1:, 0] - x, data[1:, 1] - y
    if run_id not in added_labels:
        plt.quiver(
            x, y, u, v,
            angles='xy', scale_units='xy', scale=arrow_scale,
            width=arrow_width, headwidth=head_width,
            color=color, alpha=alpha_value, label=label_txt
        )
        added_labels.add(run_id)
    else:
        plt.quiver(
            x, y, u, v,
            angles='xy', scale_units='xy', scale=arrow_scale,
            width=arrow_width, headwidth=head_width,
            color=color, alpha=alpha_value
        )

plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.title("PCA Trajectories by Run (Combined PCA Space)")
plt.grid(True)
plt.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1))
plt.tight_layout(rect=[0, 0, 0.8, 1])
plt.savefig(os.path.join(out_dir, "pca_trajectory_combined_by_run.png"))
plt.close()
print("✓ Saved PCA trajectory plot (labels use legend_replacements).")

# ===========================
# Std/Mean Trajectories (per-run) – individual figure
# ===========================
plt.figure(figsize=(9.5, 6))
added_labels = set()

for run_id, stds, means in std_mean_data:
    color = color_map_run[run_id]
    label_txt = display_label_map[run_id]

    # Lines
    plt.plot(
        stds, means,
        'o--', color=color,
        alpha=alpha_value * 0.9,
        markersize=2, linewidth=0.8
    )

    # Arrows (quiver)
    x, y = stds[:-1], means[:-1]
    u, v = stds[1:] - x, means[1:] - y
    if run_id not in added_labels:
        plt.quiver(
            x, y, u, v,
            angles='xy', scale_units='xy', scale=arrow_scale,
            width=arrow_width, headwidth=head_width,
            color=color, alpha=alpha_value, label=label_txt
        )
        added_labels.add(run_id)
    else:
        plt.quiver(
            x, y, u, v,
            angles='xy', scale_units='xy', scale=arrow_scale,
            width=arrow_width, headwidth=head_width,
            color=color, alpha=alpha_value
        )

# --- Add extra room on the Std (x) axis ---
ax = plt.gca()
all_stds = np.concatenate([s for _, s, _ in std_mean_data])
xmin, xmax = float(np.min(all_stds)), float(np.max(all_stds))
rng = xmax - xmin
pad = 0.15 * rng if rng > 0 else 0.05 * max(1.0, abs(xmax) if xmax != 0 else 1.0)
ax.set_xlim(xmin - pad, xmax + pad)
ax.margins(x=0.05)

plt.xlabel("Std")
plt.ylabel("Mean")
plt.title("Std/Mean Trajectories by Run")
plt.grid(True)
plt.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1))
plt.tight_layout(rect=[0, 0, 0.8, 1])
plt.savefig(os.path.join(out_dir, "std_mean_trajectory_by_run.png"))
plt.close()
print("✓ Saved Std/Mean plot (labels use legend_replacements).")

# ===========================
# COMBINED FIGURE (stacked) with a SINGLE SHARED LEGEND
# ===========================
fig, axes = plt.subplots(2, 1, figsize=(9.5, 12), constrained_layout=False)

# --- Top: PCA trajectories ---
ax1 = axes[0]
for run_id, data in projected_pca_data:
    color = color_map_run[run_id]
    # lines
    ax1.plot(
        data[:, 0], data[:, 1],
        'o--', color=color, alpha=alpha_value * 0.9,
        markersize=2, linewidth=0.8
    )
    # arrows
    x, y = data[:-1, 0], data[:-1, 1]
    u, v = data[1:, 0] - x, data[1:, 1] - y
    ax1.quiver(
        x, y, u, v,
        angles='xy', scale_units='xy', scale=arrow_scale,
        width=arrow_width, headwidth=head_width,
        color=color, alpha=alpha_value
    )
ax1.set_xlabel("PCA 1")
ax1.set_ylabel("PCA 2")
ax1.set_title("PCA Trajectories by Run (Combined PCA Space)")
ax1.grid(True)

# --- Bottom: Std/Mean trajectories ---
ax2 = axes[1]
for run_id, stds, means in std_mean_data:
    color = color_map_run[run_id]
    # lines
    ax2.plot(
        stds, means,
        'o--', color=color, alpha=alpha_value * 0.9,
        markersize=2, linewidth=0.8
    )
    # arrows
    x, y = stds[:-1], means[:-1]
    u, v = stds[1:] - x, means[1:] - y
    ax2.quiver(
        x, y, u, v,
        angles='xy', scale_units='xy', scale=arrow_scale,
        width=arrow_width, headwidth=head_width,
        color=color, alpha=alpha_value
    )

# padding on x for std axis
all_stds = np.concatenate([s for _, s, _ in std_mean_data])
xmin, xmax = float(np.min(all_stds)), float(np.max(all_stds))
rng = xmax - xmin
pad = 0.15 * rng if rng > 0 else 0.05 * max(1.0, abs(xmax) if xmax != 0 else 1.0)
ax2.set_xlim(xmin - pad, xmax + pad)
ax2.margins(x=0.05)

ax2.set_xlabel("Std")
ax2.set_ylabel("Mean")
ax2.set_title("Std/Mean Trajectories by Run")
ax2.grid(True)

# --- Single shared legend (using consistent colors + legend_replacements)
handles = [
    Line2D([0], [0], color=color_map_run[rid], marker='o', linestyle='--', linewidth=1.2, markersize=4)
    for rid in run_ids
]
labels = [display_label_map[rid] for rid in run_ids]
# put legend on the right, shared for both subplots
fig.legend(handles, labels, fontsize=9, loc="upper left", bbox_to_anchor=(1.02, 1.0))
plt.tight_layout(rect=[0, 0, 0.8, 1])  # leave room on the right for legend

combined_path = os.path.join(out_dir, "combined_pca_and_stdmean_stacked.png")
plt.savefig(combined_path, dpi=150)
plt.close()
print(f"✓ Saved combined stacked plot with shared legend → {combined_path}")

from matplotlib.lines import Line2D

# ===========================
# Combined figure (stacked) with one legend + bigger fonts/lines
# ===========================
LW = 2.2           # line width
MS = 3.5           # marker size
TICK_FONTSIZE = 14
LABEL_FONTSIZE = 16
TITLE_FONTSIZE = 22

fig, axes = plt.subplots(
    nrows=2, ncols=1, figsize=(10, 12),
    gridspec_kw={"height_ratios": [1, 1]}, constrained_layout=True
)
ax0, ax1 = axes

# -------- PCA Trajectories (top) --------
for run_id, data in projected_pca_data:
    color = color_map_run[run_id]
    ax0.plot(
        data[:, 0], data[:, 1], 'o--', color=color,
        alpha=alpha_value * 0.9, markersize=MS, linewidth=LW
    )
    x, y = data[:-1, 0], data[:-1, 1]
    u, v = data[1:, 0] - x, data[1:, 1] - y
    ax0.quiver(
        x, y, u, v,
        angles='xy', scale_units='xy', scale=arrow_scale,
        width=arrow_width * 1.4, headwidth=head_width * 1.1,
        color=color, alpha=alpha_value
    )

ax0.set_xlabel("PCA 1", fontsize=LABEL_FONTSIZE)
ax0.set_ylabel("PCA 2", fontsize=LABEL_FONTSIZE)
ax0.grid(True, linewidth=0.6)
ax0.tick_params(axis='both', labelsize=TICK_FONTSIZE)

# -------- Std/Mean Trajectories (bottom) --------
for run_id, stds, means in std_mean_data:
    color = color_map_run[run_id]
    ax1.plot(
        stds, means, 'o--', color=color,
        alpha=alpha_value * 0.9, markersize=MS, linewidth=LW
    )
    x, y = stds[:-1], means[:-1]
    u, v = stds[1:] - x, means[1:] - y
    ax1.quiver(
        x, y, u, v,
        angles='xy', scale_units='xy', scale=arrow_scale,
        width=arrow_width * 1.4, headwidth=head_width * 1.1,
        color=color, alpha=alpha_value
    )

# Add extra room on Std (x) axis for bottom plot (same logic as before)
all_stds = np.concatenate([s for _, s, _ in std_mean_data])
xmin, xmax = float(np.min(all_stds)), float(np.max(all_stds))
rng = xmax - xmin
pad = 0.15 * rng if rng > 0 else 0.05 * max(1.0, abs(xmax) if xmax != 0 else 1.0)
ax1.set_xlim(xmin - pad, xmax + pad)
ax1.margins(x=0.05)

ax1.set_xlabel("Std", fontsize=LABEL_FONTSIZE)
ax1.set_ylabel("Mean", fontsize=LABEL_FONTSIZE)
ax1.grid(True, linewidth=0.6)
ax1.tick_params(axis='both', labelsize=TICK_FONTSIZE)

# -------- Single legend (shared) using legend_replacements --------
custom_handles = []
custom_labels = []
for rid in run_ids:
    color = color_map_run[rid]
    label = legend_replacements.get(rid, rid)
    custom_handles.append(Line2D([0], [0], color=color, marker='o',
                                 linestyle='--', linewidth=LW, markersize=MS))
    custom_labels.append(label)

# Put legend to the right, centered vertically
fig.legend(custom_handles, custom_labels,
           loc='center right', bbox_to_anchor=(1.02, 0.5), fontsize=12)

# -------- Single title --------
fig.suptitle("weights trajectories", fontsize=TITLE_FONTSIZE, y=0.98)

# Save combined figure
combined_path = os.path.join(out_dir, "weights_trajectories_combined.png")
fig.savefig(combined_path, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"✓ Saved combined stacked plot → {combined_path}")
