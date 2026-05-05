import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
from masking.shapes import GRID_SIZE, patch_coords

def visualize_masks(image, target_indices, context_indices, title="Mask Vizualization", save_path=None):
    """
    Visualize target and context masks overlaid on an image.

    image: numpy array of shape (3, 64, 64) (CHW dimension order which is used by PyTorch) or (64, 64, 3) (HWC dimension order which is used by Matplotlib), values in [0, 1]
    target_indices: set of target patch indices (red)
    context_indices: set of context patch indices (green)
    save_path: full path to save PNG, if None just shows
    """
    patch_size = 64 // GRID_SIZE # 4 pixels per patch

    #Convert image to HWC format if needed
    if image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))

    #Clip to [0, 1] for display
    image = np.clip(image, 0, 1)
    fig, ax = plt.subplots(figsize=(6,6))
    ax.imshow(image)

    for idx in range(GRID_SIZE * GRID_SIZE):
        r, c = patch_coords(idx)
        x = c * patch_size
        y = r * patch_size

        if idx in target_indices:
            rect = mpatches.Rectangle((x, y), patch_size, patch_size, linewidth=0, facecolor='red', alpha=0.5)
            ax.add_patch(rect)
        elif idx not in context_indices:
            # Patches that are neither target nor context — dim them
            rect = mpatches.Rectangle((x, y), patch_size, patch_size, linewidth=0, facecolor='black', alpha=0.4)
            ax.add_patch(rect)
        
    # Draw patch grid lines
    for i in range(1, GRID_SIZE):
        ax.axhline(i * patch_size - 0.5, color='white', linewidth=0.3, alpha=0.5)
        ax.axvline(i * patch_size - 0.5, color='white', linewidth=0.3, alpha=0.5)
    
    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='red', alpha=0.5, label='Target (must predict)'),
        mpatches.Patch(facecolor='none', edgecolor='none', label='Context (visible)'),
        mpatches.Patch(facecolor='black', alpha=0.4, label='Neither')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
    ax.set_title(title, fontsize=10)
    ax.axis('off')

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
    else:
        plt.show()

def visualize_all_shapes(image, chromosome_configs, save_dir, seed=42):
    """
    Generate one visualization per mask shape and save to save_dir.

    chromosome_configs: list of dicts, each with keys matching chromosome attributes
    save_dir: directory to save PNGs
    """
    from masking.sampler import sample_masks

    rng = np.random.default_rng(seed)
    for cfg in chromosome_configs:
        shape_name = cfg['mask_shape']
        target_indices, context_indices, _ = sample_masks(cfg, rng)

        save_path = os.path.join(save_dir, f"{shape_name}_mask.png")
        visualize_masks(
            image=image,
            target_indices=target_indices,
            context_indices=context_indices,
            title=f'Shape: {shape_name}',
            save_path=save_path
        )
        print(f"Saved: {save_path}")