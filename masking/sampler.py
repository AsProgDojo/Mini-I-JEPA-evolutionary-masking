import numpy as np
from masking.shapes import (
    sample_rectangle,
    sample_square,
    sample_horizontal_strip,
    sample_vertical_strip,
    sample_irregular_blob,
    sample_hexagon,
    NUM_PATCHES
)

SHAPE_SAMPLERS = {
    "rectangle": sample_rectangle,
    "square" : sample_square,
    "horizontal_strip" : sample_horizontal_strip,
    "vertical_strip" : sample_vertical_strip,
    "irregular_blob" : sample_irregular_blob,
    "hexagon" : sample_hexagon
}

def sample_one_block(rng, shape, area_min, area_max, aspect_min, aspect_max):
    """Sample a single target block using the appropriate shape function."""
    if shape in ("square", "horizontal_strip", "vertical_strip", "irregular_blob", "hexagon"):
        return SHAPE_SAMPLERS[shape](rng, area_min, area_max)
    else:
        return SHAPE_SAMPLERS[shape](rng, area_min, area_max, aspect_min, aspect_max)
    
def sample_target_blocks(rng, shape, num_blocks,area_min, area_max, aspect_min, aspect_max, max_attempts=20):
    """
    Sample num_blocks non-overlapping target blocks.
    Resamples a block if it overlaps with already chosen blocks.
    Returns the union of all target patch indices.
    """
    all_target_indices = set()

    for _ in range(num_blocks):
        for attempt in range(max_attempts):
            block = sample_one_block(rng, shape, area_min, area_max, aspect_min, aspect_max)

            if block.isdisjoint(all_target_indices):
                all_target_indices.update(block)
                break
        else:
            all_target_indices.update(block)
        
    return all_target_indices

def sample_context(rng, context_area, target_indices):
    """
    Sample a context region of size context_area * NUM_PATCHES from the full grid.
    Then remove any patches that overlap with target indices.
    Returns context patch indices.
    """
    context_size = int(round(context_area * NUM_PATCHES))

    all_patches = list(range(NUM_PATCHES))
    rng.shuffle(all_patches)
    context_indices = set(all_patches[:context_size])

    # Remove target-overlapping patches from context
    context_indices -= target_indices
    return context_indices

def sample_masks(chromosome, rng):
    """
    Main entry point. Given a chromosome and rng, returns:
        - target_indices: set of patch indices that model must predict
        - context_indices: set of patch indices the encoder sees
        - all_indices: all of 256 patches (for target encoder)
    """
    shape = chromosome.shape
    num_blocks = chromosome.num_blocks
    area_min = chromosome.area_min
    area_max = chromosome.area_max
    aspect_min = chromosome.aspect_min
    aspect_max = chromosome.aspect_max
    context_area = chromosome.context_area

    target_indices = sample_target_blocks(rng, shape, num_blocks, area_min, area_max, aspect_min, aspect_max)
    context_indices = sample_context(rng, context_area, target_indices)
    all_indices = set(range(NUM_PATCHES))

    return target_indices, context_indices, all_indices