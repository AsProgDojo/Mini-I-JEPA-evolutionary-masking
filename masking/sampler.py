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
    
def sample_target_blocks(rng, shape, num_blocks, area_min, area_max, aspect_min, aspect_max):
    """
    Sample num_blocks target blocks, allowing overlap between them.
    Returns the union of all target patch indices.
    """
    all_target_indices = set()
    for _ in range(num_blocks):
        block = sample_one_block(rng, shape, area_min, area_max, aspect_min, aspect_max)
        all_target_indices.update(block)
    return all_target_indices

def sample_context(rng, context_area, target_indices):
    """
    Sample a context region as a single rectangular block of size 
    context_area * NUM_PATCHES with aspect ratio near 1.0.
    Then remove any patches that overlap with target indices.
    """
    from masking.shapes import sample_rectangle
    
    context_indices = sample_rectangle(
    rng,
    area_min=context_area,
    area_max=context_area,
    aspect_min=0.75,
    aspect_max=1.5,
    )
    
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
    shape = chromosome.mask_shape
    num_blocks = chromosome.num_target_blocks
    area_min = chromosome.area_min
    area_max = chromosome.area_max
    aspect_min = chromosome.aspect_min
    aspect_max = chromosome.aspect_max
    context_area = chromosome.context_area

    target_indices = sample_target_blocks(rng, shape, num_blocks, area_min, area_max, aspect_min, aspect_max)
    context_indices = sample_context(rng, context_area, target_indices)
    all_indices = set(range(NUM_PATCHES))

    return target_indices, context_indices, all_indices