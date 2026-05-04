import numpy as np

GRID_SIZE = 16
NUM_PATCHES = GRID_SIZE * GRID_SIZE #256

def patch_index(r, c):
    """Convert (row and col) to patch index."""
    return r * GRID_SIZE + c

def patch_coords(index):
    """Convert patch index to (row, col)."""
    r = index // GRID_SIZE
    c = index % GRID_SIZE
    return r, c

def sample_rectangle(rng, area_min, area_max, aspect_min, aspect_max):
    """Sample a rectangular block of patches."""
    area = rng.uniform(area_min, area_max) * NUM_PATCHES
    aspect = rng.uniform(aspect_min, aspect_max)

    width = int(round(np.sqrt(area * aspect)))
    height = int(round(np.sqrt(area / aspect)))

    width = np.clip(width, 1, GRID_SIZE)
    height = np.clip(height, 1, GRID_SIZE)

    top = rng.integers(0, GRID_SIZE - height + 1)
    left = rng.integers(0, GRID_SIZE - width + 1)

    indices = set()
    for r in range(top, top + height):
        for c in range(left, left + width):
            indices.add(patch_index(r, c))

    return indices

def sample_square(rng, area_min, area_max):
    """Sample a square block of patches."""
    return sample_rectangle(rng, area_min, area_max, aspect_min=1.0, aspect_max=1.0)

def sample_horizontal_strip(rng, area_min, area_max):
    """Sample a wide, short horizontal strip of patches."""
    return sample_rectangle(rng, area_min, area_max, aspect_min=3.0, aspect_max=6.0)

def sample_vertical_strip(rng, area_min, area_max):
    """Sample a tall, narrow vertical strip of patches."""
    return sample_rectangle(rng, area_min, area_max, aspect_min=0.15, aspect_max=0.33)

def _get_neighbors(index):
    """Return valid 4-connected neighbors of a patch."""
    r, c = patch_coords(index)
    neighbors = set()
    if r > 0:
        neighbors.add(patch_index(r - 1, c))
    if r < GRID_SIZE - 1:
        neighbors.add(patch_index(r + 1, c))
    if c > 0:
        neighbors.add(patch_index(r, c - 1))
    if c < GRID_SIZE - 1:
        neighbors.add(patch_index(r, c + 1))

    return neighbors
    

def sample_irregular_blob(rng, area_min, area_max):
    """
    Sample a connected irregular region using frontier growth. 
    Grows from a random seed patch untill target area is reached.
    Restarts if the frontier gets stuck before reaching target area.
    """
    target_area = int(round(rng.uniform(area_min, area_max) * NUM_PATCHES))
    target_area = max(1, target_area)

    max_attempts = 10
    for _ in range(max_attempts):
        seed = rng.integers(0, NUM_PATCHES)
        blob = {seed}
        frontier = _get_neighbors(seed) - blob

        while len(blob) < target_area and frontier:
            chosen = rng.choice(list(frontier))
            blob.add(chosen)
            frontier.update(_get_neighbors(chosen) - blob)

        if len(blob) >= target_area:
            return blob
    
    # If we fail to grow a large enough blob after max_attempts, return blob we have
    return blob

def _in_hexagon(r, c, center_r, center_c, radius):
    """Check if patch (r, c) falls within a hexagon centered at (center_r, center_c)."""
    dr = abs(r - center_r)
    dc = abs(c - center_c)

    return (dc <= radius) and (dr <= radius * np.sqrt(3) / 2) and (dr / np.sqrt(3) + dc <= radius)

def sample_hexagon(rng, area_min, area_max):
    """
    Sample a regular hexagon of patches.
    Defined by a random center and radius derived from target area.
    Aspect ratio genes are ignored since hexagons are regular.
    A regular hexagon with radius r has area ~ 2.598 * r^2 patches.
    """
    target_area = int(round(rng.uniform(area_min, area_max) * NUM_PATCHES))
    radius = np.sqrt(target_area / 2.598)

    # Random center, kept away from edges so hexagon fits reasonably
    margin = int(np.ceil(radius))
    margin = np.clip(margin, 1, GRID_SIZE // 2 - 1)
    center_r = rng.integers(margin, GRID_SIZE - margin)
    center_c = rng.integers(margin, GRID_SIZE - margin)

    indices = set()
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            if _in_hexagon(r, c, center_r, center_c, radius):
                indices.add(patch_index(r, c))

    # Fallback if hexagon is too small
    if not indices:
        indices.add(patch_index(center_r, center_c))

    return indices