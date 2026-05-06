from dataclasses import dataclass
import numpy as np

VALID_SHAPES = [
    'rectangle',
    'square',
    'horizontal_strip',
    'vertical_strip',
    'irregular_blob',
    'hexagon'
]

GENE_RANGES = {
    'num_target_blocks' : [1, 2, 4, 6],
    'target_area_min' : (0.05, 0.15),
    'target_area_max' : (0.15, 0.30),
    'aspect_ratio_min' : (0.5, 1.0),
    'aspect_ratio_max' : (1.0, 2.0),
    'context_area' : (0.7, 0.95)
}

@dataclass
class Chromosome:
    mask_shape: str
    num_target_blocks: int
    target_area_min: float
    target_area_max: float
    aspect_ratio_min: float
    aspect_ratio_max: float
    context_area: float

    def repair(self):
        """
        Enforce validity constraints after crossover or mutation
        - target_area_min < target_area_max
        - aspect_ratio_min < aspect_ratio_max
        - all continuous genes clamped to allowed ranges
        """

        # Clamp continuous genes
        self.target_area_min = float(np.clip(self.target_area_min, 0.05, 0.15))
        self.target_area_max = float(np.clip(self.target_area_max, 0.15, 0.30))
        self.aspect_ratio_min = float(np.clip(self.aspect_ratio_min, 0.5, 1.0))
        self.aspect_ratio_max = float(np.clip(self.aspect_ratio_max, 1.0, 2.0))
        self.context_area = float(np.clip(self.context_area, 0.7, 0.95))

        # Enforce ordering
        if self.target_area_min >= self.target_area_max:
            self.target_area_min, self.target_area_max = sorted([self.target_area_min, self.target_area_max])
        if self.aspect_ratio_min >= self.aspect_ratio_max:
            self.aspect_ratio_min, self.aspect_ratio_max = sorted([self.aspect_ratio_min, self.aspect_ratio_max])
        
        # Clamp discrete genes
        if self.num_target_blocks not in GENE_RANGES['num_target_blocks']:
            valid = GENE_RANGES['num_target_blocks']
            self.num_target_blocks = min(valid, key=lambda x: abs(x - self.num_target_blocks))

        # Clamp categorical gene
        if self.mask_shape not in VALID_SHAPES:
            self.mask_shape = VALID_SHAPES[0]

        return self

def random_chromosome():
    """Create a random valid chromosome"""
    return Chromosome(
        mask_shape=np.random.choice(VALID_SHAPES),
        num_target_blocks=np.random.choice(GENE_RANGES['num_target_blocks']),
        target_area_min=np.random.uniform(*GENE_RANGES['target_area_min']),
        target_area_max=np.random.uniform(*GENE_RANGES['target_area_max']),
        aspect_ratio_min=np.random.uniform(*GENE_RANGES['aspect_ratio_min']),
        aspect_ratio_max=np.random.uniform(*GENE_RANGES['aspect_ratio_max']),
        context_area=np.random.uniform(*GENE_RANGES['context_area'])
    ).repair()

def seed_chromosome():
    """Return the 3-hand crafted seed masking policies"""
    ijepa_style = Chromosome(
        mask_shape="rectangle",
        num_target_blocks=4,
        target_area_min=0.15,
        target_area_max=0.25,
        aspect_ratio_min=0.75,
        aspect_ratio_max=1.5,
        context_area=0.85,
    )

    # Random block-like policy
    random_block = Chromosome(
        mask_shape="rectangle",
        num_target_blocks=2,
        target_area_min=0.05,
        target_area_max=0.20,
        aspect_ratio_min=0.5,
        aspect_ratio_max=2.0,
        context_area=0.80,
    )

    # Irregular blob policy
    blob_policy = Chromosome(
        mask_shape="irregular_blob",
        num_target_blocks=4,
        target_area_min=0.10,
        target_area_max=0.25,
        aspect_ratio_min=0.75,
        aspect_ratio_max=1.5,
        context_area=0.85,
    )

    return [ijepa_style, random_block, blob_policy]