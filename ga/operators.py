import numpy as np
from ga.chromosome import Chromosome, VALID_SHAPES, GENE_RANGES

MUTATION_RATE = 0.01
MUTATION_RATE_HIGH = 0.05  # Used if diversity drops too early
GAUSSIAN_NOISE_STD = 0.02  # Standard deviation (std) continuous gene mutation

def tournament_selection(population, fitnesses, rng, tournament_size=3):
    """
    Select one individual via tournament selection
    Randomly pick tournament_size individuals, return the one with highest fitness
    """
    indices = rng.choice(len(population), size=tournament_size, replace=False)
    best_idx = max(indices, key=lambda i: fitnesses[i])
    return population[best_idx]

def uniform_crossover(parent1, parent2, rng):
    """
    Produce one child by uniform crossover.
    For each gene, randomly pick from parent1 or parent2 with 50% chance
    For continuous genes, occasionally average the two values with small Gaussian noise
    """

    def pick(val1, val2):
        if rng.random() < 0.5:
            return val1
        else:
            return val2
    
    def pick_continuous(val1, val2):
        if rng.random() < 0.4:
            return val1
        elif rng.random() < 0.8:
            return val2
        else:
            avg = (val1 + val2) / 2
            noise = rng.normal(0, GAUSSIAN_NOISE_STD)
            return avg + noise
    
    child = Chromosome(
        mask_shape=pick(parent1.mask_shape, parent2.mask_shape),
        num_target_blocks=pick(parent1.num_target_blocks, parent2.num_target_blocks),
        target_area_min=pick_continuous(parent1.target_area_min, parent2.target_area_min),
        target_area_max=pick_continuous(parent1.target_area_max, parent2.target_area_max),
        aspect_ratio_min=pick_continuous(parent1.aspect_ratio_min, parent2.aspect_ratio_min),
        aspect_ratio_max=pick_continuous(parent1.aspect_ratio_max, parent2.aspect_ratio_max),
        context_area=pick_continuous(parent1.context_area, parent2.context_area),
    )

    return child.repair()

def mutate(chromosome, rng, mutation_rate=MUTATION_RATE):
    """
    Mutate each gene independently with probability mutation_rate.
    - Categorical: replace with another valid shape
    - Discrete: replace with another valid num_target_blocks value
    - Continuous: add Gaussian noise, then repair with clamp
    """

    mask_shape = chromosome.mask_shape
    num_target_blocks = chromosome.num_target_blocks
    target_area_min = chromosome.target_area_min
    target_area_max = chromosome.target_area_max
    aspect_ratio_min = chromosome.aspect_ratio_min
    aspect_ratio_max = chromosome.aspect_ratio_max
    context_area = chromosome.context_area

    # Categorical mutation
    if rng.random() < mutation_rate:
        mask_shapes = [s for s in VALID_SHAPES if s != mask_shape]
        mask_shape = rng.choice(mask_shapes)
    
    # Discrete mutation
    if rng.random() < mutation_rate:
        other_blocks = [b for b in GENE_RANGES['num_target_blocks'] if b != num_target_blocks]
        num_target_blocks = rng.choice(other_blocks)

    # Continuous mutation
    if rng.random() < mutation_rate:
        target_area_min += rng.normal(0, GAUSSIAN_NOISE_STD)
    if rng.random() < mutation_rate:
        target_area_max += rng.normal(0, GAUSSIAN_NOISE_STD)
    if rng.random() < mutation_rate:
        aspect_ratio_min += rng.normal(0, GAUSSIAN_NOISE_STD)
    if rng.random() < mutation_rate:
        aspect_ratio_max += rng.normal(0, GAUSSIAN_NOISE_STD)
    if rng.random() < mutation_rate:
        context_area += rng.normal(0, GAUSSIAN_NOISE_STD)

    
    mutated = Chromosome(
        mask_shape=mask_shape,
        num_target_blocks=num_target_blocks,
        target_area_min=target_area_min,
        target_area_max=target_area_max,
        aspect_ratio_min=aspect_ratio_min,
        aspect_ratio_max=aspect_ratio_max,
        context_area=context_area,
    )

    return mutated.repair()

def elitism(population, fitnesses, num_elites=2):
    """Return top n individuals from the population based on fitness"""
    sorted_indices = sorted(range(len(population)), key=lambda i: fitnesses[i], reverse=True)
    return [population[i] for i in sorted_indices[:num_elites]]


def compute_diversity(population):
    """
    Measure diversity as mean pairwise Euclidean distance between individuals
    in normalized gene space (each gene scaled to [0, 1]).
    Categorical genes contribute 0 or 1 (differen shapes count as distance 1).
    """
    n = len(population)
    if n < 2:
        return 0.0

    def normalize(value, gene_name):
        lo, hi = GENE_RANGES[gene_name]
        return (value - lo) / (hi - lo)

    # Normalize each individual's genes to [0, 1] range
    vectors = []
    for c in population:
        shape_idx = VALID_SHAPES.index(c.mask_shape) / (len(VALID_SHAPES) - 1)
        num_blocks_list = GENE_RANGES['num_target_blocks']
        num_blocks_idx = num_blocks_list.index(c.num_target_blocks) / (len(num_blocks_list) - 1)

        vectors.append([
            shape_idx,
            num_blocks_idx,
            normalize(c.target_area_min, 'target_area_min'),
            normalize(c.target_area_max, 'target_area_max'),
            normalize(c.aspect_ratio_min, 'aspect_ratio_min'),
            normalize(c.aspect_ratio_max, 'aspect_ratio_max'),
            normalize(c.context_area, 'context_area'),
        ])

    # Compute mean pairwise distance
    total_distance = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += np.linalg.norm(vectors[i] - vectors[j])
            count += 1
    
    return total_distance / count

def get_mutation_rate(population, low_diversity_threshold=0.3):
    """
    Return appropriate mutation rate based on mean pairwise distance.
    Threshold of 0.3 in normalized space = population has converged significantly.
    """
    diversity = compute_diversity(population)

    if diversity < low_diversity_threshold:
        print(f"Low diversity (mean pairwise distance = {diversity:.3f}). Increasing mutation rate to {MUTATION_RATE_HIGH}")
        return MUTATION_RATE_HIGH
    else:
        return MUTATION_RATE