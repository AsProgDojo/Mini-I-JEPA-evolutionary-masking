# Mini I-JEPA Evolutionary Masking
A CIFAR-scale I-JEPA-inspired self-supervised learning model combined with a genetic algorithm that evolves the target-mask policy. The GA-evolved masking policy is compared against random masking and a fixed I-JEPA-style baseline using linear probe accuracy.
## Setup
```bash
conda env create -f environment.yaml
conda activate ga-mini-i-jepa
```
## Project Structure
```
data/          # Data loading and splits
masking/       # Mask shapes, sampler, visualizer
models/        # ViT encoder, predictor, EMA
training/      # JEPA pretraining loop, linear probe
ga/            # Chromosome, GA operators, GA loop
eval/          # Metrics
configs/       # debug.yaml and full.yaml
scripts/       # Entry point scripts
```
## References
- Assran et al., "Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture", arXiv:2301.08243, 2023.
