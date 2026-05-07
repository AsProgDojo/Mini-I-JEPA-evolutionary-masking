import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbed(nn.Module):
    """
    Splits an image into patches and projects eacn to embed_dim.
    Implemented as a Conv2d with stride=patch_size
    """
    def __init__(self, image_size=64, patch_size=1, in_channels=3, embed_dim=256):
        super().__init__()
        assert image_size & patch_size == 0, "Image size must be divisible by patch_size"
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
        # x shape: (batch, in_channels = 3, image_size = 64, image_size = 64)
        x = self.proj(x)              # (batch, embed_dim, 16, 16)
        x = x.flatten(2)              # (batch, embed_dim, 256)
        x = x.transpose(1, 2)         # (batch, 256, embed_dim)
        return x
    
