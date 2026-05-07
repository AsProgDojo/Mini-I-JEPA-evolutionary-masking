import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbed(nn.Module):
    """
    Splits an image into patches and projects eacn to embed_dim.
    Implemented as a Conv2d with stride=patch_size
    """
    def __init__(self, image_size=64, patch_size=4, in_channels=3, embed_dim=256):
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
    
class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention from stratch.
    """
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Single combined projection for Q, K, V (more efficient than three separate)
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, x):
        # x: (batch, seq, embed_dim)
        batch, seq, _ = x.shape

        # Project to Q, K, V
        qkv = self.qkv(x)                                                   # (batch, seq, 3 * embed_dim)
        qkv = qkv.reshape(batch, seq, 3, self.num_heads, self.head_dim)     
        qkv = qkv.permutate(2, 0, 3, 1, 4)                                  # (3, batch, num_heads, seq, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]                                    # each: (batch, num_heads, seq, head_dim)

        # scaled dot-product attention
        scores = q @ k.transpose(-2, -1)                                    # (batch, num_heads, seq, head_dim)
        scores = scores / (self.head_dim ** 0.5)
        attn = F.softmax(scores, dim = -1)

        # Apply attention to values
        out = attn @ v                                                      # (batch, num_heads, seq, head_dim)

        # Concatenate heads
        out = out.transpose(1, 2)                                           # (batch, seq, num_heads, head_dim)
        out = out.reshape(batch, seq, self.embed_dim)                       # (batch, seq, embed_dim)

        # Final projection
        out = self.out_proj(out)
        return out
    
class MLP(nn.Module):
    """
    Two-layer MLP with GELU activation. Expands by mlp_ratio then projects back.
    """
    def __init__(self, embed_dim, mlp_ratio=4):
        super().__init__()
        hidden_dim = embed_dim * mlp_ratio
        self.fc1 = nn.Linear(embed_dim, hidden_dim)     # 'fc' stand for fully connected, as in fully connected layer. It is another name for linear layer (nn.Linear)
        self.act = nn.GELU()                            # 'act' stands for activation function
        self.fc2 = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class TransformerBlock(nn.Module):
    """
    Pre-norm transformer blcok: LayerNorm before attention/MLP, with residual connections.
    """
    def __init__(self, embed_dim, num_heads, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_ratio)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class Encoder(nn.Module):
    """
    ViT encoder. Can process either the full image (target encoder) or a subset of patches
    """
    def __init__(self, image_size=64, patch_size=4, in_channels=3, embed_dim=192, depth=4, num_heads=4, mlp_ratio=4):
        super().__init__()
        self.patch_embed = PatchEmbed(image_size, patch_size, in_channels, embed_dim)
        self.num_patches = self.patch_embed.num_patches
        self_embed_dim = embed_dim

        # Learnable positional embedding, one per patch position
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Stack of transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, images, patch_indices=None):
        # images: (batch, 3, 64, 64)
        x = self.patch_embed(images)                                        # (batch, 256, embed_dim)
        x = x + self.pos_embed                                              # add positional embedding to all patches first

        # If subset specified, gather only those patches
        if patch_indices is not None:
            # patch_indices: (num_kept,) tensor of patch indices
            x = x[:, patch_indices, :]                                      # (batch, num_kept, embed_dim)
        
        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x