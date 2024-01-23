import torch.nn as nn
import torch
from einops import repeat

from modules.patch_embedding import PatchEmbedding
from modules.layers import *
from modules.raven import *


class ViT(nn.Module):
    def __init__(self, channels=3, img_size=144, patch_size=4, emb_dim=32, d_ff=32,
                n_layers=6, out_dim=37, dropout=0.1, heads=2):
        super(ViT, self).__init__()

        # Attributes
        self.channels = channels
        self.height = img_size
        self.width = img_size
        self.patch_size = patch_size
        self.n_layers = n_layers

        # Patching
        self.patch_embedding = PatchEmbedding(in_channels=channels,
                                              patch_size=patch_size,
                                              emb_size=emb_dim)
        # Learnable params
        num_patches = (img_size // patch_size) ** 2
        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_patches + 1, emb_dim), requires_grad=True)
        self.cls_token = nn.Parameter(torch.rand(1, 1, emb_dim), requires_grad=True)

        # Transformer Encoder
        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(emb_dim, heads, d_ff, dropout) for _ in range(n_layers)])

        # Classification head
        self.head = nn.Sequential(nn.LayerNorm(emb_dim), nn.Linear(emb_dim, out_dim), nn.Softmax())

    def forward(self, img): 
        # Get patch embedding vectors
        x = self.patch_embedding(img)
        b, n, _ = x.shape

        # Add cls token to inputs
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat([cls_tokens, x], dim=1)
        x += self.pos_embedding[:, :(n + 1)]

        # Transformer layers
        for i in range(self.n_layers):
            x = self.encoder_layers[i](x)

        return self.head(x[:, 0, :])