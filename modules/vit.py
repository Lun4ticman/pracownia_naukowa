import torch.nn as nn
import torch
from einops import repeat
from einops.layers.torch import Rearrange

from modules.patch_embedding import PatchEmbedding
from modules.layers import *
from modules.raven import *


class ViT(nn.Module):
    def __init__(self, channels=3, img_size=144, patch_size=4, emb_dim=32, d_ff=32,
                n_layers=12, out_dim=10, dropout=0.1, heads=12):
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
        # self.head = nn.Sequential(nn.LayerNorm(emb_dim), nn.Linear(emb_dim, out_dim), nn.Softmax())

        # Decoding layer
        self.decoder_layers = nn.ModuleList(
            [ViTDecoderLayer(emb_dim, heads, d_ff, dropout) for _ in range(n_layers)])

        self.fc = nn.Linear(emb_dim, out_dim**2)

        self.rearrange = Rearrange('b (n p) (p1 p2) -> b 1 (p p1) (n p2)', p1=patch_size, p2=patch_size, n=num_patches//2, p=num_patches//2)


    def forward(self, img, mask=None): 
        # Get patch embedding vectors
        x = self.patch_embedding(img)
        b, n, _ = x.shape

        # # Add cls token to inputs
        # cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        # x = torch.cat([cls_tokens, x], dim=1)
        # x += self.pos_embedding[:, :(n + 1)]

        # Transformer layers
        for i in range(self.n_layers):
            x = self.encoder_layers[i](x)

        # Transformer Decoder layers with masking
        for i in range(self.n_layers):
            x = self.decoder_layers[i](x, memory=x, mask=mask)

        x = self.fc(x)

        x = self.rearrange(x)

        return x