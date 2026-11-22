"""Vision Transformer for MNIST in PyTorch. Parameter names match the keys used by the
export scripts (embedding.*, encoder.N.*, norm.*, classifier.*).
"""

import torch
import torch.nn as nn


# 4 x 4 patches through a strided convolution, then CLS token and learned position embedding.
class EmbedLayer(nn.Module):
    def __init__(self, n_channels, embed_dim, image_size, patch_size):
        super().__init__()
        self.conv1 = nn.Conv2d(n_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)
        self.pos_embedding = nn.Parameter(torch.zeros(1, (image_size // patch_size) ** 2 + 1, embed_dim), requires_grad=True)

    def forward(self, x):
        x = self.conv1(x)
        x = x.reshape([x.shape[0], x.shape[1], -1])
        x = x.transpose(1, 2)
        x = torch.cat((torch.repeat_interleave(self.cls_token, x.shape[0], 0), x), dim=1)
        x = x + self.pos_embedding
        return x


# Multi-head self-attention without an output projection. head_mask (one value per head)
# multiplies the output of each head; it is used for head importance and pruning.
class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.head_mask = None

        self.wq = nn.Linear(self.embed_dim, self.embed_dim)
        self.wk = nn.Linear(self.embed_dim, self.embed_dim)
        self.wv = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, x):
        B, S, E = x.shape

        xq = self.wq(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        xk = self.wk(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        xv = self.wv(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        x_attn = torch.matmul(xq, xk.transpose(-1, -2)) / float(self.head_dim) ** 0.5
        x_attn = torch.softmax(x_attn, dim=-1)
        x = torch.matmul(x_attn, xv)

        if self.head_mask is not None:
            x = x * self.head_mask.to(dtype=x.dtype, device=x.device).view(1, self.num_heads, 1, 1)

        x = x.transpose(1, 2).reshape(B, S, E)
        return x


# Pre-norm block: x + attention(norm1(x)), then x + fc2(GELU(fc1(norm2(x)))).
class Encoder(nn.Module):
    def __init__(self, embed_dim, n_attention_heads, forward_mul):
        super().__init__()
        self.attention = SelfAttention(embed_dim, n_attention_heads)
        self.fc1 = nn.Linear(embed_dim, embed_dim * forward_mul)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(embed_dim * forward_mul, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.fc2(self.activation(self.fc1(self.norm2(x))))
        return x


# Classifier on the CLS token: fc1 -> tanh -> fc2.
class Classifier(nn.Module):
    def __init__(self, embed_dim, n_classes):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, embed_dim)
        self.activation = nn.Tanh()
        self.fc2 = nn.Linear(embed_dim, n_classes)

    def forward(self, x):
        x = x[:, 0, :]
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x


# encoder = 6 blocks followed by a LayerNorm (encoder.6); self.norm is a second LayerNorm.
class VisionTransformer(nn.Module):
    def __init__(self, n_channels, embed_dim, n_layers, n_attention_heads, forward_mul, image_size, patch_size, n_classes):
        super().__init__()
        self.embedding = EmbedLayer(n_channels, embed_dim, image_size, patch_size)
        self.encoder = nn.Sequential(*[Encoder(embed_dim, n_attention_heads, forward_mul) for _ in range(n_layers)], nn.LayerNorm(embed_dim))
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = Classifier(embed_dim, n_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = self.encoder(x)
        x = self.norm(x)
        x = self.classifier(x)
        return x

    # mask: 6 x 4 tensor (blocks x heads), or None to keep every head.
    def set_head_mask(self, mask):
        for layer, block in enumerate(self.encoder[:-1]):
            block.attention.head_mask = None if mask is None else mask[layer]


# D = 64, 6 blocks, 4 heads, MLP hidden 128, 4 x 4 patches, 10 classes.
def build_mnist_vit():
    return VisionTransformer(n_channels=1, embed_dim=64, n_layers=6, n_attention_heads=4,
                             forward_mul=2, image_size=28, patch_size=4, n_classes=10)
