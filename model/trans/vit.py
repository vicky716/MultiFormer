import torch
import torch.nn as nn
import numpy as np
from einops import rearrange, repeat
from utils.visualization import attentionheatmap_visual, attentionheatmap_visual2


class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, head_num):
        super().__init__()

        self.head_num = head_num
        self.dk = (embedding_dim // head_num) ** (1 / 2)
        self.qkv_layer = nn.Linear(embedding_dim, embedding_dim * 3, bias=False)
        '''
        self.to_qkv1 = nn.Conv2d(embedding_dim, embedding_dim * 3 // 2, kernel_size=1, padding=0, bias=False) #1x1 256x3
        self.to_qkv2 = nn.Conv2d(embedding_dim, embedding_dim * 3 // 4, kernel_size=3, padding=1, bias=False) #3x3 128x3
        self.to_qkv3 = nn.Conv2d(embedding_dim, embedding_dim * 3 // 8, kernel_size=5, padding=2, bias=False) #5x5 64x3
        self.to_qkv4 = nn.Conv2d(embedding_dim, embedding_dim * 3 // 8, kernel_size=7, padding=3, bias=False) #7x7 64x3
        '''
        self.out_attention = nn.Linear(embedding_dim, embedding_dim, bias=False)

    def forward(self, x, mask=None):
        '''
        qkv1 = self.to_qkv1(p).chunk(3, dim=2)  # x: (16, 257, 1024) qkv1: (16, 257, 3072/2/3)
        qkv2 = self.to_qkv2(x).chunk(3, dim=1) # x: (16, 257, 1024) qkv2 :(16, 257, 3072/4/3)
        qkv3 = self.to_qkv3(x).chunk(3, dim=1) # x: (16, 257, 1024) qkv3: (16, 257, 3072/8/3)
        qkv4 = self.to_qkv4(x).chunk(3, dim=1) # x: (16, 257, 1024) qkv4: (16, 257, 3072/8/3)
        q = torch.cat((qkv1[0],qkv2[0],qkv3[0],qkv4[0]),dim=2) #(16, 257, 1024)
        k = torch.cat((qkv1[1],qkv2[1],qkv3[1],qkv4[1]),dim=2) #(16, 257, 1024)
        v = torch.cat((qkv1[2],qkv2[2],qkv3[2],qkv4[2]),dim=2) #(16, 257, 1024)
        query = rearrange(q, 'b t (d h ) -> b h t d ', h=self.head_num)
        key = rearrange(k, 'b t (d h ) -> b h t d ', h=self.head_num)
        value = rearrange(v, 'b t (d h ) -> b h t d ', h=self.head_num)
        '''
        # q = rearrange(q, 'b (g d) h w -> b g (h w) d', g=self.heads)
        qkv = self.qkv_layer(x) # qkv: (16, 257, 3072)  x: (16, 257, 1024)
        query, key, value = tuple(rearrange(qkv, 'b t (d k h ) -> k b h t d ', k=3, h=self.head_num)) # (16, 8, 257, 128)
        energy = torch.einsum("... i d , ... j d -> ... i j", query, key) * self.dk

        if mask is not None:
            energy = energy.masked_fill(mask, -np.inf)

        attention = torch.softmax(energy, dim=-1)

        #attentionheatmap_visual(attention, out_dir='./Visualization/attention_af3/')

        x = torch.einsum("... i j , ... j d -> ... i d", attention, value)

        x = rearrange(x, "b h t d -> b t (h d)")
        x = self.out_attention(x)

        return x


class MLP(nn.Module):
    def __init__(self, embedding_dim, mlp_dim):
        super().__init__()

        self.mlp_layers = nn.Sequential(
            nn.Linear(embedding_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(mlp_dim, embedding_dim),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        x = self.mlp_layers(x)

        return x


class TransformerEncoderBlock(nn.Module):
    def __init__(self, embedding_dim, head_num, mlp_dim):
        super().__init__()

        self.multi_head_attention = MultiHeadAttention(embedding_dim, head_num)
        self.mlp = MLP(embedding_dim, mlp_dim)

        self.layer_norm1 = nn.LayerNorm(embedding_dim)
        self.layer_norm2 = nn.LayerNorm(embedding_dim)

        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        _x = self.multi_head_attention(x)
        _x = self.dropout(_x)
        x = x + _x
        x = self.layer_norm1(x)

        _x = self.mlp(x)
        x = x + _x
        x = self.layer_norm2(x)

        return x


class TransformerEncoder(nn.Module):
    def __init__(self, embedding_dim, head_num, mlp_dim, block_num=12):
        super().__init__()

        self.layer_blocks = nn.ModuleList(
            [TransformerEncoderBlock(embedding_dim, head_num, mlp_dim) for _ in range(block_num)])

    def forward(self, x):
        for layer_block in self.layer_blocks:
            x = layer_block(x)

        return x


class ViT(nn.Module):
    def __init__(self, img_dim, in_channels, embedding_dim, head_num, mlp_dim,
                 block_num, patch_dim, classification=True, num_classes=1):
        super().__init__()

        self.patch_dim = patch_dim
        self.classification = classification
        self.num_tokens = (img_dim // patch_dim) ** 2  # 16x 16 = 256
        self.token_dim = in_channels * (patch_dim ** 2)

        self.projection = nn.Linear(self.token_dim, embedding_dim)
        self.embedding = nn.Parameter(torch.rand(self.num_tokens + 1, embedding_dim)) # (257,1024)

        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))

        self.dropout = nn.Dropout(0.1)

        self.transformer = TransformerEncoder(embedding_dim, head_num, mlp_dim, block_num)

        if self.classification:
            self.mlp_head = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        img_patches = rearrange(x,
                                'b c (patch_x x) (patch_y y) -> b (x y) (patch_x patch_y c)',
                                patch_x=self.patch_dim, patch_y=self.patch_dim)

        batch_size, tokens, _ = img_patches.shape

        project = self.projection(img_patches) #(16, 256, 1024)
        token = repeat(self.cls_token, 'b ... -> (b batch_size) ...',
                       batch_size=batch_size) #(16, 1, 1024)

        patches = torch.cat([token, project], dim=1)
        patches += self.embedding[:tokens + 1, :] #(16, 257, 1024)

        x = self.dropout(patches)
        x = self.transformer(x)
        x = self.mlp_head(x[:, 0, :]) if self.classification else x[:, 1:, :]
        return x