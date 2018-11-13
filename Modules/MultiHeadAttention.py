import torch
import torch.nn as nn
import numpy as np
from Modules.ScaledDotProductAttention import ScaledDotProductAttention


class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim=512, num_heads=8, dropout_rate=0.0):
        super(MultiHeadAttention, self).__init__()

        self.dim_per_head = model_dim // num_heads
        self.num_heads = num_heads
        self.model_dim = model_dim
        # 这里相当于同时做了num_heads次转换
        self.linear_q = nn.Linear(self.model_dim, self.num_heads * self.dim_per_head)
        self.linear_k = nn.Linear(self.model_dim, self.num_heads * self.dim_per_head)
        self.linear_v = nn.Linear(self.model_dim, self.num_heads * self.dim_per_head)

        self.dot_product_attention = ScaledDotProductAttention(dropout_rate)

        self.linear_final = nn.Linear(self.model_dim, self.model_dim)
        self.dropout = nn.Dropout(dropout_rate)

        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, query, key, value, attn_mask=None):
        # 残差连接
        residual = query
        batch_size = query.size(0)

        # 线性变换
        query = self.linear_q(query)
        key = self.linear_k(key)
        value = self.linear_v(value)

        query = query.view(batch_size * self.num_heads, -1, self.dim_per_head)
        key = key.view(batch_size * self.num_heads, -1, self.dim_per_head)
        value = value.view(batch_size * self.num_heads, -1, self.dim_per_head)

        if attn_mask:
            # 复制一遍
            attn_mask = attn_mask.repeat(self.num_heads, 1, 1)

        scale = self.dim_per_head ** -0.5

        context, attention = self.dot_product_attention(query, key, value, scale, attn_mask)

        context = context.view(batch_size, -1, self.dim_per_head * self.num_heads)

        # final线性变换
        output = self.linear_final(context)
        output = self.dropout(output)

        output = self.layer_norm(residual + output)
        return output, attention


if __name__ == "__main__":
    q = torch.rand(100, 128, 512)
    v = q.repeat(1, 1, 1)
    k = q.repeat(1, 1, 1)
    model = MultiHeadAttention()

    output, attention = model(q, k, v)
    print(output.size())
    print(attention.size())