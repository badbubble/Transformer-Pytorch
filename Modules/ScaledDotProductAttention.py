import torch
import torch.nn as nn
import numpy as np


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout_rate=0.0):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        # 在第二个维度上进行softmax
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, scale=None, attn_mask=None):
        """
        前向传播
        :param self:
        :param q: Question张量 shape: [B, L_q, D_q]
        :param k: Keys张量 shape: [B, L_k, D_k]
        :param v: Values张量 shape: [B, L_v, D_v]
        :param scale: 缩放因子
        :param attn_mask: Masking张量 shape: [B, L_q, L_k]
        :return: 上下文张量和attention张量
        """
        # attention shape : [B, L_q, L_k]
        attention = torch.bmm(q, k.transpose(1, 2))
        if scale:
            attention *= scale
        if attn_mask:
            # attn_mask中唯一的值标记为负无穷
            attention = attention.masked_fill_(attn_mask, -np.inf)
        # 计算Softmax
        attention = self.softmax(attention)
        # 添加Dropout
        attention = self.dropout(attention)

        # 和V做点积计算context
        context = torch.bmm(attention, v)

        return context, attention


if __name__ == '__main__':
    model = ScaledDotProductAttention(0.5)
    Questions = torch.rand(100, 128, 512)
    Keys = Questions.clone()
    Values = Questions.clone()
    context, attention = model(Questions, Keys, Values, 64 ** -0.5)
    print(context.size())
    print(attention.size())