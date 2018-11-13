import torch.nn as nn
from Modules.PositionalWiseFeedForward import PositionalWiseFeedForward
from Modules.MultiHeadAttention import MultiHeadAttention
from Modules.PositionalEncoding import PositionalEncoding
from utils.mask import padding_mask


class EncoderLayer(nn.Module):
    def __init__(self, model_dim=512, num_heads=8, ffn_dim=2048, dropout_rate=0.0):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(model_dim, num_heads, dropout_rate)
        self.feed_forward = PositionalWiseFeedForward(model_dim, ffn_dim, dropout_rate)

    def forward(self, inputs, attn_mask):
        context, attention = self.attention(inputs, inputs, inputs, attn_mask)
        output = self.feed_forward(context)
        return output, attention


class Encoder(nn.Module):
    def __init__(self, vocab_size, max_seq_len, number_layers=6, model_dim=512,
                 num_heads=8, fnn_dim=2048, dropout_rate=0.0):
        super(Encoder, self).__init__()

        self.encoder_layers = nn.ModuleList([EncoderLayer(model_dim, num_heads, fnn_dim, dropout_rate)
                                            for _ in range(number_layers)])
        self.seq_emdedding = nn.Embedding(vocab_size + 1, model_dim, padding_idx=0)
        self.pos_embedding = PositionalEncoding(model_dim, max_seq_len)

    def forward(self, inputs, input_lens):
        output = self.seq_emdedding(inputs)
        output += self.pos_embedding(input_lens)
        self_attention_mask = padding_mask(inputs, inputs)
        attentions = []

        for encoder in self.encoder_layers:
            output, attention = encoder(output, self_attention_mask)
            attentions.append(attention)
        return output, attentions
