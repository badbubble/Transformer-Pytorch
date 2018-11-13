import torch.nn as nn
from Modules.PositionalWiseFeedForward import PositionalWiseFeedForward
from Modules.MultiHeadAttention import MultiHeadAttention
from Modules.PositionalEncoding import PositionalEncoding
from utils.mask import padding_mask, sequence_mask
import torch


class DecoderLayer(nn.Module):
    def __init__(self,  model_dim=512, num_heads=8,ffn_dim=2048, dropout_rate=0.0):
        super(DecoderLayer, self).__init__()

        self.attention = MultiHeadAttention(model_dim, num_heads,)
        self.feed_forward = PositionalWiseFeedForward(model_dim, ffn_dim, dropout_rate)

    def forward(self, dec_inputs, enc_outputs, self_attn_mask=None, context_attn_mask=None):
        dec_output, self_attention = self.attention(
            dec_inputs, dec_inputs, dec_inputs, self_attn_mask
        )

        dec_output, context_attention = self.attention(
            dec_output, enc_outputs, enc_outputs, context_attn_mask
        )

        dec_output = self.feed_forward(dec_output)

        return dec_output, self_attention, context_attention


class Decoder(nn.Module):
    def __init__(self, vocab_size, max_sep_len, num_layers=6, model_dim=512, num_heads=8, ffn_dim=2048,
                 dropout_rate=0.0):
        super(Decoder, self).__init__()

        self.num_layers = num_layers

        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(model_dim, num_heads, ffn_dim, dropout_rate)] for _ in
            range(num_layers)
        )

        self.seq_embedding = nn.Embedding(vocab_size + 1, model_dim, padding_idx=0)
        self.pos_embedding = PositionalEncoding(model_dim, max_sep_len)

    def forward(self, inputs, input_lens, enc_output, context_attn_mask=None):
        output = self.seq_embedding(inputs)
        output += self.pos_embedding(input_lens)

        self_attention_padding_mask = padding_mask(inputs, inputs)
        seq_mask = sequence_mask(inputs)
        self_attn_mask = torch.gt((self_attention_padding_mask + seq_mask), 0)

        self_attentions = []
        context_attentions = []

        for decoder in self.decoder_layers:
            output, self_attn, context_attn = decoder(
                output, enc_output, self_attn_mask, context_attn_mask
            )
            self_attentions.append(self_attentions)
            context_attentions.append(context_attentions)

        return output, self_attentions, context_attentions

