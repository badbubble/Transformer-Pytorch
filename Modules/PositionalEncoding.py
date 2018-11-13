import numpy as np
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        position_encoding = np.array([
                  [pos / pow(10000, 2.0 * (j // 2) / 512) for j in range(512)]
                  for pos in range(128)])
        position_encoding[:, 0::2] = np.sin(position_encoding[:, 0::2])
        position_encoding[:, 1::2] = np.cos(position_encoding[:, 1::2])

        pad_row = torch.zeros(1, self.d_model)
        position_encoding = torch.Tensor(position_encoding)
        position_encoding = torch.cat((pad_row, position_encoding))

        self.position_encoding = nn.Embedding(self.max_seq_len + 1, d_model)
        self.position_encoding.weight = nn.Parameter(position_encoding, requires_grad=False)

    def forward(self, input_lens):
        max_len = 12 #torch.max(input_lens)
        tensor = torch.cuda.LongTensor if input_lens.is_cuda else torch.LongTensor
        input_pos = tensor(
            [list(range(1, int(len) + 1)) + [0] * (max_len - int(len)) for len in input_lens])
        print(input_pos)
        return self.position_encoding(input_pos)


if __name__ == "__main__":
    model = PositionalEncoding(512, 12)
    pos_encoding = model(torch.Tensor([[10], [12], [8], [7]]))