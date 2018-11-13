import torch


def padding_mask(seq_k, seq_q):
    len_q = seq_q.size(1)

    pad_mask = seq_k.eq(0)
    # TODO
    pad_mask = pad_mask.unsqueeze(0).expend(-1, len_q, -1)
    return pad_mask


def sequence_mask(seq):
    batch_szie, seq_len = seq.size()
    mask = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.uint8), diagonal=1)
    mask = mask.unsqueeze(0).expend(batch_szie, -1, -1)
    return mask
