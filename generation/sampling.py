import torch


def greedy_sample(logits):
    next_token_id = torch.argmax(logits)
    next_token_id = next_token_id.view(1, 1)
    return next_token_id