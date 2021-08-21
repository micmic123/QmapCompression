import torch


def ste_round(x):
    return torch.round(x) - x.detach() + x
