import torch


def adjust_periodic_boundary(dr, box_len):
    half_box_len = box_len / 2
    dr = torch.where(dr > half_box_len, dr - half_box_len, dr)
    dr = torch.where(dr < -half_box_len, dr + half_box_len, dr)
    return dr
