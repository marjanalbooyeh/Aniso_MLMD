import numpy as np


def adjust_periodic_boundary(dr, box_len):
    adjusted_dr = []
    for x in dr:
        if x > (box_len/2):
            adjusted_dr.append(x - box_len/2)
        elif x < (-box_len/2):
            adjusted_dr.append(x + box_len/2)
        else:
            adjusted_dr.append(x)
    return np.asarray(adjusted_dr)