import torch

def _calculate_prior_energy(x1, x2, prior_energy_sigma, prior_energy_n):
    dr = x1 - x2
    r = torch.norm(dr, dim=1, keepdim=True)
    U_0 = torch.pow(prior_energy_sigma / r, prior_energy_n)
    return U_0


def _calculate_torque(torque_grad, R1):
    tq_x = torch.cross(torque_grad[:, :, 0], R1[:, :, 0])
    tq_y = torch.cross(torque_grad[:, :, 1], R1[:, :, 1])
    tq_z = torch.cross(torque_grad[:, :, 2], R1[:, :, 2])
    predicted_torque = tq_x + tq_y + tq_z
    return predicted_torque