import torch
import torch.nn as nn
import rowan
import rotation_matrix_ops as rmo


class BasePairNN(nn.Module):
    """Base class for neural networks trained on pair particles."""

    def __init__(self, in_dim, hidden_dim, n_layers, act_fn="ReLU",
                 dropout=0.3, batch_norm=True, device=None
                 ):
        super(BasePairNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.energy_out_dim = 1
        self.n_layers = n_layers
        self.act_fn = act_fn
        self.dropout = dropout
        self.device = device
        self.in_dim = in_dim
        self.batch_norm = batch_norm

        self.energy_net = self._get_energy_net(self.energy_out_dim)

        # initialize weights and biases
        self.energy_net.apply(self.init_net_weights)

    def init_net_weights(self, m):
        # todo: add option to initialize uniformly for weights and biases
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def _get_act_fn(self):
        act = getattr(nn, self.act_fn)
        return act()

    def _prep_features_rot_matrix(self, x1, x2, R1, R2):
        # x1: particle 1 positions
        # x2: particle 2 positions
        # R1: particle 1 orientations (rotation matrix 3x3)
        # R2: particle 2 orientations (rotation matrix 3x3)

        batch_size = x1.shape[0]
        features = []

        dr = x1 - x2
        R = torch.norm(dr, dim=1, keepdim=True)
        dr = dr / R
        inv_r = 1. / R

        orient_dot_prod = rmo.dot_product(R1, R2)
        orient_cross_prod, orient_cross_prod_norm = rmo.cross_product(R1, R2)
        rel_orient = rmo.relative_orientation(R1, R2)
        rel_pos_align, rel_pos_project = rmo.rel_pos_orientation(dr, R1, R2)
        rbf = rmo.RBF_rel_pos(dr, R1, R2)
        angle = rmo.rot_matrix_to_angle(rel_orient)

        features = torch.cat((R, inv_r, dr, orient_dot_prod,
                              orient_cross_prod.reshape(batch_size, -1),
                              orient_cross_prod_norm,
                              rel_orient.reshape(batch_size, -1),
                              rel_pos_align,
                              rel_pos_project.reshape(batch_size, -1),
                              rbf, angle),
                             dim=1)

        return features.to(self.device)

    # def _prep_input_quat(self, x1, x2, q1, q2):
    #     # x1: particle 1 positions
    #     # x2: particle 2 positions
    #     # q1: particle 1 orientations (quaternion)
    #     # q2: particle 2 orientations (quaternion)
    #
    #     dr = x1 - x2
    #     R = torch.norm(dr, dim=1, keepdim=True)
    #     dr = dr / R
    #
    #     # convert dr vector to a quaternion
    #     dr_q = torch.tensor(rowan.normalize(
    #         torch.cat((torch.zeros(dr.shape[0], 1), dr), dim=1)))
    #
    #     # calculate q1 applied to q2 and vice versa
    #     q1q2 = torch.tensor(rowan.multiply(q1, rowan.conjugate(q2)))
    #
    #     # rotate dr based on q1 and q2 and q1q2
    #     dr_q1 = torch.tensor(rowan.multiply(rowan.multiply(q1, dr_q),
    #                                         rowan.conjugate(q1)))
    #     dr_q2 = torch.tensor(rowan.multiply(rowan.multiply(q2, dr_q),
    #                                         rowan.conjugate(q2)))
    #     dr_q1q2 = torch.tensor(rowan.multiply(rowan.multiply(q1q2, dr_q),
    #                                           rowan.conjugate(q1q2)))
    #
    #     features = torch.concatenate(
    #         (R, 1. / R, dr, q1, q2, q1q2, dr_q1, dr_q2, dr_q1q2), dim=1)
    #
    #     return features.to(self.device)


class PairNN_Force_Torque(BasePairNN):
    def __init__(self, in_dim, hidden_dim, n_layers, **kwargs):
        super(PairNN_Force_Torque, self).__init__(in_dim, hidden_dim,
                                                  n_layers,
                                                  **kwargs)

    def _get_energy_net(self, out_dim):
        layers = [nn.Linear(self.in_dim, self.hidden_dim), self._get_act_fn()]
        for i in range(self.n_layers - 1):
            layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            if self.batch_norm:
                layers.append(nn.BatchNorm1d(self.hidden_dim))
            layers.append(self._get_act_fn())
            layers.append(nn.Dropout(p=self.dropout))
        layers.append(nn.Linear(self.hidden_dim, out_dim))
        return nn.Sequential(*layers)

    def forward(self, x1, x2, R1, R2):
        x = self._prep_features_rot_matrix(x1, x2, R1, R2)
        energy = self.net(x)
        force = -torch.autograd.grad(energy.sum(),
                                     x1,
                                     create_graph=True)[0]
        torque = - torch.autograd.grad(energy.sum(),
                                         R1,
                                        create_graph=True)[0]
        tq_x = torch.cross(torque[:, :, 0], R1[:, :, 0])
        tq_y = torch.cross(torque[:, :, 1], R1[:, :, 1])
        tq_z = torch.cross(torque[:, :, 2], R1[:, :, 2])
        torque_out = torch.stack((tq_x, tq_y, tq_z), dim=2)

        return force, torque_out



