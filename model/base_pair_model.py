import torch
import torch.nn as nn
import rowan
from pytorch3d.transforms import quaternion_invert, quaternion_raw_multiply
from torch.nn.functional import pad
import time


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
        self.in_dim = in_dim + 2
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

    def _prep_input_rotation_matrix(self, x1, x2, R1, R2):
        # x1: particle 1 positions
        # x2: particle 2 positions
        # R1: particle 1 orientations (rotation matrix 3x3)
        # R2: particle 2 orientations (rotation matrix 3x3)

        dr = x1 - x2
        R = torch.norm(dr, dim=1, keepdim=True)
        dr = dr / R

        # calculate dr vector in rotated frame
        dr_rot = torch.matmul(R1, dr.unsqueeze(-1)).squeeze(-1)


    def _prep_input_quat(self, x1, x2, q1, q2):
        # x1: particle 1 positions
        # x2: particle 2 positions
        # q1: particle 1 orientations (quaternion)
        # q2: particle 2 orientations (quaternion)

        dr = x1 - x2
        R = torch.norm(dr, dim=1, keepdim=True)
        dr = dr / R

        # convert dr vector to a quaternion
        dr_q = torch.tensor(rowan.normalize(torch.cat((torch.zeros(dr.shape[0], 1), dr), dim=1)))


        # calculate q1 applied to q2 and vice versa
        q1q2 = torch.tensor(rowan.multiply(q1, rowan.conjugate(q2)))


        # rotate dr based on q1 and q2 and q1q2
        dr_q1 = torch.tensor(rowan.multiply(rowan.multiply(q1, dr_q),
                                            rowan.conjugate(q1)))
        dr_q2 = torch.tensor(rowan.multiply(rowan.multiply(q2, dr_q),
                                            rowan.conjugate(q2)))
        dr_q1q2 = torch.tensor(rowan.multiply(rowan.multiply(q1q2, dr_q),
                                              rowan.conjugate(q1q2)))


        features = torch.concatenate(
            (R, 1./R, dr, q1, q2, q1q2, dr_q1, dr_q2, dr_q1q2), dim=1)


        return features.to(self.device)


class PairNN(BasePairNN):
    def __init__(self, in_dim, hidden_dim, out_dim, n_layers, **kwargs):
        super(PairNN, self).__init__(in_dim, hidden_dim, out_dim, n_layers,
                                     **kwargs)

    def _get_energy_net(self):
        layers = [nn.Linear(self.in_dim, self.hidden_dim), self._get_act_fn()]
        for i in range(self.n_layers - 1):
            layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            if self.batch_norm:
                layers.append(nn.BatchNorm1d(self.hidden_dim))
            layers.append(self._get_act_fn())
            layers.append(nn.Dropout(p=self.dropout))
        layers.append(nn.Linear(self.hidden_dim, self.out_dim))
        return nn.Sequential(*layers)

    def forward(self, x1, x2, q1, q2):
        x = self._prep_input(x1, x2, q1, q2)
        energy = self.force_net(x)
        force = -torch.autograd.grad(energy.sum(),
                                     x1,
                                     create_graph=True)[0]
        torque = self.torque_net(x)
        return force, torque


class PairNN_Force_Torque(BasePairNN):
    def __init__(self, in_dim, hidden_dim, n_layers, **kwargs):
        super(PairNN_Force_Torque, self).__init__(in_dim, hidden_dim, out_dim, n_layers,
                                     **kwargs)

    def _get_net(self):
        layers = [nn.Linear(self.in_dim, self.hidden_dim), self._get_act_fn()]
        for i in range(self.n_layers - 1):
            layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            if self.batch_norm:
                layers.append(nn.BatchNorm1d(self.hidden_dim))
            layers.append(self._get_act_fn())
            layers.append(nn.Dropout(p=self.dropout))
        layers.append(nn.Linear(self.hidden_dim, self.out_dim))
        return nn.Sequential(*layers)

    def forward(self, x1, x2, q1, q2):
        x = self._prep_input(x1, x2, q1, q2)
        out = self.net(x)
        return out

class PairNNSkipShared(BasePairNN):
    def __init__(self, in_dim, hidden_dim, out_dim, n_layers, **kwargs):
        super(PairNNSkipShared, self).__init__(in_dim, hidden_dim, out_dim,
                                               n_layers, **kwargs)
        self.activations = self._get_activations()
        self.dropouts = self._get_dropouts()
        self.input_connection = nn.Linear(self.in_dim, self.hidden_dim)

    def _get_net(self):
        layers = [nn.Linear(self.in_dim, self.hidden_dim)]
        for i in range(self.n_layers - 1):
            layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            if self.batch_norm:
                layers.append(nn.BatchNorm1d(self.hidden_dim))

        layers.append(nn.Linear(self.hidden_dim, self.out_dim))
        return nn.ModuleList(layers)

    def _get_activations(self):
        activations = []
        for i in range(self.n_layers):
            activations.append(self._get_act_fn())
        return nn.ModuleList(activations)

    def _get_dropouts(self):
        dropouts = []
        for i in range(self.n_layers):
            dropouts.append(nn.Dropout(p=self.dropout))
        return nn.ModuleList(dropouts)

    def forward(self, x1, x2, q1, q2):
        x = self._prep_input(x1, x2, q1, q2)
        # transform input to hidden dim size
        x_transform = self.input_connection(x)
        for i in range(self.n_layers):
            # add original transformed input to each layer before activation
            x = self.activations[i](self.net[i](x) + x_transform)
            x = self.dropouts[i](x)

        out = self.net[-1](x)
        return out


class PairNNGrow(BasePairNN):
    def __init__(self, in_dim, hidden_dim, out_dim, n_layers, **kwargs):
        super(PairNNGrow, self).__init__(in_dim, hidden_dim, out_dim, n_layers,
                                         **kwargs)

    def _get_net(self):
        layers = [nn.Linear(self.in_dim, self.hidden_dim[0]),
                  self._get_act_fn()]
        for i in range(1, len(self.hidden_dim)):
            layers.append(nn.Linear(self.hidden_dim[i - 1], self.hidden_dim[i]))
            if self.batch_dim:
                layers.append(nn.BatchNorm1d(self.batch_dim))
            layers.append(self._get_act_fn())
            layers.append(nn.Dropout(p=self.dropout))
        layers.append(nn.Linear(self.hidden_dim[-1], self.out_dim))
        return nn.Sequential(*layers)

    def forward(self, x1, x2, q1, q2):
        x = self._augment_input(features, x1, x2)
        out = self.net(x)
        return out
