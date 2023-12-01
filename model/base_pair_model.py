import torch
import torch.nn as nn
import rowan
from pytorch3d.transforms import quaternion_invert, quaternion_raw_multiply
from torch.nn.functional import pad
import time


class BasePairNN(nn.Module):
    """Base class for neural networks trained on pair particles."""
    def __init__(self, in_dim, hidden_dim, out_dim, n_layers, act_fn="ReLU",
                 dropout=0.3, batch_norm=True, device=None
                 ):
        super(BasePairNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.n_layers = n_layers
        self.act_fn = act_fn
        self.dropout = dropout
        self.device = device
        self.in_dim = in_dim + 2
        self.batch_norm = batch_norm


        self.net = self._get_net()
        self.net.apply(self.init_net_weights)

    def init_net_weights(self, m):
        # todo: add option to initialize uniformly for weights and biases
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
    #             m.bias.data.fill_(0.01)

    def _get_act_fn(self):
        act = getattr(nn, self.act_fn)
        return act()

    def _prep_input(self, x1, x2, q1, q2):

        dr = x1 - x2
        r = torch.norm(dr, dim=1, keepdim=True)
        x = torch.concat((features, r, 1. / r), dim=1)

        return x.to(self.device)



class PairNN(BasePairNN):
    def __init__(self, in_dim, hidden_dim, out_dim, n_layers, **kwargs):
        super(PairNN, self).__init__(in_dim, hidden_dim, out_dim, n_layers,
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