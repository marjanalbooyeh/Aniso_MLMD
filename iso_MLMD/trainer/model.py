import torch.nn as nn
import torch


class BasePairNN(nn.Module):
    """Base class for neural networks trained on pair particles."""

    def __init__(self, in_dim, hidden_dim,
                 n_layers,
                 act_fn="ReLU",
                 dropout=0.3, batch_norm=True, device=None,
                 prior_energy=True,
                 prior_energy_sigma=1.0, prior_energy_n=12
                 ):
        super(BasePairNN, self).__init__()
        self.energy_out_dim = 1
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.act_fn = act_fn
        self.dropout = dropout
        self.device = device
        self.in_dim = in_dim
        self.batch_norm = batch_norm
        self.prior_energy = prior_energy
        self.prior_energy_sigma = prior_energy_sigma
        self.prior_energy_n = prior_energy_n

        self.energy_net = self._energy_net().to(self.device)

        # initialize weights and biases
        # self.energy_net.apply(self.init_net_weights)

    def init_net_weights(self, m):
        # todo: add option to initialize uniformly for weights and biases
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def _get_act_fn(self):
        act = getattr(nn, self.act_fn)
        return act()

    def _prep_features(self, dr):
        # pairwise distances dr (B,3)

        self.R = torch.norm(dr, dim=1, keepdim=True)
        inv_R = 1. / self.R

        # concatenate all features (B, N, N_neighbors, 80)
        features = torch.cat((dr, self.R, inv_R), dim=-1)

        return features.to(self.device)

    def _energy_net(self):
        layers = [nn.Linear(self.in_dim, self.hidden_dim),
                  self._get_act_fn()]
        for i in range(self.n_layers - 1):
            layers.append(
                nn.Linear(self.hidden_dim, self.hidden_dim))
            if self.batch_norm:
                layers.append(nn.BatchNorm1d(self.hidden_dim))
            layers.append(self._get_act_fn())
            layers.append(nn.Dropout(p=self.dropout))
        layers.append(nn.Linear(self.hidden_dim, 1))
        return nn.Sequential(*layers)

    def _calculate_prior_energy(self):
        U_0 = torch.pow(self.prior_energy_sigma / self.R, self.prior_energy_n)
        return U_0

    def forward(self, dr):
        # pairwise distances dr (B, 3)
        features = self._prep_features(dr)
        energy = self.energy_net(
            features)  # (B,1)

        if self.prior_energy:
            U_0 = self._calculate_prior_energy()
            energy = energy + U_0.to(self.device)
        predicted_force = - torch.autograd.grad(energy.sum(),
                                                dr,
                                                create_graph=True)[0].to(
            self.device)
        predicted_pair_force = torch.stack((predicted_force, -predicted_force),
                                           dim=1)

        return predicted_pair_force