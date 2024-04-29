import torch
import torch.nn as nn

from aniso_per_particle.model import feature_vector, get_act_fn



class ParticleEnergyPredictorHuang(nn.Module):
    def __init__(self, in_dim,
                 prior_net_config,
                 energy_net_config,
                 dropout=0.3,
                 batch_norm=True,
                 device=None,
                 ):
        super(ParticleEnergyPredictorHuang, self).__init__()

        self.prior_hidden_dim = prior_net_config['hidden_dim']
        self.prior_n_layers = prior_net_config['n_layers']
        self.prior_act_fn = prior_net_config['act_fn']

        self.energy_hidden_dim = energy_net_config['hidden_dim']
        self.energy_n_layers = energy_net_config['n_layers']
        self.energy_act_fn = energy_net_config['act_fn']

        self.dropout = dropout
        self.device = device
        self.in_dim = in_dim
        self.batch_norm = batch_norm
        self.feat_dim = 10*6*2*2

        self.prior_net = self._MLP_net(in_dim=self.in_dim,
                                       h_dim=self.prior_hidden_dim,
                                       out_dim=1,
                                       n_layers=self.prior_n_layers,
                                       act_fn=self.prior_act_fn,
                                       dropout=self.dropout).to(self.device)

        self.energy_net = self._MLP_net(in_dim=self.feat_dim,
                                        h_dim=self.energy_hidden_dim,
                                        out_dim=1,
                                        n_layers=self.energy_n_layers,
                                        act_fn=self.energy_act_fn,
                                        dropout=self.dropout).to(self.device)

        self.prior_energy_factor_1 =torch.nn.Parameter(torch.rand(3, 1), requires_grad=True)
        self.prior_energy_factor_2 =torch.nn.Parameter(torch.rand(3, 1), requires_grad=True)

    def _MLP_net(self, in_dim, h_dim, out_dim,
                 n_layers, act_fn, dropout):
        layers = [nn.Linear(in_dim, h_dim),
                  get_act_fn(act_fn)]
        for i in range(n_layers - 1):
            layers.append(
                nn.Linear(h_dim, h_dim))
            # if self.batch_norm:
            #     layers.append(nn.BatchNorm1d(self.neighbor_hidden_dim))
            layers.append(get_act_fn(act_fn))
            # layers.append(nn.Dropout(p=dropout))
        layers.append(
            nn.Linear(h_dim, out_dim))
        return nn.Sequential(*layers)

    def forward(self, dr, orientation, n_orientation):
        # dr: (B, Nb, 3) # distance vector between particle and its neighbors
        # orientation: (B, 3, 3) # particle's orientation
        # n_orientation: (B, Nb,  3, 3) # neighbors' orientation
        B = dr.shape[0]
        Nb = dr.shape[1]
        epsilon = 1e-8
        R_cut = 2.5
        eta = torch.tensor([[2.], [1.]]).to(self.device)
        zeta = torch.tensor([[2.], [4.], [8.], [16.], [32.], [64.]]).to(self.device)
        R_s = torch.linspace(3, 9, 10).to(self.device)
        ##########################################
        # features: (B, N_neighbors, 15)
        dr = dr.reshape(B, Nb, 3, 1)
        R = torch.norm(dr, dim=2, keepdim=True)# (B, N_neighbors,1, 1)
        dr_norm = dr / R
        orientation = orientation.reshape(B, 1, 3, 3)
        features = feature_vector(dr_norm, orientation, n_orientation).to(self.device) # (B, N_neighbors, 15, 1)

        ############# Prior Net ##############
        tanh = nn.Tanh()
        prior_encoder = (((tanh(self.prior_net(features.squeeze(-1)))**2) + epsilon)
                     .reshape(B, Nb, 1, 1, 1))
        prior_out = R.reshape(B, Nb, 1, 1, 1) - prior_encoder
        prior_out = (self.prior_energy_factor_1 ** 2 + epsilon) * prior_out
        prior_out = prior_out ** (-1 *(self.prior_energy_factor_2 ** 2 + epsilon))
        prior_out = torch.sum(prior_out, dim=3)


        ############# Sym fun ##############
        fcR = (torch.cos(torch.pi * R/ R_cut) * 0.5 + 0.5).reshape(B, Nb, 1, 1, 1, 1, 1)

        R_reduced = (R - R_s).reshape(B, Nb, 1, 1,R_s.shape[0] , 1, 1)
        pre_factor = torch.exp(-eta * (R_reduced ** 2))

        pre_factor = pre_factor.reshape(B, Nb, 1, 1, R_s.shape[0], 1, eta.shape[0])

        ang = features.reshape(B, Nb, 15, 1, 1, 1)
        pos_factor_1 = (2**(1-zeta)*(1-ang)**zeta).reshape(B, Nb, self.in_dim, zeta.shape[0], 1, 1, 1)
        pos_factor_2 = (2**(1-zeta)*(1+ang)**zeta).reshape(B, Nb, self.in_dim, zeta.shape[0], 1, 1, 1)

        fca1 = pre_factor * fcR * pos_factor_1
        fca2 = pre_factor * fcR * pos_factor_2
        fca = torch.cat([fca1, fca2], dim=5)
        fcr = torch.sum(fca, dim=2)
        GAR = fcr.reshape(B, Nb, R_s.shape[0] * 2 * zeta.shape[0] * 1 * eta.shape[0])
        DES = torch.sum(GAR, dim=1)

        ############# Energy Net ##############
        leaky_relu_1 = nn.LeakyReLU()
        leaky_relu_2 = nn.LeakyReLU()
        sym_encode = leaky_relu_1(self.energy_net(DES))
        predicted_energy = leaky_relu_2(sym_encode + torch.sum(prior_out.reshape(B, Nb, 1), dim=1))




        ################## Calculate Force ##################
        neighbors_force = - torch.autograd.grad(predicted_energy.sum(dim=1).sum(),
                                                dr,
                                                create_graph=True)[0].to(
            self.device)  # (B, N, N_neighbors, 3)

        predicted_force = torch.sum(neighbors_force, dim=1).reshape(B, 3)


        ################## Calculate Torque ##################
        torque_grad = - torch.autograd.grad(predicted_energy.sum(dim=1).sum(),
                                            orientation,
                                            create_graph=True)[0].to(
            self.device) # (B, N, 3, 3)

        tq_x = torch.cross(torque_grad[:, :, :, 0].reshape(-1, 3), orientation[:, :, :, 0].reshape(-1, 3))
        tq_y = torch.cross(torque_grad[:, :, :, 1].reshape(-1, 3), orientation[:, :, :, 1].reshape(-1, 3))
        tq_z = torch.cross(torque_grad[:, :, :, 2].reshape(-1, 3), orientation[:, :, :, 2].reshape(-1, 3))
        predicted_torque = (tq_x + tq_y + tq_z).to(self.device)

        return predicted_force, predicted_torque, predicted_energy


if __name__ == '__main__':
    prior_net_config = {'hidden_dim': 3,
                        'n_layers': 2,
                        'act_fn': 'Tanh'}
    energy_net_config = {'hidden_dim': 5,
                         'n_layers': 3,
                         'act_fn': 'Tanh'}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ParticleEnergyPredictorHuang(in_dim=15,
                                         out_dim=1,
                                         prior_net_config=prior_net_config,
                                         energy_net_config=energy_net_config,
                                         dropout=0.,
                                         device=device)

    model = model.to(device)

    B = 2
    Nb = 10
    dr = torch.rand(B, Nb, 3).to(device)
    dr.requires_grad = True
    orientation = torch.rand(B, 3, 3).to(device)
    orientation.requires_grad = True
    n_orientation = torch.rand(B, Nb, 3, 3).to(device)
    predicted_force, predicted_torque, predicted_energy = model(dr, orientation, n_orientation)
    print(predicted_force.shape)