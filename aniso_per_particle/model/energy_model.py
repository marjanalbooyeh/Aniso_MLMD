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
                                       bn_dim=250).to(self.device)

        self.energy_net = self._MLP_net(in_dim=self.feat_dim,
                                        h_dim=self.energy_hidden_dim,
                                        out_dim=1,
                                        n_layers=self.energy_n_layers,
                                        act_fn=self.energy_act_fn,
                                        bn_dim=self.energy_hidden_dim).to(self.device)
        self.prior_energy_factor_1 = torch.nn.Parameter(torch.rand(3, 1), requires_grad=True)
        self.prior_energy_factor_2 = torch.nn.Parameter(torch.rand(3, 1), requires_grad=True)

        self.prior_net.apply(self.weights_init)
        self.energy_net.apply(self.weights_init)
        nn.init.uniform_(self.prior_energy_factor_1)
        nn.init.uniform_(self.prior_energy_factor_2)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.uniform_(m.weight.data)
            nn.init.uniform_(m.bias.data)
    def _MLP_net(self, in_dim, h_dim, out_dim,
                 n_layers, act_fn, bn_dim):
        layers = [nn.Linear(in_dim, h_dim),
                  get_act_fn(act_fn)]
        for i in range(n_layers):
            layers.append(
                nn.Linear(h_dim, h_dim))
            if self.batch_norm:
                layers.append(nn.BatchNorm1d(bn_dim))
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
        R_cut = 4.8
        eta = torch.tensor([[2.], [1.]]).to(self.device)
        eta_size = eta.shape[0]
        eta = eta.reshape(-1, 1, 1, 1, 1, 1, eta_size)
        zeta = torch.tensor([[2.], [4.0], [8.0], [16.0], [32.], [64.]]).to(self.device)
        R_s = torch.linspace(1, 4, 10).to(self.device)
        ##########################################
        # features: (B, N_neighbors, 15)
        dr = dr.reshape(B, Nb, 3, 1) + epsilon
        R = torch.norm(dr, dim=2, keepdim=True)# (B, N_neighbors,1, 1)
        dr_norm = dr / R
        orientation = orientation.reshape(B, 1, 3, 3)
        features = feature_vector(dr_norm, orientation, n_orientation).to(self.device) # (B, N_neighbors, 15, 1)

        ############# Prior Net ##############
        tanh = nn.Tanh()
        Ecin = (features.squeeze(-1) ** 2).reshape(B, Nb, 15)
        encoder0 = tanh(self.prior_net(Ecin))
        ECodeout = ((encoder0**2) + epsilon).reshape(B, Nb, 1, 1, 1)

        prior_out = R.reshape(B, Nb, 1, 1, 1) - ECodeout
        prior_out = (self.prior_energy_factor_1 ** 2 + epsilon) * prior_out
        prior_out = prior_out ** (-1 *(self.prior_energy_factor_2 ** 2 + epsilon))
        prior_out = torch.sum(prior_out, dim=3)


        ############# Sym fun ##############
        fcR = torch.where(R > R_cut,
                          0.0,
                          (0.5 * torch.cos(torch.pi * R / R_cut) + 0.5)).reshape(B, Nb, 1, 1, 1, 1, 1)

        R_reduced = (R - R_s).reshape(B, Nb, 1, 1,R_s.shape[0] , 1, 1)

        pre_factor = torch.exp(-eta * R_reduced ** 2)

        pre_factor = pre_factor.reshape(B, Nb, 1, 1, R_s.shape[0], 1, eta_size)

        ang = (features.squeeze(-1) ** 2).reshape(B, Nb, self.in_dim, 1, 1, 1)
        pos_factor_1 = (2**(1-zeta)*(1-ang)**zeta).reshape(B, Nb, self.in_dim, zeta.shape[0], 1, 1, 1)
        pos_factor_2 = (2**(1-zeta)*(1+ang)**zeta).reshape(B, Nb, self.in_dim, zeta.shape[0], 1, 1, 1)

        fca1 = pre_factor * fcR * pos_factor_1
        fca2 = pre_factor * fcR * pos_factor_2
        fca = torch.cat([fca1, fca2], dim=5)
        fcr = torch.sum(fca, dim=2)
        GAR = fcr.reshape(B, Nb, R_s.shape[0] * 2 * zeta.shape[0] * 1 * eta_size)
        DES = torch.sum(GAR, dim=1).reshape(-1, R_s.shape[0] * 2 * zeta.shape[0] * 1 * eta_size)

        ############# Energy Net ##############
        leaky_relu_1 = nn.LeakyReLU(negative_slope=1.0)
        leaky_relu_2 = nn.LeakyReLU(negative_slope=1.0)
        sym_encode = leaky_relu_1(self.energy_net(DES))
        predicted_energy = leaky_relu_2(sym_encode
                                        + torch.sum(prior_out.reshape(B, Nb, 1), dim=1))


        ################## Calculate Force ##################
        neighbors_force = torch.autograd.grad(predicted_energy.sum(),
                                                dr,
                                                create_graph=True)[0].to(
            self.device)  # (B, N, N_neighbors, 3)

        predicted_force = torch.sum(neighbors_force, dim=1).reshape(B, 3)


        ################## Calculate Torque ##################
        torque_grad = torch.autograd.grad(predicted_energy.sum(),
                                            orientation,
                                            create_graph=True)[0].to(
            self.device) # (B, N, 3, 3)

        tq_x = torch.cross(torque_grad[:, :, :, 0].reshape(-1, 3), orientation[:, :, :, 0].reshape(-1, 3))
        tq_y = torch.cross(torque_grad[:, :, :, 1].reshape(-1, 3), orientation[:, :, :, 1].reshape(-1, 3))
        tq_z = torch.cross(torque_grad[:, :, :, 2].reshape(-1, 3), orientation[:, :, :, 2].reshape(-1, 3))
        predicted_torque = (tq_x + tq_y + tq_z).to(self.device)

        return predicted_force, predicted_torque, predicted_energy


class ParticleEnergyPredictor_New(nn.Module):
    def __init__(self, in_dim,
                 prior_net_config,
                 energy_net_config,
                 dropout=0.3,
                 batch_norm=True,
                 device=None,
                 initial_weights=None,
                 ):
        super(ParticleEnergyPredictor_New, self).__init__()

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
                                       bn_dim=250).to(self.device)
        #
        self.energy_net = self._MLP_net(in_dim=self.in_dim + 1,
                                        h_dim=self.energy_hidden_dim,
                                        out_dim=1,
                                        n_layers=self.energy_n_layers,
                                        act_fn=self.energy_act_fn,
                                        bn_dim=self.energy_hidden_dim).to(self.device)
        self.prior_energy_factor_1 = torch.nn.Parameter(torch.rand(3, 1), requires_grad=True)
        self.prior_energy_factor_2 = torch.nn.Parameter(torch.rand(3, 1), requires_grad=True)

        if initial_weights:
            self.prior_net.apply(self.weights_init)
            self.energy_net.apply(self.weights_init)
            nn.init.uniform_(self.prior_energy_factor_1)
            nn.init.uniform_(self.prior_energy_factor_2)



    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.uniform_(m.weight.data)
            nn.init.uniform_(m.bias.data)

    def _MLP_net(self, in_dim, h_dim, out_dim,
                 n_layers, act_fn, bn_dim):

        layers = [nn.Linear(in_dim, h_dim[0]),
                  get_act_fn(act_fn)]
        for i in range(n_layers):
            layers.append(
                nn.Linear(h_dim[i], h_dim[i+1]))
            if self.batch_norm:
                layers.append(nn.BatchNorm1d(bn_dim))
            layers.append(get_act_fn(act_fn))
            # layers.append(nn.Dropout(p=dropout))
        layers.append(
            nn.Linear(h_dim[-1], out_dim))
        return nn.Sequential(*layers)


    def forward(self, dr, orientation, n_orientation):
        # dr: (B, Nb, 3) # distance vector between particle and its neighbors
        # orientation: (B, 3, 3) # particle's orientation
        # n_orientation: (B, Nb,  3, 3) # neighbors' orientation
        B = dr.shape[0]
        Nb = dr.shape[1]
        epsilon = 1e-8
        ##########################################
        # features: (B, N_neighbors, 15)
        dr = dr.reshape(B, Nb, 3, 1) + epsilon
        R = torch.norm(dr, dim=2, keepdim=True)# (B, N_neighbors,1, 1)
        dr_norm = dr / R
        orientation = orientation.reshape(B, 1, 3, 3)
        features = feature_vector(dr_norm, orientation, n_orientation).to(self.device) # (B, N_neighbors, 15, 1)

        ############# Prior Net ##############
        tanh = nn.Tanh()
        Ecin = (features.squeeze(-1)**2).reshape(B, Nb, 15)
        encoder0 = tanh(self.prior_net(Ecin))
        ECodeout = ((encoder0**2) + epsilon).reshape(B, Nb, 1, 1, 1)

        prior_out = R.reshape(B, Nb, 1, 1, 1) - ECodeout
        prior_out = (self.prior_energy_factor_1 ** 2 + epsilon) * prior_out
        prior_out = prior_out ** (-1 *(self.prior_energy_factor_2 ** 2 + epsilon))
        prior_out = torch.sum(prior_out, dim=3)

        ############# Energy Net ##############

        sym_encode = self.energy_net(torch.cat([R.squeeze(-1), Ecin], dim=-1))
        predicted_energy = (torch.sum(sym_encode.reshape(B, Nb, 1)) +
                            torch.sum(prior_out.reshape(B, Nb, 1) * 200,
                                      dim=1))

        ################## Calculate Force ##################
        neighbors_force = torch.autograd.grad(predicted_energy.sum(),
                                                dr,
                                                create_graph=True)[0].to(
            self.device)  # (B, N, N_neighbors, 3)

        predicted_force = torch.sum(neighbors_force, dim=1).reshape(B, 3)


        ################## Calculate Torque ##################
        torque_grad = torch.autograd.grad(predicted_energy.sum(),
                                            orientation,
                                            create_graph=True)[0].to(
            self.device) # (B, N, 3, 3)

        tq_x = torch.cross(torque_grad[:, :, :, 0].reshape(-1, 3), orientation[:, :, :, 0].reshape(-1, 3))
        tq_y = torch.cross(torque_grad[:, :, :, 1].reshape(-1, 3), orientation[:, :, :, 1].reshape(-1, 3))
        tq_z = torch.cross(torque_grad[:, :, :, 2].reshape(-1, 3), orientation[:, :, :, 2].reshape(-1, 3))
        predicted_torque = (tq_x + tq_y + tq_z).to(self.device)

        return predicted_force, predicted_torque, predicted_energy

class ParticleEnergyPredictor_V2(nn.Module):
    def __init__(self, in_dim,
                 prior_net_config,
                 energy_net_config,
                 dropout=0.3,
                 batch_norm=True,
                 device=None,
                 initial_weights=None,
                 ):
        super(ParticleEnergyPredictor_V2, self).__init__()

        self.prior_hidden_dim = prior_net_config['hidden_dim']
        self.prior_n_layers = prior_net_config['n_layers']
        self.prior_act_fn = prior_net_config['act_fn']
        self.prior_pre_factor = prior_net_config['pre_factor']
        self.prior_n = prior_net_config['n']

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
                                       bn_dim=250).to(self.device)


        self.energy_net = self._MLP_net(in_dim=self.in_dim,
                                        h_dim=self.energy_hidden_dim,
                                        out_dim=1,
                                        n_layers=self.energy_n_layers,
                                        act_fn=self.energy_act_fn,
                                        bn_dim=self.energy_hidden_dim).to(self.device)
        self.prior_energy_factor_1 = torch.nn.Parameter(torch.rand(3, 1), requires_grad=True)
        self.prior_energy_factor_2 = torch.nn.Parameter(torch.rand(3, 1), requires_grad=True)

        if initial_weights:
            self.prior_net.apply(self.weights_init)
            self.energy_net.apply(self.weights_init)
            nn.init.uniform_(self.prior_energy_factor_1)
            nn.init.uniform_(self.prior_energy_factor_2)



    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.uniform_(m.weight.data)
            nn.init.uniform_(m.bias.data)

    def _MLP_net(self, in_dim, h_dim, out_dim,
                 n_layers, act_fn, bn_dim):

        layers = [nn.Linear(in_dim, h_dim[0]),
                  get_act_fn(act_fn)]
        for i in range(n_layers):
            layers.append(
                nn.Linear(h_dim[i], h_dim[i+1]))
            if self.batch_norm:
                layers.append(nn.BatchNorm1d(bn_dim))
            layers.append(get_act_fn(act_fn))
            # layers.append(nn.Dropout(p=dropout))
        layers.append(
            nn.Linear(h_dim[-1], out_dim))
        return nn.Sequential(*layers)


    def forward(self, dr, orientation, n_orientation):
        # dr: (B, Nb, 3) # distance vector between particle and its neighbors
        # orientation: (B, 3, 3) # particle's orientation
        # n_orientation: (B, Nb,  3, 3) # neighbors' orientation
        B = dr.shape[0]
        Nb = dr.shape[1]
        epsilon = 1e-8
        R_cut = 4.8
        ##########################################
        # features: (B, N_neighbors, 15)
        dr = dr.reshape(B, Nb, 3, 1) + epsilon
        R = torch.norm(dr, dim=2, keepdim=True)# (B, N_neighbors,1, 1)
        dr_norm = dr / R
        orientation = orientation.reshape(B, 1, 3, 3)
        features = feature_vector(dr_norm, orientation, n_orientation).to(self.device) # (B, N_neighbors, 15, 1)

        ############# Prior Net ##############
        tanh = nn.Tanh()
        prior_features = (features.squeeze(-1)**2).reshape(B, Nb, 15)
        prior_enc = tanh(self.prior_net(prior_features))
        # physics informed repulsive potential
        prior_out = self.prior_pre_factor * torch.exp(-(R.reshape(B, Nb, 1)/R_cut)**self.prior_n) * prior_enc


        ############# Energy Net ##############

        sym_encode = self.energy_net(features.squeeze(-1))
        predicted_energy = (torch.sum(sym_encode.reshape(B, Nb, 1)) +
                            torch.sum(prior_out.reshape(B, Nb, 1),
                                      dim=1))

        ################## Calculate Force ##################
        neighbors_force = torch.autograd.grad(predicted_energy.sum(),
                                                dr,
                                                create_graph=True)[0].to(
            self.device)  # (B, N, N_neighbors, 3)

        predicted_force = torch.sum(neighbors_force, dim=1).reshape(B, 3)


        ################## Calculate Torque ##################
        torque_grad = torch.autograd.grad(predicted_energy.sum(),
                                            orientation,
                                            create_graph=True)[0].to(
            self.device) # (B, N, 3, 3)

        tq_x = torch.cross(torque_grad[:, :, :, 0].reshape(-1, 3), orientation[:, :, :, 0].reshape(-1, 3))
        tq_y = torch.cross(torque_grad[:, :, :, 1].reshape(-1, 3), orientation[:, :, :, 1].reshape(-1, 3))
        tq_z = torch.cross(torque_grad[:, :, :, 2].reshape(-1, 3), orientation[:, :, :, 2].reshape(-1, 3))
        predicted_torque = (tq_x + tq_y + tq_z).to(self.device)

        return predicted_force, predicted_torque, predicted_energy
if __name__ == '__main__':
    prior_net_config = {'hidden_dim': [64, 32, 5],
                        'n_layers': 2,
                        'act_fn': 'Tanh',
                        'pre_factor': 1.0,
                        'n': 2}
    energy_net_config = {'hidden_dim': [64, 128, 64, 32],
                         'n_layers': 3,
                         'act_fn': 'Tanh'}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ParticleEnergyPredictor_V2(in_dim=15,
                                         prior_net_config=prior_net_config,
                                         energy_net_config=energy_net_config,
                                         dropout=0.,
                                         device=device,
                                         batch_norm=False)

    model = model.to(device)
    print(model)
    B = 2
    Nb = 10
    dr = torch.rand(B, Nb, 3).to(device)
    dr.requires_grad = True
    orientation = torch.rand(B, 3, 3).to(device)
    orientation.requires_grad = True
    n_orientation = torch.rand(B, Nb, 3, 3).to(device)
    predicted_force, predicted_torque, predicted_energy = model(dr, orientation, n_orientation)
    print(predicted_force.shape)
