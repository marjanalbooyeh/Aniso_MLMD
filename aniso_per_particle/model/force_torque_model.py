import torch
import torch.nn as nn

from aniso_per_particle.model import feature_vector, get_act_fn


def cosine_features(dr, particle_orientation, neighbors_orientations):
    """
    Calculate the features for the particle and its neighbors

    Parameters
    ----------
    dr: distance vector between particle and its neighbors (B, N_neighbors, 3)
    particle_orientation: particle's orientation (B, 3, 3)
    neighbors_orientations: neighbors' orientation (B, N_neighbors, 3, 3)

    Returns
    -------
    features: (B, N_neighbors, 15)
    """
    dr_orient_dot = torch.einsum('ijk, ikh -> ijh', dr, particle_orientation)
    dr_n_orient_dot = torch.einsum('ijk, ijkh -> ijh', dr,
                                   neighbors_orientations)
    orient_n_orient_dot = torch.einsum('ihl, ijhm -> ijlm',
                                       particle_orientation,
                                       neighbors_orientations)
    features = torch.cat(
        [dr, dr_orient_dot, dr_n_orient_dot, orient_n_orient_dot], dim=-1)
    return features


# class PhysicsInformedRepulsiveLayer(nn.Module):
#     def __init__(self, lpar, lperp, sigma):
#         super(PhysicsInformedRepulsiveLayer, self).__init__()
#         self.lpar = lpar
#         self.lperp = lperp
#         self.sigma = sigma
#         self.psi = (self.lpar ** 2 - self.lperp ** 2) / (
#                 self.lpar ** 2 + self.lperp ** 2)
#         self.sigma_factor = torch.sqrt(2) * self.sigma
#
#     def forward(self, dr, dr_orient_dot, dr_n_orient_dot, orient_n_orient_dot):
#         # Apply a physics-informed repulsive potential equation
#         main_axis_orient_n_orient = torch.diagonal(orient_n_orient_dot, dim1=-1,
#                                                    dim2=-2)
#         repulsion_pre = torch.pow((1 - (self.psi ** 2) *
#                                    torch.pow(main_axis_orient_n_orient, 2)),
#                                   0.5)  # (B, N_neighbors, 3)
#         dr_orient_enc1 = ((dr_orient_dot + dr_n_orient_dot) ** 2 / (
#                 1 + self.psi * main_axis_orient_n_orient))
#         dr_orient_enc2 = ((dr_orient_dot - dr_n_orient_dot) ** 2 / (
#                 1 - self.psi * main_axis_orient_n_orient))
#         dr_orient_enc = torch.pow(
#             1 - 0.5 * self.psi * (dr_orient_enc1 + dr_orient_enc2),
#             -0.5) * self.sigma_factor  # (B, N_neighbors, 3)
#
#         repulsion_exp = torch.exp(- (dr / dr_orient_enc) ** 2)
#         replusion_factor = repulsion_pre * repulsion_exp  # (B, N_neighbors, 3)
#
#         return replusion_factor


def repulsion_calculator_v1(dr, dr_orient_dot, dr_n_orient_dot,
                            orient_n_orient_dot, n_factor, psi, sigma_factor):
    # Apply a physics-informed repulsive potential equation
    main_axis_orient_n_orient = torch.diagonal(orient_n_orient_dot, dim1=-1,
                                               dim2=-2)
    # repulsion_pre = torch.pow((1 - (psi**2)*
    #                            torch.pow(main_axis_orient_n_orient, 2)),
    #                           0.5) # (B, N_neighbors, 3)
    dr_orient_enc1 = ((dr_orient_dot + dr_n_orient_dot) ** 2 / (
            1 + psi * main_axis_orient_n_orient))
    dr_orient_enc2 = ((dr_orient_dot - dr_n_orient_dot) ** 2 / (
            1 - psi * main_axis_orient_n_orient))
    dr_orient_enc = torch.pow(1 - 0.5 * psi * (dr_orient_enc1 + dr_orient_enc2),
                              -0.5) * sigma_factor  # (B, N_neighbors, 3)

    repulsion = torch.pow(dr - dr_orient_enc,
                          n_factor) / dr  # (B, N_neighbors, 3)
    return repulsion


def repulsion_calculator_v2(dr, dr_orient_dot, dr_n_orient_dot,
                            orient_n_orient_dot, psi, sigma_factor):
    # Apply a physics-informed repulsive potential equation
    main_axis_orient_n_orient = torch.diagonal(orient_n_orient_dot, dim1=-1,
                                               dim2=-2)
    # repulsion_pre = torch.pow((1 - (psi**2)*
    #                            torch.pow(main_axis_orient_n_orient, 2)),
    #                           0.5) # (B, N_neighbors, 3)
    dr_orient_enc1 = ((dr_orient_dot + dr_n_orient_dot) ** 2 / (
            1 + psi * main_axis_orient_n_orient))
    dr_orient_enc2 = ((dr_orient_dot - dr_n_orient_dot) ** 2 / (
            1 - psi * main_axis_orient_n_orient))
    dr_orient_enc = torch.pow(1 - 0.5 * psi * (dr_orient_enc1 + dr_orient_enc2),
                              -0.5) * sigma_factor  # (B, N_neighbors, 3)

    repulsion_exp = torch.exp(- (dr / dr_orient_enc) ** 2)
    return repulsion_exp


class ParticleTorForPredictor_V1(nn.Module):
    def __init__(self, in_dim,
                 force_net_config,
                 torque_net_config,
                 ellipsoid_config,
                 dropout=0.3,
                 batch_norm=False,
                 device=None,
                 initial_weights=None,
                 ):
        super(ParticleTorForPredictor_V1, self).__init__()

        self.force_hidden_dim = force_net_config['hidden_dim']
        self.force_n_layers = force_net_config['n_layers']
        self.force_act_fn = force_net_config['act_fn']
        self.force_pre_factor = force_net_config['pre_factor']

        self.torque_hidden_dim = torque_net_config['hidden_dim']
        self.torque_n_layers = torque_net_config['n_layers']
        self.torque_act_fn = torque_net_config['act_fn']
        self.torque_pre_factor = torque_net_config['pre_factor']

        self.lpar = ellipsoid_config['lpar']
        self.lperp = ellipsoid_config['lperp']
        self.sigma = ellipsoid_config['sigma']
        self.n_factor = ellipsoid_config['n_factor']
        self.psi = (self.lpar ** 2 - self.lperp ** 2) / (
                self.lpar ** 2 + self.lperp ** 2)
        self.sigma_factor = torch.sqrt(torch.tensor(2)) * self.sigma

        self.dropout = dropout
        self.device = device
        self.in_dim = in_dim
        self.batch_norm = batch_norm

        self.force_net = self._MLP_net(in_dim=self.in_dim,
                                       h_dim=self.force_hidden_dim,
                                       out_dim=3,
                                       n_layers=self.force_n_layers,
                                       act_fn=self.force_act_fn).to(self.device)

        self.torque_net = self._MLP_net(in_dim=self.in_dim,
                                        h_dim=self.torque_hidden_dim,
                                        out_dim=3,
                                        n_layers=self.torque_n_layers,
                                        act_fn=self.torque_act_fn,).to(
            self.device)

        self.force_repulsion_net = self._MLP_net(in_dim=3,
                                                 h_dim=[32, 32],
                                                 out_dim=3,
                                                 n_layers=1,
                                                 act_fn=self.torque_act_fn,
                                                 ).to(self.device)
        self.torque_repulsion_net = self._MLP_net(in_dim=3,
                                                    h_dim=[32, 32],
                                                    out_dim=3,
                                                    n_layers=1,
                                                    act_fn=self.torque_act_fn).to(self.device)
        if initial_weights:
            self.force_net.apply(self.weights_init)
            self.torque_net.apply(self.weights_init)
            self.force_repulsion_net.apply(self.weights_init)
            self.torque_repulsion_net.apply(self.weights_init)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.uniform_(m.weight.data)
            nn.init.uniform_(m.bias.data)

    def _MLP_net(self, in_dim, h_dim, out_dim,
                 n_layers, act_fn):

        layers = [nn.Linear(in_dim, h_dim[0]),
                  get_act_fn(act_fn)]
        for i in range(n_layers):
            layers.append(
                nn.Linear(h_dim[i], h_dim[i + 1]))
            # if self.batch_norm:
            #     layers.append(nn.BatchNorm1d(bn_dim))
            layers.append(get_act_fn(act_fn))
            layers.append(nn.Dropout(p=self.dropout))
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
        dr = dr + epsilon
        R = torch.norm(dr, dim=-1, keepdim=True)  # (B, N_neighbors,1)
        dr_norm = dr / R

        ################## Prep features ##################
        dr_orient_dot = torch.einsum('ijk, ikh -> ijh', dr_norm,
                                     orientation).to(
            self.device)  # (B, N_neighbors, 3)
        dr_n_orient_dot = torch.einsum('ijk, ijkh -> ijh', dr_norm,
                                       n_orientation).to(
            self.device)  # (B, N_neighbors, 3)
        orient_n_orient_dot = torch.einsum('ihl, ijhm -> ijlm',
                                           orientation,
                                     n_orientation).to(
            self.device)  # (B, N_neighbors, 3, 3)

        features = torch.cat(
            [dr_norm, dr_orient_dot, dr_n_orient_dot,
             orient_n_orient_dot.reshape(B, Nb, 9)],
            dim=-1)  # (B, N_neighbors, 18)
        all_features = (torch.cat([R, 1 / R,
                                   features], dim=-1).to(
            self.device))  # (B, N_neighbors, 20)
        fcR = (torch.where(R > R_cut,
                           0.0,
                           (0.5 * torch.cos(torch.pi * R / R_cut) + 0.5)).
               reshape(B, Nb, 1))
        ############# Repulsion Net ##############
        repulsion_factor = repulsion_calculator_v1(dr_norm, dr_orient_dot,
                                                   dr_n_orient_dot,
                                                   orient_n_orient_dot,
                                                   n_factor=self.n_factor,
                                                   psi=self.psi,
                                                   sigma_factor=self.sigma_factor)  # (B, Nb, 3)
        force_repulsion = self.force_repulsion_net(repulsion_factor)  # (B, Nb, 3)
        torque_repulsion = self.torque_repulsion_net(repulsion_factor)  # (B, Nb, 3)

        ############# Force Net ##############
        force_enc = self.force_net(all_features)  # (B, Nb, 3)
        force_out = self.force_pre_factor * force_enc + force_repulsion

        ############# Torque Net ##############
        torque_enc = self.torque_net(all_features)  # (B, Nb, 3)
        torque_out = self.torque_pre_factor * torque_enc + torque_repulsion

        predicted_force = torch.sum(force_out, dim=1).reshape(B, 3)
        predicted_torque = torch.sum(torque_out, dim=1).reshape(B, 3)


        return predicted_force, predicted_torque


if __name__ == '__main__':
    force_net_config = {'hidden_dim': [64, 32, 5],
                        'n_layers': 2,
                        'act_fn': 'Tanh',
                        'pre_factor': 1.0,
                        'n': 2}
    torque_net_config = {'hidden_dim': [64, 128, 64, 32],
                         'n_layers': 3,
                         'act_fn': 'Tanh',
                         'pre_factor': 1.0,}
    ellipsoid_config = {'lpar': 2.176,
                        'lperp': 1.54,
                        'sigma': -1.13,
                        'n_factor': 12}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ParticleTorForPredictor_V1(in_dim=20,
                                            force_net_config=force_net_config,
                                            torque_net_config=torque_net_config,
                                            ellipsoid_config=ellipsoid_config,
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
