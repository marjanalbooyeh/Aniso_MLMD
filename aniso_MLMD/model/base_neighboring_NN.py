import torch
import torch.nn as nn
from aniso_MLMD.model.rotation_matrix_ops import dot_product, cross_product, \
    relative_orientation, rel_pos_orientation, RBF_rel_pos, rot_matrix_to_angle
from aniso_MLMD.utils import adjust_periodic_boundary

class BaseNeighborNN(nn.Module):
    """Base class for neural networks trained on pair particles."""

    def __init__(self, in_dim, hidden_dim, n_layers, box_len, act_fn="ReLU",
                 dropout=0.3, batch_norm=True, device=None
                 ):
        super(BaseNeighborNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.energy_out_dim = 1
        self.n_layers = n_layers
        self.box_len = box_len
        self.act_fn = act_fn
        self.dropout = dropout
        self.device = device
        self.in_dim = in_dim
        self.batch_norm = batch_norm

        self.energy_net = self._get_energy_net(self.energy_out_dim)

        # initialize weights and biases
        #self.energy_net.apply(self.init_net_weights)

    def init_net_weights(self, m):
        # todo: add option to initialize uniformly for weights and biases
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def _get_act_fn(self):
        act = getattr(nn, self.act_fn)
        return act()

    def _prep_features_rot_matrix(self, particle_x, neighbors_x, particle_R, neighbors_R):
        """

        :param particle_x: Particle's position, shape (batch_size, 3)
        :param neighbors_x: Neighbors' positions, shape (batch_size, n_neighbors, 3)
        :param particle_R: Particle's orientation, shape (batch_size, 3, 3)
        :param neighbors_R: Neighbors' orientations, shape (batch_size, n_neighbors, 3, 3)

        """

        batch_size = particle_x.shape[0]

        dr = (particle_x[:, None, :] - neighbors_x)
        dr = adjust_periodic_boundary(dr, self.box_len)
        R = torch.norm(dr, dim=2, keepdim=True)
        dr = dr / R
        inv_r = 1. / R

        orient_dot_prod = dot_product(R1, R2)
        orient_cross_prod, orient_cross_prod_norm = cross_product(R1, R2)
        rel_orient = relative_orientation(R1, R2)
        rel_pos_align, rel_pos_project = rel_pos_orientation(dr, R1, R2)
        rbf = RBF_rel_pos(dr, R1, R2)
        angle = rot_matrix_to_angle(rel_orient)

        features = torch.cat((R, inv_r, dr, orient_dot_prod,
                              orient_cross_prod.reshape(batch_size, -1),
                              orient_cross_prod_norm,
                              rel_orient.reshape(batch_size, -1),
                              rel_pos_align,
                              rel_pos_project.reshape(batch_size, -1),
                              rbf, angle),
                             dim=1)

        return features.to(self.device)
