import torch
import torch.nn as nn
import rowan
from pytorch3d.transforms import quaternion_invert, quaternion_raw_multiply
from torch.nn.functional import pad
import time


class BaseNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, n_layers, act_fn="ReLU",
                 dropout=0.3,
                 inp_mode="append", batch_norm=True,
                 # augment_pos="r", augment_orient="a", pool="mean",
                 device=None
                 ):
        super(BaseNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.n_layers = n_layers
        self.act_fn = act_fn
        self.dropout = dropout
        self.inp_mode = inp_mode
        # self.augment_pos = augment_pos
        # self.augment_orient = augment_orient
        # self.pool = pool
        self.device = device

        self.in_dim = in_dim + 2
        self.batch_norm = batch_norm
        # if self.augment_pos == "r":
        #     self.in_dim += 1
        # if self.augment_orient == "q1q2":
        #     self.in_dim += 12

        self.net = self._get_net()
        self.net.apply(self.init_net_weights)

    def init_net_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)

    #             m.bias.data.fill_(0.01)

    def _get_act_fn(self):
        act = getattr(nn, self.act_fn)
        return act()

    def _prep_input(self, features, x1, x2):
        if self.inp_mode == "append":
            dr = x1 - x2
            r = torch.norm(dr, dim=1, keepdim=True)
            x = torch.concat((features, r, 1. / r), dim=1)

            # if self.augment_pos == "r":
            #     r = torch.norm(dr, dim=1, keepdim=True)
            #     x = torch.concat((x, r), dim=1)
            # if self.augment_orient == "q1q2":
            #     rel_orient = quaternion_raw_multiply(q1,quaternion_invert(q2)).type(torch.FloatTensor)
            #     #apply dr to q1 and q2
            #     dr_quat = pad(dr, (1, 0), "constant", 0)
            #     dr_quat_inv = quaternion_invert(dr_quat)
            #     q1_rotated = quaternion_raw_multiply(dr_quat, quaternion_raw_multiply(q1, dr_quat_inv))
            #     q2_rotated = quaternion_raw_multiply(dr_quat, quaternion_raw_multiply(q2, dr_quat_inv))
            #     x = torch.concat((x, rel_orient, q1_rotated, q2_rotated), dim=1)

            return x.to(self.device)

        # if self.inp_mode == "stack":
        #     raise NotImplementedError


#     def _augment_input(self, x):
#         if self.augment_pos == "r" and self.inp_mode =="append":
#             # include center-to-center distance as a feature
#             x = torch.cat((x, torch.norm(x[:, :3], dim=1, keepdim=True).to(x.device)), dim=1).to(x.device)

#         if self.augment_orient == "a" and self.inp_mode == "append":
#             angles = torch.tensor(rowan.to_axis_angle(x[:, 3:7].detach().cpu().numpy())[1]).unsqueeze(1).to(x.device)
#             x = torch.cat((x, angles), dim=1).to(x.device)
#         return x


class NN(BaseNN):
    def __init__(self, in_dim, hidden_dim, out_dim, n_layers, **kwargs):
        super(NN, self).__init__(in_dim, hidden_dim, out_dim, n_layers,
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

    def forward(self, features, x1, x2):
        x = self._prep_input(features, x1, x2)
        out = self.net(x)

        #         if self.inp_mode == "stack":
        #             if self.pool == "mean":
        #                 out = torch.mean(out, dim=1)
        #             elif self.pool == "sum":
        #                 out = torch.sum(out, dim=1)
        #             elif self.pool == "max":
        #                 out = torch.max(out, dim=1)[0]
        return out


class NNSkipShared(BaseNN):
    def __init__(self, in_dim, hidden_dim, out_dim, n_layers, **kwargs):
        super(NNSkipShared, self).__init__(in_dim, hidden_dim, out_dim,
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

    def forward(self, features, x1, x2):
        x = self._prep_input(features, x1, x2)
        # transform input to hidden dim size
        x_transform = self.input_connection(x)
        for i in range(self.n_layers):
            # add original transformed input to each layer before activation
            x = self.activations[i](self.net[i](x) + x_transform)
            x = self.dropouts[i](x)

        out = self.net[-1](x)
        # if self.inp_mode == "stack":
        #     if self.pool == "mean":
        #         out = torch.mean(out, dim=1)
        #     elif self.pool == "sum":
        #         out = torch.sum(out, dim=1)
        #     elif self.pool == "max":
        #         out = torch.max(out, dim=1)[0]
        return out


class NNGrow(BaseNN):
    def __init__(self, in_dim, hidden_dim, out_dim, n_layers, **kwargs):
        super(NNGrow, self).__init__(in_dim, hidden_dim, out_dim, n_layers,
                                     **kwargs)

    def _get_net(self):
        layers = [nn.Linear(self.in_dim, self.hidden_dim[0]),
                  self._get_act_fn()]
        for i in range(1, len(self.hidden_dim)):
            layers.append(nn.Linear(self.hidden_dim[i - 1], self.hidden_dim[i]))
            # if self.batch_dim:
            #     layers.append(nn.BatchNorm1d(self.batch_dim))
            layers.append(self._get_act_fn())
            layers.append(nn.Dropout(p=self.dropout))
        layers.append(nn.Linear(self.hidden_dim[-1], self.out_dim))
        return nn.Sequential(*layers)

    def forward(self, features, x1, x2):
        x = self._augment_input(features, x1, x2)
        out = self.net(x)

        # if self.inp_mode == "stack":
        #     if self.pool == "mean":
        #         out = torch.mean(out, dim=1)
        #     elif self.pool == "sum":
        #         out = torch.sum(out, dim=1)
        #     elif self.pool == "max":
        #         out = torch.max(out, dim=1)[0]
        return out