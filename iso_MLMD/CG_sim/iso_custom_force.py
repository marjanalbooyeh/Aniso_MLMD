import hoomd
import torch
import numpy as np
import freud
import cupy
from iso_MLMD.trainer import IsoNeighborNN
import time


class IsoNeighborCustomForce(hoomd.md.force.Custom):
    def __init__(self, model_path, in_dim, hidden_dim,
                 n_layers, box_len, act_fn, dropout, prior_energy,
                 prior_energy_sigma,
                 prior_energy_n, NN=40):
        super().__init__(aniso=False)
        # load ML model
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = IsoNeighborNN(in_dim=in_dim,
                                   hidden_dim=hidden_dim,
                                   n_layers=n_layers,
                                   box_len=box_len,
                                   act_fn=act_fn,
                                   dropout=dropout,
                                   batch_norm=False,
                                   device=self.device,
                                   prior_energy=prior_energy,
                                   prior_energy_sigma=prior_energy_sigma,
                                   prior_energy_n=prior_energy_n)
        self.model.to(self.device)
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device)['model'])
        self.model.eval()
        self.box_len = box_len
        self.box = freud.box.Box.cube(self.box_len)
        self.NN = NN

    def find_neighbors(self, points, ):
        aq = freud.locality.AABBQuery(self.box, points)
        query_points = points
        query_result = aq.query(query_points,
                                dict(num_neighbors=self.NN, exclude_ii=True))
        nlist = query_result.toNeighborList()
        neighbor_list = np.asarray(
            list(zip(nlist.query_point_indices, nlist.point_indices)))
        return neighbor_list

    def set_forces(self, timestep):
        # get positions and orientations
        with self._state.gpu_local_snapshot as snap:
            rtags = cupy.array(snap.particles.rtag, copy=False)
            positions = cupy.array(snap.particles.position[rtags],
                                   copy=True)

        # get neighbor list
        neighbor_list = self.find_neighbors(positions.get())
        positions_tensor = torch.as_tensor(positions).type(
            torch.FloatTensor).to(self.device).unsqueeze(0)
        neighbor_list_tensor = torch.from_numpy(neighbor_list.astype(np.int64)).to(self.device).unsqueeze(0)
        positions_tensor.requires_grad = True
        predicted_force = self.model(positions_tensor, neighbor_list_tensor).detach()

        predicted_force_cupy = cupy.asarray(predicted_force)

        with self.gpu_local_force_arrays as arrays:
            with self._state.gpu_local_snapshot as snap:
                rtags = cupy.array(snap.particles.rtag, copy=False)
                arrays.force[rtags] = predicted_force_cupy
        del predicted_force, predicted_force_cupy, neighbor_list_tensor, positions_tensor, rtags, positions

