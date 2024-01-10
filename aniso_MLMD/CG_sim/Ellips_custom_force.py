import hoomd
import torch
import numpy as np
import rowan
from aniso_MLMD.model import PairNN_Force_Torque
from aniso_MLMD.utils import _calculate_torque, _calculate_prior_energy

class PairEllipsCustomForce(hoomd.md.force.Custom):
    def __init__(self, rigid_ids, model_path, in_dim, hidden_dim,
                 n_layers, act_fn, dropout, prior_energy, prior_energy_sigma, prior_energy_n):
        super().__init__(aniso=True)
        # load ML model
        self.ellipse_ids = [0, 1]
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = PairNN_Force_Torque(in_dim=in_dim, hidden_dim=hidden_dim,
                        n_layers=n_layers, act_fn=act_fn, dropout=dropout, batch_norm=False)
        self.model.to(self.device)
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.prior_energy = prior_energy
        self.prior_energy_sigma = prior_energy_sigma
        self.prior_energy_n = prior_energy_n

    def set_forces(self, timestep):
        # get positions and orientations
        with self._state.cpu_local_snapshot as snap:
            rtags = snap.particles.rtag[self.ellipse_ids]
            positions = np.array(snap.particles.position[rtags],
                                 copy=True)
            orientations = np.array(snap.particles.orientation[rtags],
                                    copy=True)

        orientation_rotation_matrix = rowan.to_matrix(orientations)

        x1 = torch.from_numpy(positions[0, :]).type(torch.FloatTensor).to(self.device)
        x2 = torch.from_numpy(positions[1, :]).type(torch.FloatTensor).to(self.device)
        R1 = torch.from_numpy(orientation_rotation_matrix[0, :]).type(
            torch.FloatTensor).to(self.device)
        R2 = torch.from_numpy(orientation_rotation_matrix[1, :]).type(
            torch.FloatTensor).to(self.device)
        x1.requires_grad = True
        R1.requires_grad = True
        energy_prediction = self.model(x1, x2, R1, R2)
        if self.prior_energy:
            prior_energy = _calculate_prior_energy(x1, x2, self.prior_energy_sigma, self.prior_energy_n).to(self.device)
            energy_prediction += prior_energy
        predicted_force = - torch.autograd.grad(energy_prediction.sum(),
                                                x1,
                                                create_graph=True)[0].to(self.device)
        torque_grad = - torch.autograd.grad(energy_prediction.sum(),
                                            R1,
                                            create_graph=True)[0]
        predicted_torque = _calculate_torque(torque_grad, R1).to(self.device)

        with self.cpu_local_force_arrays as arrays:
            #             print('timestep: ', timestep)
            #             print('****************************************')
            #             print(arrays.potential_energy)
            arrays.force[rtags] = predicted_force
            arrays.torque[rtags] = predicted_torque