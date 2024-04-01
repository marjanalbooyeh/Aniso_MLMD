import cupy
import freud
import hoomd
import numpy as np
import rowan
import torch

from aniso_MLMD.model import ForTorPredictorNN


class AnisoNeighborCustomForce(hoomd.md.force.Custom):
    def __init__(self, best_force_job, best_torque_job, NN=15):
        super().__init__(aniso=True)
        # load ML model
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")

        # load force model
        self.force_model = ForTorPredictorNN(in_dim=best_force_job.sp.in_dim,
                                   neighbor_hidden_dim=best_force_job.sp.neighbor_hidden_dim,
                                   n_layers=best_force_job.sp.n_layer,
                                   box_len=best_force_job.sp.box_len,
                                   act_fn=best_force_job.sp.act_fn,
                                   dropout=best_force_job.sp.dropout,
                                   batch_norm=False,
                                   device=self.device,
                                   neighbor_pool=best_force_job.sp.neighbor_pool,
                                   prior_force=best_force_job.sp.prior_force,
                                   prior_force_sigma=best_force_job.sp.prior_force_sigma,
                                   prior_force_n=best_force_job.sp.prior_force_n
                                   )
        self.force_model.to(self.device)
        self.force_model.load_state_dict(
            torch.load(best_force_job.fn("best_model_checkpoint.pth"),
                       map_location=self.device)['model'])
        self.force_model.eval()

        # load torque model
        self.torque_model = ForTorPredictorNN(in_dim=best_torque_job.sp.in_dim,
                                      neighbor_hidden_dim=best_torque_job.sp.neighbor_hidden_dim,
                                      n_layers=best_torque_job.sp.n_layer,
                                      box_len=best_torque_job.sp.box_len,
                                      act_fn=best_torque_job.sp.act_fn,
                                      dropout=best_torque_job.sp.dropout,
                                      batch_norm=False,
                                      device=self.device,
                                      neighbor_pool=best_torque_job.sp.neighbor_pool)
        self.torque_model.to(self.device)
        self.torque_model.load_state_dict(
            torch.load(best_torque_job.fn("best_model_checkpoint.pth"),
                       map_location=self.device)['model'])
        self.torque_model.eval()
        self.box_len = best_force_job.sp.box_len
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
            orientations_q = cupy.array(snap.particles.orientation[rtags],
                                       copy=True)
        orientation_R = rowan.to_matrix(orientations_q.get())
        # get neighbor list
        neighbor_list = self.find_neighbors(positions.get())
        # preparing torch tensors
        positions_tensor = torch.as_tensor(positions).type(
            torch.FloatTensor).to(self.device).unsqueeze(0)
        neighbor_list_tensor = torch.from_numpy(neighbor_list.astype(np.int64)).to(self.device).unsqueeze(0)
        orientation_R_tensor = torch.from_numpy(orientation_R).type(torch.FloatTensor).to(self.device).unsqueeze(0)
        with torch.no_grad():
        
            predicted_force = self.force_model(positions_tensor, orientation_R_tensor,
                                               neighbor_list_tensor).detach()
            predicted_torque = self.torque_model(positions_tensor, orientation_R_tensor,
                                               neighbor_list_tensor).detach()

        predicted_force_cupy = cupy.asarray(predicted_force)
        predicted_torque_cupy = cupy.asarray(predicted_torque)
        

        with self.gpu_local_force_arrays as arrays:
            with self._state.gpu_local_snapshot as snap:
                rtags = cupy.array(snap.particles.rtag, copy=False)
                arrays.force[rtags] = predicted_force_cupy
                arrays.torque[rtags] = predicted_torque_cupy
        del predicted_force, predicted_force_cupy,\
            predicted_torque, predicted_torque_cupy,\
            neighbor_list_tensor, positions_tensor, rtags, positions

