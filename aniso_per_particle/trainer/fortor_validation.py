import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')

import torch
import torch.nn as nn
import wandb
import os
import pandas as pd

from aniso_per_particle.trainer.data_loader import  ParticleConfigDataset, _get_data_loader
from aniso_per_particle.model import ParticleTorForPredictor_V1

from collections import OrderedDict
parameters = OrderedDict()
parameters["in_dim"] = 18
parameters["out_dim"] = 3


parameters["force_hidden_dim"] = [32, 128, 64, 32]
parameters["force_n_layers"] = 3
parameters["force_act_fn"] = "LeakyReLU"
parameters["force_pre_factor"] = 1.0

parameters["torque_hidden_dim"] = [32, 128, 64, 32 ]
parameters["torque_n_layers"] = 3
parameters["torque_act_fn"] = "LeakyReLU"
parameters["torque_pre_factor"] = 1.0

parameters["lpar"] = 2.176
parameters["lperp"] = 1.54
parameters["sigma"] = -1.13
parameters["n_factor"] = 12

parameters["dropout"] = 0.001
parameters["batch_norm"] = False
parameters["initial_weight"] = "xavier"
class Struct:
    def __init__(self, **entries):
        __dict__.update(entries)

config = Struct(**parameters)


def root_mean_squared_error(pred, target):
    ## ROOT MEAN SQUARED ERROR
    return torch.sqrt(
        torch.mean((pred - target) ** 2))
def validate_model(data_path, ):
    val_df = pd.read_pickle(os.path.join(data_path, 'valid.pkl'))
    valid_dataset = ParticleConfigDataset(val_df)
    valid_dataloader = _get_data_loader(valid_dataset, batch_size=1, shuffle=False)

    force_net_config = {
        "hidden_dim": config.force_hidden_dim,
        "n_layers": config.force_n_layers,
        "act_fn": config.force_act_fn,
        "pre_factor": config.force_pre_factor,
    }
    torque_net_config = {
        "hidden_dim": config.torque_hidden_dim,
        "n_layers": config.torque_n_layers,
        "act_fn": config.torque_act_fn,
        "pre_factor": config.torque_pre_factor,
    }
    ellipsoid_config = {'lpar': config.lpar,
                             'lperp': config.lperp,
                             'sigma': config.sigma,
                             'n_factor': config.n_factor}
    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu")
    model = ParticleTorForPredictor_V1(config.in_dim,
                                       force_net_config=force_net_config,
                                       torque_net_config=torque_net_config,
                                       ellipsoid_config=ellipsoid_config,
                                       dropout=config.dropout,
                                       batch_norm=config.batch_norm,
                                       device=device,
                                       initial_weights=config.initial_weight)

    model.to(device)


    model.load_state_dict(torch.load("best_checkpoint_3.pth", map_location=device)["model"])

    model.to(device)

    model.eval()
    with torch.no_grad():
        total_error = 0.
        batch_counter = 0
        for i, (
                (dr, orientation, n_orientation)
                , target_force, target_torque, energy) in enumerate(valid_dataloader):
            batch_counter += 1
            dr.requires_grad = True
            orientation.requires_grad = True
            dr = dr.to(device)
            orientation = orientation.to(device)
            n_orientation = n_orientation.to(device)

            predicted_force, predicted_torque = model(
                dr, orientation, n_orientation)

            target_force = target_force.to(device)

            target_torque = target_torque.to(device)

            force_RMSE = root_mean_squared_error(predicted_force, target_force)
            torque_RMSE = root_mean_squared_error(predicted_torque, target_torque)
            total_error += force_RMSE + torque_RMSE

            if i % 20 == 0:
                print('force_error: ', force_RMSE)
                print('torque_error:', torque_RMSE)
                print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
                print("force prediction: ", predicted_force[0])
                print("force target: ", target_force[0])
                print("torque prediction: ", predicted_torque[0])
                print("torque target: ", target_torque[0])
                print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
            del dr, orientation, n_orientation, target_force, target_torque, energy, predicted_force, predicted_torque
            torch.cuda.empty_cache()

    print('total: ', total_error / batch_counter)

if __name__ == '__main__':
    data_path = "/home/marjan/Documents/code-base/ml_datasets/PPS_800_June10/valid_1k.pkl"

    validate_model(data_path=data_path)