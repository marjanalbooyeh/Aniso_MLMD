import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')

import torch
from aniso_per_particle.model import EnergyPredictor_Residual
from aniso_per_particle.trainer.data_loader import AnisoParticleDataLoader
from collections import OrderedDict

parameters = OrderedDict()
parameters["data_path"] ="/home/marjan/Documents/code-base/ml_datasets/pps_800_N25_Aug"
parameters["batch_size"] = 1
parameters["shrink"] = True
parameters["shrink_factor"] = 1e-3
parameters["in_dim"] = 18
parameters["out_dim"] = 3

parameters["energy_hidden_dim"] = [5, 5,3, 3,3]
parameters["energy_n_layers"] = 4
parameters["energy_act_fn"] = "Tanh"


parameters["dropout"] = 0.001
parameters["batch_norm"] = True
parameters["initial_weight"] = "xavier"
parameters["nonlinearity"] ="tanh"
parameters["bn_dim"] = 25

parameters["best_model_path"] = "best_checkpoint_3.pth"
class Struct:
    def __init__(self, **entries):
        __dict__.update(entries)




def root_mean_squared_error(pred, target):
    ## ROOT MEAN SQUARED ERROR
    return torch.sqrt(
        torch.mean((pred - target) ** 2))

def create_model(config, device):
    model = EnergyPredictor_Residual(in_dim=config.in_dim,
                                     energy_net_config=config.energy_net_config,
                                     dropout=config.dropout,
                                     batch_norm=config.batch_norm,
                                     device=device,
                                     initial_weights=config.initial_weight,
                                     nonlinearity=config.nonlinearity,
                                     bn_dim=config.bn_dim)
    model.to(device)

    model.load_state_dict(torch.load(config.best_model_path, map_location=device)["model"])

    model.to(device)

    model.eval()
    return model
def validate_model(config):
    aniso_data_loader = AnisoParticleDataLoader(config.data_path,
                                                batch_size=1,
                                                overfit=False,
                                                shrink=config.shrink,
                                                shrink_factor=config.shrink_factor,
                                                train_idx="0",
                                                processor_type=None,
                                                scale_range=(0,1) )
    valid_dataloader = aniso_data_loader.get_valid_dataset()
    print('valid_dataloader size: ', len(valid_dataloader))
    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu")
    energy_net_config = {
            "hidden_dim": config.energy_hidden_dim,
            "n_layers": config.energy_n_layers,
            "act_fn": config.energy_act_fn
        }

    model = create_model(config, device)
    model.eval()

    criteria = torch.nn.MSELoss()
    batch_counter = 0
    total_error = 0
    force_total_error = 0
    torque_total_error = 0
    total_mse = 0
    for i, (
            (dr, orientation, n_orientation)
            , target_force, target_torque, energy) in enumerate(
        valid_dataloader):
        batch_counter += 1

        dr.requires_grad = True
        orientation.requires_grad = True
        dr = dr.to(device)
        orientation = orientation.to(device)
        n_orientation = n_orientation.to(device)

        predicted_force, predicted_torque, predicted_energy = model(
            dr, orientation, n_orientation)
        target_force = target_force.to(device)
        target_torque = target_torque.to(device)

        force_RMSE = root_mean_squared_error(predicted_force, target_force)
        torque_RMSE = root_mean_squared_error(predicted_torque, target_torque)

        force_total_error += force_RMSE
        torque_total_error += torque_RMSE
        total_error += force_RMSE + torque_RMSE

        total_mse += criteria(predicted_force, target_force) + criteria(predicted_torque, target_torque)

        if i % 20 == 0:
            print('force_error: ', force_RMSE)
            print('torque_error:', torque_RMSE)
            print('MSE: ', criteria(predicted_force, target_force) + criteria(predicted_torque, target_torque))
            print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
            print("force prediction: ", predicted_force[0])
            print("force target: ", target_force[0])
            print("torque prediction: ", predicted_torque[0])
            print("torque target: ", target_torque[0])
            print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
        del dr, orientation, n_orientation, target_force, target_torque, energy, predicted_force, predicted_torque
        torch.cuda.empty_cache()

    print('force_total: ', force_total_error / batch_counter)
    print('torque_total: ', torque_total_error / batch_counter)
    print('total_error: ', total_error / batch_counter)
    print('total_mse: ', total_mse / batch_counter)

if __name__ == '__main__':
    config = Struct(**parameters)
    validate_model(config)