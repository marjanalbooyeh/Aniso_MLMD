from collections import OrderedDict
parameters = OrderedDict()

# project parameters
parameters["project"] = "pps-energy-Apr22"
parameters["group"] = "energy-V2"
parameters["log"] = False
parameters["resume"] = False

# dataset parameters700
parameters["data_path"] ="/home/marjan/Documents/code-base/ml_datasets/pps_800_N25_Aug"
parameters["batch_size"] = 3
parameters["shrink"] = False
parameters["shrink_factor"] = 1e-4 * 3
parameters["overfit"] = True
parameters["train_idx"] = "0"
parameters["processor_type"] = None
parameters["scale_range"] = (0, 1)

# model parameters
parameters["in_dim"] = 18

parameters["prior_hidden_dim"] = [30, 64, 64, 32]
parameters["prior_n_layers"] = 3
parameters["prior_act_fn"] = "Tanh"

parameters["energy_hidden_dim"] = [5, 5,3, 3,3]
parameters["energy_n_layers"] = 4
parameters["energy_act_fn"] = "Tanh"

parameters["dropout"] = 0.001
parameters["batch_norm"] = True
parameters["model_type"] = "v2"
parameters["initial_weight"] = "xavier"
parameters["nonlinearity"] ="tanh"
parameters["bn_dim"] = 25
# optimizer parameters
parameters["optim"] = "Adam"
parameters["lr"] = 0.001
parameters["min_lr"] = 0.0001
parameters["use_scheduler"] = False
parameters["scheduler_type"] = "StepLR"
parameters["scheduler_patience"] = 50
parameters["scheduler_threshold"] = 20
parameters["decay"] = 0.0001
# supported loss types: "mse" (mean squared error), "mae" (means absolute error)
parameters["loss_type"] = "mse"
parameters["clipping"] = True

# run parameters
parameters["epochs"] = 5000

class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

config = Struct(**parameters)

job_id = 0

from aniso_per_particle.trainer import EnergyTrainer_V3
trainer_obj = EnergyTrainer_V3(config, job_id)
trainer_obj.run()

