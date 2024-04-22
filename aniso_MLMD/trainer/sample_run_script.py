from collections import OrderedDict
parameters = OrderedDict()

# project parameters
parameters["project"] = "pps-energy-Apr22"
parameters["group"] = "energy-V2"
parameters["log"] = False

# dataset parameters700
parameters["data_path"] ="/home/marjan/Documents/fry/ml_dataset/pps_300_N15_balanced"
parameters["batch_size"] = 10
parameters["shrink"] = False
parameters["overfit"] = True


# model parameters
parameters["in_dim"] = 19
parameters["out_dim"] = 128

parameters["neighbor_hidden_dim"] = 256
parameters["neighbor_pool"] = "sum"
parameters["neighbors_n_layers"] = 2
parameters["neighbors_act_fn"] = "Tanh"

parameters["prior_hidden_dim"] = 256
parameters["prior_n_layers"] = 2
parameters["prior_act_fn"] = "Tanh"

parameters["energy_hidden_dim"] = 64
parameters["energy_n_layers"] = 2
parameters["energy_act_fn"] = "Tanh"


parameters["dropout"] = 0.3
parameters["batch_norm"] = False

# optimizer parameters
parameters["optim"] = "Adam"
parameters["lr"] = 0.01
parameters["min_lr"] = 0.0001
parameters["use_scheduler"] = True
parameters["scheduler_type"] = "ReduceLROnPlateau"
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

from aniso_MLMD.trainer import EnergyTrainer
trainer_obj = EnergyTrainer(config, job_id)
trainer_obj.run()

