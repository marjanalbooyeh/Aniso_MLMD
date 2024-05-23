from collections import OrderedDict
parameters = OrderedDict()

# project parameters
parameters["project"] = "pps-energy-Apr22"
parameters["group"] = "energy-V2"
parameters["log"] = False
parameters["resume"] = False

# dataset parameters700
parameters["data_path"] ="/Users/marjan/Documents/code-base/cmelab/Aniso_Project/ml_datasets/pps_800"
parameters["batch_size"] = 64
parameters["shrink"] = False
parameters["overfit"] = False
parameters["train_idx"] = 0



# model parameters
parameters["in_dim"] = 15
parameters["out_dim"] = 3


parameters["prior_hidden_dim"] = [5, 32, 64, 64, 32, 5, 3]
parameters["prior_n_layers"] = 6
parameters["prior_act_fn"] = "Tanh"
parameters["prior_pre_factor"] = 0.1
parameters["prior_n"] = 1
parameters["energy_hidden_dim"] = [5, 32, 32, 32, 3]
parameters["energy_n_layers"] = 4
parameters["energy_act_fn"] = "Tanh"


parameters["dropout"] = 0.5
parameters["batch_norm"] = False
parameters["model_type"] = "v2"
parameters["initial_weight"] =  None
# optimizer parameters
parameters["optim"] = "Adam"
parameters["lr"] = 0.00015
parameters["min_lr"] = 0.0001
parameters["use_scheduler"] = True
parameters["scheduler_type"] = "ReduceLROnPlateau"
parameters["scheduler_patience"] = 50
parameters["scheduler_threshold"] = 20
parameters["decay"] = 0.0001
# supported loss types: "mse" (mean squared error), "mae" (means absolute error)
parameters["loss_type"] = "mse"
parameters["clipping"] = False

# run parameters
parameters["epochs"] = 5000

class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

config = Struct(**parameters)

job_id = 0

from aniso_per_particle.trainer import EnergyTrainer
trainer_obj = EnergyTrainer(config, job_id)
trainer_obj.run()

