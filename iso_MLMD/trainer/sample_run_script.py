from collections import OrderedDict
parameters = OrderedDict()

# project parameters
parameters["project"] = "pps-20-new"
parameters["group"] = "rotation_matrix"
parameters["notes"] = "Learning pps forces and torques"
parameters["tags"] = "random sampling"
parameters["wandb_log"] = False

# dataset parameters
parameters["data_path"] = "/home/marjan/Documents/code-base/ml_datasets/iso_N_100"
parameters["batch_size"] = 32
parameters["shrink"] = False
parameters["overfit"] = False

# model parameters
parameters["model_type"] = "neighbor"
parameters["in_dim"] = 5
parameters["hidden_dim"] = 64
parameters["n_layer"] = 2
parameters["box_len"] = 6
parameters["act_fn"] = "Tanh"
parameters["dropout"] = 0.3
parameters["batch_norm"] = False


# optimizer parameters
parameters["optim"] = "Adam"
parameters["lr"] = 0.001
parameters["min_lr"] = 0.00001
parameters["use_scheduler"] = True
parameters["scheduler_type"] = "ReduceLROnPlateau"
parameters["decay"] = 0.001
# supported loss types: "mse" (mean squared error), "mae" (means absolute error)
parameters["loss_type"] = "mse"
parameters["prior_energy"] = True
parameters["prior_energy_sigma"] = 1
parameters["prior_energy_n"] = 6

# run parameters
parameters["epochs"] = 5000

class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

config = Struct(**parameters)

job_id = 0

from iso_MLMD.trainer import IsoTrainer
trainer_obj = IsoTrainer(config, job_id)
trainer_obj.run()

