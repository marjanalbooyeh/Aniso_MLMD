from collections import OrderedDict
parameters = OrderedDict()

# project parameters
parameters["project"] = "pps-20-new"
parameters["group"] = "rotation_matrix"
parameters["notes"] = "Learning pps forces and torques"
parameters["tags"] = "random sampling"

# dataset parameters
parameters["data_path"] = "/home/marjan/Documents/code-base/ml_datasets/pps_20_new"
parameters["batch_size"] = 2
parameters["shrink"] = False
parameters["overfit"] = False

# model parameters
# supported model types: "NN", "NNSkipShared", "NNGrow"
parameters["in_dim"] = 80
parameters["neighbor_hidden_dim"] = 64
parameters["particle_hidden_dim"] = 64
parameters["n_layer"] = 2
parameters["act_fn"] = "Tanh"
parameters["dropout"] = 0.3
parameters["batch_norm"] = False
parameters["neighbor_pool"] = "mean"
parameters["particle_pool"] = "sum1"
parameters["box_len"] = 16.17887


# optimizer parameters
parameters["optim"] = "Adam"
parameters["lr"] = 0.001
parameters["min_lr"] = 0.00001
parameters["use_scheduler"] = False
parameters["scheduler_type"] = "StepLR"
parameters["decay"] = 0.001
# supported loss types: "mse" (mean squared error), "mae" (means absolute error)
parameters["loss_type"] = "mse"
parameters["prior_energy"] = True
parameters["prior_energy_sigma"] = 1
parameters["prior_energy_n"] = 6

# run parameters
parameters["epochs"] = 50000

class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

config = Struct(**parameters)

job_id = 0

from aniso_MLMD.trainer import MLTrainer
trainer_obj = MLTrainer(config, job_id)
trainer_obj.run()

