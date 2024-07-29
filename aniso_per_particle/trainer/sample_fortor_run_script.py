from collections import OrderedDict
parameters = OrderedDict()

# project parameters
parameters["project"] = "overfit-June4"
parameters["group"] = "energy-V2"
parameters["log"] = False
parameters["resume"] = False

# dataset parameters700
parameters["data_path"] ="/home/marjan/Documents/code-base/ml_datasets/PPS_800_June10"
parameters["batch_size"] = 8
parameters["shrink"] = True
parameters["overfit"] = True
parameters["shrink_factor"] = 0.01
parameters["train_idx"] = "10k"



# model parameters
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
# optimizer parameters
parameters["optim"] = "Adam"
parameters["lr"] = 0.001
parameters["min_lr"] = 0.0001
parameters["use_scheduler"] = False
parameters["scheduler_type"] = "ReduceLROnPlateau"
parameters["scheduler_patience"] = 50
parameters["scheduler_threshold"] = 20
parameters["decay"] = 0.0001
# supported loss types: "mse" (mean squared error), "mae" (means absolute error)
parameters["loss_type"] = "mse"
parameters["clipping"] = False

# run parameters
parameters["epochs"] = 50000

class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

config = Struct(**parameters)

job_id = 0

from aniso_per_particle.trainer import ForTorTrainer
trainer_obj = ForTorTrainer(config, job_id)
trainer_obj.run()

