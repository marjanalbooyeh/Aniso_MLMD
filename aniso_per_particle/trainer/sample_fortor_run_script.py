from collections import OrderedDict
parameters = OrderedDict()

# project parameters
parameters["project"] = "pps-energy-Apr22"
parameters["group"] = "energy-V2"
parameters["log"] = False
parameters["resume"] = False

# dataset parameters700
parameters["data_path"] ="/home/marjan/Documents/fry/ml_dataset/pps_N800"
parameters["batch_size"] = 8
parameters["shrink"] = True
parameters["overfit"] = False
parameters["train_idx"] = 0



# model parameters
parameters["in_dim"] = 20
parameters["out_dim"] = 3


parameters["force_hidden_dim"] = [5, 32, 64, 64, 32, 5, 3]
parameters["force_n_layers"] = 6
parameters["force_act_fn"] = "Tanh"
parameters["force_pre_factor"] = 1.0

parameters["torque_hidden_dim"] = [5, 32, 64, 64, 32, 5, 3]
parameters["torque_n_layers"] = 6
parameters["torque_act_fn"] = "Tanh"
parameters["torque_pre_factor"] = 1.0

parameters["lpar"] = 2.176
parameters["lperp"] = 1.54
parameters["sigma"] = -1.13
parameters["n_factor"] = 12

parameters["dropout"] = 0.5
parameters["batch_norm"] = False
parameters["initial_weight"] = None
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

from aniso_per_particle.trainer import ForTorTrainer
trainer_obj = ForTorTrainer(config, job_id)
trainer_obj.run()

