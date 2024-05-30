#!/usr/bin/env python
"""Initialize the project's data space.
Iterates over all defined state points and initializes
the associated job workspace directories.
The result of running this file is the creation of a signac workspace:
    - signac.rc file containing the project name
    - signac_statepoints.json summary for the entire workspace
    - workspace/ directory that contains a sub-directoy of every individual statepoint
    - signac_statepoints.json within each individual statepoint sub-directory.
"""

import logging
from collections import OrderedDict
from itertools import product

import signac


def get_parameters():
    parameters = OrderedDict()

    # project parameters
    parameters["project"] = ["ForTor-no-repulsion-May31"]
    parameters["group"] = ["fortor-0"]
    parameters["log"] = [False]
    parameters["resume"] = [False]

    # dataset parameters700
    parameters["data_path"] =["/home/marjanalbooyeh/Aniso_ML_MD_project/ml_datasets/pps_300_N15_Apr25/"]
    parameters["batch_size"] = [8]
    parameters["shrink"] = [True]
    parameters["shrink_factor"] = [0.1]
    parameters["overfit"] = False
    parameters["train_idx"] = 0

    # model parameters
    parameters["in_dim"] = [18]
    parameters["out_dim"] = [3]

    parameters["force_hidden_dim"] = [[32, 5, 5]]
    parameters["force_n_layers"] = [2]
    parameters["force_act_fn"] = ["LeakyReLU"]
    parameters["force_pre_factor"] = 1.0

    parameters["torque_hidden_dim"] = [[32, 128, 5]]
    parameters["torque_n_layers"] = [2]
    parameters["torque_act_fn"] = ["LeakyReLU"]
    parameters["torque_pre_factor"] = 1.0

    parameters["lpar"] = [2.176]
    parameters["lperp"] = [1.54]
    parameters["sigma"] = [-1.13]
    parameters["n_factor"] = [12]

    parameters["dropout"] = [0.5]
    parameters["batch_norm"] = [False]
    parameters["initial_weight"] = [None]

    # optimizer parameters
    parameters["optim"] = ["Adam"]
    parameters["lr"] = [0.00015]
    parameters["min_lr"] = [0.0001]
    parameters["use_scheduler"] = [False]
    parameters["scheduler_type"] = ["ReduceLROnPlateau"]
    parameters["scheduler_patience"] = [50]
    parameters["scheduler_threshold"] = [20]
    parameters["decay"] = [0.0001]
    # supported loss types: "mse" (mean squared error), "mae" (means absolute error)
    parameters["loss_type"] = ["mse"]
    parameters["clipping"] = [False]

    # run parameters
    parameters["epochs"] = [10000]

    return list(parameters.keys()), list(product(*parameters.values()))


custom_job_doc = {}  # add keys and values for each job document created


def main(root=None):
    project = signac.init_project("ForTor", root=root)  # Set the signac project name
    param_names, param_combinations = get_parameters()
    # Create jobs
    for params in param_combinations:
        parent_statepoint = dict(zip(param_names, params))
        parent_job = project.open_job(parent_statepoint)
        parent_job.init()
        parent_job.doc.setdefault("done", False)

    project.write_statepoints()
    return project


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
