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
    parameters["project"] = ["pps-pair-Jan4"]
    parameters["group"] = ["rotation_matrix"]
    parameters["notes"] = ["Learning pps forces and torques"]
    parameters["tags"] = ["random sampling"]

    # dataset parameters
    parameters["data_path"] =["/home/marjanalbooyeh/Aniso_ML_MD_project/ml_datasets/pps_pair"]
    parameters["batch_size"] = [16]
    parameters["shrink"] = [False]
    parameters["overfit"] = [False]

    # model parameters
    parameters["in_dim"] = [92]
    parameters["hidden_dim"] = [64, 128, 256]
    parameters["n_layer"] = [5]
    parameters["act_fn"] = ["Tanh"]
    parameters["dropout"] = [0.3]
    parameters["batch_norm"] = [False]

    # optimizer parameters
    parameters["optim"] = ["Adam"]
    parameters["lr"] = [0.001]
    parameters["min_lr"] = [0.00001]
    parameters["use_scheduler"] = [False]
    parameters["scheduler_type"] = ["StepLR"]
    parameters["decay"] = [0.001]
    # supported loss types: "mse" (mean squared error), "mae" (means absolute error)
    parameters["loss_type"] = ["mse"]
    parameters["prior_energy"] = [True]
    parameters["prior_energy_sigma"] = [1]
    parameters["prior_energy_n"] = [12]

    # run parameters
    parameters["epochs"] = [10000]

    return list(parameters.keys()), list(product(*parameters.values()))


custom_job_doc = {}  # add keys and values for each job document created


def main(root=None):
    project = signac.init_project("PPS-F", root=root)  # Set the signac project name
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
