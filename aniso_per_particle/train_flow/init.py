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
    parameters["project"] = ["huang-Apr29"]
    parameters["group"] = ["energy-0"]
    parameters["log"] = [False]
    parameters["resume"] = [False]

    # dataset parameters700
    parameters["data_path"] =["/home/marjanalbooyeh/Aniso_ML_MD_project/ml_datasets/?"]
    parameters["batch_size"] = [16]
    parameters["shrink"] = [False]
    parameters["overfit"] = [True]
    parameters["train_idx"] = [0]


    # model parameters
    parameters["in_dim"] = [15]
    parameters["out_dim"] = [3]

    parameters["prior_hidden_dim"] = [3]
    parameters["prior_n_layers"] = [2]
    parameters["prior_act_fn"] = ["Tanh"]

    parameters["energy_hidden_dim"] = [5]
    parameters["energy_n_layers"] = [3]
    parameters["energy_act_fn"] = ["Tanh"]


    parameters["dropout"] = [0.]
    parameters["batch_norm"] = [False]

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
    parameters["epochs"] = [1000]

    return list(parameters.keys()), list(product(*parameters.values()))


custom_job_doc = {}  # add keys and values for each job document created


def main(root=None):
    project = signac.init_project("Huang", root=root)  # Set the signac project name
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
