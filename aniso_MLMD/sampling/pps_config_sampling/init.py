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
    parameters["kT"] = [0.1, 
                        0.5, 0.75, 1.0, 2.0, 4.0, 6.0, 7.0
                       ]
    parameters["n_steps"] = [5e6]
    parameters["n_tries"] = [5]
    parameters["density"] = [0.3]
    parameters["target_L"]= [15.76909943]
    parameters["ff_path"] = ["/home/marjanalbooyeh/Aniso_ML_MD_project/Aniso_MLMD_March2024/Aniso_MLMD/aniso_MLMD/sampling/pps_Rigid_sims_flow/pps_ff.pkl"]
    parameters["init_rigid_snap"]=["/home/marjanalbooyeh/Aniso_ML_MD_project/Aniso_MLMD_March2024/Aniso_MLMD/aniso_MLMD/sampling/pps_Rigid_sims_flow/rigid_init_300.gsd"]
    parameters["const_rel_pos"]=["/home/marjanalbooyeh/Aniso_ML_MD_project/Aniso_MLMD_March2024/Aniso_MLMD/aniso_MLMD/sampling/pps_Rigid_sims_flow/const_rel_pos.npy"]

    return list(parameters.keys()), list(product(*parameters.values()))


custom_job_doc = {}  # add keys and values for each job document created


def main(root=None):
    project = signac.init_project("Rigid", root=root)  # Set the signac project name
    param_names, param_combinations = get_parameters()
    # Create jobs
    for params in param_combinations:
        parent_statepoint = dict(zip(param_names, params))
        parent_job = project.open_job(parent_statepoint)
        parent_job.init()
        parent_job.doc.setdefault("done", False)
        parent_job.doc.setdefault("equli", False)
        parent_job.doc.setdefault("n_runs", 0)

    project.write_statepoints()
    return project


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
