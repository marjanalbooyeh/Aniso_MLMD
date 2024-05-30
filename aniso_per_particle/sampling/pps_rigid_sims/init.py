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
    parameters["kT"] = [1.4
                       ]
    parameters["n_steps"] = [6e6]
    parameters["n_tries"] = [2]
    parameters["ff_path"] = ["/home/marjanalbooyeh/Aniso_ML_MD_project/may_23_Aniso/Aniso_MLMD/aniso_per_particle/sampling/pps_rigid_sims/assets/pps_ff.pkl"]
    parameters["init_rigid_snap"]=["/home/marjanalbooyeh/Aniso_ML_MD_project/may_23_Aniso/Aniso_MLMD/aniso_per_particle/sampling/pps_rigid_sims/assets/rigid_after_shrink_800_d_0.95.gsd"]
    parameters["const_rel_pos"]=["/home/marjanalbooyeh/Aniso_ML_MD_project/may_23_Aniso/Aniso_MLMD/aniso_per_particle/sampling/pps_rigid_sims/assets/const_rel_pos.npy"]

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

    project.write_statepoints()
    return project


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
