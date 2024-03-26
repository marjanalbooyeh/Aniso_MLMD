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

    # run parameters
    parameters["kT"] = [1.0, 3.0, 4.0, 5.0, 6.0, 7.0, 9.0]
    
    parameters["force_project_path"] = ["/home/marjanalbooyeh/Aniso_ML_MD_project/Aniso_MLMD_March2024/Aniso_MLMD/aniso_MLMD/train_flow/force_pred_Mar21"]
    parameters["force_job_id"]=["e41c33fd820854b8c720acb001e7ddaf"]
    parameters["torque_project_path"]=["/home/marjanalbooyeh/Aniso_ML_MD_project/Aniso_MLMD_March2024/Aniso_MLMD/aniso_MLMD/train_flow/torque_pred_Mar18"]
    parameters["torque_job_id"]=["27fa8f00a6efdd097b0e709cc62eb580"]
    
    parameters["cg_snapshot"]=["/home/marjanalbooyeh/Aniso_ML_MD_project/Aniso_MLMD_March2024/Aniso_MLMD/aniso_MLMD/CG_sim/CG_flow/assets/init_cg_4.0.gsd"]
    parameters["cg_dt"]=[0.0001]
    parameters["cg_log_freq"]=[1000]
    parameters["cg_n_steps"]=[3e6]

    
    parameters["pps_ff_path"]=["/home/marjanalbooyeh/Aniso_ML_MD_project/Aniso_MLMD_March2024/Aniso_MLMD/aniso_MLMD/CG_sim/CG_flow/assets/pps_ff.pkl"]
    parameters["rel_const_pos_path"]=["/home/marjanalbooyeh/Aniso_ML_MD_project/Aniso_MLMD_March2024/Aniso_MLMD/aniso_MLMD/CG_sim/CG_flow/assets/rel_const_pos.npy"]
    parameters["rigid_snapshot"]=["/home/marjanalbooyeh/Aniso_ML_MD_project/Aniso_MLMD_March2024/Aniso_MLMD/aniso_MLMD/CG_sim/CG_flow/assets/init_rigid_4.0.gsd"]
    parameters["dt"]=[0.0001]
    parameters["log_freq"]=[1000]
    parameters["n_steps"]=[5e6]


    parameters["ua_snapshot"]=["/home/marjanalbooyeh/Aniso_ML_MD_project/Aniso_MLMD_March2024/Aniso_MLMD/aniso_MLMD/CG_sim/CG_flow/assets/init_UA_4.0.gsd"]
    return list(parameters.keys()), list(product(*parameters.values()))


custom_job_doc = {}  # add keys and values for each job document created


def main(root=None):
    project = signac.init_project("ML_LJ", root=root)  # Set the signac project name
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
