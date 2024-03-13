"""Define the project's workflow logic and operation functions.
Execute this script directly from the command line, to view your project's
status, execute operations and submit them to a cluster. See also:
    $ python src/project.py --help
"""

from flow import FlowProject, directives
from flow.environment import DefaultSlurmEnvironment


class MyProject(FlowProject):
    pass

class Borah(DefaultSlurmEnvironment):
    hostname_pattern = "borah"
    template = "borah.sh"

    @classmethod
    def add_args(cls, parser):
        parser.add_argument(
            "--partition",
            default="short",
            help="Specify the partition to submit to."
        )


class Fry(DefaultSlurmEnvironment):
    hostname_pattern = "fry"
    template = "fry.sh"

    @classmethod
    def add_args(cls, parser):
        parser.add_argument(
            "--partition",
            default="batch",
            help="Specify the partition to submit to."
        )
        parser.add_argument(
            "--cpus",
            default=6,
            help="Specify cpu-cores per task."
        )


# Definition of project-related labels (classification)
@MyProject.label
def sampled(job):
    return job.doc.get("done")


@MyProject.label
def resumed(job):
    return job.doc.get("resumed")


def pair_force(sigma=1, epsilon=1):
    import hoomd
    """
    Creates non-bonded forces between A particles.
    """
    cell = hoomd.md.nlist.Cell(buffer=0.4)
    lj = hoomd.md.pair.LJ(nlist=cell)
    lj.params[('A', 'A')] = dict(epsilon=epsilon, sigma=sigma)
    lj.r_cut[('A', 'A')] = 2.5 * sigma
    return lj


@directives(executable="python -u")
@directives(ngpu=1)
@MyProject.operation
@MyProject.post(sampled)
def run(job):
    from iso_MLMD.CG_sim import IsoNeighborCustomForce
    from flowermd.base import Simulation
    import gsd.hoomd
    import signac

    import numpy as np
    with job:
        print("-----------------------")
        print("JOB ID NUMBER:")
        print(job.id)
        print("-----------------------")
        print("----------------------")
        print("Creating the model...")
        print("----------------------")

        # load model from best job
        project = signac.get_project(
            "/home/marjanalbooyeh/Aniso_ML_MD_project/Aniso_MLMD/iso_MLMD/train_flow/iso_neighbor_Feb28/")
        best_job_id = "0245343d690808044617b9b0a4c30c83"
        best_job = project.open_job(id=best_job_id)
        init_snap = \
            gsd.hoomd.open(
                '/home/marjanalbooyeh/Aniso_ML_MD_project/Aniso_MLMD/iso_MLMD/sampling/logs/N_200/restart.gsd')[
                0]

        custom_force = IsoNeighborCustomForce(model_path=best_job.fn("best_model_checkpoint.pth"),
                                              in_dim=best_job.sp.in_dim,
                                              hidden_dim=best_job.sp.hidden_dim,
                                              n_layers=best_job.sp.n_layer,
                                              box_len=init_snap.configuration.box[0],
                                              act_fn=best_job.sp.act_fn,
                                              dropout=best_job.sp.dropout,
                                              prior_energy=best_job.sp.prior_energy,
                                              prior_energy_sigma=best_job.sp.prior_energy_sigma,
                                              prior_energy_n=best_job.sp.prior_energy_n,
                                              )

        sim = Simulation(init_snap, [custom_force], dt=0.0001,
                         gsd_write_freq=1000,
                         gsd_file_name='ML_trajectory.gsd',
                         log_write_freq=1000,
                         log_file_name='ML_log.txt')
        print("ML Simulation...")
        print("----------------------")
        sim.run_NVT(n_steps=job.sp.n_steps, kT=job.sp.kT, tau_kt=0.1, write_at_start=True)
        sim.flush_writers()

        energy_log = np.array(custom_force.model.energy_log)
        np.save('energy_log.npy', energy_log)

        forces = [pair_force(sigma=1, epsilon=1)]
        lj_sim = Simulation(initial_state=init_snap,
                            forcefield=forces,
                            dt=0.0001,
                            gsd_write_freq=1000,
                            gsd_file_name='LJ_trajectory.gsd',
                            log_write_freq=1000,
                            log_file_name='LJ_log.txt')
        print("LJ Simulation...")
        print("----------------------")
        lj_sim.run_NVT(n_steps=job.sp.n_steps, kT=job.sp.kT, tau_kt=0.1, write_at_start=True)
        lj_sim.flush_writers()

        job.doc["done"] = True
        print("-----------------------------")
        print("Simulation finished")
        print("-----------------------------")


def submit_project():
    MyProject(environment=Fry).run()


if __name__ == "__main__":
    MyProject(environment=Fry).main()
