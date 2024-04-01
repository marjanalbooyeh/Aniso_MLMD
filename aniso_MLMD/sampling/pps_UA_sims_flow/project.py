"""Define the project's workflow logic and operation functions.
Execute this script directly from the command line, to view your project's
status, execute operations and submit them to a cluster. See also:
    $ python src/project.py --help
"""

from flow import FlowProject, directives
from flow.environment import DefaultSlurmEnvironment


class PPS_UA(FlowProject):
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


class R2(DefaultSlurmEnvironment):
    hostname_pattern = "r2"
    template = "r2.sh"

    @classmethod
    def add_args(cls, parser):
        parser.add_argument(
            "--partition",
            default="shortq",
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
@PPS_UA.label
def finished(job):
    return job.doc.get("done")

@PPS_UA.label
def equilibrated(job):
    return job.doc["equli"]


@directives(executable="python -u")
@directives(ngpu=1)
@PPS_UA.operation
@PPS_UA.post(finished)
def sample(job):
    import pickle 
    from flowermd.base import Simulation
    import numpy as np
    from cmeutils.sampling import is_equilibrated
    with job:
        print("-----------------------")
        print("JOB ID NUMBER:")
        print(job.id)
        print("-----------------------")
        print("----------------------")
        print('kT: ', job.sp.kT)
        with open(job.sp.ff_path, "rb") as f:
            ff = pickle.load(f)
        sim = Simulation(initial_state=job.sp.init_snapshot_path, forcefield=ff, gsd_write_freq=1e4)
        target_box = [job.sp.target_L, job.sp.target_L, job.sp.target_L]
        sim.run_update_volume(n_steps=1e5, period=100, kT=job.sp.kT, tau_kt=0.1, final_box_lengths=target_box)
        for i in range(job.sp.n_tries):
            print('####################')
            print('Trying run: ', i)
            sim.run_NVT(n_steps=job.sp.n_steps, kT=job.sp.kT, tau_kt=0.1)
            sim.flush_writers()
            for writer in sim.operations.writers:
                if hasattr(writer, 'flush'):
                    writer.flush()
            log = np.genfromtxt("sim_data.txt", names=True)
            
            potential_energy = log['mdcomputeThermodynamicQuantitiespotential_energy'][-300:]
            pe_eq = is_equilibrated(potential_energy)[0]
            if pe_eq:
                print('equlibrated')
                job.doc["equli"] = True
                break
        sim.flush_writers()
        job.doc["done"] = True
        print("-----------------------------")
        print("Simulation finished")
        print("-----------------------------")

@directives(executable="python -u")
@directives(ngpu=1)
@PPS_UA.operation
@PPS_UA.pre(finished)
@PPS_UA.post(equilibrated)
def resume(job):
    import pickle 
    from flowermd.base import Simulation
    import numpy as np
    from cmeutils.sampling import is_equilibrated
    import gsd.hoomd
    with job:
        print("-----------------------")
        print("JOB ID NUMBER:")
        print(job.id)
        print("-----------------------")
        print("----------------------")
        print("Creating the Trainer class...")
        print("----------------------")
        
        with open(job.sp.ff_path, "rb") as f:
            ff = pickle.load(f)
        last_snap = gsd.hoomd.open(job.fn("trajectory.gsd"))[-1]
        sim = Simulation(initial_state=last_snap, forcefield=ff, gsd_write_freq=1e4)
        for i in range(job.sp.n_tries):
            print('####################')
            print('Trying run: ', i)
            sim.run_NVT(n_steps=job.sp.n_steps, kT=job.sp.kT, tau_kt=0.1)
            sim.flush_writers()
            for writer in sim.operations.writers:
                if hasattr(writer, 'flush'):
                    writer.flush()
            log = np.genfromtxt("sim_data.txt", names=True)
            
            potential_energy = log['mdcomputeThermodynamicQuantitiespotential_energy'][-300:]
            pe_eq = is_equilibrated(potential_energy)[0]
            if pe_eq:
                print('equlibrated')
                job.doc["equli"] = True
                break
        sim.flush_writers()
        job.doc["done"] = True
        print("-----------------------------")
        print("Training finished")
        print("-----------------------------")


def submit_project():
    PPS_UA(environment=Fry).run()


if __name__ == "__main__":
    PPS_UA(environment=Fry).main()
