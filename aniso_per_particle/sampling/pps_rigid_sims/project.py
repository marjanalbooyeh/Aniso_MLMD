"""Define the project's workflow logic and operation functions.
Execute this script directly from the command line, to view your project's
status, execute operations and submit them to a cluster. See also:
    $ python src/project.py --help
"""

from flow import FlowProject, directives
from flow.environment import DefaultSlurmEnvironment
import pickle 
import gsd.hoomd
import numpy as np
import hoomd
from cmeutils.sampling import is_equilibrated

class PPS_Rigid(FlowProject):
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
@PPS_Rigid.label
def finished(job):
    return job.doc.get("done")

@PPS_Rigid.label
def equilibrated(job):
    return job.doc["equli"]
    
def create_rigid_simulation(kT, init_snapshot, pps_ff, rel_const_pos, dt, tau, special_LJ=True, write_freq=1000):
    const_particle_types = ['ca', 'ca', 'ca', 'ca', 'sh', 'ca', 'ca']
    rigid_simulation = hoomd.Simulation(device=hoomd.device.auto_select(), seed=1)
    rigid_simulation.create_state_from_snapshot(init_snapshot)


    rigid = hoomd.md.constrain.Rigid()
    rigid.body['rigid'] = {
        "constituent_types":const_particle_types,
        "positions": rel_const_pos,
        "orientations": [(1.0, 0.0, 0.0, 0.0)]* len(rel_const_pos),
        }
    integrator = hoomd.md.Integrator(dt=dt, integrate_rotational_dof=True)
    rigid_simulation.operations.integrator = integrator
    integrator.rigid = rigid
    rigid_centers_and_free = hoomd.filter.Rigid(("center", "free"))
    nvt = hoomd.md.methods.ConstantVolume(
        filter=rigid_centers_and_free,
        thermostat=hoomd.md.methods.thermostats.MTTK(kT=kT, tau=tau))
    integrator.methods.append(nvt)
    
    cell = hoomd.md.nlist.Cell(buffer=0, exclusions=['body'])
    
    lj = hoomd.md.pair.LJ(nlist=cell)
    
    # use aa pps simulation to define lj and special lj forces between constituent particles
    for k, v in dict(pps_ff[0].params).items():
        lj.params[k] = v
        lj.r_cut[k] = 4.8
    
    lj.params[('rigid', ['rigid', 'ca', 'sh'])]= dict(epsilon=0, sigma=0)
    lj.r_cut[('rigid', ['rigid', 'ca', 'sh'])] = 0

    integrator.forces.append(lj)
    if special_LJ:
        special_lj = hoomd.md.special_pair.LJ()
        for k, v in dict(pps_ff[1].params).items():
            special_lj.params[k] = v
            special_lj.r_cut[k] = 4.8
        
        special_lj.params[('rigid', ['rigid', 'ca', 'sh'])]= dict(epsilon=0, sigma=0)
        special_lj.r_cut[('rigid', ['rigid', 'ca', 'sh'])] = 0

        integrator.forces.append(special_lj)
        
        
    rigid_simulation.state.thermalize_particle_momenta(filter=rigid_centers_and_free,
                                             kT=kT)
    
    rigid_simulation.run(0)

    
    log_quantities = [
                        "kinetic_temperature",
                        "potential_energy",
                        "kinetic_energy",
                        "volume",
                        "pressure",
                        "pressure_tensor",
                    ]
    logger = hoomd.logging.Logger(categories=["scalar", "string", "particle"])
    logger.add(rigid_simulation, quantities=["timestep", "tps"])
    thermo_props = hoomd.md.compute.ThermodynamicQuantities(filter=rigid_centers_and_free)
    rigid_simulation.operations.computes.append(thermo_props)
    logger.add(thermo_props, quantities=log_quantities)
    
    # for f in integrator.forces:
    #     logger.add(f, quantities=["energy", "forces", "energies"])

    logger.add(rigid_simulation.operations.integrator.rigid, quantities=["torques", "forces", "energies"])
    
    gsd_writer = hoomd.write.GSD(
        filename="trajectory.gsd",
        trigger=hoomd.trigger.Periodic(int(write_freq)),
        mode="wb",
        logger=logger,
        filter=hoomd.filter.All(),
        dynamic=["momentum", "property"]
        )
    
    rigid_simulation.operations.writers.append(gsd_writer)

    table_logger = hoomd.logging.Logger(categories=["scalar", "string"])
    table_logger.add(rigid_simulation, quantities=["timestep", "tps"])
    table_logger.add(thermo_props, quantities=log_quantities)
    table_file = hoomd.write.Table(
            output=open("log.txt", mode="w", newline="\n"),
            trigger=hoomd.trigger.Periodic(int(write_freq)),
            logger=table_logger,
            max_header_len=None,
        )
    rigid_simulation.operations.writers.append(table_file)
    return rigid_simulation



@directives(executable="python -u")
@directives(ngpu=1)
@PPS_Rigid.operation
@PPS_Rigid.post(finished)
def sample(job):
    with job:
        print("-----------------------")
        print("JOB ID NUMBER:")
        print(job.id)
        print("-----------------------")
        print("----------------------")
        print('kT: ', job.sp.kT)
        print('d: ', job.sp.init_rigid_snap)
        with open(job.sp.ff_path, "rb") as f:
            pps_ff = pickle.load(f)
        init_snapshot = gsd.hoomd.open(job.sp.init_rigid_snap)[-1]
        rel_const_pos = np.load(job.sp.const_rel_pos)

        rigid_sim = create_rigid_simulation(kT=job.sp.kT, init_snapshot=init_snapshot, pps_ff=pps_ff, rel_const_pos=rel_const_pos, 
                                    dt=0.0015, tau=0.1, special_LJ=False, write_freq=1000)
        for i in range(job.sp.n_tries):
            print('####################')
            print('Trying run: ', i)
            rigid_sim.run(job.sp.n_steps)
            for writer in rigid_sim.operations.writers:
                if hasattr(writer, 'flush'):
                    writer.flush()
            log = np.genfromtxt("log.txt", names=True)
            
            potential_energy = log['mdcomputeThermodynamicQuantitiespotential_energy'][-500:]
            pe_eq = is_equilibrated(potential_energy)[0]
            if pe_eq:
                print('equlibrated')
                job.doc["equli"] = True
                # break
        
        for writer in rigid_sim.operations.writers:
            if hasattr(writer, 'flush'):
                writer.flush()
        hoomd.write.GSD.write(rigid_sim.state, filename="restart.gsd")
        job.doc["done"] = True
        print("-----------------------------")
        print("Simulation finished")
        print("-----------------------------")

@directives(executable="python -u")
@directives(ngpu=1)
@PPS_Rigid.operation
@PPS_Rigid.pre(finished)
@PPS_Rigid.post(equilibrated)
def resume(job):
    with job:
        print("-----------------------")
        print("JOB ID NUMBER:")
        print(job.id)
        print("-----------------------")
        print("----------------------")
        print("Creating the Trainer class...")
        print("----------------------")
        
        with open(job.sp.ff_path, "rb") as f:
            pps_ff = pickle.load(f)
        last_snap = gsd.hoomd.open(job.fn("restart.gsd"))[-1]
        rel_const_pos = np.load(job.sp.const_rel_pos)
        rigid_sim = create_rigid_simulation(kT=job.sp.kT, init_snapshot=last_snap, pps_ff=pps_ff, rel_const_pos=rel_const_pos, 
                                    dt=0.0015, tau=0.1, special_LJ=False, write_freq=1000)
        for i in range(job.sp.n_tries):
            print('####################')
            print('Trying run: ', i)
            rigid_sim.run(job.sp.n_steps)
            for writer in rigid_sim.operations.writers:
                if hasattr(writer, 'flush'):
                    writer.flush()
            log = np.genfromtxt("log.txt", names=True)
            
            potential_energy = log['mdcomputeThermodynamicQuantitiespotential_energy'][-500:]
            pe_eq = is_equilibrated(potential_energy)[0]
            if pe_eq:
                print('equlibrated')
                job.doc["equli"] = True
                break
        for writer in rigid_sim.operations.writers:
            if hasattr(writer, 'flush'):
                writer.flush()
        hoomd.write.GSD.write(rigid_sim.state, filename="restart.gsd")
        job.doc["done"] = True
        print("-----------------------------")
        print("Training finished")
        print("-----------------------------")


def submit_project():
    PPS_Rigid(environment=Fry).run()


if __name__ == "__main__":
    PPS_Rigid(environment=Fry).main()
