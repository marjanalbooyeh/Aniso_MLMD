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
    
def create_rigid_simulation(kT, init_snapshot, pps_ff, rel_const_pos, dt, tau, special_LJ=True, write_freq=1000, gsd_file_name="trajectory.gsd", log_file_name="log.txt"):
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
        lj.r_cut[k] = 2.5
    
    lj.params[('rigid', ['rigid', 'ca', 'sh'])]= dict(epsilon=0, sigma=0)
    lj.r_cut[('rigid', ['rigid', 'ca', 'sh'])] = 0

    integrator.forces.append(lj)
    if special_LJ:
        special_lj = hoomd.md.special_pair.LJ()
        for k, v in dict(pps_ff[1].params).items():
            special_lj.params[k] = v
            special_lj.r_cut[k] = 2.5
        
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
        filename=gsd_file_name,
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

    
def shrink_volume(simulation, target_box, n_steps):
    # box resizer
    final_box = hoomd.Box(
            Lx=target_box,
            Ly=target_box,
            Lz=target_box,
        )
    resize_trigger = hoomd.trigger.Periodic(100)
    box_ramp = hoomd.variant.Ramp(
        A=0, B=1, t_start=simulation.timestep, t_ramp=int(n_steps)
    )
    initial_box = simulation.state.box

    box_resizer = hoomd.update.BoxResize(
        box1=initial_box,
        box2=final_box,
        variant=box_ramp,
        trigger=resize_trigger,
    )
    simulation.operations.updaters.append(box_resizer)
    simulation.run(n_steps + 1, write_at_start=True)
    simulation.operations.updaters.remove(box_resizer)

    
    


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
        with open(job.sp.ff_path, "rb") as f:
            pps_ff = pickle.load(f)
        init_snapshot = gsd.hoomd.open(job.sp.init_rigid_snap)[-1]
        rel_const_pos = np.load(job.sp.const_rel_pos)

        rigid_sim = create_rigid_simulation(kT=job.sp.kT, init_snapshot=init_snapshot, pps_ff=pps_ff, rel_const_pos=rel_const_pos, 
                                    dt=0.0001, tau=0.1, special_LJ=False, write_freq=1e3)
        print('shrinking....')
        shrink_volume(rigid_sim, target_box=job.sp.target_L, n_steps=1e5)
        for i in range(job.sp.n_tries):
            print('####################')
            print('Trying run: ', i)
            rigid_sim.run(job.sp.n_steps)
            for writer in rigid_sim.operations.writers:
                if hasattr(writer, 'flush'):
                    writer.flush()
            log = np.genfromtxt(log_file_name, names=True)
            
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
        job.doc.n_runs += 1 
        with open(job.sp.ff_path, "rb") as f:
            pps_ff = pickle.load(f)
        last_snap = gsd.hoomd.open(job.fn("restart.gsd"))[-1]
        rel_const_pos = np.load(job.sp.const_rel_pos)
        rigid_sim = create_rigid_simulation(kT=job.sp.kT, init_snapshot=last_snap, pps_ff=pps_ff, rel_const_pos=rel_const_pos, 
                                    dt=0.0001, tau=0.1, special_LJ=False, write_freq=1e3, gsd_file_name=f"trajectory_{job.doc.n_runs}.gsd",
                                           log_file_name=f"log_{job.doc.n_runs}.txt")
        for i in range(job.sp.n_tries):
            print('####################')
            print('Trying run: ', i)
            rigid_sim.run(job.sp.n_steps)
            for writer in rigid_sim.operations.writers:
                if hasattr(writer, 'flush'):
                    writer.flush()
            log = np.genfromtxt(f"log_{job.doc.n_runs}.txt", names=True)
            
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
