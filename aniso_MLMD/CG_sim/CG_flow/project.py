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


def create_rigid_simulation(kT, initial_rigid_snap, const_particle_types,rel_const_pos, pps_ff, dt, log_freq):
    import hoomd
    rigid_simulation = hoomd.Simulation(device=hoomd.device.auto_select(), seed=1)
    rigid_simulation.create_state_from_snapshot(initial_rigid_snap)

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
        thermostat=hoomd.md.methods.thermostats.MTTK(kT=kT, tau=0.1))
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
    thermo_props = hoomd.md.compute.ThermodynamicQuantities(filter=hoomd.filter.All())
    rigid_simulation.operations.computes.append(thermo_props)
    logger.add(thermo_props, quantities=log_quantities)
    
    # for f in integrator.forces:
    #     logger.add(f, quantities=["energy", "forces", "energies"])

    logger.add(rigid_simulation.operations.integrator.rigid, quantities=["torques", "forces", "energies"])
    
    gsd_writer = hoomd.write.GSD(
        filename="rigid_trajectory.gsd",
        trigger=hoomd.trigger.Periodic(int(log_freq)),
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
            output=open("rigid_log.txt", mode="w", newline="\n"),
            trigger=hoomd.trigger.Periodic(int(log_freq)),
            logger=table_logger,
            max_header_len=None,
        )
    rigid_simulation.operations.writers.append(table_file)
    return rigid_simulation
    
@directives(executable="python -u")
@directives(ngpu=1)
@MyProject.operation
@MyProject.post(sampled)
def run(job):
    from aniso_MLMD.CG_sim import AnisoNeighborCustomForce
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
        # force project
        force_project = signac.get_project(job.sp.force_project_path)
        best_force_job_id = job.sp.force_job_id
        best_force_job = force_project.open_job(id=best_force_job_id)
        
        # torque project
        torque_project = signac.get_project(job.sp.torque_project_path)
        best_torque_job_id = job.sp.torque_job_id
        best_torque_job = torque_project.open_job(id=best_torque_job_id)

        custom_force = AnisoNeighborCustomForce(best_force_job=best_force_job, best_torque_job=best_torque_job)
        initial_cg_snapshot = gsd.hoomd.open(job.sp.cg_snapshot)[0]
        sim = Simulation(initial_cg_snapshot, [custom_force], 
                         dt=job.sp.cg_dt,
                         gsd_write_freq=job.sp.cg_log_freq,
                         gsd_file_name='ML_trajectory.gsd',
                         log_write_freq=job.sp.cg_log_freq,
                         log_file_name='ML_log.txt',
                        integrate_rotational_dof=True)
        print("ML Simulation...")
        print("----------------------")
        sim.run_NVT(n_steps=job.sp.cg_n_steps, kT=job.sp.kT, tau_kt=0.1, write_at_start=True)
        sim.flush_writers()


        # Rigid Body Simulation
        import pickle
        with open(job.sp.pps_ff_path, "rb") as f:
            pps_ff = pickle.load(f)
        const_particle_types = ['ca', 'ca', 'ca', 'ca', 'sh', 'ca', 'ca']
        rel_const_pos = np.load(job.sp.rel_const_pos_path)
        initial_rigid_snap = gsd.hoomd.open(job.sp.rigid_snapshot)[0]
        rigid_simulation = create_rigid_simulation(job.sp.kT, initial_rigid_snap, const_particle_types,rel_const_pos, pps_ff, job.sp.dt, job.sp.log_freq)
        print("Rigid Simulation...")
        print("----------------------")
        rigid_simulation.run(job.sp.n_steps, write_at_start=True)
        rigid_simulation.operations.writers[0].flush()
        


        ## UA simulation
        initial_ua_snap=gsd.hoomd.open(job.sp.ua_snapshot)[0]
        ua_sim = Simulation(initial_ua_snap, pps_ff, 
                        dt=job.sp.dt,
                         gsd_write_freq=job.sp.log_freq,
                         gsd_file_name='UA_trajectory.gsd',
                         log_write_freq=job.sp.log_freq,
                         log_file_name='UA_log.txt')
        print("UA Simulation...")
        print("----------------------")
        ua_sim.run_NVT(n_steps=job.sp.n_steps, kT=job.sp.kT, tau_kt=0.1, write_at_start=True)
        ua_sim.flush_writers()
                            
        job.doc["done"] = True
        print("-----------------------------")
        print("Simulation finished")
        print("-----------------------------")


def submit_project():
    MyProject(environment=Fry).run()


if __name__ == "__main__":
    MyProject(environment=Fry).main()
