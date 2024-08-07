"""Define the project's workflow logic and operation functions.
Execute this script directly from the command line, to view your project's
status, execute operations and submit them to a cluster. See also:
    $ python src/project.py --help
"""

from flow import FlowProject, directives
from flow.environment import DefaultSlurmEnvironment
from aniso_per_particle.trainer import EnergyTrainer_V3


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


@directives(executable="python -u")
@directives(ngpu=1)
@MyProject.operation
@MyProject.post(sampled)
def sample(job):
    with job:
        print("-----------------------")
        print("JOB ID NUMBER:")
        print(job.id)
        print("-----------------------")
        print("----------------------")
        print("Creating the Trainer class...")
        print("----------------------")
        trainer_obj = EnergyTrainer_V3(job.sp, job.id)
        if job.sp.log:
            job.doc["run_name"] = trainer_obj.wandb_run_name
            job.doc["run_path"] = trainer_obj.wandb_run_path
        print("Training...")
        print("----------------------")
        trainer_obj.run()

        job.doc["done"] = True
        print("-----------------------------")
        print("Training finished")
        print("-----------------------------")



def submit_project():
    MyProject(environment=Fry).run()


if __name__ == "__main__":
    MyProject(environment=Fry).main()
