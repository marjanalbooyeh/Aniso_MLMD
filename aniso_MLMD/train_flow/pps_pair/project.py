"""Define the project's workflow logic and operation functions.
Execute this script directly from the command line, to view your project's
status, execute operations and submit them to a cluster. See also:
    $ python src/project.py --help
"""

from flow import FlowProject, directives
from flow.environment import DefaultSlurmEnvironment
from aniso_MLMD.trainer import MLTrainer


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
        trainer_obj = MLTrainer(job.sp, job.id)
        job.doc["run_name"] = trainer_obj.wandb_run_name
        job.doc["run_path"] = trainer_obj.wandb_run_path
        print("Training...")
        print("----------------------")
        trainer_obj.run()

        job.doc["done"] = True
        print("-----------------------------")
        print("Training finished")
        print("-----------------------------")


@directives(executable="python -u")
@directives(ngpu=1)
@MyProject.operation
@MyProject.pre(sampled)
@MyProject.post(resumed)
def resume_job(job):
    with job:
        print("-----------------------")
        print("JOB ID NUMBER:")
        print(job.id)
        print("-----------------------")
        print("----------------------")
        print("Creating the Trainer class...")
        print("----------------------")
        trainer_obj = MLTrainer(job.sp, job.id, resume=True)
        job.doc["resume_run_name"] = trainer_obj.wandb_run_name
        job.doc["resume_run_path"] = trainer_obj.wandb_run_path
        print("Training...")
        print("----------------------")
        trainer_obj.run()

        job.doc["resumed"] = True
        print("-----------------------------")
        print("Training finished")
        print("-----------------------------")


def submit_project():
    MyProject().run()


if __name__ == "__main__":
    MyProject().main()
