import yaml
import logging
import sys
import abc
from clm_ler.utils.aws_utils import check_if_s3_dir_exists
import wandb
import os
from clm_ler.config.config import (
    WANDB_ENTITY,
    WANDB_PROJECT,
    AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY,
    AWS_DEFAULT_REGION,
)
import boto3
from wandb import util
from clm_ler.utils.logger import setup_logger

logger = setup_logger()


def get_spark():
    try:
        from pyspark.sql import SparkSession
    except Exception as e:
        raise ImportError("Couldn't import spark session. Can't return spark.")
    spark = SparkSession.builder.config(
        "spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.3.4"
    ).getOrCreate()

    # load some libraries to read from s3
    spark.conf.set("spark.hadoop.fs.s3a.endpoint", "s3.amazonaws.com")
    spark.conf.set("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
    if AWS_ACCESS_KEY_ID is not None and AWS_SECRET_ACCESS_KEY is not None:
        logger.info("Using AWS keys when creating spark.")
        spark.conf.set("spark.hadoop.fs.s3a.access.key", AWS_ACCESS_KEY_ID)
        spark.conf.set("spark.hadoop.fs.s3a.secret.key", AWS_SECRET_ACCESS_KEY)
    return spark


# the default s3 resource cannot use aws keys.
# Override the class method to use the correct keys and guarantee it works when submitting.
def init_boto(self) -> boto3.resources.base.ServiceResource:
    if self._s3 is not None:
        return self._s3
    boto: boto3 = util.get_module(
        "boto3",
        required="s3:// references requires the boto3 library, run pip install wandb[aws]",
        lazy=False,
    )
    self._s3 = boto.session.Session().resource(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_DEFAULT_REGION,  # specify the region here
    )
    self._botocore = util.get_module("botocore")
    return self._s3


# override wandb's implementation of the s3 client
wandb.sdk.artifacts.storage_handlers.s3_handler.S3Handler.init_boto = init_boto

spark = None


def _get_spark():
    from pyspark.sql import SparkSession

    global spark
    if spark is not None:
        return spark
    spark = SparkSession.builder.getOrCreate()
    # load some libraries to read from s3
    spark.conf.set("spark.hadoop.fs.s3a.endpoint", "s3.amazonaws.com")
    spark.conf.set("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
    return spark


def read_spark_table(table_name, partition_column=None):
    print(f"Reading {table_name}")
    if table_name.startswith("s3://") or table_name.startswith("s3a://"):
        return _get_spark().read.parquet(table_name.replace("s3://", "s3a://"))
    else:
        raise ValueError(f"Unsupported file type {table_name}")


def get_artifact(artifact_name, run=None):
    api = wandb.Api()
    if run is None:
        return api.artifact(artifact_name)
    else:
        return run.use_artifact(artifact_name)


class WandBJob(abc.ABC):
    def __init__(self):
        self.wandb_mode = "online"
        pass

    @abc.abstractmethod
    def get_parser(self):
        """
        The method to get the parser for the job.
        """
        pass

    @property
    @abc.abstractmethod
    def job_type(self):
        """
        A name for the type of job.
        """
        pass

    @property
    @abc.abstractmethod
    def project(self):
        """
        The project name of the job
        """
        pass

    @abc.abstractmethod
    def main(self, job_config):
        """
        The main method where the job runs.
        This takes the WandB config as an input.
        """
        pass

    @property
    @abc.abstractmethod
    def outputs(self):
        """
        This returns all outputs for this job.
        """
        pass

    @property
    @abc.abstractmethod
    def entity(self):
        """
        This returns the WandB entity name for this job.
        """
        pass

    def check_outputs(self):
        """
        Check that the outputs don't already exist on s3.
        This is to avoid overwriting them.
        """
        for output in self.outputs:
            logger.info(f"Checking {output}")
            if check_if_s3_dir_exists(output):
                raise ValueError(
                    f"This directory already exists: {output}. Please choose a different one "
                    "to avoid overwriting."
                )

    def resolve_s3_path(self, artifact_name_or_path, is_parquet_dataset):
        """
        Given an artifact ID, either a WandB artifact name or an s3 artifact uri,
        return it's location on s3.

        """
        api = None
        if self.run is None:
            logger.info(
                "No self.run was passed. Not logging artifacts, but using WandB API where "
                "needed to find artifacts."
            )
            api = wandb.Api()

        if "s3://" not in artifact_name_or_path:
            if self.run is not None:
                artifact = self.run.use_artifact(artifact_name_or_path)
            else:
                artifact = api.artifact(artifact_name_or_path)
            if is_parquet_dataset:
                s3_uri_of_artifact = get_uri_of_parquet_dataset(artifact)
            else:
                s3_uri_of_artifact = get_uri_of_file(artifact)
            return s3_uri_of_artifact
        else:
            return artifact_name_or_path

    @property
    def config(self):
        parser = self.get_parser()
        parser_args = vars(parser.parse_args())

        # store the parser arguments
        job_config = {}
        job_config["parser_args"] = parser_args

        # store and unpack all yaml config arguments
        for key in parser_args:
            if type(parser_args[key]) is str and parser_args[key].endswith("yaml"):
                assert key not in job_config
                job_config[key] = parse_yaml_args(parser_args[key])
        return job_config

    def set_mode(self):
        if "wandb_off" in self.config["parser_args"]:
            if self.config["parser_args"]["wandb_off"]:
                self.wandb_mode = "offline"
            else:
                self.wandb_mode = "online"

            return

        if "local_rank" in self.config["parser_args"]:
            if self.config["parser_args"]["local_rank"] <= 0:
                self.wandb_mode = "online"
            else:
                self.wandb_mode = "offline"
        else:
            self.wandb_mode = "online"

    def run_job(self):
        self.check_outputs()  # check that the job is safe to run.
        self.set_mode()  # set the job to online or offline if doing multi-node training
        if self.wandb_mode == "online":
            with wandb.init(
                project=self.project,
                job_type=self.job_type,
                config=self.config,
                entity=self.entity,
                mode=self.wandb_mode,
            ) as run:
                self.run = run
                self.main(run)
        else:
            self.run = None
            self.main()


class TrainEHRCLMJob(WandBJob):
    @abc.abstractmethod
    def get_parser(self):
        """
        The method to get the parser for the job.
        """
        pass

    @property
    @abc.abstractmethod
    def job_type(self):
        """
        A name for the type of job.
        """
        pass

    @property
    @abc.abstractmethod
    def outputs(self):
        """
        This returns all outputs for this job.
        """
        pass

    @abc.abstractmethod
    def main(self):
        pass

    @property
    def project(self):
        return WANDB_PROJECT

    @property
    def entity(self):
        return WANDB_ENTITY


def parse_yaml_args(config_file):
    """
    Given a yaml file, open it and parse it into nested dictionariers and lists.
    """
    logger.info(f"Using config: {config_file}")
    with open(config_file) as f:
        config = yaml.safe_load(f)
        logger.info(config)
    return config


def log_artifact(run, s3_location, name, dtype):
    artifact = wandb.Artifact(name, type=dtype)
    artifact.add_reference(s3_location)
    run.log_artifact(artifact)


def find_common_substring(uris):
    if len(uris) == 0:
        return ""

    common = uris[0]
    if len(uris) == 1:
        return common

    for uri in uris[1:]:
        common = common[: min(len(common), len(uri))]
        cutoff = len(common)
        for i, char in enumerate(common):
            if uri[i] == char:
                continue
            else:
                cutoff = i
                break
        common = common[:cutoff]

    return common


def get_uri_of_parquet_dataset(artifact):
    files = []
    for file in artifact.manifest.entries.values():
        if file.ref.startswith("s3://"):
            files.append(file.ref)
    s3_uri_of_dataset = find_common_substring(files)

    if not s3_uri_of_dataset.endswith("/"):
        s3_uri_of_dataset, _ = os.path.split(s3_uri_of_dataset)

    if (not "." in s3_uri_of_dataset) and (s3_uri_of_dataset[-1] != "/"):
        s3_uri_of_dataset += "/"
    return s3_uri_of_dataset


def get_uri_of_file(artifact):
    s3_uri_of_dataset = None
    files = []
    for file in artifact.manifest.entries.values():
        files.append(file.ref)
    assert len(files) == 1
    return files[0]


def find_upstream_artifacts(artifact_path, word_match):
    """
    Get the immediate upstream artifacts of the job that created the artifact.
    Only return those that contain the string word_match.
    Raise an error if none or more than one artifact is found.
    """
    result = get_immediate_upstream_artifacts(artifact_path, word_match=word_match)

    if len(result) == 0:
        raise ValueError(
            f"No artifacts found matching the search criteria: {word_match}"
        )
    elif len(result) > 1:
        logger.info(
            f"Found more than 1 artifact matching the search criteria, please refine the "
            "search. The first artifact found is returned."
        )

    return result[0]


def get_immediate_upstream_artifacts(
    artifact_name, word_match=None, used_artifacts=True
):
    # Initialize W&B API
    api = wandb.Api()

    # Fetch the artifact
    artifact = api.artifact(artifact_name)

    # Get the run that created this artifact
    creating_run = artifact.logged_by()

    # Get the used artifacts (upstream artifacts)
    if used_artifacts:
        upstream_artifacts = creating_run.used_artifacts()
    else:
        upstream_artifacts = creating_run.logged_artifacts()

    result = []
    # Print the upstream artifacts
    if used_artifacts:
        string = f"Upstream artifacts of {artifact_name}"
    else:
        string = f"Co-created artifacts of {artifact_name}"
    if word_match is not None:
        string += f" containing {word_match}"
    logger.info(string)
    for upstream_artifact in upstream_artifacts:
        if word_match is None or word_match in upstream_artifact.name:
            if artifact_name.endswith(upstream_artifact.name):
                continue

            project_name = creating_run.project
            artifact_full_name = f"{project_name}/{upstream_artifact.name}"
            logger.info(f"- {artifact_full_name} ({upstream_artifact.type})")
            result.append(artifact_full_name)

    return result


def _get_recursive_upstream_artifacts(artifact_name, word_match):
    """
    Recursively search the artifact create tree and return
    all artifacts with names containing the string word_match.
    """
    upstream_artifacts = get_immediate_upstream_artifacts(
        artifact_name, word_match=None
    )
    co_created_artifacts = get_immediate_upstream_artifacts(
        artifact_name, word_match=None, used_artifacts=False
    )
    found_artifacts = []
    for up_artifact_name in upstream_artifacts:
        if word_match in up_artifact_name:
            logger.info(
                f"Found upstream artifact for {artifact_name} matching word {word_match}"
            )
            logger.info(f"- {up_artifact_name}")
            found_artifacts.append(up_artifact_name)
        found_artifacts = found_artifacts + _get_recursive_upstream_artifacts(
            up_artifact_name, word_match=word_match
        )

    for co_artifact_name in co_created_artifacts:
        if word_match in co_artifact_name:
            logger.info(
                f"Found co-created artifact for {artifact_name} matching word {word_match}"
            )
            logger.info(f"- {co_artifact_name}")
            found_artifacts.append(co_artifact_name)

    return list(set(found_artifacts))


def get_recursive_upstream_artifact(artifact_name, word_match):
    artifacts = _get_recursive_upstream_artifacts(artifact_name, word_match)

    if len(artifacts) == 0:
        raise ValueError(
            f"No artifacts found matching the search criteria: {word_match}"
        )
    elif len(artifacts) > 1:
        logger.info(
            f"Found more than 1 artifact matching the search criteria, please refine the "
            "search. The first artifact found is returned."
        )

    return artifacts[0]


def ensure_s3a_path(path):
    return path.replace("s3://", "s3a://")
