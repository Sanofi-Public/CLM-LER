import argparse
from clm_ler.utils.utils import parse_yaml_args, setup_logger
import pyspark
import pyspark.sql.functions as f
import logging
import sys
from pyspark.sql import SparkSession
import os
from clm_ler.utils.aws_utils import check_if_s3_dir_exists
from clm_ler.utils.utils import (
    TrainEHRCLMJob,
    log_artifact,
    get_uri_of_parquet_dataset,
    setup_logger,
)
import wandb
from pyspark.sql.functions import expr

logger = setup_logger()


# given a preprocessed dataset and a set of labels, join the labels to the dataset
class JoinLabels(TrainEHRCLMJob):
    """
    Generates & joins the labels to the dataset for finetuning tasks.
    """

    def get_parser(self):
        parser = argparse.ArgumentParser(description="Join the labels to the dataset.")
        parser.add_argument(
            "--input",
            dest="input",
            help="The artifact name of the preprocessed dataset",
            required=True,
            type=str,
        )
        parser.add_argument(
            "--labels",
            dest="labels",
            help="The s3 location of the labels dataset",
            required=True,
            type=str,
        )
        parser.add_argument(
            "--labels_config",
            dest="labels_config",
            help="Config for which rows to keep and the criteria for the positive/negative class",
            required=True,
            type=str,
        )
        parser.add_argument(
            "--output",
            dest="output",
            help="The location on s3 for the joined dataset",
            required=True,
            type=str,
        )
        parser.add_argument(
            "--output_dataset_name",
            dest="output_dataset_name",
            help="The name of the identifier for the dataset.",
            required=True,
            type=str,
        )
        return parser

    def main(self, run):
        args = run.config["parser_args"]
        yaml_args = run.config["labels_config"]

        # start spark session
        spark = SparkSession.builder.appName("JoinLabels").getOrCreate()
        spark.conf.set("spark.hadoop.fs.s3a.endpoint", "s3.amazonaws.com")
        spark.conf.set(
            "spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem"
        )

        # get the input dataset
        input_dataset_artifact = run.use_artifact(f"{args['input']}:latest")
        input_dataset = get_uri_of_parquet_dataset(input_dataset_artifact)
        input_df = spark.read.parquet(input_dataset.replace("s3://", "s3a://"))

        # get the labels from the s3 path
        labels_df = spark.read.parquet(args["labels"])

        output_df = join_labels_to_dataset(input_df, labels_df, yaml_args)
        output_df.write.mode("overwrite").parquet(args["output"])

        log_artifact(run, args["output"], args["output_dataset_name"], "dataset")

        # end spark session
        spark.stop()

    @property
    def outputs(self):
        return [self.config["parser_args"]["output"]]

    @property
    def job_type(self):
        return "join_labels"


def join_labels_to_dataset(input_df, labels_df, labels_config):
    """
    Generates the labels based on the labels_config and joins them to the input_df.
    """
    if "inclusion_criteria" in labels_config:
        labels_df = labels_df.filter(labels_config["inclusion_criteria"])
    labels_df = labels_df.withColumn(
        "label", expr(labels_config["positive_class_criteria"])
    )

    join_column = labels_config.get("join_column_name", "person_id")

    # only keep the pid and label columns
    labels_df = labels_df.select(join_column, "label")

    # if there are duplicate person_ids, throw an error
    if labels_df.count() != labels_df.select(join_column).distinct().count():
        raise ValueError("Each person should only have one row in the labels dataset.")

    # join the labels to the dataset
    output_df = input_df.join(labels_df, on=join_column, how="inner").distinct()

    logger.info(
        f"Number of rows in the output dataset with labels: {output_df.count()}"
    )
    return output_df


if __name__ == "__main__":
    job = JoinLabels()
    job.run_job()
