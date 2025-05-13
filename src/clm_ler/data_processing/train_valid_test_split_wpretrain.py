import argparse
from clm_ler.utils.utils import parse_yaml_args
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
    find_upstream_artifacts,
    setup_logger,
)
from clm_ler.data_processing.train_valid_test_split import (
    TrainValidTestSplit,
    write_to_s3,
)
import wandb
import numpy as np

logger = setup_logger()


# implement the abstract base methods:
class TrainValidTestSplitWPretrain(TrainValidTestSplit):
    """
    If the CLM pretraining is done on the same dataset, we need to exclude the pretraining data
    from the testing test to avoid data leakage.
    This class implements the exclusion step.
    It is expected that the artifact path of the pretraining dataset ("pretrain_data_path")
    or the model ("pretrain_model_path") is specified in the .yaml file.
    If not, the default pretrain data path is specified to be
    "train_ehr_clm/clm_ler:v0".
    """

    def main(self, run):
        args = run.config["parser_args"]
        yaml_args = run.config["training_config"]

        spark = SparkSession.builder.getOrCreate()
        # load some libraries to read from s3
        spark.conf.set("spark.hadoop.fs.s3a.endpoint", "s3.amazonaws.com")
        spark.conf.set(
            "spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem"
        )

        artifact = run.use_artifact(f"{args['input']}:latest")
        s3_uri_of_dataset = get_uri_of_parquet_dataset(artifact)

        print(f"Loading {s3_uri_of_dataset}")
        dataset = spark.read.parquet(s3_uri_of_dataset.replace("s3://", "s3a://"))

        # if "pretrain_model_path" is specified in args, fetch it and obtain the upstream artifacts
        if "pretrain_model_path" in args:
            pretrain_model_artifact_path = args.get("pretrain_model_path")
            word_match = args.get("word_match", "train")
            pretrain_data_artifact_path = find_upstream_artifacts(
                pretrain_model_artifact_path, word_match
            )
        else:
            pretrain_data_artifact_path = args.get(
                "pretrain_data_path",
                "train_ehr_clm/clm_ler:v0",
            )

        pretrain_data_artifact = run.use_artifact(pretrain_data_artifact_path)
        s3_uri_of_pretrain_data = get_uri_of_parquet_dataset(pretrain_data_artifact)
        print(f"Loading pretraining data {s3_uri_of_pretrain_data}")
        pretrain_data = spark.read.parquet(
            s3_uri_of_pretrain_data.replace("s3://", "s3a://")
        )

        train_datasets, test, valid = split_wpretrain(dataset, yaml_args, pretrain_data)
        write_to_s3(train_datasets, test, valid, run)


def split_wpretrain(dataset, training_args, pretrain_data):
    """
    Split the data into train, validation, and test datasets, with the consideration
    that data used in pretraining should only be in the training dataset.
    If separate_pos_neg_classes is True, the training dataset will be split
    into training_pos and training_neg.
    """
    join_column_name = training_args.get("join_column_name", "person_id")
    pretrain_pids = pretrain_data.select(join_column_name).distinct()

    in_pretrain = dataset.join(pretrain_pids, on=join_column_name, how="inner")
    not_in_pretrain = dataset.subtract(in_pretrain)
    pretrain_ratio = in_pretrain.count() / dataset.count()

    train_ratio = training_args.get("train_ratio", 0.7)
    test_ratio = training_args.get("test_ratio", 0.15)
    valid_ratio = training_args.get("validation_ratio", 0.15)

    logger.info(f"Percent of data used in CLM pre-training: {pretrain_ratio}")
    if pretrain_ratio > train_ratio:
        raise ValueError(
            "The pretraining data is greater than the training data. Please adjust the ratios; "
            "otherwise, there will be data leakage issues."
        )

    not_in_pretrain = not_in_pretrain.unpersist()
    not_in_pretrain = not_in_pretrain.cache()

    additional_train, validation, test = not_in_pretrain.randomSplit(
        [(train_ratio - pretrain_ratio), valid_ratio, test_ratio], seed=42
    )

    additional_train = additional_train.unpersist()
    validation = validation.unpersist()
    test = test.unpersist()

    additional_train = additional_train.cache()
    validation = validation.cache()
    test = test.cache()

    train = in_pretrain.unionByName(additional_train)
    total_data = dataset.count()

    # Function to log class distribution percentages
    def log_class_distribution(df, dataset_name):
        label_col_name = training_args.get("training_label", "label")
        class_counts = df.groupBy(label_col_name).count().collect()
        total_count = df.count()

        logger.info(f"Class distribution for {dataset_name}:")
        for row in class_counts:
            class_label = row[label_col_name]
            count = row["count"]
            percentage = count / total_count * 100
            logger.info(f"Class {class_label}: {percentage:.2f}% ({count} samples)")

    # Log class distribution for each dataset
    log_class_distribution(train, "train")
    log_class_distribution(validation, "validation")
    log_class_distribution(test, "test")

    # if we specified that we want the pos/neg class to be separated in
    # the yaml file (for binary classfication), we split the training dataset here
    separate_classes_for_training = training_args.get("separate_pos_neg_classes", False)
    label_col_name = training_args.get("training_label", "label")

    test = test.repartition(max(1, test.count() // 10000))
    validation = validation.repartition(max(1, validation.count() // 10000))

    if separate_classes_for_training:
        logger.info(
            f"Split percentages: Train: {train.count()/total_data}, "
            f"Validation: {validation.count()/total_data}, "
            f"Test: {test.count()/total_data}"
        )
        train_datasets = []
        class_indexes = np.unique(
            train.select(label_col_name).distinct().toPandas()[label_col_name].values
        )  # np.unique will also sort the classes.
        assert class_indexes[0] == 0  # There is a 0'th class
        assert np.all(
            (class_indexes[1:] - class_indexes[:-1]) == 1
        )  # the classes increase by 1. i.e. like 0, 1, 2, ... N
        logger.info(f"Found these classes in the data {class_indexes}")
        for class_index in class_indexes:
            filtered_train = train.filter(f"{label_col_name}=={class_index}")
            train_datasets.append(
                filtered_train.repartition(max(1, filtered_train.count() // 10000))
            )
        for i in range(0, len(train_datasets)):
            logger.info(
                f"Split percentages of training data for class {i}: "
                f"{train_datasets[i].count()/train.count()}"
            )
        return train_datasets, test, validation
    else:
        logger.info(
            f"Split percentages: Train: {train.count()/total_data}, "
            f"Validation: {validation.count()/total_data}, "
            f"Test: {test.count()/total_data}"
        )
        return train, test, validation


if __name__ == "__main__":
    TrainValidTestSplitWPretrain().run_job()
