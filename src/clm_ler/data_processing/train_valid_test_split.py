import argparse
from clm_ler.utils.utils import parse_yaml_args
import pyspark
import pyspark.sql.functions as f
import logging
import sys
import os
from clm_ler.utils.aws_utils import check_if_s3_dir_exists
from clm_ler.utils.utils import (
    TrainEHRCLMJob,
    log_artifact,
    get_uri_of_parquet_dataset,
    setup_logger,
)
import wandb
from clm_ler.utils.utils import read_spark_table

logger = setup_logger()


def select_min_data_length(dataset, training_args):
    if "min_history_size" in training_args:
        dataset = dataset.withColumn(
            "size", f.size(f.col("sorted_event_tokens"))
        ).where(
            f.col("size") > training_args["min_history_size"]
        )  # filter out those people with very short histories.
        logger.info(
            f"""After selecting people with at least {training_args['min_history_size']} entires,
            we have this many patients: {dataset.count()}"""
        )
    return dataset


# implement the abstract base methods:
class TrainValidTestSplit(TrainEHRCLMJob):
    def get_parser(self):
        parser = argparse.ArgumentParser(
            description="Split the datasets used for training a CLM model."
        )
        parser.add_argument(
            "--input",
            dest="input",
            help="The artifact name of this dataset",
            required=True,
            type=str,
        )
        parser.add_argument(
            "--predefined_split",
            dest="predefined_split",
            help="A predefined table of person ids to split the data into train/valid/test.",
            required=False,
            type=str,
        )
        parser.add_argument(
            "--training_config",
            dest="training_config",
            help="The config for training this model.",
            required=True,
            type=str,
        )
        parser.add_argument(
            "--output",
            dest="output",
            help="The location on s3 for the training, validation and testing data.",
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
        parser.add_argument(
            "--test",
            dest="test",
            action="store_true",
            help="If test, subsample 1% of the dataset.",
        )
        parser.add_argument(
            "--pretrain_model_path",
            required=False,
            dest="pretrain_model_path",
            type=str,
        )
        parser.add_argument(
            "--pretrain_data_path", required=False, dest="pretrain_data_path", type=str
        )

        return parser

    @property
    def job_type(self):
        """
        A name for the type of job.
        """
        return "train_valid_test_split"

    @property
    def outputs(self):
        """
        This returns all outputs for this job.
        """
        return [self.config["parser_args"]["output"]]

    def main(self, run):
        args = run.config["parser_args"]
        yaml_args = run.config["training_config"]
        if "person_id_column" in yaml_args:
            person_id_column = yaml_args["person_id_column"]
        else:
            person_id_column = "person_id"

        # provide the full artifact name
        if not args["input"].startswith("s3://") and not args["input"].startswith(
            "SF://"
        ):
            artifact = run.use_artifact(f"{args['input']}")
            s3_uri_of_dataset = get_uri_of_parquet_dataset(artifact)
            print(f"Loading {s3_uri_of_dataset}")
            dataset = read_spark_table(s3_uri_of_dataset)
        else:
            print(f"Loading {args['input']}")
            dataset = read_spark_table(args["input"])

        data_splits = None
        if "predefined_split" in args and args["predefined_split"] is not None:
            logger.info("Loading pre-split data")
            data_splits = {}
            for data_type in ["train", "valid", "test"]:
                artifact_base, artifact_version = args["predefined_split"].split(":")
                artifact_name = f"{artifact_base}_{data_type}:{artifact_version}"
                logger.info(f"Using artifact {artifact_name}")
                artifact = run.use_artifact(artifact_name)
                s3_uri_of_dataset = get_uri_of_parquet_dataset(artifact)
                data_splits[data_type] = read_spark_table(s3_uri_of_dataset)
        else:
            logger.info("Not using any pre-split data")

        # load the training config
        if data_splits is None:
            train, test, valid = do_split(
                dataset, args, yaml_args, person_id_column=person_id_column
            )
        else:
            train = dataset.join(data_splits["train"], on=person_id_column, how="inner")
            valid = dataset.join(data_splits["valid"], on=person_id_column, how="inner")
            test = dataset.join(data_splits["test"], on=person_id_column, how="inner")

        assert train.join(valid, on=person_id_column, how="inner").count() == 0
        assert train.join(test, on=person_id_column, how="inner").count() == 0
        assert valid.join(test, on=person_id_column, how="inner").count() == 0

        train = select_min_data_length(train, yaml_args)
        valid = select_min_data_length(valid, yaml_args)
        test = select_min_data_length(test, yaml_args)

        train = train.withColumn("person_id", f.col(person_id_column))
        valid = valid.withColumn("person_id", f.col(person_id_column))
        test = test.withColumn("person_id", f.col(person_id_column))

        write_to_s3(train, test, valid, run)


def do_split(dataset, args, training_args, person_id_column="person_id"):
    if args["test"]:
        dataset = dataset.sample(0.01)  # sample 1% of the dataset

    early_stopping_set_size = training_args["validation_set_fraction"]

    dataset = dataset.cache()
    train_df, test_df, validation_df = dataset.randomSplit(
        [0.5 + early_stopping_set_size, 0.5, early_stopping_set_size]
    )  # train, test and validation.
    train_df = train_df.cache()
    test_df = test_df.cache()
    validation_df = validation_df.cache()
    assert train_df.join(test_df, on=person_id_column).count() == 0
    assert train_df.join(validation_df, on=person_id_column).count() == 0
    assert test_df.join(validation_df, on=person_id_column).count() == 0

    logger.info(
        f"""After doing a train, valid, test split, we
         have this many patients: {train_df.count()}, {validation_df.count(), test_df.count()}"""
    )

    repartition_size = training_args["training_partition_size"]
    validation_repartition_size = training_args["validation_partition_size"]

    train_df = train_df.repartition(repartition_size, person_id_column)
    validation_df = validation_df.repartition(
        validation_repartition_size, person_id_column
    )
    test_df = test_df.repartition(repartition_size, person_id_column)
    return train_df, test_df, validation_df


def write_to_s3(*args):
    """
    A fancier way of writing to s3, with the option to write training dataset as a single dataset
    or separately for positive and negative class.
    """
    train, test, valid, run = args
    lst_datasets = [valid, test]
    lst_names = ["valid", "test"]

    if type(train) is list:
        for i, sub_train in enumerate(train):
            lst_datasets.append(sub_train)
            lst_names.append(f"train_{i}")
    else:
        lst_datasets.append(train)
        lst_names.append("train")

    args = run.config["parser_args"]
    for dataset, destination in zip(lst_datasets, lst_names):
        logger.info(f"Number of rows in {destination}: {dataset.count()}")
        dataset_location = os.path.join(args["output"], "datasets", destination)
        dataset.write.mode("overwrite").parquet(
            dataset_location.replace("s3://", "s3a://")
        )
        log_artifact(
            run,
            dataset_location,
            args["output_dataset_name"] + "_" + destination,
            "dataset",
        )


if __name__ == "__main__":
    TrainValidTestSplit().run_job()
