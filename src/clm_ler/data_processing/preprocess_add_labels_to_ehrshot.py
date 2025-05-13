from clm_ler.data_processing.data_processing_utils import (
    apply_mapping,
    get_umls_sources,
    convert_model_vocab_to_frame,
    apply_mapping,
    derive_umls_translation,
    add_baseline_patient_information,
    align_to_first_date,
    consolidate_into_arrays,
)
from clm_ler.utils.utils import (
    parse_yaml_args,
    TrainEHRCLMJob,
    get_uri_of_parquet_dataset,
    get_uri_of_file,
    log_artifact,
    setup_logger,
    get_artifact,
    get_spark,
    ensure_s3a_path,
)
import argparse
import pyspark.sql.functions as f
from clm_ler.utils.aws_utils import (
    split_s3_file,
    download_file_from_s3,
    list_subdirectories,
)

import tarfile
import os
from pyspark.sql import Window
from pyspark.sql import functions as F


def keep_top_n_per_group(df, group_col, order_col, n):
    # Create a window partitioned by the group column and ordered by the order column
    window = Window.partitionBy(group_col).orderBy(F.desc(order_col))

    # Add a row number within each group
    df_with_rank = df.withColumn("rank", F.row_number().over(window))

    # Filter to keep only the top N rows per group
    result = df_with_rank.filter(F.col("rank") <= n).drop("rank")

    return result


def recombine_frame(frames):
    full_frame = None
    for frame in frames:
        if full_frame is None:
            full_frame = frame
        else:
            full_frame = full_frame.unionByName(frame, allowMissingColumns=True)
    return full_frame


logger = setup_logger()


class PreprocessingEHRSHOT_AddLabels_Job(TrainEHRCLMJob):
    def get_parser(self):
        parser = argparse.ArgumentParser(description="Create a training dataset")
        # parser.add_argument(
        #    "--config_file",
        #    dest="config_file",
        #    help="A config file defining the job",
        #    required=True,
        #    type=str,
        # )
        parser.add_argument(
            "--ehrshot_data_location",
            dest="ehrshot_data_location",
            help="The location of the EHRSHOT dataset on s3.",
            required=True,
            type=str,
        )
        parser.add_argument(
            "--processed_ehrshot_data",
            dest="processed_ehrshot_data",
            help=(
                "The model artifact or s3 directory of the preprocessed ehrshot data for a "
                "given model."
            ),
            required=True,
            type=str,
        )
        parser.add_argument(
            "--output_dest",
            dest="output_dest",
            help="The output location of the datasets on s3",
            required=True,
            type=str,
        )
        parser.add_argument(
            "--config_file",
            dest="config_file",
            help="A config file to remap and read datasets to a format expected by this job",
            required=True,
            type=str,
        )
        parser.add_argument(
            "--output_dataset_name",
            dest="output_dataset_name",
            help="The reference name of this dataset for WandB",
            required=True,
            type=str,
        )
        parser.add_argument(
            "--wandb_off",
            dest="wandb_off",
            required=False,
            action="store_true",
            help=(
                "When this flag is passed, WandB is not used. "
                "This is good for testing your script before moving to logging it on WandB."
            ),
        )
        return parser

    @property
    def job_type(self):
        """
        A name for the type of job.
        """
        return "data_preprocessing"

    @property
    def outputs(self):
        """
        This returns all outputs for this job.
        """
        return [self.config["parser_args"]["output_dest"]]

    def main(self, run=None):
        spark = get_spark()

        if run is None:
            logger.info("Running in WandB Offline mode!")

        data_artifact = self.config["parser_args"]["processed_ehrshot_data"]
        if "s3://" not in data_artifact:
            preprocessed_data = run.use_artifact(data_artifact)
            s3_uri_of_preprocessed_data = get_uri_of_parquet_dataset(preprocessed_data)
        else:
            s3_uri_of_preprocessed_data = data_artifact

        demographics_s3_subdir = self.config["config_file"].get(
            "demographics_data_subdir", "demographics_data"
        )
        medical_events_s3_subdir = self.config["config_file"].get(
            "medical_events_data_subdir", "translated_data"
        )

        demographics_data_s3 = os.path.join(
            s3_uri_of_preprocessed_data, demographics_s3_subdir
        ).replace("s3://", "s3a://")
        medical_events_data_s3 = os.path.join(
            s3_uri_of_preprocessed_data, medical_events_s3_subdir
        ).replace("s3://", "s3a://")

        if ".csv" not in medical_events_data_s3:
            medical_events_data = spark.read.parquet(medical_events_data_s3)
        else:
            medical_events_data = spark.read.option("header", True).csv(
                medical_events_data_s3
            )

        if ".csv" not in demographics_data_s3:
            demographics_data = spark.read.parquet(demographics_data_s3)
        else:
            demographics_data = spark.read.option("header", True).csv(
                demographics_data_s3
            )

        medical_events_data = medical_events_data.selectExpr(
            *self.config["config_file"]["medical_events_select_expr"]
        )
        demographics_data = demographics_data.selectExpr(
            *self.config["config_file"]["demographics_select_expr"]
        )

        medical_events_data = medical_events_data.filter(
            self.config["config_file"]["medical_events_filter_expr"]
        )
        demographics_data = demographics_data.filter(
            self.config["config_file"]["demographics_filter_expr"]
        )

        # get all of the kinds of prediction tasks available in the ehrshot datast
        ehr_shot_data = self.config["parser_args"]["ehrshot_data_location"]

        split_frame_s3 = os.path.join(ehr_shot_data, "splits/person_id_map.csv")
        split_frame = spark.read.option("header", True).csv(
            split_frame_s3.replace("s3://", "s3a://")
        )

        prediction_task_folders = list_subdirectories(
            os.path.join(ehr_shot_data, "benchmark"), directories_only=True
        )
        prediction_tasks = [
            el.rstrip("/").split("/")[-1] for el in prediction_task_folders
        ]

        logger.info(f"Preparing data for prediction tasks {prediction_tasks}")
        for directory, prediction_task in zip(
            prediction_task_folders, prediction_tasks
        ):
            if (
                prediction_task == "chexpert"
            ):  # TODO... see https://github.com/som-shahlab/ehrshot-benchmark/
                # blob/88dd769bfb7466a812740de6a1f824ff9efed8b1/ehrshot/utils.py#L310
                continue  # It's a multilabel prediction task with
                # a special way to decode the multiclass labels...

            logger.info(f"Processing for task {prediction_task}.")
            this_output_dest = os.path.join(
                self.config["parser_args"]["output_dest"], prediction_task
            )
            labels_s3 = os.path.join(directory, "labeled_patients.csv")
            labels_frame = spark.read.option("header", True).csv(
                labels_s3.replace("s3://", "s3a://")
            )
            array_data, metadata = add_labels(
                medical_events_data, demographics_data, labels_frame, split_frame
            )

            # write this metadata to the corresponding location on s3
            for dataset_type in array_data:
                array_data[dataset_type].write.parquet(
                    os.path.join(this_output_dest, dataset_type).replace(
                        "s3://", "s3a://"
                    )
                )

                metadata[dataset_type].write.parquet(
                    os.path.join(this_output_dest, "metadata", dataset_type).replace(
                        "s3://", "s3a://"
                    )
                )

        # finally log the artifact on s3
        if run is not None:
            log_artifact(
                run,
                self.config["parser_args"]["output_dest"],
                self.config["parser_args"]["output_dataset_name"],
                "dataset",
            )


def add_labels(medical_events_data, demographics_data, labels_frame, split_frame):
    """
    Preprocess the medical events data, demographics data, and labels data into
    the arrays expected for model training.

    How each column is defined in the arguments below:
        start: A timestamp marking the start of this medical event.
        patient_id: A string used to identify this patient.
        vocab_token: The string representing this medical event in the model's vocabulary.
        px_birth_date: The birth date of the person.
        px_gender: The gender of the person, stored as a string (e.g., Male).
        px_ethnicity: The ethnicity of the person (e.g., Hispanic).
        px_race: The race of the person (e.g., Caucasian).
        value: The prediction label.
            - 1 or 0 if label_type != 'boolean'.
            - 'True' or 'False' if label_type == 'boolean'.
        label_type: 'boolean' or 'categorical'.
        prediction_time: Prediction timestamp of form "yyyy-MM-dd HH:mm:ss".

    Args:
        medical_events_data: A Spark dataframe with columns:
            | start (timestamp) | patient_id (string or int) | vocab_token (string) |
        demographics_data: A Spark dataframe with columns:
            | px_birth_date (timestamp) | px_gender (string) | px_ethnicity (string) |
            | px_race (string) |

        labels_frame: A Spark dataframe with columns:
            | value (string) | label_type (string) | prediction_time (string) |
            | patient_id (string or int) |


    Returns:
        tuple of (output_dataframe, metadata)
            output_dataframe: A dataframe with columns:
                | patient_id (string or int) | sorted_event_tokens (array of strings) |
                | day_position_tokens (array of ints) | date (array of timestamps) |
            metadata: A dataframe containing metadata about the labels,
                including how many of each label is present in the data.
    """
    labels_frame = labels_frame.withColumn(
        "prediction_time", f.to_timestamp("prediction_time", "yyyy-MM-dd HH:mm:ss")
    )
    labels_frame.show()

    label_types = labels_frame.select("label_type").distinct().toPandas()
    assert len(label_types) == 1
    label_type = label_types["label_type"].values[0]
    if label_type == "boolean":
        labels_frame = labels_frame.withColumn(
            "value", f.expr("CASE WHEN value == 'True' THEN 1 ELSE 0 END")
        )
    else:
        labels_frame = labels_frame.withColumn(
            "value", f.expr("CAST(value AS int) AS value")
        )

    # let's concatenate the labels into the medical events frame.
    medical_events_data_to_join = medical_events_data.selectExpr(
        "patient_id as person_id", "start as date", "vocab_token as general_event_token"
    )
    labels_data_to_join = labels_frame.selectExpr(
        "patient_id as person_id",
        "prediction_time as date",
        "CONCAT('LABEL', ':', value) as general_event_token",
    )
    merged_frames = medical_events_data_to_join.unionByName(labels_data_to_join)

    merged_frames.show()

    split_frame.groupBy("split").count().show()

    event_threshold = (
        1000  # keep at most 1000 events per label for a smaller validation set.
    )

    split_data = {}
    split_demographics = {}
    for dataset_type in ["train", "val", "test"]:
        print(f"Handling {dataset_type} data")
        patients = split_frame.filter(f"split == {repr(dataset_type)}").selectExpr(
            "omop_person_id as person_id"
        )
        this_data = merged_frames.join(patients, on="person_id")
        if "person_id" not in demographics_data.columns:
            demographics_data = demographics_data.withColumn(
                "person_id", f.col("patient_id")
            )
        this_demographics = demographics_data.join(patients, on="person_id")
        split_data[dataset_type] = this_data
        split_demographics[dataset_type] = this_demographics

    label_counts_frame = (
        split_data["val"]
        .filter("CONTAINS(general_event_token, 'LABEL')")
        .groupBy("general_event_token")
        .count()
        .toPandas()
    )
    logger.info(label_counts_frame)

    labels, other_data = split_data["val"].filter(
        "CONTAINS(general_event_token, 'LABEL')"
    ), split_data["val"].filter("NOT CONTAINS(general_event_token, 'LABEL')")
    labels = labels.withColumn("random_number", f.rand()).cache()

    all_labels = []
    for i, row in label_counts_frame.iterrows():
        label, counts = row["general_event_token"], row["count"]
        these_labels = labels.filter(
            f'general_event_token == {repr(row["general_event_token"])}'
        )
        if counts > event_threshold:
            for n in range(0, 100):
                print(f"Trying {n}")
                top_n = keep_top_n_per_group(
                    these_labels, "person_id", "random_number", n
                )
                if top_n.count() > event_threshold:
                    break
            all_labels.append(top_n)
        else:
            all_labels.append(these_labels)

    all_labels = recombine_frame(all_labels)
    smaller_validation_data = recombine_frame([all_labels, other_data])

    split_data["small_val"] = smaller_validation_data
    split_demographics["small_val"] = split_demographics["val"]

    metadata = {}
    prepared_data = {}

    for dataset_type in split_data:
        this_demographics = split_demographics[dataset_type]
        this_data = split_data[dataset_type]

        this_aligned_data = align_to_first_date(this_data)
        this_aligned_data.show()
        merged_data, _ = add_baseline_patient_information(
            this_aligned_data, this_demographics
        )
        array_data = consolidate_into_arrays(merged_data)

        this_labels_data = labels_frame.selectExpr(
            "patient_id as person_id", "value"
        ).join(patients, on="person_id")
        this_labels_data = this_labels_data.groupBy("value").count()

        metadata[dataset_type] = this_labels_data
        prepared_data[dataset_type] = array_data

    return prepared_data, metadata


if __name__ == "__main__":
    PreprocessingEHRSHOT_AddLabels_Job().run_job()
