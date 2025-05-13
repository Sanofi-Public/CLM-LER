"""
This script prepares a tabular dataset of patients containing a list of diagnosis codes,
procedure codes, lab tests and prescriptions ordered by when the occured.

Take a look at config/data_files_full_example.yaml to see where the input data is taken from on s3.
Take a look at config/data_preprocess_example.yaml for default arguments related to
selecting patient events.

Usage:
    poetry run python3 -m --config_file config/data_preprocess.yaml --output_dest
    s3a://bucket/key --input_files_config config/data_files.yaml

This will produce a tabular dataset saved under s3a://bucket/key named:

person_id     sorted_event_tokens                                             days
1             ["HCPCS:pr_a", "ICD10CM:dx_a", "LOINC:lx_a:LOW", "NDC:rx_a",    [0, 0, 1, 1, 3, 368]
              "CODE:dx_a", "CPT4:pr_a"]
2             ["HCPCS:pr_a", "ICD10CM:dx_b", "LOINC:lx_a:HI", "NDC:rx_d",     [0, 0, 1, 2, 3,  5]
              "CODE:dx_a", "CPT4:pr_a"]
....
"""

import logging
import pandas as pd
from pyspark import SparkConf
from pyspark.sql import SparkSession
import logging
import os
import argparse
import pyspark.sql.functions as f
from clm_ler.utils.utils import parse_yaml_args
import wandb
from clm_ler.utils.aws_utils import check_if_s3_dir_exists
from clm_ler.utils.utils import TrainEHRCLMJob, log_artifact, setup_logger
from clm_ler.data_processing.data_processing_utils import (
    create_tokens,
    apply_yaml_selections,
    select_columns,
    align_to_first_date,
    add_baseline_patient_information,
    consolidate_into_arrays,
    merge_frames,
    process_lab_frame_into_percentiles,
    process_lab_percentiles_into_tokens,
)
from clm_ler.utils.utils import (
    find_upstream_artifacts,
    get_uri_of_parquet_dataset,
    get_spark,
)
from clm_ler.utils.utils import ensure_s3a_path, read_spark_table

logger = setup_logger()


def resolve_path(base, tail):
    if not base:
        if tail:
            return tail
        else:
            return None
    else:  # if base
        if tail:
            return os.path.join(base, tail)
        else:
            return None


class PreprocessingJob(TrainEHRCLMJob):
    def get_parser(self):
        parser = argparse.ArgumentParser(description="Create a training dataset")
        parser.add_argument(
            "--config_file",
            dest="config_file",
            help="A config file defining the job",
            required=True,
            type=str,
        )
        parser.add_argument(
            "--input_files_config",
            dest="config_file_inputs",
            help="A config file defining files to be read by the job",
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
            "--output_dataset_name",
            dest="output_dataset_name",
            help="The reference name of this dataset for WandB",
            required=True,
            type=str,
        )
        parser.add_argument(
            "--percentiles_conversions_artifact",
            dest="percentiles_conversions_artifact",
            help="Please provide the artifact names of the conversions to use",
            required=False,
            default=None,
            type=str,
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

    @property
    def outputs(self):
        """
        This returns all outputs for this job.
        """
        parser_args = self.config["parser_args"]
        conversions_fname = (
            parser_args["output_dest"].rstrip("/") + "_conversions" + "/"
        )
        percentiles_fname = (
            parser_args["output_dest"].rstrip("/") + "_percentiles" + "/"
        )
        return [
            self.config["parser_args"]["output_dest"],
            conversions_fname,
            percentiles_fname,
        ]

    def main(self, run):
        parser_args = self.config["parser_args"]
        yaml_args = self.config["config_file"]
        yaml_input_files = self.config["config_file_inputs"]

        spark = get_spark()

        # load all of the dataframes
        patient_frame = resolve_path(
            yaml_input_files["base_data_directory"],
            yaml_input_files["patient"],
        )
        if patient_frame is not None:
            patient_frame = read_spark_table(
                patient_frame, yaml_args["partition_column"]
            )

        diagnosis_frame = resolve_path(
            yaml_input_files["base_data_directory"],
            yaml_input_files["diagnosis"],
        )
        if diagnosis_frame is not None:
            diagnosis_frame = read_spark_table(
                diagnosis_frame, yaml_args["partition_column"]
            )

        prescription_frame = resolve_path(
            yaml_input_files["base_data_directory"],
            yaml_input_files["prescription"],
        )
        if prescription_frame is not None:
            prescription_frame = read_spark_table(
                prescription_frame, yaml_args["partition_column"]
            )

        procedure_frame = resolve_path(
            yaml_input_files["base_data_directory"],
            yaml_input_files["procedure"],
        )
        if procedure_frame is not None:
            procedure_frame = read_spark_table(
                procedure_frame, yaml_args["partition_column"]
            )

        lab_frame = resolve_path(
            yaml_input_files["base_data_directory"],
            yaml_input_files["lab"],
        )
        if lab_frame is not None:
            lab_frame = read_spark_table(lab_frame, yaml_args["partition_column"])

        if parser_args["percentiles_conversions_artifact"] is not None:
            logger.info("Running wth premade conversions")
            conversions_artifact = run.use_artifact(
                parser_args["percentiles_conversions_artifact"] + "_conversions:latest"
            )
            conversions_uri = get_uri_of_parquet_dataset(conversions_artifact)
            percentiles_artifact = run.use_artifact(
                parser_args["percentiles_conversions_artifact"] + "_percentiles:latest"
            )
            percentiles_uri = get_uri_of_parquet_dataset(percentiles_artifact)
            percentiles_frame = spark.read.parquet(
                percentiles_uri.replace("s3://", "s3a://")
            )
            percentiles_frame = percentiles_frame.cache()
            conversions_frame = spark.read.parquet(
                conversions_uri.replace("s3://", "s3a://")
            )
            conversions_frame = conversions_frame.cache()
            conversions_frame.show()
            percentiles_frame.show()
        else:
            percentiles_frame = None
            conversions_frame = None

        final_frame, conversions, percentiles, cutflow_frame = main(
            yaml_args,
            patient_frame,
            diagnosis_frame,
            prescription_frame,
            procedure_frame,
            lab_frame,
            spark,
            percentiles_frame=percentiles_frame,
            conversions_frame=conversions_frame,
        )
        print(cutflow_frame)

        final_frame.write.parquet(
            ensure_s3a_path(parser_args["output_dest"]), mode="overwrite"
        )
        log_artifact(
            run,
            parser_args["output_dest"],
            parser_args["output_dataset_name"],
            "dataset",
        )

        conversions_fname = (
            parser_args["output_dest"].rstrip("/") + "_conversions" + "/"
        )
        if conversions is not None:
            conversions.write.parquet(ensure_s3a_path(conversions_fname))
            log_artifact(
                run,
                conversions_fname,
                parser_args["output_dataset_name"] + "_conversions",
                "dataset",
            )

        percentiles_fname = (
            parser_args["output_dest"].rstrip("/") + "_percentiles" + "/"
        )
        if percentiles is not None:
            percentiles.write.parquet(ensure_s3a_path(percentiles_fname))
            log_artifact(
                run,
                percentiles_fname,
                parser_args["output_dataset_name"] + "_percentiles",
                "dataset",
            )

        # Save DataFrame to a CSV file
        cutflow_frame.to_csv("cutflow.csv", index=False)

        # Create an artifact
        artifact = wandb.Artifact(
            f"{parser_args['output_dataset_name']}_cutflow", type="dataset"
        )
        artifact.add_file("cutflow.csv")
        run.log_artifact(artifact)


class Cutflow:
    def __init__(self):
        self.cuts = []
        self.number_patients = []

    def add_cut(self, name, number):
        print(f"Adding cut: {name}")
        print(f"Patient count: {number}")
        assert name not in self.cuts
        assert len(self.number_patients) == 0 or self.number_patients[-1] >= number
        self.cuts.append(name)
        self.number_patients.append(number)

    def export_to_pandas(self):
        return pd.DataFrame.from_dict(
            {"cut_name": self.cuts, "number_patients": self.number_patients}
        )


def get_overall_patients(
    diagnosis_frame,
    prescription_frame,
    procedure_frame,
    lab_frame,
):
    prescription_persons = None
    diagnosis_persons = None
    procedure_persons = None
    lab_persons = None

    if diagnosis_frame is not None:
        diagnosis_persons = diagnosis_frame.select("person_id").distinct()
    if prescription_frame is not None:
        prescription_persons = prescription_frame.select("person_id").distinct()
    if procedure_frame is not None:
        procedure_persons = procedure_frame.select("person_id").distinct()
    if lab_frame is not None:
        lab_persons = lab_frame.select("person_id").distinct()

    persons = None
    for frame in [
        diagnosis_persons,
        prescription_persons,
        procedure_persons,
        lab_persons,
    ]:
        if frame is None:
            continue
        if persons is None:
            persons = frame
        else:
            persons = persons.union(frame)

    return persons.distinct()


def main(
    yaml_args,
    patient_frame,
    diagnosis_frame,
    prescription_frame,
    procedure_frame,
    lab_frame,
    spark,
    repartition=4096,
    percentiles_frame=None,
    conversions_frame=None,
):
    """
    Prepares a dataset for training a model
    """
    logger.info("Staring run.")

    (
        patient_frame,
        diagnosis_frame,
        prescription_frame,
        procedure_frame,
        lab_frame,
    ) = select_columns(
        yaml_args,
        patient_frame,
        diagnosis_frame,
        prescription_frame,
        procedure_frame,
        lab_frame,
    )

    logger.info("Caching frames")
    patient_frame = patient_frame.cache()
    diagnosis_frame = diagnosis_frame.cache()
    prescription_frame = prescription_frame.cache()
    procedure_frame = procedure_frame.cache()
    lab_frame = lab_frame.cache()

    assert (
        patient_frame.select("person_id").filter("person_id IS NULL").count() == 0
    )  # make sure every person id can be converted to bigint

    cutflow = Cutflow()
    overall_patients = get_overall_patients(
        diagnosis_frame,
        prescription_frame,
        procedure_frame,
        lab_frame,
    )
    cutflow.add_cut("overall patients", overall_patients.count())

    #######################################################################################
    (
        patient_frame,
        diagnosis_frame,
        prescription_frame,
        procedure_frame,
        lab_frame,
    ) = apply_yaml_selections(
        yaml_args,
        patient_frame,
        diagnosis_frame,
        prescription_frame,
        procedure_frame,
        lab_frame,
    )

    overall_patients = get_overall_patients(
        diagnosis_frame,
        prescription_frame,
        procedure_frame,
        lab_frame,
    )
    cutflow.add_cut("After yaml selections", overall_patients.count())

    if diagnosis_frame is not None:
        diagnosis_frame = diagnosis_frame.filter(yaml_args["event_date_filter"])
    if prescription_frame is not None:
        prescription_frame = prescription_frame.filter(yaml_args["event_date_filter"])
    if procedure_frame is not None:
        procedure_frame = procedure_frame.filter(yaml_args["event_date_filter"])
    if lab_frame is not None:
        lab_frame = lab_frame.filter(yaml_args["event_date_filter"])

    overall_patients = get_overall_patients(
        diagnosis_frame,
        prescription_frame,
        procedure_frame,
        lab_frame,
    )
    cutflow.add_cut("After date day filtering", overall_patients.count())

    #######################################################################################
    # create a dataframe of diagnosis tokens
    diagnosis_token_frame = None
    if diagnosis_frame is not None:
        logger.info("Perparing a dataframe of diagnosis tokens")
        diagnosis_token_frame = create_tokens(
            diagnosis_frame, "dx_diagnosis_code", "dx_diagnosis_code_type"
        )
        diagnosis_token_frame = diagnosis_token_frame.repartition(
            repartition, "person_id"
        ).cache()
        diagnosis_frame.unpersist()

        logger.info("Here are the diagnosis tokens")
        logger.info(diagnosis_token_frame.dtypes)
        diagnosis_token_frame.show()

    #######################################################################################
    # create a dataframe of prescription tokens
    prescription_token_frame = None
    if prescription_frame is not None:
        logger.info("Perparing a dataframe of prescription tokens")
        prescription_token_frame = create_tokens(
            prescription_frame, "rx_prescription_code", "rx_prescription_code_type"
        )

        prescription_token_frame = prescription_token_frame.repartition(
            repartition, "person_id"
        ).cache()
        prescription_frame.unpersist()

        logger.info("Here are the prescription tokens")
        logger.info(prescription_token_frame.dtypes)
        prescription_token_frame.show()

    #######################################################################################
    # create a dataframe of lab result tokens
    lab_token_frame = None
    conversions = None
    percentiles = None
    if lab_frame is not None:
        logger.info("Perparing a dataframe of lab test result tokens")
        (
            percentiled_labs_frame,
            conversions,
            percentiles,
        ) = process_lab_frame_into_percentiles(
            spark,
            lab_frame,
            {
                "min_loinc_counts": yaml_args["min_loinc_counts"],
                "min_loinc_value_count": yaml_args["min_loinc_value_counts"],
            },
            percentiles_frame=percentiles_frame,
            conversions_frame=conversions_frame,
        )
        lab_token_frame = process_lab_percentiles_into_tokens(percentiled_labs_frame)
        lab_token_frame = lab_token_frame.repartition(repartition, "person_id").cache()
        conversions = conversions.cache()
        percentiles = percentiles.cache()
        lab_frame.unpersist()

        logger.info("Here are the lab tokens")
        logger.info(lab_token_frame.dtypes)
        lab_token_frame.show()

    #######################################################################################
    # create a dataframe of procedure tokens
    proc_token_frame = None
    if procedure_frame is not None:
        logger.info("Perparing a dataframe of procedure tokens")
        proc_token_frame = create_tokens(
            procedure_frame,
            "pr_procedure_code",
            "pr_procedure_code_type",
        )
        proc_token_frame = proc_token_frame.repartition(
            repartition, "person_id"
        ).cache()
        procedure_frame.unpersist()

        logger.info("Here are the procedure tokens")
        logger.info(proc_token_frame.dtypes)
        proc_token_frame.show()

    #######################################################################################
    # concatenate the frames
    logger.info("Merging the dataframes together")
    column_order = [
        "person_id",
        "date",
        "general_event_token",
    ]
    unaligned_merged_frames = merge_frames(
        column_order,
        diagnosis_token_frame,
        prescription_token_frame,
        lab_token_frame,
        proc_token_frame,
    )

    unaligned_merged_frames = unaligned_merged_frames.cache()
    if diagnosis_token_frame is not None:
        diagnosis_frame.unpersist()
    if prescription_token_frame is not None:
        prescription_token_frame.unpersist()
    if lab_token_frame is not None:
        lab_token_frame.unpersist()
    if proc_token_frame is not None:
        proc_token_frame.unpersist()

    cutflow.add_cut(
        "patients after yaml selections",
        unaligned_merged_frames.select("person_id").distinct().count(),
    )

    # add a days_since_begin that counts the number of days for each token from their first entry
    merged_frames = align_to_first_date(unaligned_merged_frames)
    merged_frames = merged_frames.cache()
    unaligned_merged_frames.unpersist()
    cutflow.add_cut(
        "patients days aligned to first diagnosis",
        merged_frames.select("person_id").distinct().count(),
    )

    # add age at first token, gender, race etc information
    merged_frames, patient_demographics = add_baseline_patient_information(
        merged_frames, patient_frame
    )

    # some patients events are cut, giving them fewer than 365 days of entries.
    # Put this in to guarantee a complete record
    max_days = merged_frames.groupBy("person_id").agg({"days_since_begin": "max"})
    persons_to_keep = max_days.filter(
        f.col("max(days_since_begin)") >= yaml_args["number_of_days_threshold"]
    )
    merged_frames = merged_frames.join(persons_to_keep, on="person_id", how="inner")
    merged_frames = merged_frames.cache()
    persons_to_keep.unpersist()
    cutflow.add_cut(
        f"patients filtered to have at least {yaml_args['number_of_days_threshold']} days of data",
        merged_frames.select("person_id").distinct().count(),
    )

    # drop duplicate codes on the same day. We did it before, but just do again to be sure...
    merged_frames = merged_frames.dropDuplicates(
        ["person_id", "days_since_begin", "general_event_token"]
    )
    merged_frames = merged_frames.cache()
    cutflow.add_cut(
        f"duplicate events dropped",
        merged_frames.select("person_id").distinct().count(),
    )

    result = consolidate_into_arrays(merged_frames)
    result = result.cache()
    merged_frames.unpersist()

    cutflow.add_cut(
        f"consolidated into arrays",
        merged_frames.select("person_id").distinct().count(),
    )

    cutflow_frame = cutflow.export_to_pandas()

    return result, conversions, percentiles, cutflow_frame


if __name__ == "__main__":
    job = PreprocessingJob()
    job.run_job()
