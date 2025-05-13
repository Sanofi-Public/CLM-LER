from clm_ler.data_processing.data_processing_utils import (
    apply_mapping,
    get_umls_sources,
    convert_model_vocab_to_frame,
    apply_mapping,
    derive_umls_translation,
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
from clm_ler.utils.aws_utils import split_s3_file, download_file_from_s3
import tarfile
import os

logger = setup_logger()


def split_data(frame, keys, column):
    """
    Split the codes available in the vocabulary into two groups based on keys.
    Any rows with codes containing any key as a substring is included in the first dataframe.
    The remaining data is in the second dataframe.
    """
    selection = None
    for d in keys:
        if selection is None:
            selection = f.col(column).contains(d)
        else:
            selection = selection | f.col(column).contains(d)
    data_matching_keys = frame.filter(selection)
    other_data = frame.filter(~selection)
    return data_matching_keys, other_data


class PreprocessingEHRSHOTJob(TrainEHRCLMJob):
    def get_parser(self):
        parser = argparse.ArgumentParser(description="Create a training dataset")
        parser.add_argument(
            "--mapping_config",
            dest="mapping_config",
            help="A config file defining the mappings for columns for this job.",
            required=True,
            type=str,
        )
        parser.add_argument(
            "--ehrshot_data_location",
            dest="ehrshot_data_location",
            help="The location of the EHRSHOT dataset on s3.",
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
            "--model_name",
            dest="model_artifact_name",
            help="The model whose vocabulary is used to translate the data.",
            required=True,
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

        ehrshot_data_location = self.config["parser_args"]["ehrshot_data_location"]

        if not self.config["parser_args"]["model_artifact_name"].startswith("s3://"):
            model_s3_uri = get_uri_of_file(
                run.use_artifact(f"{self.config['parser_args']['model_artifact_name']}")
            )
        else:
            model_s3_uri = self.config["parser_args"]["model_artifact_name"]
        model_location = (
            "model_artifacts.tar.gz"  # change this to your model artifact name
        )
        download_file_from_s3(
            model_s3_uri,
            model_location,
        )

        with tarfile.open("model_artifacts.tar.gz", "r") as tar:
            tar.extractall()
        model_vocab = os.path.join("model_artifacts", "vocab.txt")

        data = spark.read.option("header", True).csv(
            ensure_s3a_path(os.path.join(ehrshot_data_location, "data", "ehrshot.csv"))
        )
        data = data.repartition(
            1000, "patient_id"
        )  # repartition for better parallelism

        translated_data, demographics_data = preprocess_data(
            self.config, data, model_vocab, spark
        )

        # write the data to s3 and log the artifact
        translated_data.write.parquet(
            os.path.join(
                ensure_s3a_path(self.config["parser_args"]["output_dest"]),
                "translated_data",
            ),
        )
        demographics_data.write.parquet(
            os.path.join(
                ensure_s3a_path(self.config["parser_args"]["output_dest"]),
                "demographics_data",
            ),
        )

        # only log the artifacts for online runs
        if run is not None:
            log_artifact(
                run,
                self.config["parser_args"]["output_dest"],
                self.config["parser_args"]["output_dataset_name"],
                "dataset",
            )


def preprocess_data(config, data, model_vocab, spark, umls_mapping_frame=None):
    # convert dates to datetimes in spark
    data = data.withColumn("start", f.to_timestamp("start", "yyyy-MM-dd HH:mm:ss"))
    data = data.withColumn("end", f.to_timestamp("end", "yyyy-MM-dd HH:mm:ss"))
    data.show()

    demographic_keys = ["Race", "Gender", "Ethnicity"]
    demographics_data, medical_events_data = split_data(data, demographic_keys, "code")
    medical_events_data = medical_events_data.selectExpr(
        "SPLIT(code, '/')[0] as raw_source",
        "LOWER(SPLIT(code, '/')[1]) as code",
        "start",
        "end",
        "patient_id",
        "value",
        "unit",
    )

    demographics_data.show()
    medical_events_data.show()

    demographics_data_gender = demographics_data.filter(
        f.col("code").contains("Gender/")
    ).select(
        f.col("start").alias("gender_date"),
        f.col("code").alias("gender_code"),
        "patient_id",
    )
    demographics_data_race = demographics_data.filter(
        f.col("code").contains("Race/")
    ).select(
        f.col("start").alias("race_date"),
        "patient_id",
        f.col("code").alias("race_code"),
    )
    demographics_data_ethnicity = demographics_data.filter(
        f.col("code").contains("Ethnicity/")
    ).select(
        f.col("start").alias("ethnicity_date"),
        "patient_id",
        f.col("code").alias("ethnicity_code"),
    )

    demographics_merged = demographics_data_gender.join(
        demographics_data_race, on="patient_id", how="outer"
    ).join(demographics_data_ethnicity, on="patient_id", how="outer")

    assert (
        demographics_merged.select("patient_id").distinct().count()
        == demographics_merged.count()
    )
    assert demographics_merged.count() == data.select("patient_id").distinct().count()
    assert (
        demographics_merged.filter(
            "(gender_date == race_date or gender_date IS NULL or race_date IS NULL) and "
            "(ethnicity_date == race_date or ethnicity_date IS NULL or race_date IS NULL)"
        ).count()
        == demographics_merged.count()
    )
    assert demographics_data_gender.count() == demographics_merged.count()

    # change the names of the demographics data to the be more readible.
    # This was found by matching counts of each race to the paper here:
    # https://arxiv.org/abs/2307.02028
    token_to_race = {
        "Race/5": "Race/White",
        "Race/4": "Race/Pacific_Islander",
        "Race/3": "Race/Black",
        "Race/2": "Race/Asian",
        "Race/1": "Race/American_Indian",
    }

    demographics_cleaned = apply_mapping(
        demographics_merged, token_to_race, "race_code", "race_code_mapped", spark
    )
    demographics_cleaned.show()

    # load the model's vocabulary to see how to map demographics data
    with open(model_vocab) as open_file:
        vocab = [el.replace("\n", "") for el in open_file.readlines()]

    logger.info("Searching the models vocabulary")
    token_types = ["GENDER", "RACE", "ETHNICITY"]
    for token_type in token_types:
        relevant_tokens = [el for el in vocab if token_type in el]
        logger.info(f"Tokens found matching {token_type}, {relevant_tokens}")

    logger.info("Searching the data's demographics data")
    token_types = ["Gender", "Race", "Ethnicity"]
    token_columns = ["gender_code", "race_code_mapped", "ethnicity_code"]

    for token_type, token_column in zip(token_types, token_columns):
        relevant_tokens = list(
            demographics_cleaned.select(token_column)
            .distinct()
            .toPandas()[token_column]
            .values
        )
        logger.info(f"Tokens matching {token_type}, {relevant_tokens}")

    race_mapping = config["mapping_config"]["race_mapping"]
    gender_mapping = config["mapping_config"]["gender_mapping"]
    ethnicity_mapping = config["mapping_config"]["ethnicity_mapping"]

    demographics_mapped = demographics_cleaned
    demographics_mapped = apply_mapping(
        demographics_mapped, race_mapping, "race_code_mapped", "px_race", spark
    )
    demographics_mapped = apply_mapping(
        demographics_mapped, gender_mapping, "gender_code", "px_gender", spark
    )
    demographics_mapped = apply_mapping(
        demographics_mapped, ethnicity_mapping, "ethnicity_code", "px_ethnicity", spark
    )
    demographics_mapped = demographics_mapped.withColumn(
        "px_birth_date", f.col("gender_date")
    )
    demographics_mapped = demographics_mapped.selectExpr(
        "patient_id as person_id",
        "px_race",
        "px_gender",
        "px_ethnicity",
        "px_birth_date",
    )

    logger.info("Showing the final mapped demographics.")
    demographics_mapped.show()

    logger.info("Here are the sources of codes in the data")
    logger.info(
        sorted(
            list(
                medical_events_data.select("raw_source")
                .distinct()
                .toPandas()["raw_source"]
                .values
            )
        ),
    )

    logger.info("Here are the sources of codes in UMLS")
    logger.info(get_umls_sources(umls_mapping_frame=umls_mapping_frame))

    # these are the mappings of the sources in the data to those found in UMLS.
    # This simple accounts for slight differences in naming.
    source_concept_mapping = config["mapping_config"]["data_to_umls_source_map"]
    medical_events_data_mapped = apply_mapping(
        medical_events_data, source_concept_mapping, "raw_source", "source", spark
    )

    # Similarly to the medical events data, we need to map the event sources
    # to their UMLS names in the vocabulary.
    vocab_frame = convert_model_vocab_to_frame(model_vocab, spark).selectExpr(
        "code", "source as vocab_source", "vocab_token"
    )

    # map the code names to those found in UMLS.
    source_mapping = config["mapping_config"]["vocab_to_umls_source_map"]

    vocab_frame = apply_mapping(
        vocab_frame, source_mapping, "vocab_source", "source", spark
    ).selectExpr("source", "code", "vocab_token")
    vocab_frame.show()

    translation_frame = derive_umls_translation(
        medical_events_data_mapped, vocab_frame, umls_mapping_frame=umls_mapping_frame
    )

    filtered_translations_frame = translation_frame.selectExpr(
        "original_source as source",
        "original_code as code",
        "vocab_source",
        "vocab_code",
        "vocab_token",
    ).dropDuplicates(["source", "code", "vocab_token"])

    translated_medical_events_data = medical_events_data_mapped.join(
        filtered_translations_frame,
        on=["source", "code"],
        how="left",
    )
    translated_medical_events_data.show()

    counts = translated_medical_events_data.groupBy("raw_source").agg(
        f.sum(f.when(f.col("vocab_token").isNull(), 1).otherwise(0)).alias(
            "null_count"
        ),
        f.count("*").alias("total_count"),
    )
    logger.info(
        "This is the fraction of medical events by raw source without an equivalent token "
        "found for the model's vocabulary."
    )
    logger.info(
        counts.selectExpr(
            "null_count/total_count AS null_frac", "raw_source", "total_count"
        )
        .orderBy(f.col("null_frac"), ascending=False)
        .toPandas()
    )

    counts = (
        translated_medical_events_data.dropDuplicates(["raw_source", "code"])
        .groupBy("raw_source")
        .agg(
            f.sum(f.when(f.col("vocab_token").isNull(), 1).otherwise(0)).alias(
                "null_count"
            ),
            f.count("*").alias("total_count"),
        )
    )
    logger.info(
        "This is the fraction of medical codes by raw source without an equivalent token "
        "found for the model's vocabulary."
    )
    logger.info(
        counts.selectExpr(
            "null_count/total_count AS null_frac", "raw_source", "total_count"
        )
        .orderBy(f.col("null_frac"), ascending=False)
        .toPandas()
    )

    fraction_of_valid_tests = (
        translated_medical_events_data.filter(
            "source == 'LOINC' and value is not NULL and unit is not NULL"
        ).count()
        / translated_medical_events_data.filter("source == 'LOINC'").count()
    )
    logger.info(
        f"This percentage of loinc tests had valid values {fraction_of_valid_tests}"
    )

    loinc_code_selection = (
        "unit is not NULL and value is not NULL and vocab_source == 'LOINC'"
    )
    new_code = "CONCAT(vocab_token, ':', value, ':', unit)"
    translated_medical_events_data = translated_medical_events_data.withColumn(
        "vocab_token",
        f.expr(
            f"CASE WHEN {loinc_code_selection} THEN {new_code} ELSE vocab_token END"
        ),
    )

    return translated_medical_events_data, demographics_mapped


if __name__ == "__main__":
    job = PreprocessingEHRSHOTJob()
    job.run_job()
