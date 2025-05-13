from tokenizers import Tokenizer
from tokenizers.models import WordPiece
import numpy as np
import boto3
from clm_ler.utils.aws_utils import split_s3_file
import pyspark.sql.functions as f
import argparse
from pyspark.sql import SparkSession
import os
from clm_ler.utils.utils import parse_yaml_args
from clm_ler.utils.utils import (
    TrainEHRCLMJob,
    log_artifact,
    get_uri_of_parquet_dataset,
    setup_logger,
)
import wandb
from clm_ler.utils.aws_utils import get_s3_client
from transformers import PreTrainedTokenizerFast

logger = setup_logger()


# implement the abstract base methods:
class PrepareTokenizer(TrainEHRCLMJob):
    def get_parser(self):
        """
        Parse the input arguments for preparing a tokenizer model.
        """
        parser = argparse.ArgumentParser(
            description="Prepare a tokenizer vocabulary file for training a model"
        )
        parser.add_argument(
            "--input",
            dest="input",
            help="The name of the input dataset",
            required=True,
            type=str,
        )
        parser.add_argument(
            "--training_config",
            dest="training_config",
            help="The config for training this model",
            required=True,
            type=str,
        )
        parser.add_argument(
            "--output",
            dest="output",
            help="The location on s3 for the output model",
            required=True,
            type=str,
        )
        parser.add_argument(
            "--output_name",
            dest="output_name",
            help="The name of the output dataset",
            required=True,
            type=str,
        )

        return parser

    @property
    def job_type(self):
        """
        A name for the type of job.
        """
        return "prepare_tokenizer"

    @property
    def outputs(self):
        """
        This returns all outputs for this job.
        """
        return [self.config["parser_args"]["output"]]

    def main(self, run):
        args = run.config["parser_args"]
        assert args["output"].endswith(".txt")

        spark = SparkSession.builder.getOrCreate()
        # load some libraries to read from s3
        spark.conf.set("spark.hadoop.fs.s3a.endpoint", "s3.amazonaws.com")
        spark.conf.set(
            "spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem"
        )

        artifact = run.use_artifact(f"{run.project}/{args['input']}:latest")
        s3_uri_of_dataset = get_uri_of_parquet_dataset(artifact)

        dframe = spark.read.parquet(s3_uri_of_dataset.replace("s3://", "s3a://"))
        training_args = run.config["training_config"]
        tokenizer, vocabularly = prepare_tokenizer(dframe, training_args)

        local_save_directory = "vocab_folder"
        if not os.path.exists(local_save_directory):
            os.makedirs(local_save_directory)
        tokenizer.save_pretrained(local_save_directory)

        with open(
            os.path.join(local_save_directory, "vocab.txt"), "w", encoding="utf-8"
        ) as f:
            for token, idx in sorted(vocabulary.items(), key=lambda x: x[1]):
                f.write(token + "\n")

        # upload the file to s3
        get_s3_client().upload_file(
            os.path.join(local_save_directory, "vocab.txt"),
            *split_s3_file(args["output"]),
        )
        log_artifact(
            run,
            args["output"],
            args["output_name"],
            "dataset",
        )


def prepare_tokenizer(dframe, training_args):
    """
    Given a spark dataframe containing a column called
    "sorted_event_tokens" of type array<string>, build a tokenizer
    containing all events as a token. Furthermore, add the unknown
    and mask tokens.
    """

    exploded_df = dframe.select(
        "person_id", f.explode("sorted_event_tokens").alias("token")
    )

    # in this case, treat loinc codes as individual tokens.
    # Use token_type_ids to encode percentile information.
    if training_args["split_percentile_tokens"]:
        # Use substring_index to get the part of the string before the last colon
        exploded_df = exploded_df.withColumn(
            "token", f.substring_index("token", ":", 2)
        )

    count_frame = exploded_df.groupBy("token").count().dropDuplicates().toPandas()
    count_frame = count_frame.sort_values("count")[::-1]
    count_frame["cum_count"] = np.cumsum(count_frame["count"].values).astype(np.float64)
    count_frame["cum_count_frac"] = count_frame.eval(
        f"cum_count/{count_frame['cum_count'].values[-1]}"
    )
    # selected_tokens = count_frame[
    #    count_frame["cum_count_frac"] <= training_args["tokens_kept_fraction"]
    # ]  # 0.9995]
    count_frame["index"] = np.arange(0, len(count_frame))
    selected_tokens = count_frame[
        count_frame["index"] < training_args["vocabulary_size"]
    ]

    logger.info(selected_tokens)

    token_set = list(selected_tokens["token"].values)

    if "single_age_token" in training_args and training_args["single_age_token"]:
        token_set = [el for el in token_set if not el.startswith("AGE:")]
        token_set.append("AGE")

    logger.info("Keeping a vocabulary that covers this much data")
    logger.info(selected_tokens["cum_count_frac"].values[-1])

    if (
        "skip_special_tokens" not in training_args
        or not training_args["skip_special_tokens"]
    ):

        assert "[MASK]" not in token_set
        token_set.append("[MASK]")
        assert "[CLS]" not in token_set
        token_set.append("[CLS]")

    assert "[UNK]" not in token_set
    token_set.append("[UNK]")

    vocabulary = {el: i for i, el in enumerate(list(token_set))}

    # 1. Create the WordPiece model
    wordpiece_model = WordPiece(
        vocab=vocabulary, unk_token="[UNK]"  # a dict {token: id}
    )

    # 2. Create a Tokenizer object that uses the WordPiece model
    tokenizer_backend = Tokenizer(wordpiece_model)

    # 3. Wrap it
    base_tokenizer_model = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_backend,
        unk_token="[UNK]",
        mask_token="[MASK]",
        cls_token="[CLS]",
    )
    return base_tokenizer_model, vocabulary


if __name__ == "__main__":
    PrepareTokenizer().run_job()
