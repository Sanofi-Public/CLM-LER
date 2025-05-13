import tokenizers
import torch
from torch.utils.data import IterableDataset, DataLoader
from transformers import (
    EarlyStoppingCallback,
    IntervalStrategy,
    BertForMaskedLM,
    BertConfig,
    LongformerForMaskedLM,
    LongformerConfig,
    AutoModelForMaskedLM,
    Trainer,
    TrainingArguments,
)
from clm_ler.model_training.train_utils import (
    resolve_sequence_length,
    CustomDataset,
)
import itertools
import argparse
import os
from clm_ler.config.config import PROJECT_DIR
from clm_ler.utils.aws_utils import split_s3_file, download_file_from_s3
from clm_ler.utils.utils import (
    parse_yaml_args,
    TrainEHRCLMJob,
    get_uri_of_parquet_dataset,
    get_uri_of_file,
    log_artifact,
    setup_logger,
    get_artifact,
)
from torch.nn.utils.rnn import pad_sequence
from pyarrow.parquet import ParquetFile
import numpy as np
import random
import boto3
import logging
import sys
import pandas as pd
import glob
import tarfile
import pickle as pkl
import multiprocessing
import math
import json
from transformers import PreTrainedTokenizerFast


logger = setup_logger()

from transformers import TrainerCallback


class CustomTrainer(Trainer):
    # why is this here?
    # This is the original function copied from huggingface's trainer with a bug fix. The
    # persistent workers flag is on. There is an option to configure this in the package,
    # however it is bugged only for the validation set. It causes memory leaks by recreating
    # the workers for every iteration of the validation.
    # The workaround here is to persist only the dataloaders for training.
    # persisting workers works well with the cache=True option for our dataloaders.
    # Caching the data lets all of it be laoded in to memory and shuffled.
    # If the dataloaders are killed (persistent_workers=False), then the cache
    # needs to be recreated for every iteration.
    # For some very large datasets, this can be a time consuming operation.
    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler
        (adapted to distributed training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        data_collator = self._get_collator_with_removed_columns(
            data_collator, description="training"
        )

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_num_workers > 0,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))


## create a delayed early stopping callback to avoid the trap at the beginning of training
class DelayedEarlyStoppingCallback(TrainerCallback):
    """
    An early stopping callback that takes effect after a certain number of epochs
    """

    def __init__(self, early_stopping_callback, delay_epochs=0.5):
        self.early_stopping_callback = early_stopping_callback
        self.delay_epochs = delay_epochs

    def on_evaluate(self, args, state, control, **kwargs):
        # Check if the delay period has passed
        if state.epoch >= self.delay_epochs:
            # Trigger early stopping
            self.early_stopping_callback.on_evaluate(args, state, control, **kwargs)


class TrainModel(TrainEHRCLMJob):
    def get_parser(self):
        """
        Parse the input arguments for training a model.
        """
        parser = argparse.ArgumentParser(
            description="Train a model for masked language modelling"
        )
        parser.add_argument(
            "--input",
            dest="input",
            help="The name of the dataset on s3.",
            required=True,
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
            "--tokenizer",
            dest="tokenizer",
            help="The name of the tokenizer artifact.",
            required=False,
            type=str,
            default=None,
        )
        parser.add_argument(
            "--output",
            dest="output",
            help="The location on s3 for the output model and data.",
            required=True,
            type=str,
        )
        parser.add_argument(
            "--input_model",
            dest="input_model",
            help=(
                "If this is supplied, load the model artifact found here "
                "and start training from it."
            ),
            required=False,
            type=str,
            default=None,
        )
        parser.add_argument(
            "--output_name",
            dest="output_name",
            help="The name of the output model artifact.",
            required=True,
            type=str,
        )
        parser.add_argument(
            "--test",
            dest="test",
            action="store_true",
        )
        parser.add_argument(
            "--local_rank",
            type=int,
            default=-1,
            help="Local rank for distributed training",
            dest="local_rank",
        )
        return parser

    @property
    def job_type(self):
        """
        A name for the type of job.
        """
        return "mlm_pretraining"

    @property
    def outputs(self):
        """
        This returns all outputs for this job.
        """
        return [self.config["parser_args"]["output"]]

    def main(self, run=None):
        args = self.config["parser_args"]
        training_args = self.config["training_config"]

        local_save_directory = os.path.join(
            PROJECT_DIR, os.path.split(args["output"])[-1]
        )
        assert not os.path.exists(local_save_directory)

        head, tail = os.path.split(local_save_directory)
        local_training_logs_directory = os.path.join(head, tail + "_training_logs")
        logger.info(f"Saving training logs in {local_training_logs_directory}")
        assert not os.path.exists(local_training_logs_directory)

        train_artifactid = f"{args['input'] + '_train:latest'}"
        print(f"Using a training dataset from {train_artifactid}")
        dataset_artifact_train = get_artifact(train_artifactid, run=run)

        valid_artifactid = f"{args['input'] + '_valid:latest'}"
        print(f"Using a validation dataset from {valid_artifactid}")
        dataset_artifact_valid = get_artifact(valid_artifactid, run=run)

        if args["input_model"] is not None:
            assert (
                args["tokenizer"] is None
            )  # we will get it from the pre-trained model
            model_artifact = run.use_artifact(f"{args['input_model'] + ':latest'}")
            model_s3_uri = get_uri_of_file(
                run.use_artifact(
                    f"{config['parser_args']['model_artifact_name']}:latest"
                )
            )
            model_location = "model_artifacts.tar.gz"
            download_file_from_s3(
                model_s3_uri,
                model_location,
            )
            with tarfile.open(model_location, "r") as tar:
                tar.extractall()
            local_fname = model_location.split(".")[0]
            tokenizer = PreTrainedTokenizerFast.from_pretrained(
                os.path.join(local_fname, "vocab.txt")
            )
            model = AutoModelForMaskedLM(local_fname)
        else:
            assert args["tokenizer"] is not None
            model = None
            vocab_artifact = get_artifact(f"{args['tokenizer']}:latest", run=run)
            vocab_artifact.download(root="./")
            tokenizer = PreTrainedTokenizerFast.from_pretrained("vocab.txt")
            vocabulary = get_vocab_from_file("vocab.txt")

        training_data = get_uri_of_parquet_dataset(dataset_artifact_train)
        validation_data = get_uri_of_parquet_dataset(dataset_artifact_valid)

        model, training_args, trainer = train_model(
            training_data,
            validation_data,
            training_args,
            tokenizer,
            vocabulary,
            local_training_logs_directory,
            model=model,
            skip_wandb=args["local_rank"] > 0,
            wait_for_download=args["local_rank"] > 0,
        )

        # only save the model on the main process.
        if args["local_rank"] <= 0:
            # save them all together now
            model.save_pretrained(local_save_directory)
            tokenizer.save_pretrained(local_save_directory)

            with open(
                os.path.join(local_save_directory, "vocab.txt"), "w", encoding="utf-8"
            ) as f:
                for token, idx in sorted(vocabulary.items(), key=lambda x: x[1]):
                    f.write(token + "\n")

            # zip-up the folder and upload to s3
            head, tail = os.path.split(local_save_directory)
            tar_filename = os.path.join(head, tail + ".tar.gz")
            with tarfile.open(tar_filename, "w") as tar:
                tar.add(
                    local_save_directory, arcname=os.path.basename(local_save_directory)
                )

            model_artifact_on_s3 = os.path.join(
                args["output"], os.path.split(tar_filename)[-1]
            )
            # upload the file to s3
            boto3.client("s3").upload_file(
                tar_filename,
                *split_s3_file(model_artifact_on_s3),
            )
            log_artifact(
                run,
                args["output"],
                args["output_name"],
                "model",
            )


class MetricsCallback(TrainerCallback):
    """
    A training callback for logging evaluations.
    """

    def __init__(self, logging_directory):
        self.training_history = []
        self.logging_directory = logging_directory

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        self.training_history.append(metrics)

    def on_train_end(self, args, state, control, **kwargs):
        with open(os.path.join(self.logging_directory, "eval_history.json"), "w") as f:
            json.dump(self.training_history, f)


class SaveMetricsCallback(TrainerCallback):
    """
    A training callback for logging training metrics and eval metrics in real time.
    """

    def __init__(self, logging_directory):
        self.logging_directory = logging_directory
        self.metrics_history = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        current_metrics = {"step": state.global_step, **logs}
        self.metrics_history.append(current_metrics)

        # Save the metrics history to a file
        with open(
            os.path.join(self.logging_directory, "training_metrics.json"), "w"
        ) as f:
            json.dump(self.metrics_history, f)

    def on_train_end(self, args, state, control, **kwargs):
        with open(
            os.path.join(self.logging_directory, "training_metrics.json"), "w"
        ) as f:
            json.dump(self.metrics_history, f)


def preprocess_logits_for_metrics(logits, labels):
    """
    Given a np.array of logits of shap batch_size x sequence_length x vocab_size
    and labels of size batch_size x sequence_length,
    preprocess the logits by selecting those that are predictions
    for masked language modelling (labels != -100).
    Afterwards, select the argmax along the vocabulary axis to select the
    predicted token. return the predictions and labels
    """
    predictions = torch.argmax(logits, -1)
    return predictions  # , labels


def compute_metrics(eval_pred):
    """
    Given a tuple of arbitrary length, assume that the first element
    are the clean predictions and labels from the above function.
    Calculate the accuracy of how many tokens are correctly predicted.
    """
    predictions, labels = eval_pred.predictions, eval_pred.label_ids
    mask = labels > -50
    predictions, labels = predictions[mask], labels[mask]
    return {"accuracy": np.sum(1 * (predictions == labels)) / len(predictions)}


class CustomDataCollator:
    """
    A "data collator" in the hugging-face library is a
    class that is meant to merge instances of data into a batch.
    This data collator also handles the masking
    of input tokens for masked language modelling.
    """

    def __init__(
        self,
        tokenizer,
        tokenizer_vocabulary,
        mask_token="[MASK]",
        prediction_fraction=0.15,
        masking_fraction=0.8,
        random_replacement_fraction=0.1,
        override_maxlen=None,
    ):
        """
        tokenizer (tokenizers.models.Model):
            The tokenizer model to handle tokenization
        tokenizer_vocabulary (dict[string]:int):
            A dictionary of keys that are the tokens in
            the vocabulary, and the values are the token id.
        mask_token:
            the masking token in the vocabulary, normally "[MASK]"
        prediction_fraction:
            The fraction of tokens to perform
            masked-language-modelling for
        masking_fraction:
            The fraction of the masked-language-modelling
            tokens that are replaced by the mask token
        random_replacement_fraction:
            The fraction of the masked-language-mdelling
            tokens that are replaced by random words.
        """

        self.tokenizer = tokenizer
        self.tokenizer_vocabulary = tokenizer_vocabulary

        # get all non special token ids, not including the [mask] or [unk] tokens, for example.
        self.nonspecial_token_ids = [
            token_id
            for token, token_id in tokenizer_vocabulary.items()
            if token[0] != "[" and token[-1] != "]"
        ]

        self.prediction_fraction = (
            prediction_fraction  # percentage of tokens to perform MLM for
        )

        # fraction of MLM tokens replaced by the mask token
        self.masking_fraction = masking_fraction
        # fraction of MLM tokens replaced by a random token
        self.random_replacement_fraction = random_replacement_fraction

        # fraction of MLM tokens not replaced by their original word
        self.unchanged = 1.0 - self.masking_fraction - self.random_replacement_fraction
        assert self.unchanged >= 0.0

        self.mask_token_id = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.tokenize(mask_token)[0]
        )

        self.override_maxlen = override_maxlen

        super().__init__()

    def __call__(self, list_of_data):
        length_cutoff = max(len(el["input_ids"]) for el in list_of_data)
        if self.override_maxlen is not None:
            length_cutoff = self.override_maxlen

        inputs = np.zeros((len(list_of_data), length_cutoff), dtype=int)
        position_ids = np.zeros((len(list_of_data), length_cutoff), dtype=int)
        attention_mask = np.zeros((len(list_of_data), length_cutoff), dtype=int)

        labels = (
            np.zeros(inputs.shape, dtype=int) - 100
        )  # set the labels to -100 by default (i.e. don't perform MLM on these tokens)

        for i, data in enumerate(list_of_data):
            encoded_inputs = data["input_ids"]
            one_position_id_list = data["position_ids"]
            assert len(encoded_inputs) == len(one_position_id_list)

            ##########################################
            # randomly select self.prediction_fraction
            # fraction of the tokens for masked language modelling
            # transfer these tokens to the labels as targets for training.
            indices = np.arange(0, len(encoded_inputs))
            random_probs = np.random.uniform(0, 1, size=len(indices))
            mlm_indices = indices[random_probs < self.prediction_fraction]
            labels[i, mlm_indices] = encoded_inputs[
                mlm_indices
            ]  # label the masked tokens with the inputs

            # select self.masking_fraction of the mlm tokens to be replaced by the mask token
            random_probs = np.random.uniform(0, 1, size=len(mlm_indices))
            masking_choices = random_probs <= self.masking_fraction
            masking_indices = mlm_indices[masking_choices]

            # select self.random_replacement_fraction
            # of the tokens to be replaced by the random words
            random_replacement_choices = np.logical_not(masking_choices) & (
                random_probs
                < (self.masking_fraction + self.random_replacement_fraction)
            )  # replace these indices with random words
            random_replacement_indices = mlm_indices[random_replacement_choices]

            # now that we've selected the tokens to be masked,
            # replaced or kept the same, modify the input accoringly

            # these tokens are to be replaced by the mask token
            encoded_inputs[masking_indices] = self.mask_token_id

            # replace these inputs with random other words
            encoded_inputs[random_replacement_indices] = np.random.choice(
                self.nonspecial_token_ids, size=len(random_replacement_indices)
            )

            # do nothing for those tokens that are to be kept as their original
            # pass

            # package everything back into the output arrays created before this loop:
            inputs[i, : len(encoded_inputs)] = encoded_inputs
            position_ids[i, : len(one_position_id_list)] = one_position_id_list
            attention_mask[i, : len(encoded_inputs)] = 1

        to_return = {
            "input_ids": torch.from_numpy(inputs),
            "attention_mask": torch.from_numpy(attention_mask),
            "position_ids": torch.from_numpy(position_ids),
            "labels": torch.from_numpy(labels),
        }

        if "token_type_ids" in list_of_data[0]:
            token_type_ids = torch.zeros(to_return["input_ids"].size(), dtype=torch.int)

            for i, item in enumerate(list_of_data):
                these_ids = torch.tensor(item["token_type_ids"])
                token_type_ids[i, 0 : len(these_ids)] = these_ids

            to_return["token_type_ids"] = token_type_ids

        return to_return


def train_model(
    training_data_location,
    validation_data_location,
    training_args,
    tokenizer,
    vocabulary,
    output_dir,
    skip_wandb=False,
    wait_for_download=False,
    model=None,
):
    """
    Given a training data location on s3 or local disk, a validation data location
    on s3 or local disk,train a CLM model given the training_args yaml file found
    in config/training.yaml, for example.
    Also provide the tokenizer and vocabulary of the CLM model.

    training_data_location (string):
        the location of the training data.
        Either a string like s3://path/to/your/data, or a simple directory
        for where the data is on disk.
    validation_data_location (string):
        Same as above, but for the validation dataset.
    training_args:
        A dictionary of training arguments for the model.
        These arguments configure the training of the model.
        You'll probably have to look at the code to understand exactly what the arguments mean,
        or look at the comments in one of the config files like config/train.yaml.
    tokenizer (tokenizers.models.Model):
        The tokenizer model to handle tokenization
    vocabulary (dict of str: int):
        The mapping of tokens to the token_id. The input to a transformer model requires a series
        of token ids.
    output_dir (string):
        The directory to save the model in.
    skip_wandb (bool):
        Whether to skip the wandb reporting. This is needed in test cases, for example.
    model:
        optional: a model to start trainining. If this is none, then create a new model to train
    """

    percentile_split = None
    type_vocab_size = 2
    if training_args["split_percentile_tokens"]:
        percentile_split = training_args["percentile_count"]
        type_vocab_size = percentile_split + 1

    if model is None:  # the model was not supplied. Create it from the config.
        if training_args["model_flavour"] == "bert":
            config_class = BertConfig
            model_class = BertForMaskedLM
        elif training_args["model_flavour"] == "longformer":
            config_class = LongformerConfig
            model_class = LongformerForMaskedLM
        else:
            raise ValueError(
                f"Couldn't find the right kind of model for {training_args['model_flavour']}. "
                "Please extend the code above to fix."
            )

        # Initialize an untrained CLM model for Masked Language Modeling (MLM)
        config = config_class(
            vocab_size=len(vocabulary),  # Default vocab size for CLM
            hidden_size=training_args[
                "hidden_size"
            ],  # Default hidden size for CLM base model
            num_hidden_layers=training_args[
                "num_hidden_layers"
            ],  # Default number of layers for CLM base model
            num_attention_heads=training_args[
                "num_attention_heads"
            ],  # Default number of attention heads for CLM base model
            max_position_embeddings=training_args["max_position_embeddings"],
            hidden_dropout_prob=training_args["hidden_dropout_prob"],
            attention_probs_dropout_prob=training_args["attention_probs_dropout_prob"],
            type_vocab_size=type_vocab_size,
            position_embedding_type=training_args["position_embedding_type"],
        )
        torch.cuda.empty_cache()
        model = model_class(config=config)

    if "override_maxlen" not in training_args:
        max_sequence_length = resolve_sequence_length(model)
    else:
        max_sequence_length = training_args["override_maxlen"]

    local_training_data_location = os.path.join(PROJECT_DIR, "TMP_training_data")
    local_validation_data_location = os.path.join(PROJECT_DIR, "TMP_validation_data")

    max_length = training_args["override_maxlen"]

    training_dataset = CustomDataset(
        training_data_location,
        tokenizer,
        tmp_data_location=local_training_data_location,
        percentile_split=percentile_split,
        wait_for_download=wait_for_download,
        trim_randomly=True,
        max_position_embedding=model.config.max_position_embeddings,
        max_length=max_length,
    )
    validation_dataset = CustomDataset(
        validation_data_location,
        tokenizer,
        tmp_data_location=local_validation_data_location,
        percentile_split=percentile_split,
        wait_for_download=wait_for_download,
        max_position_embedding=model.config.max_position_embeddings,
        max_length=max_length,
    )

    args = TrainingArguments(
        report_to="wandb" if not skip_wandb else None,
        fp16=training_args["fp16"],
        bf16=training_args["bf16"],
        eval_strategy=IntervalStrategy.STEPS,
        eval_steps=training_args["eval_steps"],
        logging_steps=training_args["logging_steps"],
        learning_rate=training_args["learning_rate"],
        per_device_train_batch_size=training_args["batch_size"],
        per_device_eval_batch_size=training_args["eval_batch_size"],
        num_train_epochs=training_args["num_training_epochs"],
        weight_decay=training_args["weight_decay"],
        metric_for_best_model=training_args["early_stop_metric"],
        gradient_accumulation_steps=training_args["gradient_accumulation_steps"],
        output_dir=output_dir,
        logging_strategy="steps",
        logging_dir=output_dir,
        load_best_model_at_end=True,
        eval_accumulation_steps=1,
        save_steps=training_args["eval_steps"],
        save_total_limit=training_args[
            "early_stop_patience"
        ],  # prevent to many models from being saved
        dataloader_num_workers=training_args["num_dataloader_workers"],
    )

    data_collator = CustomDataCollator(
        tokenizer,
        vocabulary,
        prediction_fraction=training_args["prediction_fraction"],
        masking_fraction=training_args["masking_fraction"],
        random_replacement_fraction=training_args["random_replacement_fraction"],
        override_maxlen=max_length,
    )

    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=training_args[
            "early_stop_patience"
        ]  # Number of evaluations with no improvement after which training will be stopped
    )
    delayed_early_stopping_callback = DelayedEarlyStoppingCallback(
        early_stopping_callback, delay_epochs=training_args["early_stopping_delay"]
    )

    metrics_callback = MetricsCallback(output_dir)
    logging_callback = SaveMetricsCallback(output_dir)

    trainer = CustomTrainer(
        model=model,
        args=args,
        train_dataset=training_dataset,
        eval_dataset=validation_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        callbacks=[delayed_early_stopping_callback, metrics_callback, logging_callback],
    )
    train_result = trainer.train()

    # Save training metrics
    trainer.save_metrics("all", train_result.metrics)

    return model, args, trainer  # return everything needed to save the model.


def get_vocab_from_file(vocab_file):
    with open(vocab_file, "r") as f:
        vocab = {el.replace("\n", ""): i for i, el in enumerate(f.readlines())}
    return vocab


if __name__ == "__main__":
    TrainModel().run_job()
