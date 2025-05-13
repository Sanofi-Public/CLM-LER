from clm_ler.utils.utils import TrainEHRCLMJob, log_artifact, setup_logger
from clm_ler.utils.aws_utils import download_file_from_s3
from clm_ler.model_training.train import CustomDataset
from sklearn.metrics import classification_report
import tokenizers
import itertools
import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset
from torch.nn.utils.rnn import pad_sequence
from clm_ler.utils.utils import (
    get_uri_of_parquet_dataset,
    get_uri_of_file,
    setup_logger,
)
from clm_ler.model_training.train_utils import (
    LabelledDataCollator,
    LabelledDataset,
    ProportionalDataset,
    compute_metrics,
    benchmark_model_for_classification,
)
from transformers import (
    EarlyStoppingCallback,
    IntervalStrategy,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
    PreTrainedTokenizerFast,
)

import pickle as pkl
import pandas as pd
import argparse
import tarfile
import os
import subprocess
from clm_ler.utils.aws_utils import (
    split_s3_file,
    download_file_from_s3,
    upload_file_to_s3,
)
import logging
import sys
from tqdm import tqdm
import wandb
import torch.nn.functional as F
from torch import nn
from torch.nn import Softmax
from clm_ler.config.config import PROJECT_NAME

logger = setup_logger()


class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        self.loss_fct = kwargs.pop("loss_fct")
        super().__init__(*args, **kwargs)

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss = self.loss_fct(logits, labels[:, 0])
        return (loss, outputs) if return_outputs else loss

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


class FinetuneCLMModel(TrainEHRCLMJob):
    def get_parser(self):
        parser = argparse.ArgumentParser(
            description="Preprocess the data from t1d for our model."
        )
        parser.add_argument(
            "--input_data_artifact_name",
            dest="input",
            help="The name if the input training dataset",
            required=True,
            type=str,
        )
        parser.add_argument(
            "--model_artifact_name",
            dest="model_artifact_name",
            help="The model artifact to use as a base model",
            required=True,
            type=str,
        )
        parser.add_argument(
            "--training_args",
            dest="training_args",
            help="The yaml file for setting the training params. See config/config_t1d.yaml",
            required=True,
            type=str,
        )
        parser.add_argument(
            "--output_model_artifact_name",
            dest="output_model_artifact_name",
            help="The model artifact that is the output",
            required=True,
            type=str,
        )
        parser.add_argument(
            "--output_model_location",
            dest="output_model_location",
            help="The location of the output data on s3",
            required=True,
            type=str,
        )
        return parser

    @property
    def job_type(self):
        """
        A name for the type of job.
        """
        return "tune_clm_model"

    @property
    def project(self):
        return PROJECT_NAME

    @property
    def outputs(self):
        """
        This returns all outputs for this job.
        """
        return [self.config["parser_args"]["output_model_location"]]

    def main(self, run):
        config = self.config
        training_datasets = []
        class_index = 0
        training_data_base = config["parser_args"]["input"] + "_train_{}:latest"
        while True:
            training_data_artifact = training_data_base.format(class_index)
            artifact = None
            try:
                artifact = run.use_artifact(training_data_artifact)
            except Exception as e:
                logger.info(
                    f"Failed to find data for class {class_index}. Stopping search."
                )
                break
            training_datasets.append(get_uri_of_parquet_dataset(artifact))
            class_index += 1

        logger.info(f"Found this many datasets: {len(training_datasets)}")

        valid_data_artifact = config["parser_args"]["input"] + "_valid:latest"
        test_data_artifact = config["parser_args"]["input"] + "_test:latest"

        valid_data_location = get_uri_of_parquet_dataset(
            run.use_artifact(valid_data_artifact)
        )
        test_data_location = get_uri_of_parquet_dataset(
            run.use_artifact(test_data_artifact)
        )

        if not config["parser_args"]["model_artifact_name"].startswith("s3://"):
            model_s3_uri = get_uri_of_file(
                run.use_artifact(
                    f"{config['parser_args']['model_artifact_name']}:latest"
                )
            )
        else:
            model_s3_uri = config["parser_args"]["model_artifact_name"]
        model_location = "model_artifacts.tar.gz"
        download_file_from_s3(
            model_s3_uri,
            model_location,
        )

        with tarfile.open("model_artifacts.tar.gz", "r") as tar:
            tar.extractall()
        local_fname = "model_artifacts.tar.gz".split(".")[0]
        tokenizer = PreTrainedTokenizerFast.from_pretrained(
            os.path.join("model_artifacts", "vocab.txt")
        )

        model = BertForSequenceClassification.from_pretrained(
            "model_artifacts",
            num_labels=len(training_datasets),
        )

        model, all_predictions, all_labels, all_person_ids = main(
            self.config,
            training_datasets,
            valid_data_location,
            test_data_location,
            tokenizer,
            model,
        )

        with open("predictions.pkl", "wb") as f:
            pkl.dump(
                {
                    "predictions": all_predictions,
                    "labels": all_labels,
                    "person_ids": all_person_ids,
                },
                f,
            )

        # Create an artifact
        artifact = wandb.Artifact(
            f"{config['parser_args']['output_model_artifact_name']}_test_predictions",
            type="dataset",
        )
        artifact.add_file("predictions.pkl")
        run.log_artifact(artifact)

        pred_label = np.argmax(all_predictions, axis=1)
        report = classification_report(all_labels, pred_label, output_dict=True)

        # Assuming 'report' is your classification report dictionary
        report_df = pd.DataFrame(report).transpose()

        report_df.reset_index(inplace=True)
        report_df.rename(columns={"index": "class"}, inplace=True)

        # Convert DataFrame to W&B Table
        report_table = wandb.Table(dataframe=report_df)

        # Log the classification report table
        run.log({"Classification Report": report_table})

        # save the model and write it to s3
        model_name = config["parser_args"]["output_model_artifact_name"]
        model.save_pretrained(model_name)
        # Securely copy the file
        os.makedirs(model_name, exist_ok=True)  # ensure model directory exists
        subprocess.run(
            ["cp", "model_artifacts/vocab.txt", os.path.join(model_name, "vocab.txt")],
            check=True,
        )

        tar_filename = model_name + ".tar.gz"
        with tarfile.open(tar_filename, "w") as tar:
            tar.add(model_name, arcname=os.path.basename(model_name))

        output_fname = os.path.join(
            config["parser_args"]["output_model_location"], tar_filename
        )
        upload_file_to_s3(output_fname, tar_filename)

        log_artifact(run, output_fname, model_name, "model")


def main(
    config,
    training_datasets,
    valid_data_location,
    test_data_location,
    tokenizer,
    model,
):
    # check if we need to split lab tokens by inspecting the number of type ids
    model_config = model.config
    logger.info(f"Number of token type IDs: {model_config.type_vocab_size}")
    percentile_split = None
    if model_config.type_vocab_size > 2:
        percentile_split = model_config.type_vocab_size - 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    training_args = config["training_args"]

    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=training_args["early_stopping_patience"]
    )

    train_iterable_datasets = [
        LabelledDataset(
            training_args["training_label"],
            data,
            tokenizer,
            512,
            tmp_data_location="tmp_train_pos",
            percentile_split=percentile_split,
        )
        for data in training_datasets
    ]

    valid_dataset = LabelledDataset(
        training_args["training_label"],
        valid_data_location,
        tokenizer,
        512,
        tmp_data_location="tmp_valid",
        percentile_split=percentile_split,
    )

    test_dataset = LabelledDataset(
        training_args["training_label"],
        test_data_location,
        tokenizer,
        512,
        tmp_data_location="tmp_test",
        include_person_ids=True,
        percentile_split=percentile_split,
    )

    proportions_per_label = config["training_args"]["proportions_per_label"]
    logger.info(f"These proportions: {proportions_per_label}")
    logger.info(f"For these datasets: {training_datasets}")
    assert len(proportions_per_label) == len(train_iterable_datasets)
    for i in range(0, len(proportions_per_label)):
        assert f"_{i}" in training_datasets[i]

    train_dataset = ProportionalDataset(train_iterable_datasets, proportions_per_label)

    total_proportions = sum(proportions_per_label)
    class_weights = np.array([total_proportions / p for p in proportions_per_label])
    class_weights = class_weights / sum(class_weights)  # normalize to 1.0
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(
        model.device
    )  # Adjust weights according to class imbalance
    loss_function = torch.nn.CrossEntropyLoss(weight=class_weights, reduction="mean")

    # let's eval every 10% of an epoch
    train_batch_size = training_args["training_batch_size"]
    eval_steps_in_batch = len(train_dataset) // train_batch_size
    eval_steps = int(training_args["eval_step_fraction"] * eval_steps_in_batch)
    logging_steps = eval_steps // training_args["training_logging_steps"]

    training_args = TrainingArguments(
        per_device_train_batch_size=train_batch_size,
        eval_strategy=IntervalStrategy.STEPS,
        save_strategy=IntervalStrategy.STEPS,
        num_train_epochs=training_args["number_epochs"],
        learning_rate=training_args["learning_rate"],
        logging_dir="./logs",
        logging_steps=logging_steps,
        save_steps=eval_steps,
        eval_steps=eval_steps,
        warmup_steps=eval_steps,
        load_best_model_at_end=True,
        output_dir="./output",
        metric_for_best_model=training_args["metric_for_best_model"],  # "1_f1-score",
        save_total_limit=11,
    )

    # Initialize the Trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=LabelledDataCollator(),
        compute_metrics=compute_metrics,
        callbacks=[early_stopping_callback],
        loss_fct=loss_function,
    )
    trainer.train()

    all_predictions, all_labels, all_person_ids = benchmark_model_for_classification(
        model, test_dataset, eval_batch_size=128
    )

    return model, all_predictions, all_labels, all_person_ids


if __name__ == "__main__":
    FinetuneCLMModel().run_job()
