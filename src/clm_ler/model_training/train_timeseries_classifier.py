import os
from clm_ler.config.config import PROJECT_NAME
import argparse
from torch.utils.data import DataLoader, IterableDataset
from clm_ler.model_training.train_utils import (
    glob_parquet_files_to_pandas,
    UnitConverter,
    UnitStandardizer,
    MultiLabelledDatasetWithLabs,
    glob_parquet_files_to_pandas,
)
from clm_ler.utils.utils import (
    get_recursive_upstream_artifact,
    TrainEHRCLMJob,
    log_artifact,
)
from clm_ler.utils.logger import setup_logger
from clm_ler.utils.aws_utils import (
    download_file_from_s3,
    recursively_download_files_from_s3,
    list_subdirectories,
    upload_file_to_s3,
    split_s3_file,
)

import tokenizers
from clm_ler.model_training.train_utils import (
    ProportionalDataset,
    LabelledDataCollator,
    benchmark_model_for_classification,
)
from transformers import (
    TrainingArguments,
    TrainingArguments,
    AutoModelForSequenceClassification,
    IntervalStrategy,
    Trainer,
    EarlyStoppingCallback,
    PreTrainedTokenizerFast,
)
from clm_ler.model_training.train_utils import compute_metrics
import torch
import pandas as pd
import tarfile
import sklearn
from sklearn.metrics import classification_report
from torch.nn import Softmax
import pickle as pkl
import numpy as np
import wandb

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


class FinetuneTimeseriesCLMModel(TrainEHRCLMJob):
    def get_parser(self):
        parser = argparse.ArgumentParser(
            description="Fine tune a model on timeseries data."
        )
        parser.add_argument(
            "--input_data_artifact_name",
            dest="input_data_artifact_name",
            help="The name of the input training dataset",
            required=True,
            type=str,
        )
        parser.add_argument(
            "--subdataset_name",
            dest="subdataset_name",
            help=(
                "The name of the subdataset to use, assuming the input dataset is structured "
                "like dataset/task1, dataset/task2, ... dataset/taskn. "
                "This will read the corresponding dataset if it is set to 'taskn', for example."
            ),
            required=False,
            type=str,
        )
        parser.add_argument(
            "--model_artifact_name",
            dest="model_artifact_name",
            help=(
                "The model artifact to use as a base model, either the WandB artifact name "
                "or the S3 URI."
            ),
            required=True,
            type=str,
        )
        parser.add_argument(
            "--percentiles_conversion_artifact_name",
            dest="percentiles_artifact_name",
            help=(
                "The model artifact to use as a base model, either the WandB artifact name "
                "or the S3 URI."
            ),
            required=False,
            type=str,
        )
        parser.add_argument(
            "--unit_conversions_artifact_name",
            dest="unit_conversions_artifact_name",
            help=(
                "The model artifact to use as a base model, either the WandB artifact name "
                "or the S3 URI."
            ),
            required=False,
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
        parser.add_argument(
            "--tmp_save_directory",
            dest="tmp_save_directory",
            help="The location where intermediate files are saved during the hyperparameter scan.",
            required=True,
            type=str,
        )
        parser.add_argument(
            "--wandb_off",
            dest="wandb_off",
            required=False,
            action="store_true",
            help=(
                "When this flag is passed, WandB is not used. This is good for testing "
                "your script before moving to logging it on WandB."
            ),
        )
        parser.add_argument(
            "--local_rank",
            type=int,
            default=-1,
            help="Local rank for distributed training",
            dest="local_rank",
        )
        parser.add_argument(
            "--validate_on_full_dataset",
            dest="validate_on_full_dataset",
            help="Use the full validaiton set for validation",
            action="store_true",
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

    def main(self, run=None):
        s3_uri_of_preprocessed_data = self.resolve_s3_path(
            self.config["parser_args"]["input_data_artifact_name"], True
        )
        training_data = os.path.join(
            s3_uri_of_preprocessed_data,
            self.config["parser_args"]["subdataset_name"],
            "train",
        )
        validation_data = os.path.join(
            s3_uri_of_preprocessed_data,
            self.config["parser_args"]["subdataset_name"],
            "small_val",
        )
        full_validation_data = os.path.join(
            s3_uri_of_preprocessed_data,
            self.config["parser_args"]["subdataset_name"],
            "val",
        )
        test_data = os.path.join(
            s3_uri_of_preprocessed_data,
            self.config["parser_args"]["subdataset_name"],
            "test",
        )

        metadata = os.path.join(
            s3_uri_of_preprocessed_data,
            self.config["parser_args"]["subdataset_name"],
            "metadata",
            "train",
        )
        if metadata[-1] != "/":
            metadata += "/"
        # we need to download this and get
        logger.info(f"Looking in {metadata} for metadata")
        sub_directories = [
            el for el in list_subdirectories(metadata) if "parquet" in el
        ]
        logger.info(sub_directories)
        assert len(sub_directories) == 1
        metadata = sub_directories[0]
        local_fname = "metadata.parquet"
        download_file_from_s3(metadata, local_fname)
        metadata = pd.read_parquet(local_fname)

        logger.info(f"Using model {self.config['parser_args']['model_artifact_name']}")
        model_artifact_uri = self.resolve_s3_path(
            self.config["parser_args"]["model_artifact_name"], False
        )
        if "s3://" not in self.config["parser_args"]["model_artifact_name"]:
            if self.config["parser_args"]["unit_conversions_artifact_name"] is not None:
                unit_conversions_s3_uri = self.resolve_s3_path(
                    self.config["parser_args"]["unit_conversions_artifact_name"],
                    True,
                )
            else:
                unit_conversions_s3_uri = self.resolve_s3_path(
                    get_recursive_upstream_artifact(
                        self.config["parser_args"]["model_artifact_name"], "conversion"
                    ),
                    True,
                )

        if "s3://" not in self.config["parser_args"]["model_artifact_name"]:
            if self.config["parser_args"]["percentiles_artifact_name"] is not None:
                percentiles_s3_uri = self.resolve_s3_path(
                    self.config["parser_args"]["percentiles_artifact_name"],
                    True,
                )
            else:
                percentiles_s3_uri = self.resolve_s3_path(
                    get_recursive_upstream_artifact(
                        self.config["parser_args"]["model_artifact_name"], "percentile"
                    ),
                    True,
                )

        # download the artifacts
        model_location = (
            "model_artifacts.tar.gz"  # replace this with your model artifact
        )
        download_file_from_s3(
            model_artifact_uri,
            model_location,
        )

        if os.path.exists("model_artifacts"):
            raise ValueError("Please delete the old file!")

        with tarfile.open("model_artifacts.tar.gz", "r") as tar:
            tar.extractall()
        local_modelname = "model_artifacts.tar.gz".split(".")[0]

        conversions_dir = os.path.join(local_modelname, "conversions")
        recursively_download_files_from_s3(
            unit_conversions_s3_uri, conversions_dir, use_cli=True
        )
        conversions_frame = glob_parquet_files_to_pandas(conversions_dir)

        percentiles_dir = os.path.join(local_modelname, "percentiles")
        recursively_download_files_from_s3(
            percentiles_s3_uri, percentiles_dir, use_cli=True
        )
        percentiles_frame = glob_parquet_files_to_pandas(percentiles_dir)

        tokenizer = PreTrainedTokenizerFast.from_pretrained(
            os.path.join(local_modelname, "vocab.txt")
        )

        if self.config["parser_args"]["validate_on_full_dataset"]:
            validation_data = full_validation_data

        (model, predictions) = train_timeseries_model_hyperparameter_scan(
            self.config["parser_args"]["tmp_save_directory"],
            local_modelname,
            tokenizer,
            metadata,
            training_data,
            validation_data,
            full_validation_data,
            test_data,
            conversions_frame,
            percentiles_frame,
            self.config["training_args"],
            wait_for_download=self.config["parser_args"]["local_rank"] == 0,
        )

        with open("predictions.pkl", "wb") as f:
            pkl.dump(predictions, f)

        # Create an artifact
        if run is not None:
            artifact = wandb.Artifact(
                f"{self.config['parser_args']['output_model_artifact_name']}_test_predictions",
                type="dataset",
            )
            artifact.add_file("predictions.pkl")
            run.log_artifact(artifact)

        pred_label = np.argmax(all_predictions, axis=1)
        report = classification_report(all_labels, pred_label, output_dict=True)

        report_df = pd.DataFrame(report).transpose()
        report_df.reset_index(inplace=True)
        report_df.rename(columns={"index": "class"}, inplace=True)

        # Log the classification report table
        if run is not None:
            report_table = wandb.Table(dataframe=report_df)
            run.log({"Classification Report": report_table})

        # save the model and write it to s3
        model_name = self.config["parser_args"]["output_model_artifact_name"]
        model.save_pretrained(model_name)
        os.system(f"cp model_artifacts/vocab.txt {model_name}/vocab.txt")
        os.system(f"cp predictions.pkl {model_name}/predictions.pkl")

        report_df.to_csv(os.path.join(model_name, "performance_report.csv"))

        tar_filename = model_name + ".tar.gz"
        with tarfile.open(tar_filename, "w") as tar:
            tar.add(model_name, arcname=os.path.basename(model_name))

        output_fname = os.path.join(
            self.config["parser_args"]["output_model_location"], tar_filename
        )
        upload_file_to_s3(output_fname, tar_filename)

        if run is not None:
            log_artifact(run, output_fname, model_name, "model")


def train_timeseries_model_hyperparameter_scan(
    save_directory,
    model_directory,
    tokenizer,
    metadata,
    training_data,
    smaller_validation_data,
    validation_data,
    test_data,
    unit_conversions_frame,
    percentiles_conversion_frame,
    config,
    wait_for_download=False,
):
    assert not os.path.exists(save_directory)

    tuneable_parameters = config["tuneable_parameters"]
    params = {key: config[key] for key in tuneable_parameters}

    best_metric = None
    best_model = None
    best_config = None
    best_all_predictions, best_all_labels, best_all_person_ids = None, None, None
    metric = "eval_roc_auc_average"

    for updated_params in sklearn.model_selection.ParameterGrid(params):
        new_params = {}
        for key in config:
            if key in updated_params:
                new_params[key] = updated_params[key]
            else:
                new_params[key] = config[key]

        logger.info(f"Loading model from {model_directory}")
        model = AutoModelForSequenceClassification.from_pretrained(
            model_directory, num_labels=len(metadata)
        )
        descriptor = "_".join(
            [f"{param}_{new_params[param]}" for param in tuneable_parameters]
        )

        model, all_predictions, all_labels, all_person_ids = train_timeseries_model(
            model,
            tokenizer,
            metadata,
            training_data,
            smaller_validation_data,
            validation_data,
            unit_conversions_frame,
            percentiles_conversion_frame,
            new_params,
            wait_for_download=wait_for_download,
        )

        class Predictions:
            def __init__(self, all_labels, all_predictions):
                self.predictions = all_predictions
                self.label_ids = all_labels

        predictions = Predictions(all_labels, all_predictions)
        metrics = compute_metrics(predictions)

        if best_metric is None or metrics[metric] > best_metric:
            if best_metric is None:
                best_metric = 0.0

            logger.info(
                "Best model found with {} of {:.02f}, greater than {:.02f}".format(
                    metric, metrics[metric], best_metric
                )
            )
            best_metric = metrics[metric]
            model_dir = os.path.join(save_directory, descriptor)
            model.save_pretrained(model_dir)
            best_model = model_dir
            best_config = new_params
            best_all_predictions = all_predictions
            best_all_labels = all_labels
            best_all_person_ids = all_person_ids

        del model  # clear the memory on the gpu

    best_model = AutoModelForSequenceClassification.from_pretrained(best_model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_model.to(device)
    converter = UnitConverter(unit_conversions_frame)
    percentiler = UnitStandardizer(percentiles_conversion_frame)
    test_dataset = MultiLabelledDatasetWithLabs(
        converter,
        percentiler,
        test_data + "/",  # test on the test dataset
        tokenizer,
        percentile_split=percentiler.get_percentile_split_number(),
        max_position_embedding=best_model.config.max_position_embeddings,
        max_length=best_config["max_sequence_length"],
        wait_for_download=wait_for_download,
        include_person_ids=True,
    )
    test_predictions, test_labels, test_person_ids = benchmark_model_for_classification(
        best_model,
        test_dataset,
        eval_batch_size=best_config["eval_batch_size"],
    )
    return best_model, {
        "val": (best_all_predictions, best_all_labels, best_all_person_ids),
        "test": (test_predictions, test_labels, test_person_ids),
    }


def train_timeseries_model(
    model,
    tokenizer,
    metadata,
    training_data,
    smaller_validation_data,
    validation_data,
    unit_conversions_frame,
    percentiles_conversion_frame,
    config,
    wait_for_download=False,
):
    logger.info("Using the following config:")
    logger.info(config)
    train_batch_size = config["train_batch_size"]
    accumulation_steps = config["accumulation_steps"]
    logging_steps = config["logging_steps"]
    epochs = config["epochs"]
    learning_rate = config["learning_rate"]
    eval_steps = config["eval_steps"]
    max_length = config["max_sequence_length"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    converter = UnitConverter(unit_conversions_frame)
    percentiler = UnitStandardizer(percentiles_conversion_frame)

    class_to_dataloader = {}
    class_to_count = {}
    for i, row in metadata.iterrows():
        class_to_dataloader[row["value"]] = MultiLabelledDatasetWithLabs(
            converter,
            percentiler,
            training_data + "/",
            tokenizer,
            percentile_split=percentiler.get_percentile_split_number(),
            label=int(row["value"]),
            max_position_embedding=model.config.max_position_embeddings,
            max_length=max_length,
            wait_for_download=wait_for_download,
        )
        class_to_count[row["value"]] = row["count"]

    class_dataloader_list = []
    proportions = []
    for i in range(0, len(class_to_dataloader)):
        class_dataloader_list.append(class_to_dataloader[i])
        proportions.append(class_to_count[i])

    min_proportion = min(proportions)
    proportions = [
        min(el / min_proportion, config["max_proportion"]) for el in proportions
    ]

    training_dataset = ProportionalDataset(class_dataloader_list, proportions)

    collator = LabelledDataCollator()

    validation_dataset = MultiLabelledDatasetWithLabs(
        converter,
        percentiler,
        smaller_validation_data + "/",
        tokenizer,
        percentile_split=percentiler.get_percentile_split_number(),
        max_position_embedding=model.config.max_position_embeddings,
        max_length=max_length,
        wait_for_download=wait_for_download,
    )

    training_args = TrainingArguments(
        report_to=None,
        fp16=config["fp16"],
        per_device_train_batch_size=train_batch_size,
        eval_strategy=IntervalStrategy.EPOCH,
        per_device_eval_batch_size=config["eval_batch_size"],
        gradient_accumulation_steps=accumulation_steps,
        save_strategy=IntervalStrategy.EPOCH,
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        logging_dir="./logs",
        logging_steps=logging_steps,
        save_steps=eval_steps,
        eval_steps=eval_steps,
        warmup_steps=eval_steps,
        load_best_model_at_end=True,
        output_dir="./output",
        metric_for_best_model="eval_roc_auc_average",  # "1_f1-score",
        save_total_limit=config["early_stopping_patience"] + 1,
        dataloader_num_workers=4,
    )

    loss_function = torch.nn.CrossEntropyLoss()

    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=config["early_stopping_patience"]
    )

    # Initialize the Trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=training_dataset,
        eval_dataset=validation_dataset,
        data_collator=LabelledDataCollator(),
        compute_metrics=compute_metrics,
        callbacks=[early_stopping_callback],
        loss_fct=loss_function,
    )
    trainer.train()

    # load the full validation set
    full_validation_dataset = MultiLabelledDatasetWithLabs(
        converter,
        percentiler,
        validation_data + "/",
        tokenizer,
        percentile_split=percentiler.get_percentile_split_number(),
        max_position_embedding=model.config.max_position_embeddings,
        max_length=max_length,
        include_person_ids=True,
    )

    all_predictions, all_labels, all_person_ids = benchmark_model_for_classification(
        model,
        full_validation_dataset,
        eval_batch_size=config["eval_batch_size"],
    )

    # iterate throught he full validaion set and aggregate performance statistics
    return model, all_predictions, all_labels, all_person_ids


if __name__ == "__main__":
    FinetuneTimeseriesCLMModel().run_job()
