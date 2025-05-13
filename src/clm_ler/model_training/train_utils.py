from torch.utils.data import IterableDataset
import os
import numpy as np
import pandas as pd
from clm_ler.config.config import PROJECT_DIR
from clm_ler.utils.aws_utils import (
    split_s3_file,
    download_file_from_s3,
    upload_file_to_s3,
    recursively_download_files_from_s3,
)
from torch.utils.data import DataLoader, IterableDataset
from clm_ler.utils.utils import setup_logger
import boto3
import logging
import sys
import torch
from tqdm import tqdm
from pyarrow.parquet import ParquetFile
import itertools
from torch.nn.utils.rnn import pad_sequence
import math
import random
import glob
from sklearn.metrics import classification_report
from transformers import (
    BertModel,
    LongformerModel,
    BertForMaskedLM,
    LongformerForMaskedLM,
)
from sklearn.metrics import precision_recall_curve, roc_auc_score
import numpy as np
import bisect
import time
import pint
import re
from scipy.special import softmax
from clm_ler.data_processing.data_processing_utils import (
    handle_unit_conversion,
)
from torch.nn import Softmax
import numpy as np

logger = setup_logger()


def _compute_metrics(preds, scores, labels):
    all_possible_labels = np.unique(labels)
    roc_scores = {}
    for one_label_vs_rest in all_possible_labels:
        one_vs_rest_labels = 1 * (labels == one_label_vs_rest)
        one_vs_rest_scores = scores[:, one_label_vs_rest]
        roc_scores[f"eval_roc_auc_{one_label_vs_rest}_vs_rest"] = roc_auc_score(
            one_vs_rest_labels, one_vs_rest_scores
        )

    report = classification_report(labels, preds, output_dict=True)
    flatten_dict = {}  # a dictionary to hold all metrics
    for key in report:
        if isinstance(report[key], dict):
            for key2 in report[key]:
                flatten_dict[f"eval_{key}_{key2}"] = report[key][key2]
        else:
            flatten_dict[f"eval_{key}"] = report[key]

    flatten_dict.update(roc_scores)
    average_roc = sum([roc_scores[el] for el in roc_scores]) / len(roc_scores)
    flatten_dict["eval_roc_auc_average"] = average_roc

    return flatten_dict


def compute_metrics_causal_lm(predictions):
    assert all([len(el) == 1 for el in predictions.predictions])
    assert all([len(el) == 1 for el in predictions.label_ids])
    logits = np.concatenate([el[0] for el in predictions.predictions])
    labels = np.concatenate([el[0] for el in predictions.label_ids])
    preds = np.argmax(logits, axis=1)
    scores = softmax(logits, axis=1)

    assert len(labels) == len(preds)
    return _compute_metrics(preds, scores, labels[:, 0])


def compute_metrics(predictions=None):
    preds = np.argmax(predictions.predictions, axis=1)
    scores = softmax(predictions.predictions, axis=1)
    labels = predictions.label_ids[:, 0]
    return _compute_metrics(preds, scores, labels)


class CustomDataset(IterableDataset):
    """
    An object that handles loading data from parquet files containing tables formatted like:

    person_id  | sorted_event_tokens                                      | day_position_tokens  |
    -----------------------------------------------------------------------------------------------
    0          | ["AGE:25", "ETHNICITY:UNK", "GENDER:F", "HCPCS:pr_a",    | [0, 0, 0, 1, 2, 3]   |
                      "ICD10CM:dx_a", "LOINC:lx_a:LOW"]
    1          | ["AGE:26", "ETHNICITY:UNK", "GENDER:F", "HCPCS:pr_a",    | [0, 0, 0, 1, 1, 1]   |
    ...           "ICD10CM:dx_a", "LOINC:lx_a:LOW"]                         ...

    """

    def __init__(
        self,
        data_location,
        tokenizer,
        max_length=None,
        tmp_data_location=os.path.join(PROJECT_DIR, "TMP_training_data"),
        subsample=None,
        include_person_ids=False,
        shuffle=True,
        percentile_split=None,
        convert_to_ids=True,
        wait_for_download=False,
        max_position_embedding=None,
        trim_randomly=False,
        use_cli_file_download=True,
    ):
        """
        Args:
            data_location (string):
                A directory containing the parquet files to load data from.
                The data will be loaded from all files matching
                {data_location}/*.parquet.
                This class also supports loading data directly to s3.
                The parquet file is assumed to have a table of the following form:
                person_id (int), sorted_event_tokens (array<string>),
                day_position_tokens (array<int>).
            tokenizer (tokenizers.models.Model):
                The tokenizer model to handle tokenization
            max_length (int):
                The sequence length to pad all outputs to. e.g.
            tmp_data_location (string):
                If the data_location is on s3, and not local,
                then downlaod the data from s3 to this directory.
            subsample (float < 1.0):
                The fraction of the dataset to subsample
            include_person_ids:
                Whether to include person ids in the loaded data.
            shuffle:
                Whether to shuffle the dataframes before returning.
            percentile_split:
                The number of percentiles expected in the training data.
            convert_to_ids:
                Whether or not to apply the tokenizer to the output. i.e. to convert to input_ids
            wait_for_download:
                If this worker is not responsible for downloading the data,
                the wait for the data files to appear from s3
            max_position_id:
                The maximum allowable position_id when loading data.
            trim_randomly:
                If true, randomly trim the sequence of medical events to be a maximum
                length of max_length and be shorter than the maximum allowable position_id
        """

        self.wait_for_download = wait_for_download
        self.data_location = data_location
        assert self.data_location.endswith("/")
        self.tokenizer = tokenizer
        self.max_length = max_length
        if self.max_length is None:
            self.max_length = 100000000
        self.tmp_data_location = tmp_data_location
        self.subsample = subsample
        if not os.path.exists(tmp_data_location):
            os.makedirs(tmp_data_location)
        self.include_person_ids = include_person_ids
        self.shuffle = shuffle
        self.percentile_split = percentile_split
        self.convert_to_ids = convert_to_ids
        self.max_position_embedding = max_position_embedding
        self.trim_randomly = trim_randomly
        self.use_cli_file_download = use_cli_file_download

        super().__init__()

        if (
            self.use_cli_file_download
            and not self.wait_for_download
            and "s3://" in self.data_location
        ):
            recursively_download_files_from_s3(
                self.data_location, self.tmp_data_location, use_cli=True
            )

        self.files = sorted([self.download_file_if_needed(f) for f in self.get_files()])

    def handle_multi_processing(self, files):
        """
        Determine which files this worker is responsible for.
        If running w/o multiprocessing, then just read all files.
        If running multiprocessing then check which worker
        this is using torch.utils.data.get_worker_info().
        Make sure not to read the same files as other workers
        to avoid duplicates. Check the documentation here:
        https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset
        """
        worker_info = torch.utils.data.get_worker_info()
        start = 0
        end = len(files)
        if worker_info is not None:  # in a worker process
            per_worker = int(math.ceil((end - start) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = start + worker_id * per_worker
            iter_end = min(iter_start + per_worker, end)
            logger.debug(
                f"Worker {worker_info.id} handling files {files[iter_start:iter_end]}"
            )
        else:
            iter_start = start
            iter_end = end
        return files[iter_start:iter_end]

    def get_files(self):
        """
        Return a list of all data parquet files found in the directory self.data_location
        """
        if self.data_location.startswith("s3://"):
            bucket, prefix = split_s3_file(self.data_location)
            s3 = boto3.resource("s3")
            my_bucket = s3.Bucket(bucket)
            files = [
                f"s3://{bucket}/{my_bucket_object.key}"
                for my_bucket_object in my_bucket.objects.filter(Prefix=prefix)
                if my_bucket_object.key.endswith(".parquet")
            ]
            logger.debug(f"Found files {files}")
        else:
            files = glob.glob(os.path.join(self.data_location, "*.parquet"))
        files = sorted(
            files
        )  # sort them to make sure each worker has the same files when handling multiprocessing
        return files

    def download_file_if_needed(self, data_location):
        if not self.wait_for_download:
            if data_location.startswith("s3://"):
                local_fname = os.path.join(
                    self.tmp_data_location, os.path.split(data_location)[-1]
                )
                if not os.path.exists(local_fname):
                    logger.debug(f"downloading file {data_location}")
                    bucket, key = split_s3_file(data_location)
                    boto3.client("s3").download_file(bucket, key, local_fname)
                else:
                    logger.debug(
                        f"Reading local copy of this data file {data_location} at {local_fname}"
                    )
            else:  # this is local data file. Don't download it. Just read it.
                local_fname = data_location
        else:
            local_fname = os.path.join(
                self.tmp_data_location, os.path.split(data_location)[-1]
            )
            while not os.path.exists(local_fname):
                time.sleep(
                    10
                )  # wait 10 seconds for this file to appear. Another job should be downloading it!

        return local_fname

    def iterate_files(self, no_randomness=False):
        files = self.handle_multi_processing(self.files)
        if self.shuffle and not no_randomness:
            random.shuffle(files)
        for f in files:
            yield f

    def get_data_length(self):
        total = 0
        for local_fname in self.iterate_files():
            if self.subsample is None:
                total += ParquetFile(local_fname).metadata.num_rows
            else:
                total += int(
                    ParquetFile(local_fname).metadata.num_rows * self.subsample
                )
        return total

    def __len__(self):
        return self.get_data_length()

    def stable_sort_with_baseline_priority(
        self, row, position_key="day_position_tokens"
    ):
        """
        Stably sort all arrays in the dictionary, ensuring position_id=0 elements come first. Do not change the order of any other tokens.
        Args:
            row: Dictionary of numpy arrays
            position_key: Key for the position_id array
        Returns:
            Dictionary with sorted arrays
        """
        # Create a primary key: 0 for position_id=0, 1 for everything else
        position_array = row[position_key]
        primary_key = np.where(position_array == 0, 0, 1)

        # Create a secondary key: original indices to maintain stable ordering
        secondary_key = np.arange(len(position_array))

        # Use lexsort for stable sorting (sorts by last key first, then previous keys)
        # This gives us a stable sort where position_id=0 elements come first
        sort_indices = np.lexsort((secondary_key, primary_key))

        # Apply the sorting to all arrays in the dictionary
        sorted_dict = {}
        for key, array in row.items():
            if isinstance(array, np.ndarray) and len(array) == len(sort_indices):
                sorted_dict[key] = array[sort_indices]
            else:
                # Keep non-array or differently sized items unchanged
                sorted_dict[key] = array

        return sorted_dict

    def random_trim(self, row):
        if self.max_length is None:
            return row

        if len(row["day_position_tokens"]) <= self.max_length:
            return row

        row = self.stable_sort_with_baseline_priority(row)

        baseline_tokens = sum(1 * (row["day_position_tokens"] == 0))
        events_length = self.max_length - baseline_tokens

        starting_random_time = baseline_tokens + random.randint(
            0, len(row["day_position_tokens"]) - events_length
        )
        ending_random_time = starting_random_time + events_length

        index = np.arange(0, len(row["day_position_tokens"]))
        mask = (index < baseline_tokens) | (
            (index >= starting_random_time) & (index < ending_random_time)
        )

        for key in row.keys():
            if isinstance(row[key], np.ndarray):
                row[key] = row[key][mask]

        non_demographic_tokens = row["day_position_tokens"] != 0
        if np.any(non_demographic_tokens):
            row["day_position_tokens"][non_demographic_tokens] = (
                row["day_position_tokens"][non_demographic_tokens]
                - min(row["day_position_tokens"][non_demographic_tokens])
                + 1
            )

        row = self.trim_to_max_position_id(row)

        to_return = self.correct_age(row)
        return to_return

    def correct_age(self, row):
        """
        After truncating the row to stay within the max_position_embedding and max_sequence_length,
        the age may not correspond to the first medical event in the series.
        Correct the person's age using the date information of their birth date and the
        first medical event in the window.
        """
        event_token_indices = [
            i
            for i, el in enumerate(row["sorted_event_tokens"])
            if row["day_position_tokens"][i] != 0
        ]

        # cases where the first prediction event happened with only baseline information available.
        if not len(event_token_indices) > 0:
            assert np.all(row["day_position_tokens"] == 0)
            return row

        first_event_token_index = event_token_indices[0]
        first_event_date = row["date"][first_event_token_index]
        birth_date = row["date"][0]

        # Convert both dates to days since the epoch
        first_event_days = first_event_date.astype("datetime64[D]").astype(int)
        birth_days = birth_date.astype("datetime64[D]").astype(int)

        # Calculate age in years
        age = (first_event_days - birth_days) // 365

        age_token = f"AGE:{age}"
        age_tokens = [el.startswith("AGE") for el in row["sorted_event_tokens"]]

        row["sorted_event_tokens"][age_tokens] = age_token
        return row

    def trim_to_max_length(self, row):
        """
        Trim rows of data so that arrays are shorter than max_sequence_length.

        Example:
        If max_sequence_length is 4:

        >>> data = {
                "person_id": 0,
                "sorted_event_tokens": [
                    "AGE:25", "ETHNICITY:UNK", "GENDER:F",
                    "LOINC:1234-1:1.5:g", "ICD10CM:dx_a", "ICD10CM:dx_b"
                ],
                "label": 1,
            }

        >>> self.trim_to_max_length(data)
        {
            "person_id": 0,
            "sorted_event_tokens": ["AGE:25", "ETHNICITY:UNK", "GENDER:F", "ICD10CM:dx_b"]
        }
        """
        if self.max_length is None:
            return row

        if len(row["day_position_tokens"]) < self.max_length - 1:
            return row

        row = self.stable_sort_with_baseline_priority(row)

        number_of_baseline = sum(1 * (row["day_position_tokens"] == 0))

        indices = np.arange(0, len(row["day_position_tokens"]))[::-1]
        include = (indices < ((self.max_length - 1) - number_of_baseline)) | (
            row["day_position_tokens"] == 0
        )

        for key in row.keys():
            if isinstance(row[key], np.ndarray):
                row[key] = row[key][include]

        original_baseline = row["day_position_tokens"] == 0
        not_baseline = np.logical_not(original_baseline)
        if np.any(not_baseline):
            row["day_position_tokens"][not_baseline] = (
                row["day_position_tokens"][not_baseline]
                - min(row["day_position_tokens"][not_baseline])
                + 1
            )

        return self.correct_age(row)

    def trim_to_max_position_id(self, row):
        """
        Trim rows of data so that arrays are shorter than max_sequence_length.

        Example:
        If max_position_id is 10:

        >>> data = {
                "person_id": 0,
                "sorted_event_tokens": [
                    "AGE:25", "ETHNICITY:UNK", "GENDER:F",
                    "LOINC:1234-1:1.5:g", "ICD10CM:dx_a", "ICD10CM:dx_b", "ICD10CM:dx_c"
                ],
                "day_position_tokens": [0, 0, 0, 1, 2, 101, 110],
                "label": 1,
            }

        >>> self.trim_to_max_length(data)

        {
            "person_id": 0,
            "sorted_event_tokens": [
                "AGE:25", "ETHNICITY:UNK", "GENDER:F", "ICD10CM:dx_b", "ICD10CM:dx_c"
            ],
            "day_position_tokens": [0, 0, 0, 1, 10],
            "label": 1,
        }
        """
        if self.max_position_embedding is None:
            return row

        if max(row["day_position_tokens"]) < self.max_position_embedding:
            return row

        row = self.stable_sort_with_baseline_priority(row)

        position_ids = row["day_position_tokens"]
        original_baseline_tokens = position_ids == 0

        new_position_ids = (
            (position_ids - max(position_ids)) + self.max_position_embedding - 1
        )
        new_position_ids[original_baseline_tokens] = 0

        mask = new_position_ids >= 0
        for key in row.keys():
            if isinstance(row[key], np.ndarray):
                row[key] = row[key][mask]
        row["day_position_tokens"] = new_position_ids[mask]

        non_demographic_tokens = row["day_position_tokens"] != 0
        if np.any(non_demographic_tokens):
            row["day_position_tokens"][non_demographic_tokens] = (
                row["day_position_tokens"][non_demographic_tokens]
                - min(row["day_position_tokens"][non_demographic_tokens])
                + 1
            )

        return self.correct_age(row)

    def drop_duplicates(self, row):
        """
        Drop events that happen on the same day and have the token.
        """
        same = np.zeros(len(row["sorted_event_tokens"])) < 1
        for column in ["sorted_event_tokens", "day_position_tokens"]:
            selection = row[column][:-1] == row[column][1:]
            same[:-1] = same[:-1] & selection
        same[-1] = False
        for key in row.keys():
            if isinstance(row[key], np.ndarray):
                row[key] = row[key][np.logical_not(same)]
        return row

    def iterate_dataframes(self, no_randomness=False):
        for local_fname in self.iterate_files(no_randomness=no_randomness):
            df = pd.read_parquet(local_fname)
            if self.shuffle and not no_randomness:
                df = df.sample(frac=1.0)
            elif self.subsample is not None and not no_randomness:
                df = df.sample(frac=self.subsample)
            logger.debug(
                f"The dataframe after reading the file was this long {len(df)}"
            )
            if len(df) == 0:
                continue
            yield df

    def process_data(self, row, add_cls_token=True):
        inputs = row["sorted_event_tokens"]
        position_ids = row["day_position_tokens"]

        # apply the filters, if they are defined
        if inputs[0] != "[CLS]" and add_cls_token:
            inputs = itertools.chain(["[CLS]"], inputs)  # add the cls token and encode
            inputs = [el for el in inputs]
            position_ids = [0] + list(position_ids)  # account for the cls token

        if self.percentile_split is not None:
            percentiles = [
                el.split(":")[-1] if el.count(":") > 1 else None for el in inputs
            ]
            inputs = [":".join(el.split(":")[0:2]) for el in inputs]
            assert all(
                [
                    int(el.split("-")[0]) % self.percentile_split == 0
                    for el in percentiles
                    if el is not None
                ]
            )
            encoded_percentiles = [
                int(el.split("-")[1]) // self.percentile_split if el is not None else 0
                for el in percentiles
            ]
            encoded_percentiles = np.array(encoded_percentiles)
            assert len(encoded_percentiles) == len(inputs)

        to_return = {}
        if self.convert_to_ids:
            encoded_inputs = np.array(
                [
                    self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(el)[0])
                    for el in inputs
                ]
            )
            encoded_inputs = encoded_inputs[
                0 : min(self.max_length, len(encoded_inputs))
            ]
            to_return["input_ids"] = encoded_inputs
        else:
            encoded_inputs = inputs
            encoded_inputs = encoded_inputs[
                0 : min(self.max_length, len(encoded_inputs))
            ]
            to_return["inputs"] = encoded_inputs

        position_ids = position_ids[0 : min(self.max_length, len(encoded_inputs))]
        to_return["attention_mask"] = np.ones(len(encoded_inputs), np.int8)
        to_return["position_ids"] = np.array(position_ids, np.int32)

        if self.percentile_split is not None:
            to_return["token_type_ids"] = encoded_percentiles[
                0 : len(encoded_inputs)
            ]  # pad them to the max length

        if self.include_person_ids:
            to_return["person_id"] = [row["person_id"]]

        if "age_in_years" in row:
            to_return["age_in_years"] = row["age_in_years"]

        return to_return

    def __iter__(self):
        """
        Iterate through the entire dataset, returning entries one-by-one
        """

        for df in self.iterate_dataframes():
            for i, row in df.iterrows():

                row = self.stable_sort_with_baseline_priority(row)

                if self.trim_randomly:
                    row = self.random_trim(row)

                row = self.trim_to_max_length(row)
                row = self.trim_to_max_position_id(row)
                row = self.correct_age(row)

                to_return = self.process_data(row)
                yield to_return


def resolve_sequence_length(model):
    if type(model) is BertModel or type(model) is BertForMaskedLM:
        max_sequence_length = 512
    elif type(model) is LongformerForMaskedLM or type(model) is LongformerModel:
        max_sequence_length = 4092
    else:
        raise ValueError(
            f"Unknown model type {model}. Please extend the code in {__file__} by specifying the "
            "max sequence length for the model."
        )
    return max_sequence_length


class LabelledDataset(CustomDataset):
    """
    A class to iterate through a dataset of labelled dataset for training a CLM model.
    """

    def __init__(self, labelled_data_column_name, *args, **kwargs):
        self.labelled_data_column = labelled_data_column_name
        super().__init__(*args, **kwargs)

    def __iter__(self):
        """
        Iterate through the entire dataset, returning entries one-by-one
        """
        for df in self.iterate_dataframes():
            for i, row in df.iterrows():
                row = {
                    key: row[key] for key in row.index
                }  # pandas series don't play well when indexing length 1 arrays. Convet to dict
                to_return = self.process_data(row)
                to_return["labels"] = np.array([row[self.labelled_data_column]])
                yield to_return


class ProportionalDataset(IterableDataset):
    """
    This class is meant to load two datasets at fixed proportion.
    i.e. if proportion_1 is 1 and proportion_2 is 10, then 10 data
    point from dataset 2 will be returned for every one data point from dataset_1
    """

    def __init__(self, datasets, proportions):
        assert len(datasets) == len(proportions)
        self.datasets = datasets
        total_proportions = sum(proportions)
        self.proportions = proportions
        self.samplings = [proportion / total_proportions for proportion in proportions]
        self.iterators = [iter(data) for data in self.datasets]
        self.cumulative_sum = np.cumsum(self.samplings)

    def __len__(self):
        return int(
            min([len(data) for data in self.datasets]) * sum(self.proportions)
        )  # This is how many iterations are needed to see the all samples of the smallest class.

    def __iter__(self):
        for i in range(0, len(self)):
            index = bisect.bisect_right(self.cumulative_sum, random.uniform(0.0, 1.0))
            try:
                yield next(self.iterators[index])
            except StopIteration:
                self.iterators[index] = iter(self.datasets[index])
                yield next(self.iterators[index])


class LabelledDataCollator:
    """
    A data collator that takes a set of data points from the CustomDataset defined above.
    It collates them into a batch and pads accordingly.
    """

    def __call__(self, lst_of_points):
        input_ids = [torch.tensor(item["input_ids"]) for item in lst_of_points]
        attention_mask = [
            torch.tensor(item["attention_mask"]) for item in lst_of_points
        ]
        position_ids = [torch.tensor(item["position_ids"]) for item in lst_of_points]
        labels = [torch.tensor(item["labels"]) for item in lst_of_points]

        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
        attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
        position_ids = pad_sequence(position_ids, batch_first=True, padding_value=0)
        labels = torch.stack(labels)

        to_return = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "labels": labels,
        }

        if "person_id" in lst_of_points[0]:
            person_ids = []
            for point in lst_of_points:
                person_ids += point["person_id"]
            to_return["person_id"] = person_ids

        if "token_type_ids" in lst_of_points[0]:
            token_type_ids = [
                torch.tensor(item["token_type_ids"]) for item in lst_of_points
            ]
            token_type_ids = pad_sequence(
                token_type_ids, batch_first=True, padding_value=0
            )
            to_return["token_type_ids"] = token_type_ids

        return to_return


def glob_parquet_files_to_pandas(directory):
    files = glob.glob(os.path.join(directory, "*.parquet"))
    frame = None
    for f in files:
        this_frame = pd.read_parquet(f)
        if frame is None:
            frame = this_frame
        else:
            frame = pd.concat([frame, this_frame])
    return frame


def clean_units(unit):
    # First, replace patterns like "cm3" with "cm**3"
    unit = re.sub(r"([a-zA-Z]+)(\d+)", r"\1**\2", unit)

    # Then, replace patterns like "cm*3" with "cm**3"
    unit = re.sub(r"([a-zA-Z]+)\*(\d+)", r"\1**\2", unit)

    return unit


class UnitConverter:
    """
    This class handles the conversions of units of LOINC lab tests to standard units.
    """

    def __init__(self, unit_conversions_dataframe, ureg=None):
        """
        unit_conversions_dataframe:
            A dataframe containing a column of lx_loinc_code, standard_unit,
            other_unit and conversion_factor.
            This Dataframe is used as a lookup table to convert a given lab test
            with a given unit to the standard unit, using the conversion factor.
        """
        self.unit_conversions = unit_conversions_dataframe
        if ureg is None:
            self.ureg = pint.UnitRegistry()
        self.unit_dict = self.get_unit_dict()
        self.cache = {}

    def get_unit_dict(self):
        conversion_dict = {}
        for i, row in self.unit_conversions.iterrows():
            if "lx_loinc_code" in row:
                conversion_dict[row["lx_loinc_code"]] = clean_units(
                    row["lx_result_unit"]
                )
            elif "lx_lab_code" in row:
                conversion_dict[row["lx_lab_code"]] = clean_units(row["lx_result_unit"])
            else:
                raise ValueError(
                    "Couldn't find the lab test column in the conversions data"
                )
        return conversion_dict

    def translate(self, value, loinc_code, other_unit):
        if loinc_code not in self.unit_dict:
            return None

        cache_str = f"CODE_{loinc_code}_UNIT_{other_unit}"
        if cache_str not in self.cache:
            conversion = handle_unit_conversion(
                self.ureg, self.unit_dict[loinc_code], clean_units(other_unit)
            )
            self.cache[cache_str] = conversion
        else:
            conversion = self.cache[cache_str]

        if conversion is None:
            return None

        try:
            float(value)
        except Exception as e:
            return None

        return float(value) * conversion


class UnitStandardizer:
    """
    This class handles the conversions of LOINC lab tests to standard percentile ranges.
    """

    def __init__(self, percentile_conversions_dataframe):
        """
        percentile_conversions_dataframe:
            A DataFrame containing a column `lx_loinc_code` and additional columns
            representing the boundaries for different percentiles.
        """
        self.percentiles_lookup = percentile_conversions_dataframe
        self.percentiles_dict = self.get_percentiles_dict()

    def get_percentiles_dict(self):
        percentiles_dict = {}
        for i, row in self.percentiles_lookup.iterrows():
            if "lx_loinc_code" in row:
                percentiles_dict[row["lx_loinc_code"]] = row["percentile_boundaries"]
            elif "lx_lab_code" in row:
                percentiles_dict[row["lx_lab_code"]] = row["percentile_boundaries"]
            else:
                raise ValueError(
                    "Couldn't find the lab test column in the percentiles data"
                )
        return percentiles_dict

    def percentile(self, value, loinc_code):
        if loinc_code not in self.percentiles_dict or value is None:
            return None
        for i, boundary in enumerate(self.percentiles_dict[loinc_code]):
            if boundary > value:
                return i + 1
        return len(self.percentiles_dict[loinc_code]) + 1

    def get_percentile_split_number(self):
        length = None
        for key in self.percentiles_dict:
            if length is None:
                length = len(self.percentiles_dict[key]) + 1
            else:
                assert length == len(self.percentiles_dict[key]) + 1
        return length


def keep_indices_from_row(row, indices):
    """
    Given a dictionary representing a row of a DataFrame, select only the indices of entries
    that are `np.ndarrays`.

    Args:
        row (dict of str, object): A dictionary representing a row of the dataset.
        indices (np.ndarray<int>): An array of indices used to select entries that are arrays
            from the row.

    Returns:
        dict: A dictionary with the same keys as `row`, but with `np.ndarray` entries filtered
        based on the selected indices.
    """
    row = row.copy()
    data_columns = [el for el in row.keys()]
    for column in data_columns:
        if isinstance(row[column], np.ndarray):
            row[column] = row[column][indices]
    return row


class MultiLabelDataset(CustomDataset):
    """
    A class to handle data loading of timeseries data labelled like
    [event:A, event:B, label:0, event:C, event:D, label:1] as

    timeseries, label = [event:A, event:B], [0]
    timeseries, label = [event:A, event:B, event:C, event:D], [1]

    N.B. The labels are expected to be contained in the timeseries data format expected by CustomDataset with
    integer labels like label:0, label:1, label:2, label:3, etc.

    This class also ensures that data is loaded in a randomized way when passing shuffle=True. i.e.
    that when iterating through the dataset, the data is randomly sorted.
    """

    def __init__(self, *args, label=None, **kwargs):
        """
        label: int. When not set to None, then iterate through the dataset only taking labels with label:{label}.
               This can be used in conjunction with the proportional dataloader to balance the ratio of each label relative to eachother.
        """
        super().__init__(*args, **kwargs)
        self.label = label

    def iterate_dataframes_and_files(self):
        for file in super().iterate_files(no_randomness=True):
            frame = pd.read_parquet(file)
            yield file, frame

    @property
    def person_label_file_lookup(self):
        """
        Store a lookup table of the file, person id and label count of each label.
        This is later used to load the data in a randomized way.
        """
        person_label_file_lookup = {}
        person_label_file_lookup["file"] = []
        person_label_file_lookup["person_id"] = []
        person_label_file_lookup["label"] = []
        person_label_file_lookup["label_count"] = []

        for file, frame in self.iterate_dataframes_and_files():
            for i, row in frame.iterrows():
                label_counter = 0
                for el in row["sorted_event_tokens"]:
                    if "LABEL" in el:
                        label = int(el.split(":")[1])
                        if label == self.label or self.label is None:
                            person_label_file_lookup["file"].append(file)
                            person_label_file_lookup["person_id"].append(
                                row["person_id"]
                            )
                            person_label_file_lookup["label"].append(label)
                            person_label_file_lookup["label_count"].append(
                                label_counter
                            )
                        label_counter += 1

        return pd.DataFrame.from_dict(person_label_file_lookup)

    def __len__(self):
        if self.label is not None:
            length = 0
            for _, frame in self.iterate_dataframes_and_files():
                for i, row in frame.iterrows():
                    length += sum(
                        [
                            1 * (el == f"LABEL:{self.label}")
                            for el in row["sorted_event_tokens"]
                        ]
                    )
            return length
        else:
            length = 0
            for _, frame in self.iterate_dataframes_and_files():
                for i, row in frame.iterrows():
                    length += sum(
                        [
                            1 * el.startswith("LABEL")
                            for el in row["sorted_event_tokens"]
                        ]
                    )
            return length

    def iterate_all(self):
        start = 0
        step = 500
        stop = step

        if self.shuffle:
            person_labels = self.person_label_file_lookup.sample(frac=1.0)
        else:
            person_labels = self.person_label_file_lookup

        while start < len(person_labels):
            batch = person_labels.iloc[start : min(stop, len(person_labels))]
            files = set(batch.file.values)
            person_ids = set(batch.person_id.values)
            file_to_frame = {}

            for f in files:
                frame = pd.read_parquet(f)
                frame = frame[frame["person_id"].isin(person_ids)]
                file_to_frame[f] = frame

            for i, metadata_row in batch.iterrows():
                query = f"person_id == {repr(metadata_row['person_id'])}"
                frame = file_to_frame[metadata_row["file"]].query(query)
                assert len(frame) == 1
                row = [el for i, el in frame.iterrows()][0]
                row = {
                    key: row[key] for key in row.index
                }  # pandas series don't play well when indexing length 1 arrays. Convet to dict

                row, label = self.get_nth_label(row, metadata_row["label_count"])

                assert label == metadata_row["label"]

                yield row, label

            start += step
            stop += step

    def get_nth_label(self, row, n):
        """
        retrieve the n'th label from a row
        """
        row = row.copy()
        indices_to_keep = []
        label_counter = -1
        for index, el in enumerate(row["sorted_event_tokens"]):
            if el.startswith("LABEL"):
                label_counter += 1
                if label_counter == n:
                    label = int(el.split(":")[1])
                    return keep_indices_from_row(row, indices_to_keep), label
            else:
                indices_to_keep.append(index)

    def __iter__(self):
        for to_return, label in self.iterate_all():
            if self.label is not None and self.label != label:
                continue
            to_return = self.drop_duplicates(to_return)
            to_return = self.trim_to_max_length(to_return)
            to_return = self.trim_to_max_position_id(to_return)
            to_return = self.process_data(to_return)
            to_return["labels"] = [label]

            yield to_return


def trim_meds_to_max_length(patient, event_length_max=None):
    """
    Given a patient in meds format, trim their medical history to the maximum input length allowed by the model.
    Trim the patient from the left, preserving demographic and age features.
    Trimming from the left is important to ensure the most recent medical events are used to make the next prediction.
    """

    if event_length_max is None:
        return patient

    patient_events = patient["events"]

    # flatten the medical events
    flat_events = []
    for event in patient_events:
        time = event["time"]
        for measurement in event["measurements"]:
            flat_event = {"time": time}
            assert "time" not in measurement
            for key in measurement:
                flat_event[key] = measurement[key]
            flat_events.append(flat_event)

    birth_event = None
    demographic_tokens = ["gender", "ethnicity", "race", "region"]
    demographic_events = []
    for event in flat_events:
        if event["code"] == "SNOMED/184099003":
            birth_event = event
        if any([el in event["code"].lower() for el in demographic_tokens]):
            demographic_events.append(event)

    non_demographic_events = []
    for event in flat_events:
        if event["code"] == birth_event["code"]:
            continue
        if any([event["code"] == el["code"] for el in demographic_events]):
            continue
        non_demographic_events.append(event)

    # now truncate the events
    baseline_events_length = 1 + len(
        demographic_events
    )  # the birth event + the demographic events
    truncation_length = min(
        event_length_max - baseline_events_length, len(non_demographic_events)
    )
    assert truncation_length >= 0

    all_events = [birth_event] + demographic_events

    if truncation_length > 0:
        all_events = all_events + non_demographic_events[-1 * truncation_length :]

    # in case the demographic events and birth event happened after some of the diagnoses, re-sort the events
    all_events = sorted(all_events, key=lambda x: x["time"])

    # convert back into the meds format
    formatted_events = []
    current_time = None
    measurements = []
    for event in all_events:
        if event["time"] != current_time and current_time is not None:
            formatted_events.append(
                {
                    "time": current_time,
                    "measurements": measurements,
                }
            )
            measurements = []
            current_time = event["time"]

        elif current_time is None:
            current_time = event["time"]

        measurements.append({key: event[key] for key in event if "time" != key})

    # the final set of events.
    formatted_events.append(
        {
            "time": current_time,
            "measurements": measurements,
        }
    )

    patient["events"] = formatted_events

    return patient


SEEN_OUTLIERS = set()


def convert_to_meds_format(row, event_length_max=None):
    """
    Given a row of data from the table containing keys for sorted_event_tokens, dates,
    and patient_ids, return this patient's data in the MEDS schema:
    https://github.com/Medical-Event-Data-Standard/meds.
    """
    global SEEN_OUTLIERS

    patient = {}

    patient["patient_id"] = row["person_id"]
    current_date = None
    current_dates_data = {}
    events = []
    seen_events = set()
    total_events = 0

    for event, date in zip(row["sorted_event_tokens"], row["date"]):
        # the model uses this birth code to determine the birthday.
        if event == "SNOMED/3950001":
            event = "SNOMED/184099003"

        date = pd.to_datetime(date).to_pydatetime()

        if current_date is None or date.date() != current_date.date():
            if current_date is not None:
                events.append(current_dates_data)

            current_date = date
            current_dates_data = {}
            current_dates_data["time"] = current_date
            current_dates_data["measurements"] = []
            seen_events = set()

        if event not in seen_events:
            event_value = None
            event_text = None

            if ":" in event:
                original_event = event
                split_event = event.split(":")
                event, unparsed_event_value = split_event[0], ":".join(split_event[1:])
                if len(split_event) > 2 and original_event not in SEEN_OUTLIERS:
                    logger.warning(
                        f"Strange event {original_event} was parsed as follows... event: {event}, text or value: {unparsed_event_value}"
                    )
                    SEEN_OUTLIERS.add(original_event)

                try:
                    event_value = float(unparsed_event_value)
                    event_text = None
                except Exception as e:
                    event_value = None
                    event_text = str(unparsed_event_value)

            event_data = {"code": event}
            event_data["numeric_value"] = event_value
            event_data["text_value"] = event_text

            current_dates_data["measurements"].append(event_data)
            seen_events.add(event)
            total_events += 1

    events.append(current_dates_data)
    patient["events"] = events
    return trim_meds_to_max_length(patient, event_length_max=event_length_max)


class MultiLabelDatasetMedsFormat(MultiLabelDataset):
    def __iter__(self):
        for to_return, label in self.iterate_all():
            row = to_return
            if self.label is not None and self.label != label:
                continue

            # convert the data to the MEDS format.
            # group events by their timestamp.
            to_return = convert_to_meds_format(
                to_return, event_length_max=self.max_length
            )

            to_return = {"batch": to_return, "labels": [label]}

            if self.include_person_ids:
                to_return["person_id"] = [row["person_id"]]

            yield to_return


class MultiLabelledDatasetWithLabs(MultiLabelDataset):
    def __init__(
        self,
        unit_converter,
        unit_standardizer,
        *args,
        **kwargs,
    ):
        """
        A dataloader that handles the loading of data from parquet files containing
        tables formatted like:

        Example table:

        person_id | sorted_event_tokens    | day_position_tokens| date
        --------- | -----------------------| --------------------- | ----------------------------
        0         | [                      | [0, 0, 0, 1, 2, 3, 4] | ["29-11-1994", "29-11-1994",
                  |   "AGE:25",            |                       | "29-11-1994", "29-11-1994",
                  |   "ETHNICITY:UNK",     |                       | "01-10-2020", "02-10-2020",
                  |   "GENDER:F",          |                       | "03-10-2020", "04-10-2020"]
                  |   "label:0",
                  |   "LOINC:1234-1:1.5:g",
                  |   "ICD10CM:dx_a",
                  |   "LOINC:lx_a:LOW"
                  | ]                      |                       |

        Args:
            unit_converter (UnitConverter): Defined in the clm_ler package.
                A class that handles conversions of units and lab test values
                like "LOINC:1234-1:1.5:g" to that expected by the model.
            unit_standardizer (UnitPercentiler): Defined in the clm_ler package.
                A class that handles conversions of lab test results to percentiles
                indexed from 1 - N, where N is the number of percentiles.
            *args: Arguments passed to the clm_ler.data_processing.data_processing_utils.
                CustomDataset init function.
            label (Union[None, int]): If set to an integer, then only return labeled data with the
                corresponding label, skipping the rest.
            **kwargs: Additional args passed to the clm_ler.data_processing.
                data_processing_utils.CustomDataset init function.
        """
        super().__init__(*args, **kwargs)

        self.unit_converter = unit_converter
        self.unit_standardizer = unit_standardizer

        if self.unit_standardizer is not None:
            self.percentile_split = unit_standardizer.get_percentile_split_number()
        else:
            self.percentile_split = None

    def convert_labs_with_units_to_percentiles(self, row):
        """
        Iterate through all events in the row of data and convert LOINC codes
        and their values and units to those expected by the model using the unit_converter
        and then to the percentiles expected by the model using the unit_standardizer.

        If no percentile or converted value can be derived for a given loinc lab test, skip it.
        """
        indices_to_keep = []
        for index, el in enumerate(row["sorted_event_tokens"]):
            if el.startswith("LOINC"):
                lab_test_split = tuple(row["sorted_event_tokens"][index].split(":"))
                if len(lab_test_split) != 4:
                    continue
                source, loinc_code, value, unit = lab_test_split
                converted_value = self.unit_converter.translate(value, loinc_code, unit)
                percentile = self.unit_standardizer.percentile(
                    converted_value, loinc_code
                )
                if percentile is None:
                    continue
                else:
                    percentile_string = (
                        f"{str(self.percentile_split * (percentile - 1))}-"
                    )
                    percentile_string += f"{str(self.percentile_split * percentile)}"

                    row["sorted_event_tokens"][index] = ":".join(
                        ["LOINC", loinc_code, percentile_string]
                    )
                    indices_to_keep.append(index)
            else:
                indices_to_keep.append(index)

        final_row = keep_indices_from_row(row, indices_to_keep)
        return final_row

    def __iter__(self):
        for to_return, label in self.iterate_all():
            to_return = self.convert_labs_with_units_to_percentiles(to_return)
            if self.label is not None and self.label != label:
                continue
            to_return = self.drop_duplicates(to_return)
            to_return = self.trim_to_max_length(to_return)
            to_return = self.trim_to_max_position_id(to_return)
            to_return = self.process_data(to_return)
            to_return["labels"] = [label]
            yield to_return


def take_batch_to_device(batch, device, skip_keys=[]):
    if isinstance(batch, dict):
        new_batch = {}
        for key in batch.keys():
            if key in skip_keys:
                continue
            new_batch[key] = take_batch_to_device(
                batch[key], device, skip_keys=skip_keys
            )
        return new_batch
    elif isinstance(batch, list):
        new_batch = [
            take_batch_to_device(el, device, skip_keys=skip_keys) for el in batch
        ]
        return new_batch
    else:
        return batch.to(device)


def benchmark_model_for_classification(
    model,
    test_dataset,
    eval_batch_size=128,
    device=None,
    causal_lm=False,
    collator=LabelledDataCollator(),
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=eval_batch_size,
        collate_fn=collator,
    )

    all_predictions = []
    all_labels = []
    all_person_ids = []
    sm = Softmax(dim=1)
    for batch in tqdm(
        test_dataloader,
        desc="Processing Batches",
        total=len(test_dataset) // eval_batch_size,
    ):
        with torch.no_grad():
            batch, person_id = (
                take_batch_to_device(batch, device, skip_keys=["person_id"]),
                batch["person_id"],
            )
            result = model(**batch)

            if causal_lm:
                result.logits = result.logits[0]

            predictions = sm(result.logits).cpu().numpy()
            all_predictions.append(predictions)

            if not causal_lm:
                all_labels.append(batch["labels"].cpu().numpy())
            else:
                all_labels.append(batch["labels"][0].cpu().numpy())

            all_person_ids.append(np.array(person_id))

    all_predictions = np.concatenate(all_predictions)
    all_labels = np.concatenate(all_labels)
    all_person_ids = np.concatenate(all_person_ids)

    return all_predictions, all_labels, all_person_ids
