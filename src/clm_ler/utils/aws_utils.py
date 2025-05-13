import os
import boto3
import re

from clm_ler.config.config import (
    AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY,
    AWS_DEFAULT_REGION,
)
from clm_ler.utils.logger import setup_logger

logger = setup_logger()


def check_if_s3_dir_exists(directory):
    """
    Checks if a specified directory exists in an S3 bucket by verifying the presence
    of at least one object with the directory prefix.

    Parameters:
    - directory (str): S3 directory path in the format "s3://bucket-name/path/to/directory/"
                       Ensure the path ends with a slash (/).

    Returns:
    - bool: True if the directory exists, False otherwise.

    Example:
    >>> check_if_s3_dir_exists("s3://mybucket/mydirectory/")
    True
    """
    s3 = get_s3_client()
    bucket, key = split_s3_file(directory)
    print(f"Listing in bucket {bucket} and key {key}")
    response = s3.list_objects_v2(Bucket=bucket, Prefix=key, MaxKeys=1)
    directory_exists = "Contents" in response
    return directory_exists


def get_s3_client():
    """
    Get an s3 client using environment variables for AWS access
    """
    if AWS_ACCESS_KEY_ID is not None:
        s3_client = boto3.client(
            "s3",
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            region_name=AWS_DEFAULT_REGION,  # specify the region here
        )
    else:
        s3_client = boto3.client(
            "s3",
        )
    return s3_client


def split_s3_file(s3_filename):
    """
    Given a s3_filename string of the form s3://{bucket_name}/{this/is/the/key}
    return the bucket and key as a tuple: ("bucket_name", "this/is/the/key")
    """
    s3_filename_split = s3_filename.replace("s3://", "").split("/")
    bucket = s3_filename_split[0]
    key = "/".join(s3_filename_split[1:])
    return bucket, key


def download_file_from_s3(s3_filename, local_fname):
    """
    Given the s3 uri of a file, download it to local_fname on the local system's directory.
    """
    bucket, key = split_s3_file(s3_filename)
    return get_s3_client().download_file(bucket, key, local_fname)


def recursively_find_s3_files_and_local_location(s3_filename, local_fname):
    """ """
    logger.info(f"Searching for all files in {s3_filename}")
    directories = list_subdirectories(s3_filename)
    logger.info(f"Found {directories}")
    files = {}
    for dir in directories:
        dir = dir.rstrip("/").split("/")[-1]
        logger.info(f"Checking if {dir} is a subdirectory.")
        new_directory = os.path.join(s3_filename, dir)
        new_local_directory = os.path.join(local_fname, dir)
        if new_directory.endswith(".parquet"):
            files[new_directory] = new_local_directory
            continue

        sub_directories = list_subdirectories(new_directory)
        if len(sub_directories) == 0:
            files[new_directory] = new_local_directory
        else:
            files.update(
                recursively_find_s3_files_and_local_location(
                    new_directory, new_local_directory
                )
            )
    logger.info(f"Found {files}")
    return files


def check_local_and_remote_agree(s3_filename, local_fname):
    """ """
    directories = list_subdirectories(s3_filename)
    local_directories = [
        os.path.split(el)[0] for el in glob.glob(os.path.join(local_fname, "*"))
    ]
    for dir in directories:
        assert dir in local_directories
        new_remote_dir = os.path.join(s3_filename, dir)
        new_local_dir = os.path.join(local_fname, dir)
        check_local_and_remote_agree(new_remote_dir, new_local_dir)

    for dir in local_directories:
        assert dir in directories
        new_remote_dir = os.path.join(s3_filename, dir)
        new_local_dir = os.path.join(local_fname, dir)
        check_local_and_remote_agree(new_remote_dir, new_local_dir)


def recursively_download_files_from_s3(s3_filename, local_fname, use_cli=False):
    """ """
    logger.info(f"downloading {s3_filename} to {local_fname}")
    if use_cli:
        logger.info("Using the CLI instead!")
    if use_cli:
        if local_fname[-1] != "/":
            local_fname += "/"
        command = f"aws s3 sync {s3_filename} {local_fname}"
        logger.info(f"Running {command}")
        os.system(command)
        return

    files = recursively_find_s3_files_and_local_location(s3_filename, local_fname)
    for remote_file in files:
        local_file = files[remote_file]
        head, tail = os.path.split(local_file)
        if not os.path.exists(head):
            os.makedirs(head)
        download_file_from_s3(remote_file, local_file)


def upload_file_to_s3(s3_filename, local_fname):
    """
    Given a file located at local_filename at the local system's directory,
    upload it to the bucket and key given by s3_filename uri
    (e.g. s3://{bucket_name}/{this/is/the/key}).
    """
    bucket, key = split_s3_file(s3_filename)
    return get_s3_client().upload_file(local_fname, bucket, key)


def list_subdirectories(directory, directories_only=False):
    """
    List all directories of a bucket under a prefix on S3.

    Here is an example of using this function:

    >>> from clm_ler.utils.aws_utils import split_s3_file
    >>> ehr_shot_data = (
    ...     "s3://some_path/ehr_shot_data/"
    ...     "EHRSHOT_ASSETS/benchmark/"
    ... )
    >>> list_subdirectories(ehr_shot_data)
    [
        "some_path/ehr_shot_data/EHRSHOT_ASSETS/benchmark/chexpert/",
        "some_path/ehr_shot_data/EHRSHOT_ASSETS/benchmark/guo_icu/",
        "some_path/ehr_shot_data/EHRSHOT_ASSETS/benchmark/guo_los/",
        "some_path/ehr_shot_data/EHRSHOT_ASSETS/benchmark/guo_readmission/",
        "some_path/ehr_shot_data/EHRSHOT_ASSETS/benchmark/lab_anemia/",
        "some_path/ehr_shot_data/EHRSHOT_ASSETS/benchmark/lab_hyperkalemia/",
        "some_path/ehr_shot_data/EHRSHOT_ASSETS/benchmark/lab_hypoglycemia/",
        "some_path/ehr_shot_data/EHRSHOT_ASSETS/benchmark/lab_hyponatremia/",
        "some_path/ehr_shot_data/EHRSHOT_ASSETS/benchmark/lab_thrombocytopenia/",
        "some_path/ehr_shot_data/EHRSHOT_ASSETS/benchmark/new_acutemi/",
        "some_path/ehr_shot_data/EHRSHOT_ASSETS/benchmark/new_celiac/",
        "some_path/ehr_shot_data/EHRSHOT_ASSETS/benchmark/new_hyperlipidemia/",
        "some_path/ehr_shot_data/EHRSHOT_ASSETS/benchmark/new_hypertension/",
        "some_path/ehr_shot_data/EHRSHOT_ASSETS/benchmark/new_lupus/",
        "some_path/ehr_shot_data/EHRSHOT_ASSETS/benchmark/new_pancan/",
    ]
    """
    bucket_name, prefix = split_s3_file(directory)
    if len(prefix) > 0 and prefix[-1] != "/":
        prefix += "/"
    s3 = get_s3_client()
    paginator = s3.get_paginator("list_objects_v2")
    result = paginator.paginate(Bucket=bucket_name, Prefix=prefix, Delimiter="/")

    subdirectories = []
    for page in result:
        if "CommonPrefixes" in page:
            for prefix in page["CommonPrefixes"]:
                subdirectories.append(prefix["Prefix"])
        if "Contents" in page:
            for content in page["Contents"]:
                if directories_only:
                    continue
                subdirectories.append(content["Key"])

    subdirectories = [f"s3://{bucket_name}/{el}" for el in subdirectories]

    return subdirectories
