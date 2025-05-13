import os

PROJECT_DIR = os.path.split(
    os.path.split(os.path.split(os.path.split(__file__)[0])[0])[0]
)[0]
INTERMEDIATE_FILES_S3_LOCATION = os.getenv(
    "INTERMEDIATE_FILES_S3_LOCATION",
    "<some-s3-path>",
)
AWS_ACCESS_KEY_ID = os.getenv("CLM_AWS_ACCESS_KEY_ID", None)
AWS_SECRET_ACCESS_KEY = os.getenv("CLM_AWS_SECRET_ACCESS_KEY", None)
AWS_DEFAULT_REGION = os.getenv("CLM_AWS_DEFAULT_REGION", "eu-west-1")
WANDB_API_KEY = os.getenv("WANDB_API_KEY", None)
WANDB_USERNAME = os.getenv("WANDB_USERNAME", None)
WANDB_ENTITY = os.getenv("WANDB_ENTITY", None)
WANDB_PROJECT = os.getenv("WANDB_PROJECT", None)
PROJECT_NAME = WANDB_PROJECT
