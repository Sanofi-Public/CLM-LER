"""
Get the environment variables
"""

import os

import boto3

AWS_PROFILE = os.getenv(
    "AWS_PROFILE",
    None,
)

profile_exists = False
if AWS_PROFILE is not None:
    try:
        profile_exists = boto3.Session(profile_name=AWS_PROFILE)
        profile_exists = True
    except Exception as e:
        print(
            f"Got error {e} when loading profile {AWS_PROFILE}. Trying env variables instead."
        )
        del os.environ["AWS_PROFILE"]
        AWS_PROFILE = None

if profile_exists:
    print(f"Profile is set to {AWS_PROFILE}")
    session = boto3.Session(profile_name=AWS_PROFILE)
    credentials = session.get_credentials()
    os.environ["AWS_ACCESS_KEY_ID"] = credentials.access_key
    os.environ["AWS_SECRET_ACCESS_KEY"] = credentials.secret_key
    os.environ["AWS_DEFAULT_REGION"] = session.region_name
    os.environ["AWS_REGION"] = session.region_name
    AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID", None)
    AWS_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", None)
    AWS_DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION", None)
    AWS_REGION = os.getenv("AWS_REGION", None)
else:
    AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID", None)
    AWS_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", None)
    AWS_DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION", None)
    AWS_REGION = os.getenv("AWS_REGION", None)
