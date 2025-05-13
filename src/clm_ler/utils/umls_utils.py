"""A class to read UMLS data stored on s3"""

import os

import boto3
import numpy as np
import pandas as pd

from clm_ler.utils.aws_utils import get_s3_client, split_s3_file
from clm_ler.utils.logger import setup_logger

logger = setup_logger()

try:
    import pyspark

    PYSPARK_SUPPORT = True
    logger.info("Pyspark installation found!")
except ImportError:
    PYSPARK_SUPPORT = False
    logger.info("No pyspark installation available.")


import pyspark


project_path = os.path.split(os.path.split(__file__)[0])[0]
LOCAL_DATA_LOCATION = os.path.join(project_path, ".UMLS_TMP")


def _init_spark():
    """
    Create spark session.

    Args:
        testing (bool, optional): Flag indicating whether the spark session is for testing purposes.
        Defaults to False.

    Returns:
        pyspark.sql.SparkSession: The initialized Spark session.
    """
    spark = pyspark.sql.SparkSession.builder.getOrCreate()
    return spark


class UmlsRRFReader:
    """
    A class that handles the loading of data from UMLS into Pandas Dataframes.
    See this documentation about the files that this class reads:

    https://www.ncbi.nlm.nih.gov/books/NBK9685/
    """

    def __init__(
        self,
        remote_data_location="s3://some/path/to/UMLS_data/META/",
        local_data_location=LOCAL_DATA_LOCATION,
        spark=None,
    ):
        """
        Keyword Arguments :
            remote_data_location: A string containing the location of the UMLS RRF data
                files on s3
            local_data_location: A folder on the local machine where data will be
                downloaded from s3 to.
            spark:
                Pass the spark context if needed. Will make a default one otherwise.
        """
        self.remote_data_location = remote_data_location
        self.local_data_location = local_data_location
        if not os.path.exists(self.local_data_location):
            os.makedirs(self.local_data_location)

        # The MRFILES contains the column names for each file, along with other metadata.
        self.metadata = pd.read_csv(
            self.download_datafile("MRFILES.RRF"), delimiter="|", header=None
        )
        # The MRCOLS file contains information about columns in each file.
        self.column_descriptions = self.get_frame("MRCOLS.RRF", col_descriptions=False)
        if spark is None and PYSPARK_SUPPORT:
            logger.info(
                "Pyspark support available but no context passed. Creating a new one."
            )
            self.spark = _init_spark()
        else:
            self.spark = spark

    def download_datafile(self, fname):
        """
        Given a filename like "MRFILES.txt" in the UMLS language system,
        download it from the s3 location of the data (self.remote_data_location)
        to the local machine's directory at self.local_data_location.
        """
        local_path = os.path.join(self.local_data_location, fname)

        if self.remote_data_location is None:
            logger.debug(
                "The remote location was none. "
                "Assuming files are already at the local destination"
            )
            return local_path

        full_s3_path = os.path.join(self.remote_data_location, fname)
        bucket, key = split_s3_file(full_s3_path)
        if not os.path.exists(local_path):
            logger.info(f"File {fname} was not cached. Downloading to {local_path}.")
            get_s3_client().download_file(bucket, key, local_path)
        else:
            logger.info(f"File {fname} was already cached at {local_path}")
        return local_path

    def get_column_descriptions(self, column_names, filename):
        """Get the descriptions of the columns in a file."""
        # query the dataframe of column names for column descriptions
        queries = []
        for column_name in column_names:
            queries.append(f"(COL == '{column_name}' )")
        query_string = f"(FIL == '{filename}') and (" + " or ".join(queries) + ")"
        column_description_frame = self.column_descriptions.query(query_string)
        return column_description_frame

    def get_available_files(self):
        return self.metadata

    def get_spark_frame(
        self,
        filename,
        col_descriptions=False,
        usecols=None,
        selection=None,
    ):
        """
        Given a filename of a file in the UMLS, ensure that the file is downloaded to
        self.local_data_location. Load the datafile file as a pandas dataframe.

        Args:
            filename: filename of datafile from UMLS

        Keyword Args:
            col_descriptions: if True, return an additional dataframe containing a
                description of all columns in the main frame
            usecols: If None, return a dataframe containing all columns in the data
                file. If it is a list of strings, return a dataframe containing only
                these columns.
            selection: a string that will be passed to used to query the dataframe.

            Returns:
                A spark dataframe
                if col_descriptions is True, then return a tuple of Pandas dataframe and
                another frame providing descriptions of the columns.
        """
        if self.spark is None:
            raise ValueError(
                "Please initialize this class with a spark context or ensure that pyspark in installed."
            )

        column_names = (
            self.metadata.query(f"@self.metadata[0] == '{filename}'")
            .values[0][2]
            .split(",")
        )

        # Check if usecols are in column_names
        if usecols is not None:
            for c in usecols:
                assert c in column_names
        else:
            usecols = column_names

        if self.remote_data_location is not None:
            data_path = os.path.join(self.remote_data_location, filename).replace(
                "s3://", "s3a://"
            )
        else:
            data_path = os.path.join(self.local_data_location, filename)

        dataframe = self.spark.read.option("delimiter", "|").csv(
            data_path, header=False, inferSchema=True
        )
        columns = dataframe.columns
        logger.info(f"Dropping {columns[-1]}")
        dataframe = dataframe.drop(columns[-1])
        dataframe = dataframe.toDF(*column_names)  # Assign column names
        dataframe = dataframe.select(*usecols)

        if selection is not None:
            dataframe = dataframe.filter(selection)

        if col_descriptions:
            column_description_frame = self.get_column_descriptions(
                column_names, filename
            )
            return dataframe, column_description_frame
        else:
            return dataframe

    def get_frame(
        self,
        filename,
        col_descriptions=False,
        usecols=None,
        chunksize=None,
        selection=None,
    ):
        """
        Given a filename of a file in the UMLS, ensure that the file is downloaded to
        self.local_data_location. Load the datafile file as a pandas dataframe.

        Args:
            filename: filename of datafile from UMLS

        Keyword Args:
            col_descriptions: if True, return an additional dataframe containing a
                description of all columns in the main frame
            usecols: If None, return a dataframe containing all columns in the data
                file. If it is a list of strings, return a dataframe containing only
                these columns.
            chunksize: If None: load the entire dataframe at once. If set to some
                integer N, load the dataset in chunks of size N, apply the selection,
                and then concatenate the final frame. This can reduce the overall memory
                footprint of loading the dataset, since so little of it could pass
                selection.
            selection: a string that will be passed to used to query the dataframe.

        Returns:
            Pandas dataframe
            if col_descriptions is True, then return a tuple of Pandas dataframe and
            another frame providing descriptions of the columns.

        Example output:
        >>> reader.get_frame(
            "MRREL.RRF",
            col_descriptions=True,
            usecols=["CUI2", "CUI1", "SAB", "RELA", "REL"]
        )
        (              CUI1 REL      CUI2            RELA     SAB
            0         C0000005  RB  C0036775             NaN  MSHFRE
            1         C0000005  RB  C0036775             NaN     MSH
            2         C0000039  SY  C0000039  translation_of  MSHSWE
            3         C0000039  SY  C0000039  translation_of  MSHCZE
            4         C0000039  SY  C0000039  translation_of  MSHPOR
            ...            ...  ..       ...             ...     ...
            43842945  C5779496  RB  C4047263     has_version     SRC
            43842946  C5779497  RB  C4047262     has_version     SRC
            43842947  C5779498  RB  C4047261     has_version     SRC
            43842948  C5779499  RB  C5698419     has_version     SRC
            43842949  C5779500  RB  C4047260     has_version     SRC
            [43842950 rows x 5 columns],
                    COL                                                DES  REF  MIN  \
            6        AUI1                   Unique identifier for first atom  NaN    0
            8        AUI2                  Unique identifier for second atom  NaN    0
            26       CUI1                Unique identifier for first concept  NaN    8
            29       CUI2               Unique identifier for second concept  NaN    8
            73        CVF                                  Content view flag  NaN    0
            83        DIR                Source asserted directionality flag  NaN    0
            206      RELA                      Additional relationship label  NaN    0
            213       REL                                 Relationship label  NaN    2
            215        RG                                 Relationship group  NaN    0
            218       RUI                 Unique identifier for relationship  NaN    9
            224       SAB                                Source abbreviation  NaN    2
            237        SL                      Source of relationship labels  NaN    2
            242      SRUI          Source attributed relationship identifier  NaN    0
            247    STYPE1  The name of the column in MRCONSO.RRF that con...  NaN    3
            248    STYPE2  The name of the column in MRCONSO.RRF that con...  NaN    3
            286  SUPPRESS                                  Suppressible flag  NaN    1
        """
        column_names = (
            self.metadata.query(f"@self.metadata[0] == '{filename}'")
            .values[0][2]
            .split(",")
        )

        # if using a subset of the total columns, make sure that they are available.
        if usecols is not None:
            for c in usecols:
                assert c in column_names
            columns = usecols
        else:
            columns = column_names

        logger.info(f"Reading UMLS file {filename}")
        local_fname = self.download_datafile(filename)
        logger.info(f"Reading local copy of file {filename} at {local_fname}")

        if chunksize is not None:
            logger.info("Loading dataset in chunks.")
            iterable_reader = pd.read_csv(
                local_fname,
                names=column_names,
                delimiter="|",
                index_col=False,
                usecols=usecols,
                chunksize=chunksize,
            )
        else:
            logger.info("Loading dataset all at once")
            iterable_reader = [
                pd.read_csv(
                    local_fname,
                    names=column_names,
                    delimiter="|",
                    index_col=False,
                    usecols=usecols,
                )
            ]

        full_frame = []
        counter = 0
        for frame in iterable_reader:
            frame = frame[columns]
            if counter % 10 == 0:
                logger.debug(f"Read chunk {counter} of file {filename}")
                logger.debug(f"Applied selection {selection}")
            if selection is not None:
                frame = frame.query(selection)
            full_frame.append(frame)
            counter += 1

        full_frame = pd.concat(full_frame)

        if col_descriptions:
            column_description_frame = self.get_column_descriptions(
                column_names, filename
            )
            return full_frame, column_description_frame

        return full_frame
