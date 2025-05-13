import pyspark.sql.functions as f
import logging
from pyspark.sql.window import Window
from clm_ler.utils.utils import setup_logger
import pint
import pandas as pd
import binascii
from clm_ler.config.config import (
    INTERMEDIATE_FILES_S3_LOCATION,
    AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY,
)
import os
import boto3
import pyspark
import json

logger = setup_logger()

try:
    # propogage the credentials to the package
    import clm_ler.utils.umls_config as umls_config

    umls_config.AWS_ACCESS_KEY = AWS_ACCESS_KEY_ID
    umls_config.AWS_SECRET_KEY = AWS_SECRET_ACCESS_KEY
    from clm_ler.utils.umls_utils.data_reader import UmlsRRFReader
except Exception as e:
    logger.info("Couldn't import umls_tools. Skipping.")


def check_for_negative_days(dframe, day_column):
    exploded_df = dframe.select("person_id", f.explode(day_column).alias("day_tokens"))
    logger.info("Making sure there are no negative days")
    assert exploded_df.filter(f.col("day_tokens") < 0.0).count() == 0


def select_columns(
    yaml_args,
    patient_frame,
    diagnosis_frame,
    prescription_frame,
    procedure_frame,
    lab_frame,
):
    if patient_frame is not None:
        table_selection_args = yaml_args["tables"]
        patient_args = table_selection_args["patient"]
        patient_frame = patient_frame.selectExpr(*patient_args["select_expr"])

    if diagnosis_frame is not None:
        table_selection_args = yaml_args["tables"]
        diagnosis_args = table_selection_args["diagnosis"]
        diagnosis_frame = diagnosis_frame.selectExpr(*diagnosis_args["select_expr"])

    if prescription_frame is not None:
        table_selection_args = yaml_args["tables"]
        prescription_args = table_selection_args["prescription"]
        prescription_frame = prescription_frame.selectExpr(
            *prescription_args["select_expr"]
        )

    if procedure_frame is not None:
        table_selection_args = yaml_args["tables"]
        procedure_args = table_selection_args["procedure"]
        procedure_frame = procedure_frame.selectExpr(*procedure_args["select_expr"])

    if lab_frame is not None:
        table_selection_args = yaml_args["tables"]
        lab_args = table_selection_args["lab"]
        lab_frame = lab_frame.selectExpr(*lab_args["select_expr"])

    return (
        patient_frame,
        diagnosis_frame,
        prescription_frame,
        procedure_frame,
        lab_frame,
    )


def apply_yaml_selections(
    yaml_args,
    patient_frame,
    diagnosis_frame,
    prescription_frame,
    procedure_frame,
    lab_frame,
):
    if patient_frame is not None:
        table_selection_args = yaml_args["tables"]
        patient_args = table_selection_args["patient"]
        patient_frame = patient_frame.where(patient_args["filter_expr"])

    if diagnosis_frame is not None:
        table_selection_args = yaml_args["tables"]
        diagnosis_args = table_selection_args["diagnosis"]
        diagnosis_frame = diagnosis_frame.where(diagnosis_args["filter_expr"])

    if prescription_frame is not None:
        table_selection_args = yaml_args["tables"]
        prescription_args = table_selection_args["prescription"]
        prescription_frame = prescription_frame.where(prescription_args["filter_expr"])

    if procedure_frame is not None:
        table_selection_args = yaml_args["tables"]
        procedure_args = table_selection_args["procedure"]
        procedure_frame = procedure_frame.where(procedure_args["filter_expr"])

    if lab_frame is not None:
        table_selection_args = yaml_args["tables"]
        lab_args = table_selection_args["lab"]
        lab_frame = lab_frame.where(lab_args["filter_expr"])

    return (
        patient_frame,
        diagnosis_frame,
        prescription_frame,
        procedure_frame,
        lab_frame,
    )


def create_tokens(frame, token_column, token_type_column):
    frame = frame.withColumn(
        "general_event_token",
        f.concat(
            f.col(token_type_column),
            f.lit(":"),
            f.col(token_column),
        ),
    )
    return frame


def merge_frames(column_order, *args):
    assert len(args) > 0
    merged_frames = None
    for arg in args:
        if arg is None:
            continue
        if merged_frames is None:
            merged_frames = arg.select(*column_order)
        else:
            merged_frames = merged_frames.union(arg.select(*column_order))
    return merged_frames


def align_to_first_date(merged_frames):
    min_date = merged_frames.groupBy("person_id").agg(f.min("date").alias("first_date"))
    merged_frames = merged_frames.join(min_date, on="person_id")
    merged_frames = merged_frames.withColumn(
        "days_since_begin", f.datediff(f.col("date"), f.col("first_date")) + f.lit(1)
    )

    columns_after_merging = [
        "person_id",
        "general_event_token",
        "days_since_begin",
        "date",
    ]
    merged_frames = merged_frames.select(*columns_after_merging)
    return merged_frames


def handle_unit_conversion(ureg, base_unit, other_unit):
    """
    Converts one unit to another and returns the conversion factor.

    This function calculates the conversion factor between two units using the
    provided unit registry. If the units are the same, the conversion factor is 1.0.
    If the units are different, the function attempts to convert the `other_unit`
    to the `base_unit` and returns the conversion factor. If the conversion fails
    (due to an incompatible unit type or any other exception), the function
    returns `None`.

    Parameters:
    -----------
    ureg : pint.UnitRegistry
        A Pint unit registry used to perform the unit conversion.
    base_unit : str
        The base unit to which the other unit will be converted.
    other_unit : str
        The unit that needs to be converted to the base unit.

    Returns:
    --------
    float or None
        The conversion factor from `other_unit` to `base_unit`. If the units are
        identical, returns 1.0. If the conversion fails, returns `None`.
    """
    if base_unit == other_unit:
        return 1.0
    try:
        base_unit_ureg = ureg(base_unit)
        other_unit_ureg = ureg(other_unit)
        other_unit_ureg = other_unit_ureg.to(base_unit_ureg)
        conversion = other_unit_ureg.magnitude / base_unit_ureg.magnitude
    except Exception as e:
        conversion = None
    return conversion


def tmp_file_location():
    fname = binascii.hexlify(os.urandom(16)).decode()
    return os.path.join(INTERMEDIATE_FILES_S3_LOCATION, fname)


def write_pandas_to_tmp(pandas_frame):
    """
    Write a pandas dataframe to temporary storage on s3 and return the location of the file.
    This function assumes you have s3fs and pyarrow installed. If you have a local pyspark
    installation, this function will crate a temporary file to save to.
    """
    if AWS_ACCESS_KEY_ID is not None and AWS_SECRET_ACCESS_KEY is not None:
        # Write the DataFrame to S3 as a Parquet file
        s3_path = tmp_file_location()
        pandas_frame.to_parquet(
            s3_path,
            engine="pyarrow",
            index=False,
            storage_options={
                "key": AWS_ACCESS_KEY_ID,
                "secret": AWS_SECRET_ACCESS_KEY,
            },
        )
        return s3_path

    else:
        tmp_path = f"tmp_file_{tmp_file_location().split('/')[-1]}.parquet"
        pandas_frame.to_parquet(tmp_path)
        return tmp_path


def get_units_and_conversions(lab_frame):
    """
    Processes a lab DataFrame to identify and handle unit conversions for specific LOINC codes.

    Given a DataFrame containing columns `lx_lab_code` and `lx_result_unit`, this function
    creates a new DataFrame that manages conversions between different units for each LOINC code.

    The function performs the following steps:
    1. Selects and groups the LOINC codes and result units, filtering out invalid codes and
       formatting units for conversion.
    2. Processes units to handle cases where exponents are incorrectly formatted, ensuring
       compatibility with unit conversion libraries.
    3. Converts the resulting Spark DataFrame to a Pandas DataFrame, ordered by the frequency
       of unit occurrences.
    4. Creates a dictionary to map each LOINC code to its associated units and identifies the
       most common unit for each code.
    5. Utilizes the Pint library to calculate conversion factors between the most common unit
       and other units for each LOINC code.
    6. Constructs and returns a Pandas DataFrame containing the LOINC code, the base unit,
       other units, and their respective conversion factors.

    Parameters:
        lab_frame (DataFrame): A Spark DataFrame containing columns `lx_lab_code`
            and `lx_result_unit`.

    Returns:
        pandas.DataFrame: A DataFrame containing columns `lx_lab_code`, `lx_result_unit`,
            `lx_result_other_unit`, and `conversion_factor`, representing unit conversions
            for each LOINC code.
    """
    all_unit_code_combos = (
        lab_frame.select("lx_lab_code_type", "lx_lab_code", "lx_result_unit")
        .dropna()
        .groupBy("lx_lab_code_type", "lx_lab_code", "lx_result_unit")
        .count()
    )

    # People list exponents immediately after the unit e.g. as mm3, when it should be mm**3 to be
    # understandable by the unit converter.
    # Handle that conversion here. similarly, they do mm*3 when they mean mm**3.

    # cases like mm3
    all_unit_code_combos = all_unit_code_combos.withColumn(
        "processed_unit",
        f.regexp_replace("lx_result_unit", r"([a-zA-Z]+)(\d+)", r"$1**$2"),
    )
    # cases like mm*3
    all_unit_code_combos = all_unit_code_combos.withColumn(
        "processed_unit",
        f.regexp_replace("processed_unit", r"([a-zA-Z]+)\*(\d+)", r"$1**$2"),
    )

    # convert to a pandas dataframe, ordered by the frequency of that unit
    all_unit_code_combos_pandas = all_unit_code_combos.orderBy(
        "lx_lab_code", f.desc("count"), "processed_unit"
    ).toPandas()

    # keep track of all appearances of different units for a given loinc code and test
    codes_to_units = {}
    for i, row in all_unit_code_combos_pandas.iterrows():
        if row["lx_lab_code"] not in codes_to_units:
            codes_to_units[row["lx_lab_code"]] = []
        codes_to_units[row["lx_lab_code"]].append(
            (row["processed_unit"], row["lx_result_unit"])
        )

    code_to_conversion_to_unit = {}
    conversion_table = {}
    conversion_table["lx_lab_code"] = []
    conversion_table["lx_result_unit"] = []
    conversion_table["lx_result_other_unit"] = []
    conversion_table["conversion_factor"] = []

    code_to_base_unit = {
        code: codes_to_units[code][0] for code in codes_to_units
    }  # a dictionary of (loinc-code):(most common unit used in the dataset for this loinc code)

    # try to find unit conversions between the most common unit and the others.
    ureg = pint.UnitRegistry()
    for code in code_to_base_unit:
        conversions = []
        for other_unit_cleaned, other_unit in codes_to_units[code]:
            base_unit_cleaned, base_unit = code_to_base_unit[code]
            conversion = handle_unit_conversion(
                ureg, base_unit_cleaned, other_unit_cleaned
            )
            conversion_table["lx_lab_code"].append(code)
            conversion_table["lx_result_unit"].append(base_unit)
            conversion_table["lx_result_other_unit"].append(other_unit)
            conversion_table["conversion_factor"].append(conversion)

    pandas_frame = pd.DataFrame(conversion_table)
    return pandas_frame


def ensure_correct_coding(lab_frame):
    """
    Filter a lab frame to select entries with correctly formatted loinc codes
    """
    # a regex pattern to match loinc codes like {4 or more numbers}-{a number}. e.g. 9472-1
    pattern = r"^\d{4,}-\d{1}$"

    # Filter DataFrame using the regex pattern. remove all 0 codes.
    lab_frame = lab_frame.filter(
        lab_frame.lx_lab_code.rlike(pattern) | (lab_frame.lx_lab_code_type != "LOINC")
    ).filter("lx_lab_code!='0000-0' and lx_lab_code!='00000-0'")

    lab_frame = lab_frame.filter("lx_lab_code IS NOT NULL")

    return lab_frame


def ensure_numerical_units(lab_frame):
    """
    Filter a lab frame for having units and a value
    """
    lab_frame = lab_frame.dropna(subset=["lx_result_unit", "lx_result_val"])
    return lab_frame


def ensure_floating_number(lab_frame):
    # Assuming df is your DataFrame and 'string_column' is the column to convert
    lab_frame = lab_frame.withColumn(
        "lx_result_val_converted", f.col("lx_result_val").cast("double")
    )

    lab_frame = lab_frame.select(
        [el for el in lab_frame.columns if el != "lx_result_val"]
        + [f.col("lx_result_val_converted").alias("lx_result_val")]
    )

    # Filter out rows where the conversion to float failed (i.e., where float_column is null)
    lab_frame = lab_frame.dropna(subset=["lx_result_val"])

    return lab_frame


def process_lab_frame_into_percentiles(
    spark_context, lab_frame, yaml_args, percentiles_frame=None, conversions_frame=None
):
    """
    Processes a lab DataFrame to handle unit conversions and calculate percentiles for
    lab test results.

    This function performs the following operations:
    1. Retrieves unit conversion factors for the lab tests using `get_units_and_conversions`.
    2. Temporarily writes the conversion factors to a file and reads it back into a Spark DataFrame.
    3. Converts the lab test results into consistent units using the conversion factors.
    4. Filters the converted lab results based on specified minimum counts for LOINC codes
    and values, as defined in `yaml_args`.
    5. Calculates percentiles for the filtered lab test results.
    6. Converts the lab test results into their corresponding percentile values.

    Parameters:
        spark_context (SparkContext): The Spark context used for reading and processing data.
        lab_frame (DataFrame): A Spark DataFrame containing lab test results, including LOINC codes
            and result units.
        yaml_args (dict): A dictionary of arguments loaded from a YAML file, including:
            - `min_loinc_counts`: Minimum count of LOINC codes to include in filtering.
            - `min_loinc_value_count`: Minimum count of lab values for each LOINC code to include
            in filtering.

    Returns:
        tuple:
            - percentiled_labs_frame (DataFrame): A Spark DataFrame with lab test results converted
            to percentile values.
            - conversions_spark (DataFrame): A Spark DataFrame containing the unit conversions used.
            - percentiles (DataFrame): A Spark DataFrame containing the calculated percentiles for
            the lab test results.
    """

    before_clean_count = lab_frame.count()
    lab_frame = ensure_correct_coding(lab_frame)
    lab_frame = ensure_numerical_units(lab_frame)
    lab_frame = ensure_floating_number(lab_frame)

    after_clean_count = lab_frame.count()
    percent_passing = 0.0
    if before_clean_count > 0:
        percent_passing = 100.0 * after_clean_count / before_clean_count
    logger.info(
        "{:.02f} % of lab tests pass data quality selections: coding, "
        "having nan values and conversions to float.".format(percent_passing)
    )

    if conversions_frame is None:
        conversions = get_units_and_conversions(lab_frame)
        tmp_file_for_conversions = write_pandas_to_tmp(conversions)
        conversions_spark = spark_context.read.parquet(
            tmp_file_for_conversions.replace("s3//", "s3a://")
        )
    else:
        conversions_spark = conversions_frame

    original_test_count = lab_frame.count()
    converted_lab_frame = convert_units(lab_frame, conversions_spark)

    converted_lab_frame = converted_lab_frame.cache()
    logger.info("Showing the labs frame after unit conversions")
    converted_lab_frame.dtypes
    converted_lab_frame.show()
    lab_frame.unpersist()

    final_test_count = converted_lab_frame.count()
    percent_passing = 0.0
    if original_test_count > 0:
        percent_passing = (final_test_count / original_test_count) * 100.0
    logger.info(
        "{:.02f} % of lab tests passed unit conversions".format(percent_passing)
    )

    clean_converted_lab_frame = filter_lab_results(
        converted_lab_frame,
        yaml_args["min_loinc_counts"],
        yaml_args["min_loinc_value_count"],
    )

    if percentiles_frame is None:
        percentiles = calculate_percentiles(clean_converted_lab_frame)
    else:
        percentiles = percentiles_frame

    percentiled_labs_frame = convert_lab_tests_to_percentiles(
        clean_converted_lab_frame, percentiles
    )

    return percentiled_labs_frame, conversions_spark, percentiles


def process_lab_percentiles_into_tokens(percentiled_labs_frame):
    percentiled_labs_frame = percentiled_labs_frame.withColumn(
        "general_event_token",
        f.concat(
            f.col("lx_lab_code_type"),
            f.lit(":"),
            f.col("lx_lab_code"),
            f.lit(":"),
            f.col("percentile_group"),
        ),
    )
    lab_token_frame = percentiled_labs_frame.select(
        "general_event_token",
        "date",
        "person_id",
    )
    return lab_token_frame


def convert_units(lab_frame, conversions):
    """
    Given a dataframe of lab test results and another frame of converions for units,
    convert the lab test values in lab_frame and return the dataframe.
    """
    converted_frame = lab_frame.select(
        "person_id",
        "date",
        f.col("lx_result_unit").alias("lx_result_other_unit"),
        "lx_lab_code",
        "lx_result_val",
        "lx_lab_code_type",
    ).join(
        conversions.select(
            "lx_lab_code",
            "lx_result_other_unit",
            "conversion_factor",
            "lx_result_unit",
        ),
        on=["lx_lab_code", "lx_result_other_unit"],
    )
    converted_frame = converted_frame.withColumn(
        "lx_result_val_converted", f.col("lx_result_val") * f.col("conversion_factor")
    )
    converted_frame = converted_frame.dropna(subset=["lx_result_val_converted"])
    return converted_frame


def filter_lab_results(converted_lab_frame, min_code_count, min_unique_values):
    codes_that_occur_enough = (
        converted_lab_frame.groupBy("lx_lab_code")
        .count()
        .filter(f"count > {min_code_count}")
        .select("lx_lab_code")
    )
    codes_that_have_multiple_values = (
        converted_lab_frame.groupBy("lx_lab_code")
        .agg(f.countDistinct("lx_result_val").alias("value_counts"))
        .filter(f"value_counts > {min_unique_values}")
    )
    converted_lab_frame = converted_lab_frame.join(
        codes_that_occur_enough, on="lx_lab_code"
    ).join(codes_that_have_multiple_values, on="lx_lab_code")

    return converted_lab_frame


def calculate_percentiles(converted_lab_frame, percentile_step=10):
    # the percentile step must divide into 100%.
    # This is so that the steps cover the percentages from 0-100.
    assert 100 % percentile_step == 0

    # Calculate the 10% percentile for each group
    quantiles = [
        i / percentile_step for i in range(1, 100 // percentile_step)
    ]  # 0.1, 0.2, ..., 0.9

    percentiles_df = converted_lab_frame.groupBy("lx_lab_code").agg(
        f.expr(
            f'percentile_approx(lx_result_val_converted, array({", ".join(map(str, quantiles))}))'
        ).alias("percentile_boundaries")
    )

    # Split the array into separate columns
    percentile_cols = [
        percentiles_df["percentile_boundaries"][i].alias(
            f"percentile_{(i+1)*percentile_step}"
        )
        for i in range(len(quantiles))
    ]

    result_df = percentiles_df.select(*(percentiles_df.columns + percentile_cols))

    return result_df


def convert_lab_tests_to_percentiles(
    converted_lab_frame, percentile_lookup, percentile_step=10
):
    assert 100 // percentile_step >= 3  # at least three

    classified_df = converted_lab_frame.join(percentile_lookup, on="lx_lab_code")
    calculation = f.when(
        (f.col("lx_result_val_converted") < f.col(f"percentile_{percentile_step}")),
        f"0-{percentile_step}",
    )
    for p in range(2, 100 // percentile_step):
        calculation = calculation.when(
            (
                f.col("lx_result_val_converted")
                < f.col(f"percentile_{p * percentile_step}")
            ),
            f"{(p - 1) * percentile_step}-{p * percentile_step}",
        )
    calculation = calculation.otherwise(f"{100-percentile_step}-100")
    classified_df = classified_df.withColumn("percentile_group", calculation)
    return classified_df


def add_baseline_patient_information(merged_frames, patient_frame):
    #######################################################################################
    # add to the merged frames event tokens for the person, representing features that they
    # have from the beginning.
    # add gender, when it is there

    min_date = merged_frames.groupBy("person_id").agg(f.min("date").alias("first_date"))

    merged_frames = merged_frames.select(
        ["person_id", "general_event_token", "days_since_begin", "date"]
    )

    patient_frame = patient_frame.withColumn("days_since_begin", f.lit(0))
    patient_frame = patient_frame.withColumn(
        "gender_token", f.concat(f.lit("GENDER:"), f.col("px_gender"))
    )
    merged_frames = merged_frames.unionByName(
        patient_frame.filter("px_gender IS NOT NULL").selectExpr(
            "person_id",
            "gender_token as general_event_token",
            "days_since_begin",
            "px_birth_date as date",
        )
    )

    # calculate the age of each person for the first event that appears
    patient_frame = patient_frame.join(
        min_date.select("person_id", "first_date"), on="person_id"
    )
    patient_frame = patient_frame.withColumn(
        "px_age_at_start_years_shifted",
        (f.datediff(f.col("first_date"), f.col("px_birth_date")) / 365).cast("integer"),
    )
    patient_frame = patient_frame.withColumn("days_since_begin", f.lit(0))
    patient_frame = patient_frame.withColumn(
        "age_token",
        f.concat(f.lit("AGE:"), f.col("px_age_at_start_years_shifted")),
    )
    merged_frames = merged_frames.unionByName(
        patient_frame.filter("px_birth_date IS NOT NULL").selectExpr(
            [
                "person_id",
                "age_token as general_event_token",
                "days_since_begin",
                "px_birth_date as date",
            ]
        )
    )

    # add their race:
    patient_frame = patient_frame.withColumn("days_since_begin", f.lit(0))
    patient_frame = patient_frame.withColumn(
        "race_token", f.concat(f.lit("RACE:"), f.col("px_race"))
    )
    merged_frames = merged_frames.unionByName(
        patient_frame.filter("px_race IS NOT NULL").selectExpr(
            [
                "person_id",
                "race_token as general_event_token",
                "days_since_begin",
                "px_birth_date as date",
            ]
        )
    )

    # add their ethnicity:
    patient_frame = patient_frame.withColumn("days_since_begin", f.lit(0))
    patient_frame = patient_frame.withColumn(
        "ethnicity_token", f.concat(f.lit("ETHNICITY:"), f.col("px_ethnicity"))
    )
    merged_frames = merged_frames.union(
        patient_frame.filter("px_ethnicity IS NOT NULL").selectExpr(
            [
                "person_id",
                "ethnicity_token as general_event_token",
                "days_since_begin",
                "px_birth_date as date",
            ]
        )
    )

    # add their region:
    patient_frame = patient_frame.withColumn("days_since_begin", f.lit(0))
    patient_frame = patient_frame.withColumn(
        "region_token", f.concat(f.lit("REGION:"), f.col("px_region"))
    )
    merged_frames = merged_frames.union(
        patient_frame.filter("px_region IS NOT NULL").selectExpr(
            [
                "person_id",
                "region_token as general_event_token",
                "days_since_begin",
                "px_birth_date as date",
            ]
        )
    )

    return merged_frames, patient_frame


def consolidate_into_arrays(merged_frames):
    # flatten the data for each person into arrays sorted by when they occured
    collect_list_columns = [f.col("days_since_begin"), f.col("general_event_token")]
    columns = ["person_id", "day_position_tokens", "sorted_event_tokens"]

    has_date = False
    if "date" in merged_frames.columns:
        has_date = True
        collect_list_columns = [f.col("date")] + collect_list_columns
        columns.append("date")

    result = (
        merged_frames.where(merged_frames.general_event_token.isNotNull())
        .groupBy("person_id")
        .agg(
            f.array_sort(f.collect_list(f.struct(*collect_list_columns))).alias(
                "collected_list"
            )
        )
        .withColumn("day_position_tokens", f.col("collected_list.days_since_begin"))
        .withColumn("sorted_event_tokens", f.col("collected_list.general_event_token"))
    )
    if has_date:
        result = result.withColumn("date", f.col("collected_list.date"))

    result = result.select(*columns)
    return result


def find_random_match(dfA, dfB, matching_columns, id_column="person_id"):
    """
    Given two dataframes dfA and dfB with matching columns,
    randomly match a value of dfB to a dfA that have the same values of matching_columns.
    This can create a control group of patients who match in demographics, but do not share
    the same person id.
    """
    # Define the columns on which you want to join
    dfA = dfA.alias("a")
    dfB = dfB.alias("b")

    conditions = [dfA[c] == dfB[c] for c in matching_columns]
    conditions.append(dfA["person_id"] != dfB["person_id"])
    final_condition = conditions[0]
    for condition in conditions[1:]:
        final_condition = condition & final_condition

    # Perform the join
    joined_df = dfA.join(dfB, final_condition, "inner").select(
        "a.person_id",
        "b.person_id",
        *[f.col("b." + c).alias(c) for c in matching_columns],
    )

    # Define a window specification for randomly ordering rows within each partition
    windowSpec = Window.partitionBy([f.col(c) for c in matching_columns]).orderBy(
        f.rand()
    )

    # Apply the window function to assign a row number based on random ordering
    randomly_ranked_df = joined_df.withColumn(
        "row_number", f.row_number().over(windowSpec)
    )
    randomly_ranked_df = randomly_ranked_df.cache()

    # Filter to keep only the first row within each partition, effectively selecting a random match
    random_match_df = randomly_ranked_df.filter(f.col("row_number") == 1)
    random_match_df = random_match_df.cache()
    randomly_ranked_df.unpersist()

    return random_match_df


def convert_mapping_to_pandas_frame(mapping, column, new_column):
    """
    Given a dictionary mapping, convert it to a pandas dataframe

    Args:
        mapping (dict of str:str): a mapping of values in the column to their new values
        column (str): the column that the mapping is applied to
        new_column (str): the column name for the new values

    Returns:
        A pandas dataframe containing two columns: column and new_column.
        The column contains the original values and the new_columns contains the mappings.
    """
    frame = {key: [] for key in [column, new_column]}
    for key in mapping:
        frame[column].append(key)
        frame[new_column].append(mapping[key])
    frame = pd.DataFrame.from_dict(frame)
    return frame


def convert_to_spark_frame(frame, spark):
    """
    Given a pandas dataframe and a spark context, convert the dataframe to a spark dataframe
    """
    return spark.createDataFrame(frame)


def get_missing_mappings(frame, column, mapping):
    """
    Given a pandas dataframe and a mapping, return any missing mappings.

    Args:
        frame (Pandas DataFrame): A spark dataframe that the mapping is applied to.
            Should contain the column.
        column (str): The column that the mapping is applied to.
        mapping (dict of str:str): A mapping of values in the column to their new values.

    Returns:
        Tuple of (missing_mappings, missing_sources).
            missing_mappings (list of str): The keys of the mapping that are not present in any
                row of the column of the frame.
            missing_sources (list of str): The rows of the frame in the column that do not have
                any key in the mapping.
    """
    all_sources = set(frame.select(column).distinct().toPandas()[column].values)
    all_mappings = set([key for key in mapping])

    missing_mappings = [el for el in all_mappings if el not in all_sources]
    missing_sources = [el for el in all_sources if el not in all_mappings]

    return missing_mappings, missing_sources


def apply_mapping(frame, mapping, column, new_column, spark):
    """
    Given a spark dataframe, take all values in the column, convert them according to the mapping,
    and store the new values in a column named new_column.

    Args:
        frame (Pandas DataFrame): A spark dataframe that the mapping is applied to.
            Should contain the column.
        mapping (dict of str:str): A mapping of values in the column to their new value.
        column (str): The column that the mapping is applied to.
        new_column (str): The column name for the new values.
        spark (SparkContext): A pyspark context, created by
            pyspark.sql.SparkSession.builder.getOrCreate(), for example.

    Returns:
        A dataframe with a new column added with the mapping applied.
        If values in the original dataframe column did not have a key in the mapping,
        they are NULL in the new_column. The missingness of various columns
        is logged to the output for debugging.
    """
    logger.info("Appling the following mapping")
    logger.info(json.dumps(mapping))
    missing_mappings, missing_sources = get_missing_mappings(frame, column, mapping)

    logger.info(
        "The following mappings were present in the mapping, but missing from the dataframe column."
    )
    logger.info(
        "They were: " + json.dumps({key: mapping[key] for key in missing_mappings})
    )
    logger.info("The following column values did not have a mapping in the frame.")
    logger.info("They were: " + str(missing_sources))

    pandas_mapping = convert_mapping_to_pandas_frame(mapping, column, new_column)
    spark_mapping = convert_to_spark_frame(pandas_mapping, spark)
    return frame.join(spark_mapping, on=column, how="left")


def add_lowercase_and_period_codes(all_mappings_to_join):
    # sometimes ICD9 and 10 codes are stored in lowercase and without a period. Add them to the translations frame to cover these cases.
    dot_codes = all_mappings_to_join.filter("CONTAINS(code, '.')")
    no_dot_codes = dot_codes.selectExpr(
        "CUI", "REPLACE(code, '.', '') AS code", "source"
    )
    all_mappings_to_join = all_mappings_to_join.union(no_dot_codes)

    # sometimes codes have their characters lowered. Create lowercase versions of the codes, too.
    lowered_codes = all_mappings_to_join.selectExpr(
        "CUI", "LOWER(code) AS lowered_code", "code", "source"
    ).filter("lowered_code != code")
    lowered_codes = lowered_codes.selectExpr("CUI", "lowered_code as code", "source")
    all_mappings_to_join = all_mappings_to_join.union(lowered_codes)
    return all_mappings_to_join


def _get_umls_translations_dataframe(umls_concepts, umls_sat):
    """
    This function handles a few different cases for codes found in UMLS corresponding to
    prescriptions.

    UMLS concepts and prescriptions are stored in different files: MRCONSO.RRF contains
    everything except NDC code prescriptions.
    MRSAT.RRF contains NDC codes mapped to UMLS concepts.
    """

    # Some NDC codes are stored as less-than 11 digit numbers with hyphens
    # like 123-123-11, for example.
    # There is an 9 digit standard format that we should convert to,
    # in order to be consisten with most EHR data.
    # This SQL string does the conversion by splitting the hyphens, padding by zeros,
    # and dropping the last two numbers (the package codes).
    # This would convert an NDC code like 123-123-11 to "001230123"
    hyphen_selection_string = (
        "LPAD(SPLIT(ATV, '-')[0], 5, '0') || LPAD(SPLIT(ATV, '-')[1], 4, '0')"
    )
    # some NDC codes are already in the 11 or 9 digit format, but we want to keep only the
    # first 9 digits.
    non_hyphen_selection_string = "SUBSTRING(ATV, 1, 9)"

    # select the mappings of NDC codes to UMLS concepts (CUIs are concept identifiers in UMLS).
    # The mappings of NDC codes to UMLS concepts is provided by RXNORM or MTHSPL.
    mrsat_ndc_rxnorm_mappings = umls_sat.select("CUI", "ATN", "SAB", "ATV").filter(
        "ATN == 'NDC' and (SAB== 'MTHSPL' or SAB == 'RXNORM')"
    )

    # Apply the standardization of codes to 9 digit formats as described above.
    mrsat_ndc_rxnorm_mappings = mrsat_ndc_rxnorm_mappings.withColumn(
        "code",
        f.expr(
            f"CASE WHEN CONTAINS(ATV, '-') THEN {hyphen_selection_string} "
            f"ELSE {non_hyphen_selection_string} END"
        ),
    )

    # rename the columns
    umls_concepts_to_join = umls_concepts.selectExpr(
        "CUI", "CODE as code", "SAB as source"
    )

    # rename the columns
    mrsat_ndc_rxnorm_mappings_to_join = mrsat_ndc_rxnorm_mappings.selectExpr(
        "CUI", "code", "'NDC' as source"
    )

    # combine the non-prescription conepts with the other concepts like diagnoses.
    all_mappings_to_join = umls_concepts_to_join.union(
        mrsat_ndc_rxnorm_mappings_to_join
    )

    all_mappings_to_join = add_lowercase_and_period_codes(all_mappings_to_join)

    return all_mappings_to_join


def get_umls_translations_dataframe():
    """
    Create a dataframe from UMLS that can convert data to concepts within UMLS.
    Full implementation is in _get_umls_translations_dataframe, which is used for unit testing.

    Returns:
        A spark dataframe containing colums code, source and CUI. The code is the source code,
        e.g. J45 for asthma in ICD10CM.
        The source is the source of the code, e.g. ICD10CM for the last example.
        The CUI is the UMLS concept identifier that can be used for translation.
        Included in the translations are NDC codes mapped to drug concepts.
        These NDC codes are formatted in the 9-digit format, dropping
        the two-digit package code at the end. e.g. 1234-123 is included as a code
        formatted like 012340123, padded by 0's to reach length 9.
    """
    data_reader = UmlsRRFReader()
    umls_concepts = data_reader.get_spark_frame("MRCONSO.RRF")
    umls_sat = data_reader.get_spark_frame("MRSAT.RRF")
    return _get_umls_translations_dataframe(umls_concepts, umls_sat)


def get_umls_sources(umls_mapping_frame=None):
    """
    Return a list of all sources available for UMLS translation.
    """
    if umls_mapping_frame is None:
        umls_concepts = get_umls_translations_dataframe()
    else:
        umls_concepts = umls_mapping_frame
    sources = sorted(
        list(umls_concepts.select("source").distinct().toPandas()["source"].values)
    )
    return sources


def get_unique_values(frame, column):
    """
    Given a spark dataframe and a column, return a set of the values found in that column.
    """
    return set(frame.select(column).distinct().toPandas()[column].values)


def identify_missing_values(frame1, frame2, column):
    """
    Given two spark dataframes, find the column values in the first
    and missing from the second and vice versa.
    Return these two as a tuple, respectively.
    """
    frame1_unique = get_unique_values(frame1, column)
    frame2_unique = get_unique_values(frame2, column)
    return [el for el in frame1_unique if el not in frame2_unique], [
        el for el in frame2_unique if el not in frame1_unique
    ]


def translate(translation_frame, frame):
    """
    Given a dataframe called translation frame with columns code, source and CUI,
    join it with another dataframe containing a code and source columns.
    This adds to the frame the CUI's needed for translation in UMLS.
    """
    _, missing_from_translation = identify_missing_values(
        translation_frame, frame, "source"
    )
    logger.info(
        "The following sources in the data were missing from the translation frame"
    )
    logger.info(missing_from_translation)
    umls_concepts = frame.join(translation_frame, on=["code", "source"])
    umls_concepts = umls_concepts.dropDuplicates(["code", "source", "CUI"])
    return umls_concepts


def find_matching_translations(
    translation_frame, frame1, frame2, frame1_prefix, frame2_prefix
):
    """
    Join frame1 and frame2 using equivalent codes as identified by the translation frame.
    This could, for example, mean matching J45 (asthma in ICD10CM) to the equivalent code
    in SNOMEDCT_US.

    Args:
        translation_frame (Spark Dataframe): A spark dataframe containing columns code,
        source, and CUI.
            This is used to identify equivalent codes in different ontologies
            (e.g. ICD10CM and SNOMEDCT_US).
        frame1 (Spark Dataframe): A spark dataframe to be matched with frame2. Should contain
            columns for code and source.
        frame2 (Spark Dataframe): A spark dataframe to be matched with frame1. Should contain
            columns for code and source.
        frame1_prefix (str): When a column is shared between both frames (except the CUI), then
            add this prefix to columns from frame1.
        frame2_prefix (str): When a column is shared between both frames (except the CUI), then
            add this prefix to columns from frame2.

    Returns:
        A spark dataframe of all matching translations for codes in frame1 to codes in frame2.
    """
    logger.info("Translating frame1 to UMLS concepts")
    frame1_translated = translate(translation_frame, frame1)
    logger.info("Translating frame 2 to UMLS concepts")
    frame2_translated = translate(translation_frame, frame2)

    frame1_columns = frame1_translated.columns
    frame2_columns = frame2_translated.columns

    matching = list(set(frame1_columns).intersection(set(frame2_columns)))
    if "CUI" in matching:
        matching.remove("CUI")
    frame1_columns = [
        f"{el} as {frame1_prefix}_{el}" if (el in matching) else el
        for el in frame1_columns
    ]
    frame2_columns = [
        f"{el} as {frame2_prefix}_{el}" if (el in matching) else el
        for el in frame2_columns
    ]

    frame1 = frame1_translated.selectExpr(*frame1_columns)
    frame2 = frame2_translated.selectExpr(*frame2_columns)

    return frame1.join(frame2, on="CUI")


def get_umls_translations(frame1, frame2, frame1_prefix, frame2_prefix):
    """
    See the docstring for find_matching_translations.
    This functions performs the same task, but loads a default UMLS translation frame
    provided by get_umls_translations_dataframe.
    """
    translations = get_umls_translations_dataframe()
    return find_matching_translations(
        translations, frame1, frame2, frame1_prefix, frame2_prefix
    )


def derive_umls_translation(data, target_vocab, umls_mapping_frame=None):
    """
    Given two dataframes: the data and the target vocabulary,
    translate the data to the target vocabulary according to UMLS.
    The resulting dataframe will have columns original_source,
    original_code, vocab_source and vocab_coude.
    This encodes the mappings from all original codes and sources
    in the data to those in the target vocabulary.

    Arguments:
        data (spark dataframe): A dataframe with columns source and code.
        These columns include what UMLS source (SAB) a code came from and the corresponding code.

        target_vocab (spark dataframe): A dataframe with columns source and code.
        These columns include what UMLS source (SAB) a code came from and the corresponding code.

        spark (spark context): A spark context used to create intermediate dataframes.
    """
    codes_in_data = data.select("source", "code").distinct()
    codes_in_data_in_vocab = codes_in_data.join(
        target_vocab, how="inner", on=["source", "code"]
    )
    codes_needing_translation = codes_in_data.join(
        target_vocab, how="left_anti", on=["source", "code"]
    )
    n_codes = codes_in_data.count()
    n_needs_translation = codes_needing_translation.count()
    logger.info(
        "This fraction of codes need translation {:.2f}".format(
            n_needs_translation / n_codes
        )
    )
    if umls_mapping_frame is None:
        translations = get_umls_translations(
            codes_needing_translation, target_vocab, "original", "vocab"
        )
    else:
        translations = find_matching_translations(
            umls_mapping_frame,
            codes_needing_translation,
            target_vocab,
            "original",
            "vocab",
        )
    translations = translations.dropDuplicates(["original_source", "original_code"])
    n_with_translation = translations.filter(
        "vocab_code IS NOT NULL and vocab_source IS NOT NULL"
    ).count()
    logger.info(
        "This fraction of codes found a translation to the model's vocabulary {:.2f}".format(
            n_with_translation / n_needs_translation
        )
    )

    codes_in_data_in_vocab = codes_in_data_in_vocab.selectExpr(
        "source as original_source",
        "code as original_code",
        "source as vocab_source",
        "code as vocab_code",
        "vocab_token",
    )
    translations = translations.unionByName(
        codes_in_data_in_vocab, allowMissingColumns=True
    )
    logger.info(
        "This fraction of codes found an equivalent in the model's vocab, either by translation "
        "or direct mapping {:.2f}".format(
            (n_codes - n_needs_translation + n_with_translation) / n_codes
        )
    )
    return translations


def convert_model_vocab_to_frame(vocab_file, spark):
    """
    Convert a list of words in the model's vocabulary to a spark dataframe
    with a source and code column.
    """

    with open(vocab_file) as f:
        vocab = f.readlines()
        vocab = [el.replace("\n", "") for el in vocab]

    vocab_codes = {}
    vocab_codes["source"] = []
    vocab_codes["code"] = []
    vocab_codes["vocab_token"] = []

    for i, word in enumerate(vocab):
        if ":" not in word:
            continue
        assert len(word.split(":")) == 2
        source = word.split(":")[0]
        code = word.split(":")[1]

        vocab_codes["source"].append(source)
        vocab_codes["code"].append(code)
        vocab_codes["vocab_token"].append(word)
    vocab_codes_frame = spark.createDataFrame(pd.DataFrame.from_dict(vocab_codes))
    return vocab_codes_frame
