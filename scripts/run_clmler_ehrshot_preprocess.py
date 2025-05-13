# python3 -m clm_ler.data_processing.preprocess_ehrshot_data --ehrshot_data_location <some-s3-path>/ehr_shot_data/EHRSHOT_ASSETS/
# --model_name <some-s3-path>/model_artifacts/model_artifacts.tar.gz --output_dest <some-s3-path>/ehrshot_data_outputs
# --output_dataset_name ehrshot_data_processed_for_model --wandb_off
import os

this_project = "ehrshot_benchmarks"
os.environ["WANDB_PROJECT"] = this_project
from clm_ler.config import config as config

if __name__ == "__main__":

    ehrshot_data_location = " <some-s3-path>/ehr_shot_data/EHRSHOT_ASSETS/"
    model_name = "<some-model-name>"
    model_artifact_name = f"<some-longer-model-name-with-a-path>/{model_name}:v0"

    job_name = f"ehrshot_benchmarks_{model_name}"
    job_directory = f" <some-s3-path>/ehrshot_benchmarking/{job_name}"
    mapping_config = "config/mapping_config_to_clm_ler.yaml"

    output_dest = os.path.join(
        job_directory, f"ehrshot_data_processed_for_{model_name}"
    )
    output_artifact_name = f"preprocessed_data_{job_name}"

    processed_data_directory = output_dest
    processed_data_artifact = f"<some-path-similar-to-model_artifact_name>/{this_project}/{processed_data_artifact}:latest"

    # python3 -m src.data_processing.preprocess_ehrshot_data --ehrshot_data_location {ehrshot_data_location} --output_dest {output_dest} --output_dataset_name {output_artifact_name} --model_name {model_artifact_name} --output_dest {output_dest} --mapping_config {mapping_config} || exit $?

    ## run the second job, which involes producing a labelled dataset
    labelled_data_directory = os.path.join(
        job_directory, f"labelled_data_for_{model_name}"
    )
    labelled_data_artifact = f"labelled_data_for_{job_name}"

    script = f"src/clm_ler/data_processing/preprocess_add_labels_to_ehrshot.py"
    column_config = "config/config_add_labels_translated_data.yaml"

    # python3 -m src.data_processing.preprocess_add_labels_to_ehrshot --ehrshot_data_location {ehrshot_data_location} --output_dest {labelled_data_directory} --output_dataset_name {labelled_data_artifact} --config_file {column_config} --processed_ehrshot_data {processed_data_artifact} || exit $?
