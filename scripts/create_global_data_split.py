import os

os.environ["WANDB_PROJECT"] = "train_ehr_clm"
import config

input_file = "config/data_files_full_example.yaml"
split_config = "config/data_split_config_example.yaml"
job_name = "full_global_data"
patient_data_path = "<some-path-to-patient-table>"

output_data_directory = f"<some-path-to-s3>/global_data_split_${job_name}"
output_model_directory = f"<some-path-to-s3>/trained_model/${job_name}"

split_person_id_artifact_name = f"{job_name}_split_person_ids"

# split the data by person id
# python3 -m src.data_processing.train_valid_test_split --input {patient_data_path} --training_config {split_config} --output {output_model_directory}/split_person_frame --output_dataset_name {split_person_id_artifact_name} || exit $?
