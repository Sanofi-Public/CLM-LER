import os

os.environ["WANDB_PROJECT"] = "train_ehr_clm"

if __name__ == "__main__":
    input_file = "config/data_files_full_example.yaml"
    config_file = "config/data_preprocess_example.yaml"
    job_name = "<insert-job-name>"
    output_artifact_name = f"preprocess_data_{job_name}"
    output_data_directory = f"<some-path>/preprocessed_data_{job_name}"

    # split the data by person id

    # python3 -m src.data_processing.preprocess_data --config_file {config_file} --input_files_config {input_file} --output_dest {output_data_directory} --output_dataset_name {output_artifact_name} || exit $?

    ### split the data according to a pre-made split
    data_artifact = output_artifact_name + ":latest"
    train_config = "config/train.yaml"
    presplit_data_artifact = "full_global_data_split_person_ids:v0"  # an example artifact storing a pre-made person_id split
    split_data_s3_local_base = f"<some-s3-path>/trained_model/{job_name}"
    split_data_s3_local = f"{split_data_s3_local_base}/datasets"
    output_name = f"{job_name}_split"

    # python3 -m src.data_processing.train_valid_test_split --input={data_artifact} --predefined_split={presplit_data_artifact} --training_config={train_config} --output={split_data_s3_local} --output_dataset_name={output_name} || exit $?

    #### Create the vocabulary artifact
    vocab_artifact = f"{job_name}_vocab"
    train_dataset = f"{output_name}_train"
    output_location = f"{split_data_s3_local_base}/vocab.txt"

    # python3 -m src.model_training.prepare_tokenizer --input={train_dataset} --training_config={train_config} --output={output_location} --output_name={vocab_artifact} || exit $?

    # now we have everything we need to train the model. Please do so on an instance with a GPU! :)
    # python3 -m clm_ler.model_training.train --input s3://your/bucket/for/the/data --output s3://bucket/for/your/output --training_config config/train.yaml --tokenizer s3://your-vocab-file-from-the-last-step
