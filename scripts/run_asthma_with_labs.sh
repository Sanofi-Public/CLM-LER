set -xe

export WANDB_PROJECT=test_train_ehr_clm

input_file=config/data_files_full_example.yaml
dataprocess_config=config/data_preprocess_example.yaml
percentiles_conversion_artifact=train_ehr_clm/clm_ler

s3_path_prefix=s3://some/path


## These jobs created the datasets.
jobname=clm_finetuning_asthma_labs_full_multiclass
preprocessed_dataset=${jobname}_dataset
#python3 -m src.data_processing.preprocess_data --config_file ${dataprocess_config} --input_files_config ${input_file} --output_dest ${s3_path_prefix}/preprocessed_data_${jobname} --output_dataset_name ${preprocessed_dataset} --percentiles_conversions_artifact ${percentiles_conversion_artifact} || exit $?

# Define an array with the possible values
values=("multiclass" "tpp2_to_tpp1", "control_to_tpp")

for value in "${values[@]}"; do
config_file=config/train_asthma_${value}.yaml
jobname=clm_ler_finetuning_${value}

labels_location=s3://some/path/to/labels/
preprocessed_dataset_wlabels=${jobname}_dataset_wlabels
python3 -m src.data_processing.join_labels --input ${WANDB_PROJECT}/${preprocessed_dataset} --labels ${labels_location} --labels_config ${config_file} --output ${s3_path_prefix}/preprocessed_data_wlabels_${jobname} --output_dataset_name ${preprocessed_dataset_wlabels}

model_artifact=train_ehr_clm/clm_ler
output_model_artifact=${jobname}_model
 
split_dataset=${preprocessed_dataset_wlabels}_split
python3 -m src.data_processing.train_valid_test_split_wpretrain --input ${WANDB_PROJECT}/${preprocessed_dataset_wlabels} --training_config ${config_file} --output ${s3_path_prefix}/trained_model/${jobname} --output_dataset_name ${split_dataset} --pretrain_model_path ${model_artifact}:latest

output_model_destination=${s3_path_prefix}/asthma_finetuned_models/${output_model_artifact}
python3 -m src.model_training.train_imbalanced_classifier --training_args ${config_file} --input_data_artifact_name ${split_dataset} --output_model_location ${output_model_destination} --model_artifact_name ${model_artifact} --output_model_artifact_name ${output_model_artifact}

done
